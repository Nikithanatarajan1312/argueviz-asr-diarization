"""
audio_v5.py — ARgueVis Speech Pipeline
=======================================
Architecture: VAD → Enrolled-speaker classify (ECAPA-TDNN, ~50ms) → Whisper ASR → JSONL + WebSocket

Changes from v4:
  FIX #1  — Post-enrollment segment no longer labeled U:
              try_enroll() returning True immediately triggers classify() on same audio.
  FIX #2  — AB overlap detection removed entirely:
              RMS-based overlap was unreliable (loud ≠ overlap).
              Classifier now always returns hard A/B — dominant speaker wins.
  FIX #3  — Audio peak-normalized before Whisper:
              audio /= (max_abs + 1e-6) eliminates matmul overflow/divide-by-zero warnings.
  FIX #4  — SIM_THRESHOLD raised to 0.40:
              Observed A sims: 0.50–0.77; observed B sims: 0.02–0.09. Large margin.
              Still tunable via config comment.
  FIX #5  — EMA anchor update kept and verified correct (no change needed).

Setup:
  pip install faster-whisper speechbrain torch torchaudio sounddevice webrtcvad-wheels numpy

Optional WebSocket UI:
  pip install fastapi uvicorn

HuggingFace token NOT required (speechbrain model is public).
"""

import asyncio
import json
import time
import queue
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any  # type: ignore

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning,  message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning,  message=".*SpeechBrain could not find.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered.*")

import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# =========================
# Config
# =========================
SAMPLE_RATE       = 16000
CHANNELS          = 1
FRAME_MS          = 20
FRAME_SAMPLES     = SAMPLE_RATE * FRAME_MS // 1000   # 320 samples
INPUT_DEVICE      = None   # None = system default; set to int for specific device

DROP_TIMEOUT_SEC  = 0.02
RING_SECONDS      = 90

# --- VAD ---
VAD_MODE          = 1      # 0–3; 1 = balanced
VAD_SPEECH_PAD_MS = 80
VAD_END_SIL_MS    = 400

# --- Segment merging ---
MERGE_GAP_MS      = 120
MERGE_MIN_SEG_SEC = 0.5

# --- Whisper ---
WHISPER_SIZE      = "base"
WHISPER_COMPUTE   = "int8"   # use "float16" if CUDA available

# --- Enrollment ---
# The AR wearer speaks normally; the first ENROLL_SPEECH_SEC of VAD speech
# is silently accumulated to build their voice embedding (no explicit prompt needed).
ENROLL_SPEECH_SEC  = 15.0   # seconds of VAD speech needed to enroll
ENROLL_MIN_SEG_SEC = 0.5    # ignore very short blips during enrollment

# --- Speaker classification ---
# FIX #4: raised from 0.25 → 0.40 based on observed sim distributions:
#   Wearer (A): 0.50–0.77   Partner (B): 0.02–0.09   → large margin, 0.40 is safe.
# Tune down to 0.30 if wearer's segments are being labeled B.
# Tune up to 0.50 if partner keeps being labeled A.
SIM_THRESHOLD         = 0.40

# EMA anchor update: corrects for gradual mic/environment drift over long sessions.
# FIX #5: kept exactly as-is — design is correct.
ANCHOR_EMA_ALPHA      = 0.05    # weight of new embedding in running average
ANCHOR_UPDATE_MIN_SIM = 0.55   # only update on high-confidence A detections

# --- Output ---
TRANSCRIPT_JSONL  = "transcript.jsonl"
WS_PORT           = 8000

# =========================
# WebSocket state
# =========================
_ws_connections: Set[Any] = set()
_ws_lock = threading.Lock()
_ws_loop: Optional[asyncio.AbstractEventLoop] = None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _broadcast_segment(segment: dict) -> None:
    if not HAS_FASTAPI or _ws_loop is None:
        return

    async def _send_all() -> None:
        dead = []
        with _ws_lock:
            conns_snapshot = list(_ws_connections)
        for ws in conns_snapshot:
            try:
                await ws.send_json(segment)
            except Exception:
                dead.append(ws)
        if dead:
            with _ws_lock:
                for ws in dead:
                    _ws_connections.discard(ws)

    try:
        asyncio.run_coroutine_threadsafe(_send_all(), _ws_loop)
    except Exception:
        pass


if HAS_FASTAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        global _ws_loop
        _ws_loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="ARgueVis transcript stream", lifespan=_lifespan)

    @app.websocket("/ws")
    async def websocket_transcript(websocket: WebSocket) -> None:
        await websocket.accept()
        with _ws_lock:
            _ws_connections.add(websocket)
        try:
            while True:
                await asyncio.sleep(3600)
        except Exception:
            pass
        finally:
            with _ws_lock:
                _ws_connections.discard(websocket)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return """<html><body>
        <p>ARgueVis transcript stream</p>
        <p>WebSocket: <code>ws://localhost:%d/ws</code></p>
        </body></html>""" % WS_PORT


# =========================
# Data structures
# =========================
@dataclass
class SpeechSegment:
    seg_id: int
    start_sample: int
    end_sample: int


# =========================
# Ring buffer
# =========================
class RingBuffer:
    def __init__(self, seconds: int, sample_rate: int):
        self.capacity     = int(seconds * sample_rate)
        self.buf          = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx    = 0
        self.total_written = 0
        self.lock         = threading.Lock()

    def append_pcm16(self, pcm: np.ndarray) -> int:
        """Append int16 mono PCM. Returns global sample index BEFORE this append."""
        x = (pcm.astype(np.float32) / 32768.0).reshape(-1)
        n = len(x)
        with self.lock:
            base = self.total_written
            end  = self.write_idx + n
            if end <= self.capacity:
                self.buf[self.write_idx:end] = x
            else:
                first = self.capacity - self.write_idx
                self.buf[self.write_idx:] = x[:first]
                self.buf[:end % self.capacity] = x[first:]
            self.write_idx    = end % self.capacity
            self.total_written += n
        return base

    def get_range(self, start_global: int, end_global: int) -> np.ndarray:
        """Extract audio slice by global sample indices."""
        with self.lock:
            total     = self.total_written
            cap       = self.capacity
            buf_start = max(0, total - cap)
            s = max(start_global, buf_start)
            e = min(end_global, total)
            if e <= s:
                return np.zeros(0, dtype=np.float32)
            n         = e - s
            start_idx = (self.write_idx - (total - s)) % cap
            if start_idx + n <= cap:
                return self.buf[start_idx:start_idx + n].copy()
            first = cap - start_idx
            return np.concatenate([
                self.buf[start_idx:].copy(),
                self.buf[:n - first].copy()
            ])

    def now_sec(self) -> float:
        with self.lock:
            return self.total_written / SAMPLE_RATE

    def now_sample(self) -> int:
        with self.lock:
            return self.total_written


# =========================
# VAD segmenter
# =========================
class VadSegmenter:
    def __init__(self):
        self.vad              = webrtcvad.Vad(VAD_MODE)
        self.in_speech        = False
        self.seg_start_sample = 0
        self.last_voice_sample = 0
        self.seg_id           = 0
        self.pad_samples      = int(VAD_SPEECH_PAD_MS * SAMPLE_RATE / 1000)
        self.end_sil_samples  = int(VAD_END_SIL_MS    * SAMPLE_RATE / 1000)

    def process_frame(
        self, pcm16_frame: np.ndarray, frame_start_global: int
    ) -> Optional[SpeechSegment]:
        is_speech = self.vad.is_speech(pcm16_frame.tobytes(), SAMPLE_RATE)
        frame_end = frame_start_global + len(pcm16_frame)

        if is_speech:
            self.last_voice_sample = frame_end
            if not self.in_speech:
                self.in_speech        = True
                self.seg_start_sample = max(0, frame_start_global - self.pad_samples)

        if self.in_speech:
            if (frame_end - self.last_voice_sample) >= self.end_sil_samples:
                self.in_speech = False
                self.seg_id   += 1
                return SpeechSegment(
                    seg_id       = self.seg_id,
                    start_sample = self.seg_start_sample,
                    end_sample   = self.last_voice_sample + self.pad_samples,
                )
        return None


# =========================
# Speaker Enrolling + Classification
# =========================
class SpeakerClassifier:
    """
    Passively enrolls the AR wearer's voice from the first ENROLL_SPEECH_SEC
    of VAD speech, then classifies each segment as A (wearer) or B (partner).

    No explicit user prompt needed — enrollment happens silently in the background.

    FIX #2: No AB overlap detection. Classifier always returns a hard A or B
            label (dominant speaker wins). RMS-based overlap detection was
            unreliable since loud speech ≠ overlapping speech.

    FIX #4: SIM_THRESHOLD = 0.40 (raised from 0.25).

    FIX #5: EMA anchor update kept — adapts to mic/room drift over long sessions.
    """

    def __init__(self):
        print("[ENROLL] Loading SpeechBrain ECAPA-TDNN speaker encoder...")
        from speechbrain.pretrained import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source   = "speechbrain/spkrec-ecapa-voxceleb",
            run_opts = {"device": "cpu"},
            savedir  = "pretrained_models/spkrec-ecapa-voxceleb",
        )
        self.anchor_A: Optional[torch.Tensor] = None
        self.enrolled  = False
        self._lock     = threading.Lock()

        # Enrollment accumulation
        self._enroll_chunks:     List[np.ndarray] = []
        self._enroll_speech_sec: float = 0.0
        self._enroll_done        = threading.Event()

        print("[ENROLL] Encoder ready. Accumulating wearer speech passively...")

    # ------------------------------------------------------------------
    def _embed(self, audio_f32: np.ndarray) -> torch.Tensor:
        """Return L2-normalised ECAPA embedding for float32 mono audio."""
        wav = torch.tensor(audio_f32, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav).squeeze()
        return F.normalize(emb, dim=0)

    # ------------------------------------------------------------------
    def try_enroll(self, audio_f32: np.ndarray) -> bool:
        """
        Feed a VAD segment during the enrollment phase.
        Returns True (and sets self.enrolled) the moment enrollment completes.
        Thread-safe.
        """
        if self.enrolled:
            return True

        dur = len(audio_f32) / SAMPLE_RATE
        if dur < ENROLL_MIN_SEG_SEC:
            return False

        with self._lock:
            self._enroll_chunks.append(audio_f32.copy())
            self._enroll_speech_sec += dur

            if self._enroll_speech_sec >= ENROLL_SPEECH_SEC:
                combined      = np.concatenate(self._enroll_chunks)
                self.anchor_A = self._embed(combined)
                self.enrolled = True
                self._enroll_done.set()
                print(
                    f"[ENROLL] ✓ Wearer voice enrolled "
                    f"({self._enroll_speech_sec:.1f}s accumulated). "
                    f"Classification active — threshold={SIM_THRESHOLD}"
                )
                return True
        return False

    # ------------------------------------------------------------------
    def classify(self, audio_f32: np.ndarray) -> Tuple[str, float]:
        """
        Classify segment as A (wearer) or B (partner).
        Returns (label, cosine_similarity_to_anchor).

        FIX #2: Hard A/B only — no AB overlap label.
        """
        assert self.enrolled, "classify() called before enrollment complete"

        emb = self._embed(audio_f32)
        with self._lock:
            sim = F.cosine_similarity(
                emb.unsqueeze(0), self.anchor_A.unsqueeze(0)
            ).item()

        label = "A" if sim >= SIM_THRESHOLD else "B"

        # EMA drift correction: update anchor on high-confidence A detections
        if label == "A" and sim >= ANCHOR_UPDATE_MIN_SIM:
            with self._lock:
                updated       = (1 - ANCHOR_EMA_ALPHA) * self.anchor_A + ANCHOR_EMA_ALPHA * emb
                self.anchor_A = F.normalize(updated, dim=0)

        return label, sim

    # ------------------------------------------------------------------
    @property
    def enrollment_progress(self) -> float:
        """0.0 → 1.0"""
        with self._lock:
            return min(1.0, self._enroll_speech_sec / ENROLL_SPEECH_SEC)


# =========================
# Main
# =========================
def main():
    print("[INFO] Loading Whisper model...")
    asr_model  = WhisperModel(WHISPER_SIZE, compute_type=WHISPER_COMPUTE)
    classifier = SpeakerClassifier()
    ring       = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    seg_q: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()
    _stop      = threading.Event()

    transcript_file = open(TRANSCRIPT_JSONL, "a", encoding="utf-8")
    transcript_lock = threading.Lock()

    audio_frame_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2000)
    _last_mic_print    = [0.0]
    _low_energy_skips  = [0]

    # ---- Audio callback ----
    def audio_callback(indata, frames, time_info, status):
        x = indata[:, 0].copy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1.0, 1.0)
        if time.time() - _last_mic_print[0] >= 1.0:
            _last_mic_print[0] = time.time()
            rms = _rms(x)
            if classifier.enrolled:
                tag = "enrolled"
            else:
                pct = int(classifier.enrollment_progress * 100)
                tag = f"enrolling {pct}%"
            print(f"[MIC] rms={rms:.4f}  [{tag}]")
        pcm = (x * 32767.0).astype(np.int16)
        try:
            audio_frame_q.put(pcm, timeout=DROP_TIMEOUT_SEC)
        except queue.Full:
            pass

    # ---- Ingest loop: mic frames → ring buffer + VAD ----
    vad = VadSegmenter()

    def ingest_loop():
        leftover   = np.zeros(0, dtype=np.int16)
        first_sent = False
        while not _stop.is_set():
            try:
                pcm = audio_frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if pcm is None:
                break

            pcm_base   = ring.append_pcm16(pcm)
            chunk_base = pcm_base - len(leftover)
            chunk      = np.concatenate([leftover, pcm])

            idx = 0
            while idx + FRAME_SAMPLES <= len(chunk):
                frame       = chunk[idx:idx + FRAME_SAMPLES]
                frame_start = chunk_base + idx
                seg         = vad.process_frame(frame, frame_start)
                if seg is not None:
                    dur = (seg.end_sample - seg.start_sample) / SAMPLE_RATE
                    print(f"[VAD] seg_id={seg.seg_id} dur={dur:.2f}s")
                    seg_q.put(seg)
                    if not first_sent:
                        first_sent = True
                        print("[INFO] First VAD segment queued.")
                idx += FRAME_SAMPLES
            leftover = chunk[idx:]

        seg_q.put(None)   # shutdown sentinel

    ingest_t = threading.Thread(target=ingest_loop, daemon=True)
    ingest_t.start()

    # ---- ASR + classification loop ----
    min_seg_samples   = int(MERGE_MIN_SEG_SEC * SAMPLE_RATE)
    min_print_samples = int(0.3 * SAMPLE_RATE)
    merge_gap_sec     = MERGE_GAP_MS / 1000.0
    last_text         = {"t": "", "end": 0.0}

    def transcribe_and_emit(seg: SpeechSegment) -> None:
        # Extract core audio for classification (no silence tail)
        core_audio = ring.get_range(seg.start_sample, seg.end_sample)
        if len(core_audio) < min_print_samples:
            return

        # ---- Speaker classification (before ASR) ----
        if not classifier.enrolled:
            # FIX #1: try_enroll returns True the moment enrollment completes.
            # If it just completed on this very segment, classify immediately
            # instead of labeling U — we no longer waste that segment.
            just_enrolled = classifier.try_enroll(core_audio)
            if just_enrolled:
                speaker, sim = classifier.classify(core_audio)
            else:
                speaker, sim = "U", 0.0
        else:
            speaker, sim = classifier.classify(core_audio)

        # ---- Prepare audio for Whisper ----
        # Append silence tail to help Whisper finalize the last word
        tail  = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([core_audio, tail])
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # FIX #3: peak-normalize before ASR to prevent matmul overflow/divide-by-zero
        peak  = np.max(np.abs(audio))
        audio = audio / (peak + 1e-6)

        if float(np.std(audio)) < 1e-4:
            return

        energy = _rms(audio)
        if energy < 0.0003:
            _low_energy_skips[0] += 1
            print(f"[DEBUG] Low energy skip #{_low_energy_skips[0]} (rms={energy:.5f})")
            return

        # ---- ASR ----
        t0 = time.time()
        segments_out, _ = asr_model.transcribe(
            audio,
            language                   = "en",
            vad_filter                 = False,
            beam_size                  = 3,
            best_of                    = 3,
            temperature                = 0.0,
            condition_on_previous_text = True,
            no_speech_threshold        = 0.6,
            log_prob_threshold         = -1.0,
            compression_ratio_threshold = 2.4,
            word_timestamps            = False,
            repetition_penalty         = 1.1,
        )
        text   = " ".join((s.text or "").strip() for s in segments_out).strip()
        asr_ms = (time.time() - t0) * 1000

        if not text:
            return

        start_sec = seg.start_sample / SAMPLE_RATE
        end_sec   = seg.end_sample   / SAMPLE_RATE

        # De-dup Whisper hallucination repeats
        if text == last_text["t"] and (start_sec - last_text["end"]) < 0.8:
            return
        last_text["t"]   = text
        last_text["end"] = end_sec

        sim_str = f" sim={sim:.3f}" if speaker not in ("U",) else ""
        print(f"[{speaker}]{sim_str} [{asr_ms:.0f}ms] {text}")

        record = {
            "seg_id":  seg.seg_id,
            "start":   round(start_sec, 2),
            "end":     round(end_sec, 2),
            "speaker": speaker,
            "text":    text,
        }

        with transcript_lock:
            transcript_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            transcript_file.flush()

        _broadcast_segment(record)

    # ---- ASR loop: segment merging + dispatch ----
    def asr_loop():
        pending_seg:    Optional[SpeechSegment] = None
        pending_last_t: float = 0.0

        while True:
            item = seg_q.get()
            if item is None:
                # Flush pending segment on shutdown
                if pending_seg is not None:
                    transcribe_and_emit(pending_seg)
                break

            seg     = item
            seg_len = seg.end_sample - seg.start_sample
            now_t   = time.time()

            if seg_len < min_print_samples:
                continue

            is_short = seg_len < min_seg_samples

            if is_short:
                if pending_seg is None:
                    pending_seg    = replace(seg)
                    pending_last_t = now_t
                elif now_t - pending_last_t <= merge_gap_sec:
                    # Merge into pending
                    pending_seg    = replace(pending_seg, end_sample=seg.end_sample)
                    pending_last_t = now_t
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_seg_samples:
                        transcribe_and_emit(pending_seg)
                        pending_seg = None
                else:
                    # Gap too large — flush and start fresh
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_emit(pending_seg)
                    pending_seg    = replace(seg)
                    pending_last_t = now_t
                continue

            # Long segment
            if pending_seg is not None:
                if now_t - pending_last_t > merge_gap_sec:
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_emit(pending_seg)
                    pending_seg = None
                    transcribe_and_emit(seg)
                else:
                    merged = replace(pending_seg, end_sample=seg.end_sample)
                    transcribe_and_emit(merged)
                    pending_seg = None
            else:
                transcribe_and_emit(seg)

    asr_t = threading.Thread(target=asr_loop, daemon=True)
    asr_t.start()

    # ---- WebSocket server ----
    if HAS_FASTAPI:
        def _run_ws():
            uvicorn.run(app, host="0.0.0.0", port=WS_PORT, log_level="error")
        ws_t = threading.Thread(target=_run_ws, daemon=True)
        ws_t.start()
        for _ in range(50):
            if _ws_loop is not None:
                break
            time.sleep(0.05)
        print(f"[INFO] WebSocket at ws://localhost:{WS_PORT}/ws")

    # ---- Input device selection ----
    def find_device(name_hint: str) -> Optional[int]:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
                return i
        return None

    dev = INPUT_DEVICE
    if isinstance(INPUT_DEVICE, str):
        dev = find_device(INPUT_DEVICE)
        if dev is None:
            print(f"[WARN] Device '{INPUT_DEVICE}' not found — using system default")
            dev = sd.default.device[0]
    elif INPUT_DEVICE is None:
        dev = sd.default.device[0]

    try:
        info = sd.query_devices(dev, "input")
        print(f"[INFO] Input device [{dev}]: {info.get('name')}")
    except Exception as e:
        print(f"[WARN] Could not query device info: {e}")

    print("[INFO] Starting mic stream.")
    print(f"[INFO] Speak normally — first {ENROLL_SPEECH_SEC:.0f}s of VAD speech enrolls the wearer.")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = "float32",
            blocksize  = FRAME_SAMPLES,
            callback   = audio_callback,
            device     = dev,
        ):
            while True:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        _stop.set()
        try:
            audio_frame_q.put_nowait(None)
        except queue.Full:
            pass
        ingest_t.join(timeout=2.0)
        asr_t.join(timeout=3.0)
        with transcript_lock:
            transcript_file.close()
        print(f"[INFO] Transcript saved → {TRANSCRIPT_JSONL}")
        if _low_energy_skips[0]:
            print(f"[INFO] Low-energy segments skipped: {_low_energy_skips[0]}")


if __name__ == "__main__":
    main()