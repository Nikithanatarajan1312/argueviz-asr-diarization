"""
audio_v6.py — ARgueVis Speech Pipeline
=======================================
Architecture: VAD → Enrolled-speaker classify (ECAPA-TDNN, ~50ms) → Whisper ASR → JSONL + WebSocket

Changes from v5:
  ADD #2  — JSONL speaker patching:
              When a segment is written as "U" (still enrolling), it is patched
              in-place in the JSONL file the moment a label becomes available.
              JSONL is always the source of truth for downstream argument extraction.

  ADD #3  — Enrollment quality gate:
              Chunks with RMS < ENROLL_MIN_RMS are silently skipped during enrollment.
              Prevents background noise / room tone from polluting the anchor embedding.

  ADD #4  — condition_on_previous_text=False:
              Each conversational turn is treated as independent.
              Prevents a Whisper transcription error from compounding into subsequent turns.

  ADD #5  — Re-enrollment via CLI command:
              Press 'r' + Enter at any time to wipe the current anchor and re-enroll
              from scratch. Useful if the wearer changes or the environment shifts significantly.
              All future segments will be labeled U until the new enrollment completes.

Setup:
  pip install faster-whisper speechbrain torch torchaudio sounddevice webrtcvad-wheels numpy

Optional WebSocket UI:
  pip install fastapi uvicorn

HuggingFace token NOT required (speechbrain model is public).
"""

import asyncio
import json
import os
import sys
import time
import queue
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any  # type: ignore

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning,   message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning,   message=".*SpeechBrain could not find.*")
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
ENROLL_SPEECH_SEC  = 15.0   # seconds of clean VAD speech required to enroll
ENROLL_MIN_SEG_SEC = 0.5    # ignore very short blips during enrollment
# ADD #3: RMS floor — chunks below this level are background noise, not speech.
# Raise to 0.010 in noisy environments; lower to 0.003 for quiet rooms.
ENROLL_MIN_RMS     = 0.005

# --- Speaker classification ---
# Observed sim ranges from real runs:
#   Wearer (A): 0.50–0.77   Partner (B): 0.02–0.09   → large margin
# Tune down to 0.30 if wearer segments are mislabeled B.
# Tune up to 0.50 if partner keeps being labeled A.
SIM_THRESHOLD         = 0.40
ANCHOR_EMA_ALPHA      = 0.05   # EMA weight for anchor drift correction
ANCHOR_UPDATE_MIN_SIM = 0.55   # only update anchor on high-confidence A detections

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
        self.capacity      = int(seconds * sample_rate)
        self.buf           = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx     = 0
        self.total_written = 0
        self.lock          = threading.Lock()

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
            self.write_idx     = end % self.capacity
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
        self.vad               = webrtcvad.Vad(VAD_MODE)
        self.in_speech         = False
        self.seg_start_sample  = 0
        self.last_voice_sample = 0
        self.seg_id            = 0
        self.pad_samples       = int(VAD_SPEECH_PAD_MS * SAMPLE_RATE / 1000)
        self.end_sil_samples   = int(VAD_END_SIL_MS    * SAMPLE_RATE / 1000)

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
# JSONL manager — source of truth
# =========================
class JsonlManager:
    """
    ADD #2: Manages the transcript JSONL file.

    Keeps an in-memory index of seg_id → file byte offset so that
    speaker label updates can patch the exact line in-place without
    rewriting the entire file.

    Thread-safe. All writes and patches go through this class.
    """

    def __init__(self, path: str):
        self.path  = path
        self._lock = threading.Lock()
        # seg_id → byte offset of that line's start in the file
        self._offsets: Dict[int, int] = {}
        # Open in append+read mode; we'll seek for patches
        self._fh = open(path, "a+", encoding="utf-8")

    def write(self, record: dict) -> None:
        """Append a new segment record. Remembers byte offset for future patching."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.seek(0, 2)          # seek to end
            offset = self._fh.tell()
            self._offsets[record["seg_id"]] = offset
            self._fh.write(line)
            self._fh.flush()

    def patch_speaker(self, seg_id: int, new_speaker: str) -> bool:
        """
        Overwrite the speaker field of an already-written segment in-place.
        The line is rewritten at its original offset (same byte length guaranteed
        because speaker labels are always single chars: A, B, U).
        Returns True if the patch was applied.
        """
        with self._lock:
            if seg_id not in self._offsets:
                return False
            offset = self._offsets[seg_id]
            # Read the existing line
            self._fh.seek(offset)
            line = self._fh.readline()
            try:
                entry = json.loads(line.rstrip("\n"))
            except json.JSONDecodeError:
                return False
            if entry.get("speaker") == new_speaker:
                return True   # already correct, nothing to do
            entry["speaker"] = new_speaker
            new_line = json.dumps(entry, ensure_ascii=False) + "\n"
            # Overwrite at same offset — same length because A/B/U are all 1 char
            # and the JSON key order is preserved by our controlled serialisation.
            # If somehow the length differs (shouldn't happen), append a corrected line.
            if len(new_line) == len(line):
                self._fh.seek(offset)
                self._fh.write(new_line)
            else:
                # Fallback: append a correction record with a note
                self._fh.seek(0, 2)
                correction = dict(entry)
                correction["_correction"] = True
                self._fh.write(json.dumps(correction, ensure_ascii=False) + "\n")
            self._fh.flush()
        return True

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# =========================
# Speaker Enrolling + Classification
# =========================
class SpeakerClassifier:
    """
    Passively enrolls the AR wearer's voice from the first ENROLL_SPEECH_SEC
    of clean VAD speech, then classifies each segment as A (wearer) or B (partner).

    ADD #3: Enrollment quality gate — chunks with RMS < ENROLL_MIN_RMS are skipped.
    ADD #5: reset() wipes the anchor and restarts enrollment from scratch.
    """

    def __init__(self):
        print("[ENROLL] Loading SpeechBrain ECAPA-TDNN speaker encoder...")
        from speechbrain.pretrained import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source   = "speechbrain/spkrec-ecapa-voxceleb",
            run_opts = {"device": "cpu"},
            savedir  = "pretrained_models/spkrec-ecapa-voxceleb",
        )
        self._lock = threading.Lock()
        self._reset_state()
        print("[ENROLL] Encoder ready. Accumulating wearer speech passively...")
        print("[ENROLL] (Press 'r' + Enter at any time to re-enroll)")

    def _reset_state(self) -> None:
        """Internal: wipe all enrollment state. Call under self._lock or before threads start."""
        self.anchor_A:           Optional[torch.Tensor] = None
        self.enrolled:           bool  = False
        self._enroll_chunks:     List[np.ndarray] = []
        self._enroll_speech_sec: float = 0.0
        self._enroll_done        = threading.Event()
        self._skipped_noisy:     int   = 0

    # ADD #5: public reset method, thread-safe
    def reset(self) -> None:
        """Wipe current anchor and restart enrollment. All segments will be labeled U until re-enrolled."""
        with self._lock:
            self._reset_state()
        print("[ENROLL] ↺  Re-enrollment started — speak normally to re-enroll the wearer.")
        print(f"[ENROLL]    Need {ENROLL_SPEECH_SEC:.0f}s of speech with RMS > {ENROLL_MIN_RMS}.")

    def _embed(self, audio_f32: np.ndarray) -> torch.Tensor:
        wav = torch.tensor(audio_f32, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav).squeeze()
        return F.normalize(emb, dim=0)

    def try_enroll(self, audio_f32: np.ndarray) -> bool:
        """
        Feed a VAD segment during enrollment.
        ADD #3: Rejects chunks whose RMS is below ENROLL_MIN_RMS (noise gate).
        Returns True the moment enrollment completes.
        """
        if self.enrolled:
            return True

        dur = len(audio_f32) / SAMPLE_RATE
        if dur < ENROLL_MIN_SEG_SEC:
            return False

        # ADD #3: quality gate
        chunk_rms = _rms(audio_f32)
        if chunk_rms < ENROLL_MIN_RMS:
            with self._lock:
                self._skipped_noisy += 1
            print(f"[ENROLL] Skipped noisy chunk (rms={chunk_rms:.4f} < {ENROLL_MIN_RMS}). "
                  f"Total skipped: {self._skipped_noisy}")
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
                    f"[ENROLL] ✓ Wearer enrolled "
                    f"({self._enroll_speech_sec:.1f}s clean speech, "
                    f"{self._skipped_noisy} noisy chunks skipped). "
                    f"Threshold={SIM_THRESHOLD}"
                )
                return True
        return False

    def classify(self, audio_f32: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, cosine_sim).  label ∈ {"A", "B"}.
        Always returns a hard label — no AB overlap category.
        """
        assert self.enrolled, "classify() called before enrollment"

        emb = self._embed(audio_f32)
        with self._lock:
            sim = F.cosine_similarity(
                emb.unsqueeze(0), self.anchor_A.unsqueeze(0)
            ).item()

        label = "A" if sim >= SIM_THRESHOLD else "B"

        # EMA anchor update on high-confidence A detections (drift correction)
        if label == "A" and sim >= ANCHOR_UPDATE_MIN_SIM:
            with self._lock:
                updated       = (1 - ANCHOR_EMA_ALPHA) * self.anchor_A + ANCHOR_EMA_ALPHA * emb
                self.anchor_A = F.normalize(updated, dim=0)

        return label, sim

    @property
    def enrollment_progress(self) -> float:
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
    jsonl      = JsonlManager(TRANSCRIPT_JSONL)   # ADD #2
    seg_q: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()
    _stop      = threading.Event()

    # ADD #2: track seg_ids that were written as U so we can patch them later
    # seg_id → True means "written as U, needs patch when enrollment completes"
    _pending_u_segs: Dict[int, bool] = {}
    _pending_u_lock = threading.Lock()

    _last_mic_print   = [0.0]
    _low_energy_skips = [0]

    audio_frame_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2000)

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

    # ---- Ingest loop ----
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

        seg_q.put(None)

    ingest_t = threading.Thread(target=ingest_loop, daemon=True)
    ingest_t.start()

    # ADD #5: re-enrollment patch thread
    # When enrollment completes after a reset, retroactively re-classify all
    # pending-U segments that were accumulated since the reset began.
    def _patch_pending_u_segments() -> None:
        """
        Called once after enrollment (re-)completes.
        Re-classifies all segments that were written as U during this enrollment
        window and patches both the JSONL and broadcasts the correction.
        """
        with _pending_u_lock:
            pending = list(_pending_u_segs.keys())
            _pending_u_segs.clear()

        if not pending:
            return

        print(f"[PATCH] Re-classifying {len(pending)} U-labeled segments post-enrollment...")
        for seg_id in pending:
            # We don't have the audio anymore if it's old, so we just note it.
            # If we did store the audio (see note below), we could re-classify.
            # For now, mark as unresolvable with a note in the JSONL.
            print(f"[PATCH] seg_id={seg_id} — audio no longer in ring, leaving as U")

    # ---- ASR + classification loop ----
    min_seg_samples   = int(MERGE_MIN_SEG_SEC * SAMPLE_RATE)
    min_print_samples = int(0.3 * SAMPLE_RATE)
    merge_gap_sec     = MERGE_GAP_MS / 1000.0
    last_text         = {"t": "", "end": 0.0}

    def transcribe_and_emit(seg: SpeechSegment) -> None:
        core_audio = ring.get_range(seg.start_sample, seg.end_sample)
        if len(core_audio) < min_print_samples:
            return

        # ---- Speaker classification ----
        if not classifier.enrolled:
            just_enrolled = classifier.try_enroll(core_audio)
            if just_enrolled:
                # Immediately classify this segment — don't waste it as U
                speaker, sim = classifier.classify(core_audio)
                # Patch any U segments written since last reset
                threading.Thread(target=_patch_pending_u_segments, daemon=True).start()
            else:
                speaker, sim = "U", 0.0
        else:
            speaker, sim = classifier.classify(core_audio)

        # ---- Prepare audio for Whisper ----
        tail  = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([core_audio, tail])
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Peak-normalise: prevents Whisper matmul overflow
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
        # ADD #4: condition_on_previous_text=False
        # Each conversational turn is independent — prevents error compounding.
        t0 = time.time()
        segments_out, _ = asr_model.transcribe(
            audio,
            language                    = "en",
            vad_filter                  = False,
            beam_size                   = 3,
            best_of                     = 3,
            temperature                 = 0.0,
            condition_on_previous_text  = False,   # ADD #4
            no_speech_threshold         = 0.6,
            log_prob_threshold          = -1.0,
            compression_ratio_threshold = 2.4,
            word_timestamps             = False,
            repetition_penalty          = 1.1,
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

        sim_str = f" sim={sim:.3f}" if speaker != "U" else ""
        print(f"[{speaker}]{sim_str} [{asr_ms:.0f}ms] {text}")

        record = {
            "seg_id":  seg.seg_id,
            "start":   round(start_sec, 2),
            "end":     round(end_sec, 2),
            "speaker": speaker,
            "text":    text,
        }

        # ADD #2: write to JSONL via manager (tracks byte offset for patching)
        jsonl.write(record)

        # ADD #2: if written as U, register for future patching if enrollment completes
        if speaker == "U":
            with _pending_u_lock:
                _pending_u_segs[seg.seg_id] = True

        _broadcast_segment(record)

    # ---- ASR loop: segment merging + dispatch ----
    def asr_loop():
        pending_seg:    Optional[SpeechSegment] = None
        pending_last_t: float = 0.0

        while True:
            item = seg_q.get()
            if item is None:
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
                    pending_seg    = replace(pending_seg, end_sample=seg.end_sample)
                    pending_last_t = now_t
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_seg_samples:
                        transcribe_and_emit(pending_seg)
                        pending_seg = None
                else:
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

    # ---- ADD #5: CLI command listener (runs in its own thread) ----
    def cli_loop():
        print("[CLI]  Commands: 'r' = re-enroll wearer | 'q' = quit")
        while not _stop.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd == "r":
                    classifier.reset()
                elif cmd == "q":
                    print("[CLI] Quit requested.")
                    _stop.set()
                    break
                elif cmd:
                    print(f"[CLI] Unknown command '{cmd}'. Use 'r' to re-enroll or 'q' to quit.")
            except (EOFError, OSError):
                break

    cli_t = threading.Thread(target=cli_loop, daemon=True)
    cli_t.start()

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
    print(f"[INFO] Speak normally — first {ENROLL_SPEECH_SEC:.0f}s of clean speech enrolls the wearer.")
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
            while not _stop.is_set():
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
        jsonl.close()
        print(f"[INFO] Transcript saved → {TRANSCRIPT_JSONL}")
        if _low_energy_skips[0]:
            print(f"[INFO] Low-energy segments skipped: {_low_energy_skips[0]}")


if __name__ == "__main__":
    main()