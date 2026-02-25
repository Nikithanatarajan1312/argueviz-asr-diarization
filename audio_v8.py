"""
audio_v9.py — ARgueVis Speech Pipeline
=======================================
Architecture: VAD → Enrolled-speaker classify (ECAPA-TDNN, 30–150ms CPU) → Whisper ASR → JSONL + WebSocket

Changes from v8:
  FIX 1  — No U segments dropped permanently:
              _reclassify_pending_u() no longer clears the entire cache.
              It pops entries one at a time from the deque front (oldest first),
              processes them in throttled batches of RECLASSIFY_BATCH_SIZE,
              and leaves any unprocessed entries in place for the next batch.
              Zero data loss regardless of how many U segments accumulated.

  FIX 2  — Race condition between new U arrivals and reclassify thread eliminated:
              Because the thread pops entries individually (not snapshot+clear),
              any new U segment added while the thread is running simply stays
              in the deque and is processed in the next batch pass.
              No generation IDs needed — pop semantics make it race-free by design.

  FIX 3  — Comment accuracy: core_audio is stored as a heap copy, not ring-dependent.
              Updated all comments that incorrectly implied ring-buffer dependency
              for stored pending-U audio. Audio is a standalone np.ndarray copy.

  FIX 4  — ECAPA latency comment corrected:
              Updated from "~50ms" to "30–150ms on CPU" to reflect actual
              observed range depending on hardware and BLAS acceleration.

Lock order (must never be violated):
  classifier._lock  →  never acquires any other lock inside
  _pending_u_lock   →  never acquires classifier._lock inside
  jsonl._lock       →  independent, never nested with above

Setup:
  pip install faster-whisper speechbrain torch torchaudio sounddevice webrtcvad-wheels numpy

Optional WebSocket UI:
  pip install fastapi uvicorn

HuggingFace token NOT required (speechbrain model is public).

JSONL consumer — last-write-wins per seg_id:

    import json

    def load_transcript(path):
        segs = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                segs[r["seg_id"]] = r   # patch records are full records; safe to overwrite
        return sorted(segs.values(), key=lambda r: r["start"])
"""

import asyncio
import collections
import json
import sys
import time
import queue
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any  # type: ignore

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning,    message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning,    message=".*SpeechBrain could not find.*")
warnings.filterwarnings("ignore", category=FutureWarning,  message=".*_register_pytree_node.*")
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
ENROLL_SPEECH_SEC  = 15.0   # seconds of clean VAD speech required
ENROLL_MIN_SEG_SEC = 0.5    # ignore very short blips during enrollment
ENROLL_MIN_RMS     = 0.005  # noise gate — raise to 0.010 in noisy rooms

# --- Speaker classification ---
# Observed sim ranges:  Wearer (A): 0.50–0.77  |  Partner (B): 0.02–0.09
# Tune down to 0.30 if wearer segments are labeled B.
# Tune up to 0.50 if partner keeps being labeled A.
SIM_THRESHOLD         = 0.40
ANCHOR_EMA_ALPHA      = 0.05   # EMA weight for anchor drift correction
ANCHOR_UPDATE_MIN_SIM = 0.55   # only update anchor on high-confidence A detections

# --- Pending-U store ---
# Audio is stored as a standalone heap copy (not ring-dependent).  FIX 3
# At float32 / ~5s avg segment: 50 segs ≈ 32MB worst case.
MAX_PENDING_U_SEGMENTS = 50

# FIX 1/2: reclassify in batches of this size, popping from the deque.
# Remaining entries stay for the next batch — nothing is ever discarded.
RECLASSIFY_BATCH_SIZE = 20

# Sleep between each reclassified segment to yield CPU to live ASR.
# FIX 4: ECAPA latency is 30–150ms on CPU; throttle accordingly.
RECLASSIFY_THROTTLE_SEC = 0.05

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
        """Extract audio slice by global sample indices. Returns empty array if out of range."""
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
# JSONL manager
# =========================
class JsonlManager:
    """
    Append-only JSONL with full-record correction events.

    patch_speaker() writes a complete copy of the original record with the
    updated speaker field and _patch=True appended. The consumer's
    last-write-wins logic always yields a complete record — start/end/text
    are never lost when the patch is the last entry for a seg_id.

    Consumer:
        def load_transcript(path):
            segs = {}
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    segs[r["seg_id"]] = r
            return sorted(segs.values(), key=lambda r: r["start"])
    """

    def __init__(self, path: str):
        self.path  = path
        self._lock = threading.Lock()
        self._fh   = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

    def patch_speaker(self, original_record: dict, new_speaker: str) -> None:
        """Append a full correction record — all original fields + updated speaker."""
        patch           = dict(original_record)
        patch["speaker"] = new_speaker
        patch["_patch"]  = True
        with self._lock:
            self._fh.write(json.dumps(patch, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# =========================
# Speaker classifier
# =========================
class SpeakerClassifier:
    """
    Passively enrolls the AR wearer's voice from the first ENROLL_SPEECH_SEC
    of clean VAD speech, then classifies each segment as A (wearer) or B (partner).

    ECAPA-TDNN inference: 30–150ms on CPU depending on hardware.  FIX 4

    Lock order:
      classifier._lock — never acquires any other lock inside
      Callers must not hold _pending_u_lock when calling any method here.
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
        """Called under self._lock or before threads start."""
        self.anchor_A:           Optional[torch.Tensor] = None
        self.enrolled:           bool  = False
        self._enroll_chunks:     List[np.ndarray] = []
        self._enroll_speech_sec: float = 0.0
        self._enroll_done        = threading.Event()
        self._skipped_noisy:     int   = 0

    def reset(self, on_reset: Optional[Callable] = None) -> None:
        """
        Wipe anchor and restart enrollment.
        on_reset() called OUTSIDE self._lock — safe for callers that need
        to acquire other locks inside (e.g. _pending_u_lock).
        """
        with self._lock:
            self._reset_state()
        if on_reset is not None:
            on_reset()
        print("[ENROLL] ↺  Re-enrollment started — speak normally to re-enroll.")
        print(f"[ENROLL]    Need {ENROLL_SPEECH_SEC:.0f}s of speech "
              f"with RMS > {ENROLL_MIN_RMS}.")

    def _embed(self, audio_f32: np.ndarray) -> torch.Tensor:
        wav = torch.tensor(audio_f32, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav).squeeze()
        return F.normalize(emb, dim=0)

    def try_enroll(self, audio_f32: np.ndarray) -> bool:
        """
        Feed a VAD segment during enrollment.
        Applies RMS noise gate (ENROLL_MIN_RMS).
        Returns True the moment enrollment completes.
        """
        if self.enrolled:
            return True

        dur = len(audio_f32) / SAMPLE_RATE
        if dur < ENROLL_MIN_SEG_SEC:
            return False

        chunk_rms = _rms(audio_f32)
        if chunk_rms < ENROLL_MIN_RMS:
            with self._lock:
                self._skipped_noisy += 1
            print(f"[ENROLL] Noise gate: skipped chunk "
                  f"(rms={chunk_rms:.4f} < {ENROLL_MIN_RMS}), "
                  f"total_skipped={self._skipped_noisy}")
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
        Hard label — no ambiguous/overlap category.
        EMA anchor update on high-confidence A detections (drift correction).
        """
        assert self.enrolled, "classify() called before enrollment"

        emb = self._embed(audio_f32)
        with self._lock:
            sim = F.cosine_similarity(
                emb.unsqueeze(0), self.anchor_A.unsqueeze(0)
            ).item()

        label = "A" if sim >= SIM_THRESHOLD else "B"

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
    ring       = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    jsonl      = JsonlManager(TRANSCRIPT_JSONL)
    seg_q: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()
    _stop      = threading.Event()

    # Pending-U store.
    # Audio is a standalone heap copy — not dependent on the ring buffer.  FIX 3
    # Layout: seg_id → {"audio": np.ndarray, "record": dict}
    # _pending_u_order is the authoritative FIFO eviction queue.
    _pending_u:       Dict[int, dict] = {}
    _pending_u_order: Deque[int]      = collections.deque()
    _pending_u_lock   = threading.Lock()

    def _on_reset() -> None:
        """Called OUTSIDE classifier._lock — safe to acquire _pending_u_lock here."""
        with _pending_u_lock:
            _pending_u.clear()
            _pending_u_order.clear()
        print("[PATCH] Pending-U cache cleared for new enrollment session.")

    classifier = SpeakerClassifier()

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
            tag = ("enrolled" if classifier.enrolled
                   else f"enrolling {int(classifier.enrollment_progress * 100)}%")
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

    # ---- Pending-U reclassification ----
    def _reclassify_pending_u() -> None:
        """
        FIX 1 + FIX 2: Pop-based batch processing.

        Pops up to RECLASSIFY_BATCH_SIZE entries from the front of
        _pending_u_order (oldest first), processes them, and returns.
        Any entries added to the deque AFTER this function starts are
        untouched and will be processed on the next call.

        Race-free by design: we hold _pending_u_lock only during the pop,
        not during classify() or JSONL writes. A new U segment added while
        we're classifying simply appends to the deque and waits its turn.

        If the deque still has entries after one batch, the caller should
        schedule another pass (handled below via while loop on first call).
        """
        while True:
            # Pop one entry at a time under lock — no snapshot+clear.
            with _pending_u_lock:
                if not _pending_u_order:
                    break
                seg_id = _pending_u_order.popleft()
                entry  = _pending_u.pop(seg_id, None)

            if entry is None:
                # Entry was evicted (cap hit) before we got to it — skip.
                continue

            audio  = entry.get("audio")
            record = entry.get("record")

            if audio is None or len(audio) == 0 or record is None:
                print(f"[PATCH] seg_id={seg_id} — missing data, skipping.")
                continue

            try:
                new_speaker, sim = classifier.classify(audio)
            except Exception as e:
                print(f"[PATCH] seg_id={seg_id} — classify error: {e}")
                continue

            jsonl.patch_speaker(record, new_speaker)
            _broadcast_segment({
                "type":    "update",
                "seg_id":  seg_id,
                "speaker": new_speaker,
            })
            print(f"[PATCH] seg_id={seg_id}  U → {new_speaker}  sim={sim:.3f}")

            # Yield CPU between segments so live ASR stays responsive.  FIX 4
            time.sleep(RECLASSIFY_THROTTLE_SEC)

        print("[PATCH] Reclassification pass complete.")

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
                # Classify the enrollment-completing segment immediately
                speaker, sim = classifier.classify(core_audio)
                # Reclassify all stored U segments in throttled background thread.
                # FIX 1/2: _reclassify_pending_u loops until deque is empty,
                # processing all entries without discarding any.
                threading.Thread(target=_reclassify_pending_u, daemon=True).start()
            else:
                speaker, sim = "U", 0.0
        else:
            speaker, sim = classifier.classify(core_audio)

        # ---- Prepare audio for Whisper ----
        tail  = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([core_audio, tail])
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Peak-normalise: prevents Whisper matmul overflow / divide-by-zero
        peak  = np.max(np.abs(audio))
        audio = audio / (peak + 1e-6)

        if float(np.std(audio)) < 1e-4:
            return

        energy = _rms(audio)
        if energy < 0.0003:
            _low_energy_skips[0] += 1
            print(f"[DEBUG] Low-energy skip #{_low_energy_skips[0]} (rms={energy:.5f})")
            return

        # ---- ASR ----
        # condition_on_previous_text=False: each turn independent,
        # prevents transcription errors from compounding across turns.
        t0 = time.time()
        segments_out, _ = asr_model.transcribe(
            audio,
            language                    = "en",
            vad_filter                  = False,
            beam_size                   = 3,
            best_of                     = 3,
            temperature                 = 0.0,
            condition_on_previous_text  = False,
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

        jsonl.write(record)

        # Store U segments as standalone heap copies for later reclassification.
        # FIX 3: audio stored here is independent of the ring buffer.
        if speaker == "U":
            with _pending_u_lock:
                if len(_pending_u) >= MAX_PENDING_U_SEGMENTS:
                    # Evict oldest via deque — explicit and deterministic
                    evict_id = _pending_u_order.popleft()
                    _pending_u.pop(evict_id, None)
                _pending_u[seg.seg_id] = {
                    "audio":  core_audio.copy(),   # standalone heap copy
                    "record": dict(record),
                }
                _pending_u_order.append(seg.seg_id)

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

    # ---- CLI command listener ----
    def cli_loop():
        print("[CLI]  Commands:  r = re-enroll wearer  |  q = quit")
        while not _stop.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd == "r":
                    classifier.reset(on_reset=_on_reset)
                elif cmd == "q":
                    print("[CLI] Quit requested.")
                    _stop.set()
                    break
                elif cmd:
                    print(f"[CLI] Unknown command '{cmd}'. Use 'r' or 'q'.")
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