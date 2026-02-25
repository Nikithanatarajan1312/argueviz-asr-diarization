import asyncio
import collections
import json
import os
import time
import queue
import threading
import warnings
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Deque, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any  # type: ignore[misc, assignment]

# Suppress known-harmless dependency warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*SpeechBrain could not find.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

import numpy as np
import torch
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# =========================
# Config
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 20                          # VAD frame size
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
PCM_DTYPE = np.int16
INPUT_DEVICE = None                    # None = system default mic

DROP_TIMEOUT_SEC = 0.02
RING_SECONDS = 90
ASR_WINDOW_SECONDS = 7.0
ASR_HOP_SECONDS = 1.2
ASR_MIN_FINAL_LAG = 1.4

VAD_MODE = 1
VAD_SPEECH_PAD_MS = 80
VAD_END_SIL_MS = 400

# --- Merge short segments (ASR stage) ---
MERGE_GAP_MS = 120
MERGE_MAX_GAP_SAMPLES = int(MERGE_GAP_MS * SAMPLE_RATE / 1000)
MERGE_MIN_SEG_SECONDS = 0.5

# --- ASR model ---
WHISPER_SIZE = "base"
WHISPER_COMPUTE = "int8"

ENABLE_LIVE_ASR = False

# --- Diarization ---
ENABLE_DIAR = True
DIAR_INTERVAL_SEC = 3.0
DIAR_WINDOW_SEC = 25
DIAR_LAG_TOL = 4.0
NUM_SPEAKERS = 2

HF_TOKEN = None  # or os.environ["HF_TOKEN"]

# --- Transcript persistence ---
TRANSCRIPT_JSONL = "transcript_v2.jsonl"

# --- WebSocket ---
WS_PORT = 8000

# --- Labeler retries ---
MAX_LABELER_TRIES = 12
RETRY_BACKOFF = 0.7

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

    app = FastAPI(title="ArgueViz transcript stream", lifespan=_lifespan)

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
        <p>WebSocket transcript at ws://localhost:%d/ws</p>
        </body></html>""" % WS_PORT


# =========================
# Data structures
# =========================
@dataclass
class SpeechSegment:
    seg_id: int
    start_sample: int
    end_sample: int


@dataclass
class DiarSegment:
    seg_id: int
    start_sec: float
    end_sec: float
    speaker: str
    conf: float


@dataclass
class AsrSegment:
    start_sec: float
    end_sec: float
    text: str


# =========================
# Ring buffer
# =========================
class RingBuffer:
    def __init__(self, seconds: int, sample_rate: int):
        self.capacity = int(seconds * sample_rate)
        self.buf = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def append_pcm16(self, pcm: np.ndarray) -> int:
        """Append int16 mono PCM. Returns the global sample index of the
        first sample in `pcm` (i.e. total_written *before* this append)."""
        x = (pcm.astype(np.float32) / 32768.0).reshape(-1)
        n = len(x)
        with self.lock:
            base = self.total_written          # FIX #1: capture before writing
            end = self.write_idx + n
            if end <= self.capacity:
                self.buf[self.write_idx:end] = x
            else:
                first = self.capacity - self.write_idx
                self.buf[self.write_idx:] = x[:first]
                self.buf[:end % self.capacity] = x[first:]
            self.write_idx = end % self.capacity
            self.total_written += n
        return base

    def get_last(self, seconds: float) -> Tuple[np.ndarray, int]:
        """Return (audio_float32, start_sample_global) for last `seconds`."""
        n = int(seconds * SAMPLE_RATE)
        with self.lock:
            n = min(n, min(self.total_written, self.capacity))
            start_global = self.total_written - n
            start_idx = (self.write_idx - n) % self.capacity
            if start_idx + n <= self.capacity:
                out = self.buf[start_idx:start_idx + n].copy()
            else:
                first = self.capacity - start_idx
                out = np.concatenate([
                    self.buf[start_idx:].copy(),
                    self.buf[:n - first].copy()
                ])
        return out, start_global

    def get_range(self, start_global: int, end_global: int) -> np.ndarray:
        """Extract a contiguous slice by global sample indices."""
        with self.lock:
            total = self.total_written
            cap = self.capacity
            # Clamp to what's actually in the buffer
            buf_start = max(0, total - cap)
            s = max(start_global, buf_start)
            e = min(end_global, total)
            if e <= s:
                return np.zeros(0, dtype=np.float32)
            n = e - s
            start_idx = (self.write_idx - (total - s)) % cap
            if start_idx + n <= cap:
                return self.buf[start_idx:start_idx + n].copy()
            else:
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
    """
    Wraps webrtcvad and manages segment boundaries.

    Fix #2: The segmenter no longer controls audio collection directly.
    It returns SpeechSegment with accurate start/end global sample indices,
    and the caller uses RingBuffer.get_range() to extract audio precisely.
    """

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.in_speech = False
        self.seg_start_sample = 0
        self.last_voice_sample = 0
        self.seg_id = 0
        self.pad_samples = int(VAD_SPEECH_PAD_MS * SAMPLE_RATE / 1000)
        self.end_sil_samples = int(VAD_END_SIL_MS * SAMPLE_RATE / 1000)

    def process_frame(
        self, pcm16_frame: np.ndarray, frame_start_sample_global: int
    ) -> Optional[SpeechSegment]:
        is_speech = self.vad.is_speech(pcm16_frame.tobytes(), SAMPLE_RATE)
        frame_end = frame_start_sample_global + len(pcm16_frame)

        if is_speech:
            self.last_voice_sample = frame_end
            if not self.in_speech:
                self.in_speech = True
                # Back-pad segment start
                self.seg_start_sample = max(0, frame_start_sample_global - self.pad_samples)

        if self.in_speech:
            silence_samples = frame_end - self.last_voice_sample
            if silence_samples >= self.end_sil_samples:
                self.in_speech = False
                self.seg_id += 1
                # FIX #3: end_sample is last_voice + forward pad — not double-padded
                end_sample = self.last_voice_sample + self.pad_samples
                seg = SpeechSegment(
                    seg_id=self.seg_id,
                    start_sample=self.seg_start_sample,
                    end_sample=end_sample,
                )
                return seg
        return None


# =========================
# Helpers
# =========================
def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlap_with(prev: List[DiarSegment], s: float, e: float, ab: str) -> float:
    return sum(
        overlap(s, e, d.start_sec, d.end_sec)
        for d in prev if d.speaker == ab
    )


# Rate limiter for noisy debug prints
_last_no_overlap_print: float = 0.0


def best_speaker_for_interval(
    diar: List[DiarSegment], s0: float, s1: float, min_ratio: float = 0.05
) -> str:
    global _last_no_overlap_print
    if not diar:
        return "U"
    dur = max(1e-6, s1 - s0)
    by_spk: dict = {}
    for d in diar:
        ov = overlap(s0, s1, d.start_sec, d.end_sec)
        if ov > 0:
            by_spk[d.speaker] = by_spk.get(d.speaker, 0.0) + ov

    if not by_spk:
        # FIX #8: proper time-based rate limiter instead of float modulo trick
        now = time.time()
        if now - _last_no_overlap_print >= 1.0:
            _last_no_overlap_print = now
            diar_range = f"{diar[0].start_sec:.1f}-{diar[-1].end_sec:.1f}"
            print(f"[OVL] seg {s0:.2f}-{s1:.2f} NO OVERLAP with diar {diar_range}")
        return "U"

    best_spk = max(by_spk.items(), key=lambda x: x[1])
    ratio = best_spk[1] / dur
    print(
        f"[OVL] seg {s0:.2f}-{s1:.2f} dur={dur:.2f} "
        f"overlaps={by_spk} -> {best_spk[0]} (ratio={ratio:.3f})"
    )
    return best_spk[0] if ratio >= min_ratio else "U"


def _extract_annotation(diar_out: Any) -> Any:
    if hasattr(diar_out, "itertracks"):
        return diar_out
    for attr in ("speaker_diarization", "diarization", "annotation"):
        if hasattr(diar_out, attr):
            ann = getattr(diar_out, attr)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
    if isinstance(diar_out, Mapping) and "annotation" in diar_out:
        return diar_out["annotation"]
    if hasattr(diar_out, "__getitem__"):
        try:
            ann = diar_out["annotation"]
            if hasattr(ann, "itertracks"):
                return ann
        except Exception:
            pass
    raise RuntimeError(f"Unknown diar output type: {type(diar_out)}")


# =========================
# Main
# =========================
def main():
    print("[INFO] Loading faster-whisper model...")
    model = WhisperModel(WHISPER_SIZE, compute_type=WHISPER_COMPUTE)

    diar_pipeline = None
    if ENABLE_DIAR:
        try:
            from pyannote.audio import Pipeline
            token = HF_TOKEN or os.environ.get("HF_TOKEN")
            diar_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
            diar_pipeline.to(torch.device("cpu"))
            print("[INFO] PyAnnote diarization loaded.")
        except Exception as e:
            print("[WARN] PyAnnote diarization disabled:", e)
            diar_pipeline = None

    ring = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    seg_q_asr: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()

    diar_segments: List[DiarSegment] = []
    diar_lock = threading.Lock()
    speaker_map_lock = threading.Lock()
    speaker_map: dict = {}

    # FIX #5: last_diar_update is now actually used in labeler gating
    last_diar_update: float = 0.0
    last_diar_update_lock = threading.Lock()

    diar_running = threading.Lock()

    asr_history: List[dict] = []
    asr_hist_lock = threading.Lock()

    # --- Exit signal ---
    _stop_event = threading.Event()

    def _total_overlap_with_label(
        prev: List[DiarSegment], s: float, e: float, label: str
    ) -> float:
        return sum(
            overlap(s, e, d.start_sec, d.end_sec)
            for d in prev if d.speaker == label
        )

    vad = VadSegmenter()

    transcript_file = open(TRANSCRIPT_JSONL, "a", encoding="utf-8")
    transcript_lock = threading.Lock()

    audio_frame_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2000)
    _last_mic_print = [0.0]

    def audio_callback(indata, frames, time_info, status):
        x = indata[:, 0].copy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if time.time() - _last_mic_print[0] >= 1.0:
            _last_mic_print[0] = time.time()
            print(f"[MIC] rms {rms:.4f}")
        pcm = (x * 32767.0).astype(np.int16)
        try:
            audio_frame_q.put(pcm, timeout=DROP_TIMEOUT_SEC)
        except queue.Full:
            pass

    # ---- Ingest loop ----
    # FIX #1: timestamps computed correctly relative to ring's clock.
    # FIX #2: audio extracted from ring by global sample range, not collected inline.
    def ingest_loop():
        leftover = np.zeros((0,), dtype=np.int16)
        first_vad_sent = False

        while not _stop_event.is_set():
            try:
                pcm = audio_frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if pcm is None:
                break

            # Capture base_sample BEFORE appending so leftover lines up correctly
            pcm_base = ring.append_pcm16(pcm)
            # leftover was already in the ring from a previous append; its global
            # start is pcm_base - len(leftover)
            chunk_base = pcm_base - len(leftover)
            chunk = np.concatenate([leftover, pcm])

            idx = 0
            while idx + FRAME_SAMPLES <= len(chunk):
                frame = chunk[idx:idx + FRAME_SAMPLES]
                frame_start_global = chunk_base + idx
                seg = vad.process_frame(frame, frame_start_global)

                if seg is not None:
                    dur_sec = (seg.end_sample - seg.start_sample) / SAMPLE_RATE
                    print(f"[VAD] finalized seg_id={seg.seg_id} dur={dur_sec:.2f}s")
                    seg_q_asr.put(seg)
                    if not first_vad_sent:
                        first_vad_sent = True
                        print("[INFO] First VAD segment sent to ASR.")

                idx += FRAME_SAMPLES
            leftover = chunk[idx:]

        # Flush any open VAD segment on shutdown
        seg_q_asr.put(None)

    ingest_t = threading.Thread(target=ingest_loop, daemon=True)
    ingest_t.start()

    # ---- Diarization loop ----
    def diar_loop():
        nonlocal last_diar_update
        if diar_pipeline is None:
            return
        printed_diar_dir = False

        while not _stop_event.is_set():
            time.sleep(DIAR_INTERVAL_SEC)
            audio, start_global = ring.get_last(DIAR_WINDOW_SEC)
            actual_window_sec = len(audio) / SAMPLE_RATE

            if actual_window_sec < 3.0:
                continue
            if _rms(audio) < 0.001:
                continue
            if not diar_running.acquire(blocking=False):
                continue
            try:
                wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                inp = {"waveform": wav, "sample_rate": SAMPLE_RATE}
                kwargs = {}
                if ring.now_sec() > 12:
                    kwargs["num_speakers"] = NUM_SPEAKERS
                diar = diar_pipeline(inp, **kwargs)

                if not printed_diar_dir:
                    printed_diar_dir = True
                    print("[DIAR DEBUG] dir:", [a for a in dir(diar) if "diar" in a.lower()])

                offset_sec = start_global / SAMPLE_RATE
                ann = _extract_annotation(diar)
                raw = [
                    (turn.start + offset_sec, turn.end + offset_sec, sp)
                    for turn, _, sp in ann.itertracks(yield_label=True)
                ]

                dur: dict = {}
                for s, e, sp in raw:
                    dur[sp] = dur.get(sp, 0.0) + (e - s)
                labels_sorted = sorted(dur.keys(), key=lambda k: dur[k], reverse=True)
                top = labels_sorted[:2]
                print("[DIAR DEBUG] labels:", labels_sorted)

                if len(labels_sorted) < 2:
                    print(
                        "[DIAR] Only 1 speaker detected — need 2 voices for A/B. "
                        "Try 2 people or wait for turn-taking."
                    )
                    print("[DIAR DEBUG] durations:", {k: round(v, 2) for k, v in dur.items()})

                with speaker_map_lock:
                    local_map = dict(speaker_map)
                print("[DIAR DEBUG] speaker_map(before):", local_map)

                with diar_lock:
                    prev = list(diar_segments)

                scores: dict = {}
                for sp in top:
                    regions = [(s, e) for (s, e, x) in raw if x == sp]
                    ovA = sum(overlap_with(prev, s, e, "A") for s, e in regions)
                    ovB = sum(overlap_with(prev, s, e, "B") for s, e in regions)
                    scores[sp] = (ovA, ovB)

                if len(top) == 2:
                    sp0, sp1 = top
                    if sp0 in local_map and sp1 in local_map:
                        new_map = {sp0: local_map[sp0], sp1: local_map[sp1]}
                    else:
                        keep = scores[sp0][0] + scores[sp1][1]
                        swap = scores[sp0][1] + scores[sp1][0]
                        if swap > keep:
                            new_map = {sp0: "B", sp1: "A"}
                        else:
                            new_map = {sp0: "A", sp1: "B"}
                elif len(top) == 1:
                    new_map = {top[0]: local_map.get(top[0], "A")}
                else:
                    new_map = {}
                for sp in labels_sorted[2:]:
                    new_map[sp] = "U"

                # FIX #6: speaker map stability — only lock in assignments once
                # *both* speakers have substantial duration, preventing permanent
                # suppression of "B" when pyannote reuses cluster IDs.
                min_dur_for_lock = 1.0
                both_substantial = (
                    len(top) == 2
                    and dur.get(top[0], 0) >= min_dur_for_lock
                    and dur.get(top[1], 0) >= min_dur_for_lock
                )

                with speaker_map_lock:
                    stable = dict(speaker_map)
                    have_A = any(v == "A" for v in stable.values())
                    have_B = any(v == "B" for v in stable.values())

                    for sp, ab in new_map.items():
                        if sp in stable:
                            # Always allow re-confirmation of existing assignments
                            continue
                        # New label: assign carefully
                        if both_substantial:
                            if ab == "A" and not have_A:
                                stable[sp] = "A"
                                have_A = True
                            elif ab == "B" and not have_B:
                                stable[sp] = "B"
                                have_B = True
                            elif not have_A:
                                # Force assign missing A before anything else
                                stable[sp] = "A"
                                have_A = True
                            elif not have_B:
                                # Force assign missing B
                                stable[sp] = "B"
                                have_B = True
                            else:
                                stable[sp] = ab
                        else:
                            # Not enough audio yet — tentative assignment
                            stable[sp] = ab

                    speaker_map.clear()
                    speaker_map.update(stable)
                    local_map = dict(speaker_map)

                print("[DIAR DEBUG] speaker_map(after):", local_map)

                out = [
                    DiarSegment(
                        seg_id=0,
                        start_sec=s,
                        end_sec=e,
                        speaker=local_map.get(sp, "U"),
                        conf=1.0,
                    )
                    for s, e, sp in raw
                ]

                window_start = start_global / SAMPLE_RATE
                # FIX #9: use actual audio length, not DIAR_WINDOW_SEC constant
                window_end = window_start + actual_window_sec

                with diar_lock:
                    diar_segments[:] = [
                        d for d in diar_segments
                        if d.end_sec <= window_start or d.start_sec >= window_end
                    ]
                    diar_segments.extend(out)
                    diar_segments.sort(key=lambda d: d.start_sec)

                    print(f"[DIAR] Added {len(out)} segments, total: {len(diar_segments)}")

                    cutoff = ring.now_sec() - 60.0
                    before = len(diar_segments)
                    diar_segments[:] = [d for d in diar_segments if d.end_sec >= cutoff]
                    removed = before - len(diar_segments)
                    if removed:
                        print(f"[DIAR] Cutoff removed {removed} segments")
                    if diar_segments:
                        print(
                            f"[DIAR] Timeline: "
                            f"{diar_segments[0].start_sec:.1f}-"
                            f"{diar_segments[-1].end_sec:.1f} "
                            f"({len(diar_segments)} segs)"
                        )

                # FIX #5: update timestamp so labeler can gate on freshness
                with last_diar_update_lock:
                    last_diar_update = time.time()

            except Exception as e:
                print("[DIAR WARN]", type(e).__name__, e)
            finally:
                diar_running.release()

    if ENABLE_DIAR and diar_pipeline is not None:
        diar_t = threading.Thread(target=diar_loop, daemon=True)
        diar_t.start()

    # ---- Labeler loop: retroactively label "U" segments once diar covers them ----
    def labeler_loop() -> None:
        while not _stop_event.is_set():
            time.sleep(0.5)
            with diar_lock:
                dcopy = list(diar_segments)
            if not dcopy:
                continue

            diar_start = dcopy[0].start_sec
            diar_end = dcopy[-1].end_sec

            # FIX #5: gate on diar freshness
            with last_diar_update_lock:
                diar_age = time.time() - last_diar_update
            if diar_age > DIAR_INTERVAL_SEC + DIAR_LAG_TOL:
                continue  # diar is stale; don't label with potentially wrong data

            now = time.time()
            updates = []
            # FIX #10: prune _done segments from asr_history to avoid wasted iterations
            with asr_hist_lock:
                # Remove done entries beyond the 200-entry window (keep recents)
                asr_history[:] = [
                    s for s in asr_history
                    if not s.get("_done") or s.get("speaker") != "U"
                ]
                if len(asr_history) > 200:
                    asr_history[:] = asr_history[-200:]

                for seg in asr_history:
                    if seg["speaker"] != "U":
                        continue
                    if seg.get("_done"):
                        continue
                    if seg["end"] > diar_end:
                        continue
                    if seg["start"] < diar_start:
                        seg["_done"] = True  # too old, diar won't cover it
                        continue
                    if now < seg.get("_next_try", 0):
                        continue
                    seg["_tries"] = seg.get("_tries", 0) + 1
                    spk = best_speaker_for_interval(dcopy, seg["start"], seg["end"], min_ratio=0.05)
                    if spk != "U":
                        seg["speaker"] = spk
                        seg["_done"] = True
                        updates.append({
                            "type": "update",
                            "seg_id": seg["seg_id"],
                            "speaker": spk,
                        })
                    else:
                        seg["_next_try"] = now + RETRY_BACKOFF * seg["_tries"]
                        if seg["_tries"] >= MAX_LABELER_TRIES:
                            seg["_done"] = True  # give up

            for u in updates:
                _broadcast_segment(u)
                print(f"[UPDATE] seg {u['seg_id']} -> {u['speaker']}")
                # Rewrite the JSONL entry
                with transcript_lock:
                    # Re-read, patch, rewrite
                    try:
                        with open(TRANSCRIPT_JSONL, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        with open(TRANSCRIPT_JSONL, "w", encoding="utf-8") as f:
                            for line in lines:
                                try:
                                    entry = json.loads(line)
                                    if entry.get("seg_id") == u["seg_id"]:
                                        entry["speaker"] = u["speaker"]
                                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                except json.JSONDecodeError:
                                    f.write(line)
                    except FileNotFoundError:
                        pass

    labeler_t = threading.Thread(target=labeler_loop, daemon=True)
    labeler_t.start()

    # ---- ASR loop ----
    # FIX #2: audio is extracted from the ring buffer by global sample index.
    # FIX #3: no double-padding (fixed in VadSegmenter).
    # FIX #4: SpeechSegment is never mutated in place; replace() used for merging.
    def asr_loop_segments():
        min_seg_samples = int(MERGE_MIN_SEG_SECONDS * SAMPLE_RATE)
        min_print_samples = int(0.3 * SAMPLE_RATE)
        merge_gap_sec = MERGE_GAP_MS / 1000.0

        pending_seg: Optional[SpeechSegment] = None
        pending_last_t: float = 0.0

        last_text: dict = {"t": "", "end": 0.0}
        _first_segment_logged = False

        # FIX #7: track low-energy skips properly with a total count
        _low_energy_skips = 0

        def transcribe_and_print(seg: SpeechSegment) -> None:
            nonlocal _low_energy_skips

            # FIX #2: extract audio from ring by global sample range
            audio = ring.get_range(seg.start_sample, seg.end_sample)
            if len(audio) < min_print_samples:
                return

            # Append a short silence tail to help Whisper finalize
            tail = int(0.2 * SAMPLE_RATE)
            audio = np.concatenate([audio, np.zeros(tail, dtype=np.float32)])

            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            if not np.isfinite(audio).all():
                return
            if float(np.std(audio)) < 1e-4:
                return

            energy = _rms(audio)
            print(f"[ASR] energy {energy:.4f}")
            if energy < 0.0003:
                _low_energy_skips += 1
                print(
                    f"[DEBUG] Segment skipped (low energy {energy:.4f}). "
                    f"Total skips: {_low_energy_skips}"
                )
                return

            segments_out, _ = model.transcribe(
                audio,
                language="en",
                vad_filter=False,
                beam_size=3,
                best_of=3,
                temperature=0.0,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                word_timestamps=False,
                repetition_penalty=1.1,
            )
            text = " ".join((s.text or "").strip() for s in segments_out).strip()
            if not text:
                return

            start_sec = seg.start_sample / SAMPLE_RATE
            end_sec = seg.end_sample / SAMPLE_RATE

            # De-dup Whisper repeats
            if text == last_text["t"] and (start_sec - last_text["end"]) < 0.8:
                return
            last_text["t"] = text
            last_text["end"] = end_sec

            speaker = "U"
            if ENABLE_DIAR and diar_pipeline is not None:
                with diar_lock:
                    copy_diar = list(diar_segments)
                if copy_diar and copy_diar[-1].end_sec >= start_sec - DIAR_LAG_TOL:
                    speaker = best_speaker_for_interval(
                        copy_diar, start_sec, end_sec, min_ratio=0.05
                    )

            segment = {
                "seg_id": seg.seg_id,
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "speaker": speaker,
                "text": text,
                "_tries": 0,
                "_next_try": time.time() + 1.0,
                "_done": False,
            }

            with asr_hist_lock:
                asr_history.append(segment)
                if len(asr_history) > 200:
                    asr_history[:] = asr_history[-200:]

            pub = {k: v for k, v in segment.items() if not k.startswith("_")}
            with transcript_lock:
                transcript_file.write(
                    json.dumps(pub, ensure_ascii=False) + "\n"
                )
                transcript_file.flush()
            _broadcast_segment(pub)
            print(f"[{speaker}] {text}")

        while True:
            item = seg_q_asr.get()
            if item is None:
                # Flush pending on shutdown
                if pending_seg is not None:
                    transcribe_and_print(pending_seg)
                break

            seg: SpeechSegment = item
            seg_len = seg.end_sample - seg.start_sample

            if seg_len < min_print_samples:
                continue

            if not _first_segment_logged:
                _first_segment_logged = True
                print("[INFO] First speech segment received (VAD + mic OK).")

            now_t = time.time()
            is_short = seg_len < min_seg_samples

            if is_short:
                if pending_seg is None:
                    # FIX #4: store a copy, never mutate the original
                    pending_seg = replace(seg)
                    pending_last_t = now_t
                    continue
                # Try to merge into pending
                if now_t - pending_last_t <= merge_gap_sec:
                    # FIX #4: use replace() to extend end_sample safely
                    pending_seg = replace(pending_seg, end_sample=seg.end_sample)
                    pending_last_t = now_t
                    pending_len = pending_seg.end_sample - pending_seg.start_sample
                    if pending_len >= min_seg_samples:
                        transcribe_and_print(pending_seg)
                        pending_seg = None
                else:
                    # Gap too large: flush old, start new
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_print(pending_seg)
                    pending_seg = replace(seg)
                    pending_last_t = now_t
                continue

            # Long segment
            if pending_seg is not None:
                if now_t - pending_last_t > merge_gap_sec:
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_print(pending_seg)
                    pending_seg = None
                    transcribe_and_print(seg)
                else:
                    # FIX #4: merge via replace()
                    merged = replace(pending_seg, end_sample=seg.end_sample)
                    transcribe_and_print(merged)
                    pending_seg = None
            else:
                transcribe_and_print(seg)

    asr_t = threading.Thread(target=asr_loop_segments, daemon=True)
    asr_t.start()

    # ---- Live ASR (sliding window, optional) ----
    if ENABLE_LIVE_ASR:
        last_printed = ""

        def asr_loop_live():
            nonlocal last_printed
            while not _stop_event.is_set():
                time.sleep(ASR_HOP_SECONDS)
                audio, _ = ring.get_last(ASR_WINDOW_SECONDS)
                if len(audio) < int(1.0 * SAMPLE_RATE):
                    continue
                segs, _ = model.transcribe(
                    audio,
                    language="en",
                    vad_filter=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
                text = " ".join((s.text or "").strip() for s in segs).strip()
                if not text or text == last_printed:
                    continue
                if text.startswith(last_printed):
                    delta = text[len(last_printed):].strip()
                    if delta:
                        print(f"[live] {delta}")
                else:
                    print(f"[live] {text}")
                last_printed = text

        asr_live_t = threading.Thread(target=asr_loop_live, daemon=True)
        asr_live_t.start()

    # ---- WebSocket server ----
    if HAS_FASTAPI:
        def _run_ws_server() -> None:
            uvicorn.run(app, host="0.0.0.0", port=WS_PORT)

        _ws_thread = threading.Thread(target=_run_ws_server, daemon=True)
        _ws_thread.start()
        for _ in range(50):
            if _ws_loop is not None:
                break
            time.sleep(0.05)
        print(f"[INFO] WebSocket transcript at ws://localhost:{WS_PORT}/ws")

    # ---- Device helpers ----
    def find_headphone_device() -> Optional[int]:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                if any(k in dev["name"].lower() for k in ("headphone", "headset")):
                    return i
        return None

    def list_input_devices() -> None:
        print("\n[INFO] Available input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                marker = " (default)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{marker}")
        print()

    print("[INFO] Starting mic stream. Press Ctrl+C to stop.")
    try:
        dev = INPUT_DEVICE
        if isinstance(INPUT_DEVICE, str) and INPUT_DEVICE.lower() in ("headphone", "headset"):
            dev = find_headphone_device()
            if dev is None:
                print("[WARN] No headphone/headset device found, using system default")
                dev = sd.default.device[0]
            else:
                print(f"[INFO] Auto-detected headphone device: {dev}")
        elif INPUT_DEVICE is None:
            dev = sd.default.device[0]

        try:
            info = sd.query_devices(dev, "input")
            print(f"[INFO] Using input device {dev}: {info.get('name')}")
        except Exception as e:
            print("[WARN] Could not query input device info:", e)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=audio_callback,
            device=dev,
        ):
            while True:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        _stop_event.set()

        try:
            audio_frame_q.put_nowait(None)
        except queue.Full:
            pass

        # FIX #11: join ingest before closing file; sentinel already posted above
        ingest_t.join(timeout=2.0)
        # seg_q_asr sentinel is posted by ingest_loop on exit; wait for ASR to drain
        asr_t.join(timeout=3.0)

        with transcript_lock:
            transcript_file.close()


if __name__ == "__main__":
    main()