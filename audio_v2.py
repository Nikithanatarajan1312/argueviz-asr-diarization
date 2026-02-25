import asyncio
import json
import os
import time
import queue
import threading
import warnings
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any  # type: ignore[misc, assignment]

# Suppress known-harmless dependency warnings (we use in-memory audio; no torchcodec needed)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*SpeechBrain could not find.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor")

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
FRAME_MS = 20                    # VAD frame size
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
PCM_DTYPE = np.int16
INPUT_DEVICE = None  # set to 0, 1, ... for specific device; None = system default mic
# Set to "headphone" or "headset" to auto-detect headphones, or device index number
DROP_TIMEOUT_SEC = 0.02  # block briefly in callback to reduce drops and timestamp drift

RING_SECONDS = 90                # keep last N seconds of audio in RAM
ASR_WINDOW_SECONDS = 7.0         # sliding window for transcription
ASR_HOP_SECONDS = 1.2            # how often we run ASR
ASR_MIN_FINAL_LAG = 1.4          # finalize segments older than this from "now"

VAD_MODE = 1                     # 0..3 (1 = less garbage segments; use 0 if missing speech)
VAD_SPEECH_PAD_MS = 80           # pad around speech boundaries
VAD_END_SIL_MS = 400             # end segment after silence (shorter = less mixing speakers)

# --- Merge short segments (ASR stage) ---
MERGE_GAP_MS = 120
MERGE_MAX_GAP_SAMPLES = int(MERGE_GAP_MS * SAMPLE_RATE / 1000)
MERGE_MIN_SEG_SECONDS = 0.5

# --- ASR model ---
# CPU: base + int8 is a good starting point
WHISPER_SIZE = "base"
WHISPER_COMPUTE = "int8"         # "int8" for CPU, "float16" if CUDA

# --- Live ASR (sliding window): set True for near-real-time + segment output (2x Whisper load)
ENABLE_LIVE_ASR = False

# --- Diarization: PyAnnote (no enrollment, fuse by timestamp overlap) ---
ENABLE_DIAR = True               # set False to skip diarization
# Run diar every N sec on last M sec of audio; more context = more stable
DIAR_INTERVAL_SEC = 3.0          # run PyAnnote every 3s (2s too aggressive if each call takes ~3–8s on CPU)
DIAR_WINDOW_SEC = 25             # last 25s of audio (enough context, stays current)
DIAR_LAG_TOL = 4.0               # seconds grace buffer (CPU diar is slow; don't treat as stale within this)
NUM_SPEAKERS = 2                 # force 2 speakers only after enough audio (see diar_loop)
# HuggingFace token for pyannote models (accept agreement at huggingface.co/pyannote/speaker-diarization-3.1)
HF_TOKEN = None                  # or set os.environ["HF_TOKEN"]

# --- Transcript persistence (JSONL for ArgueVis / structured argument data) ---
TRANSCRIPT_JSONL = "transcript.jsonl"

# --- WebSocket broadcast (Milestone A: stream transcript to UI) ---
WS_PORT = 8000
_ws_connections: Set[Any] = set()
_ws_lock = threading.Lock()
_ws_loop: Optional[asyncio.AbstractEventLoop] = None


def _rms(audio: np.ndarray) -> float:
    """RMS level of audio (float or int); used for gates."""
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
        return """<!DOCTYPE html><html><head><title>ArgueViz</title></head><body>
        <p>WebSocket transcript at <a href="/ws">ws://localhost:%d/ws</a></p>
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
    speaker: str                  # "A" or "B"
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
        self.total_written = 0  # total samples written since start
        self.lock = threading.Lock()

    def append_pcm16(self, pcm: np.ndarray):
        # pcm: int16 mono
        x = (pcm.astype(np.float32) / 32768.0).reshape(-1)
        n = len(x)
        with self.lock:
            end = self.write_idx + n
            if end <= self.capacity:
                self.buf[self.write_idx:end] = x
            else:
                first = self.capacity - self.write_idx
                self.buf[self.write_idx:] = x[:first]
                self.buf[:end % self.capacity] = x[first:]
            self.write_idx = end % self.capacity
            self.total_written += n

    def get_last(self, seconds: float) -> Tuple[np.ndarray, int]:
        """Return (audio_float32, start_sample_index_global) for last `seconds`."""
        n = int(seconds * SAMPLE_RATE)
        with self.lock:
            n = min(n, min(self.total_written, self.capacity))
            start_global = self.total_written - n
            start_idx = (self.write_idx - n) % self.capacity
            if start_idx + n <= self.capacity:
                out = self.buf[start_idx:start_idx + n].copy()
            else:
                first = self.capacity - start_idx
                out = np.concatenate([self.buf[start_idx:].copy(),
                                      self.buf[:n - first].copy()])
        return out, start_global

    def now_sec(self) -> float:
        with self.lock:
            return self.total_written / SAMPLE_RATE

    def now_sample(self) -> int:
        """Current total samples written (same clock as diar uses)."""
        with self.lock:
            return self.total_written

# =========================
# VAD segmenter
# =========================
class VadSegmenter:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.in_speech = False
        self.seg_start_sample = 0
        self.last_voice_sample = 0
        self.seg_id = 0

        self.pad_samples = int(VAD_SPEECH_PAD_MS * SAMPLE_RATE / 1000)
        self.end_sil_samples = int(VAD_END_SIL_MS * SAMPLE_RATE / 1000)

    def process_frame(self, pcm16_frame: np.ndarray, frame_start_sample_global: int) -> Optional[SpeechSegment]:
        # webrtcvad expects 16-bit mono bytes at 8/16/32/48k
        is_speech = self.vad.is_speech(pcm16_frame.tobytes(), SAMPLE_RATE)

        if is_speech:
            self.last_voice_sample = frame_start_sample_global + len(pcm16_frame)

        if not self.in_speech and is_speech:
            self.in_speech = True
            self.seg_start_sample = max(0, frame_start_sample_global - self.pad_samples)

        if self.in_speech:
            # end if enough silence since last voice
            cur_end = frame_start_sample_global + len(pcm16_frame)
            if (cur_end - self.last_voice_sample) >= self.end_sil_samples:
                self.in_speech = False
                self.seg_id += 1
                seg = SpeechSegment(
                    seg_id=self.seg_id,
                    start_sample=self.seg_start_sample,
                    end_sample=min(cur_end + self.pad_samples, frame_start_sample_global + len(pcm16_frame) + self.pad_samples),
                )
                return seg

        return None

# =========================
# Helpers
# =========================
def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlap_with(prev: List[DiarSegment], s: float, e: float, ab: str) -> float:
    tot = 0.0
    for d in prev:
        if d.speaker != ab:
            continue
        tot += overlap(s, e, d.start_sec, d.end_sec)
    return tot


def best_speaker_for_interval(diar: List[DiarSegment], s0: float, s1: float, min_ratio: float = 0.05) -> str:
    if not diar:
        return "U"
    dur = max(1e-6, s1 - s0)
    by_spk = {}
    for d in diar:
        ov = overlap(s0, s1, d.start_sec, d.end_sec)
        if ov > 0:
            by_spk[d.speaker] = by_spk.get(d.speaker, 0) + ov
    if not by_spk:
        # No overlap (rate-limit print to avoid spam)
        if abs((s0 * 10) % 10) < 0.01:  # ~once per second-ish
            diar_range = f"{diar[0].start_sec:.1f}-{diar[-1].end_sec:.1f}" if diar else "empty"
            print(f"[OVL] seg {s0:.2f}-{s1:.2f} NO OVERLAP with diar {diar_range}")
        return "U"
    best_spk = max(by_spk.items(), key=lambda x: x[1])
    ratio = best_spk[1] / dur
    # Always print overlap details to debug why U is returned
    print(f"[OVL] seg {s0:.2f}-{s1:.2f} dur={dur:.2f} overlaps={by_spk} -> {best_spk[0]} (ratio={ratio:.3f}, min={min_ratio:.3f})")
    return best_spk[0] if ratio >= min_ratio else "U"


def _extract_annotation(diar_out: Any) -> Any:
    """
    Handles pyannote outputs across versions:
    - Annotation directly (has .itertracks)
    - DiarizeOutput with various attribute names
    - dict-like / Mapping outputs containing 'annotation'
    - objects supporting __getitem__ with key 'annotation'
    """
    # Already an Annotation
    if hasattr(diar_out, "itertracks"):
        return diar_out

    # common DiarizeOutput attribute names across versions
    for attr in ("speaker_diarization", "diarization", "annotation"):
        if hasattr(diar_out, attr):
            ann = getattr(diar_out, attr)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann

    # Mapping output
    if isinstance(diar_out, Mapping) and "annotation" in diar_out:
        return diar_out["annotation"]

    # Fallback: __getitem__
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

    # PyAnnote diarization (no enrollment; run on sliding window, fuse by overlap)
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
            print("[INFO] PyAnnote diarization loaded (no enrollment).")
        except Exception as e:
            print("[WARN] PyAnnote diarization disabled:", e)
            diar_pipeline = None

    ring = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    seg_q_asr: "queue.Queue[tuple]" = queue.Queue()  # (SpeechSegment, seg_audio or None)

    # Diar segments: list in global time; updated by diar thread
    diar_segments: List[DiarSegment] = []
    diar_lock = threading.Lock()
    speaker_map_lock = threading.Lock()
    speaker_map = {}  # maps pyannote label -> "A"/"B" (stable across updates)
    last_speaker = "A"
    last_speaker_lock = threading.Lock()
    last_diar_update = 0.0
    last_diar_update_lock = threading.Lock()
    diar_running = threading.Lock()
    asr_history: List[dict] = []  # keep last ~N segments for backfill labeling
    asr_hist_lock = threading.Lock()

    def _total_overlap_with_label(prev: List[DiarSegment], s: float, e: float, label: str) -> float:
        tot = 0.0
        for d in prev:
            if d.speaker != label:
                continue
            tot += overlap(s, e, d.start_sec, d.end_sec)
        return tot

    vad = VadSegmenter()

    # JSONL transcript log (structured argument data for ArgueVis)
    transcript_file = open(TRANSCRIPT_JSONL, "a", encoding="utf-8")
    transcript_lock = threading.Lock()

    # audio callback -> frames
    audio_frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

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

    # Consumer thread: write to ring + VAD frames + segment queue
    first_vad_sent = [False]

    def ingest_loop():
        leftover = np.zeros((0,), dtype=np.int16)
        current_seg_audio = []

        while True:
            pcm = audio_frame_q.get()
            if pcm is None:
                break

            ring.append_pcm16(pcm)

            chunk = np.concatenate([leftover, pcm])
            # Use ring's sample clock so VAD segment timestamps match diar
            base_sample = ring.now_sample() - len(chunk)
            idx = 0
            while idx + FRAME_SAMPLES <= len(chunk):
                frame = chunk[idx:idx + FRAME_SAMPLES]
                frame_start = base_sample + idx

                seg = vad.process_frame(frame, frame_start)

                # collect only after VAD has entered speech (process_frame updates vad.in_speech)
                if vad.in_speech:
                    current_seg_audio.append(frame.copy())

                if seg is not None:
                    dur_sec = (seg.end_sample - seg.start_sample) / SAMPLE_RATE
                    print(f"[VAD] finalized seg_id={seg.seg_id} dur={dur_sec:.2f}s")
                    if current_seg_audio:
                        seg_audio = np.concatenate(current_seg_audio).astype(np.float32) / 32768.0
                        tail = int(0.2 * SAMPLE_RATE)
                        seg_audio = np.concatenate([seg_audio, np.zeros(tail, dtype=np.float32)])
                        seg_q_asr.put((seg, seg_audio))
                        if not first_vad_sent[0]:
                            first_vad_sent[0] = True
                            print("[INFO] First VAD segment sent to ASR (transcript + diar will appear next).")
                    else:
                        seg_q_asr.put((seg, None))
                    current_seg_audio = []
                idx += FRAME_SAMPLES

            leftover = chunk[idx:]

    ingest_t = threading.Thread(target=ingest_loop, daemon=True)
    ingest_t.start()

    def diar_loop():
        nonlocal last_diar_update
        if diar_pipeline is None:
            return

        printed_diar_dir = False
        while True:
            time.sleep(DIAR_INTERVAL_SEC)

            audio, start_global = ring.get_last(DIAR_WINDOW_SEC)
            if len(audio) < int(3.0 * SAMPLE_RATE):
                continue
            if _rms(audio) < 0.001:
                continue
            # std gate removed for debugging (relaxed)
            # if float(np.std(audio)) < 1e-5:
            #     continue

            if not diar_running.acquire(blocking=False):
                continue
            try:
                wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                inp = {"waveform": wav, "sample_rate": SAMPLE_RATE}
                kwargs = {}
                # Only force 2 speakers after enough audio; let pyannote infer early to avoid warnings
                if ring.now_sec() > 12:
                    kwargs["num_speakers"] = NUM_SPEAKERS
                diar = diar_pipeline(inp, **kwargs)
                if not printed_diar_dir:
                    printed_diar_dir = True
                    print("[DIAR DEBUG] dir:", [a for a in dir(diar) if "diar" in a.lower()])
                offset_sec = start_global / SAMPLE_RATE

                ann = _extract_annotation(diar)
                raw = []
                for turn, _, sp in ann.itertracks(yield_label=True):
                    raw.append((turn.start + offset_sec, turn.end + offset_sec, sp))
                # --- pick top-2 current speakers by duration ---
                dur = {}
                for s, e, sp in raw:
                    dur[sp] = dur.get(sp, 0.0) + (e - s)
                labels_sorted = sorted(dur.keys(), key=lambda k: dur[k], reverse=True)
                top = labels_sorted[:2]  # we only support A/B
                print("[DIAR DEBUG] labels:", labels_sorted)
                if len(labels_sorted) < 2:
                    print("[DIAR] Only 1 speaker detected in window — need 2 voices for A/B. Try 2 people or wait for turn-taking.")
                print("[DIAR DEBUG] durations:", {k: round(v, 2) for k, v in dur.items()})
                with speaker_map_lock:
                    local_map = dict(speaker_map)
                    print("[DIAR DEBUG] speaker_map(before):", dict(speaker_map))

                with diar_lock:
                    prev = list(diar_segments)  # previous A/B timeline

                # compute overlap scores with previous A/B for each current label
                scores = {}
                for sp in top:
                    regions = [(s, e) for (s, e, x) in raw if x == sp]
                    ovA = sum(overlap_with(prev, s, e, "A") for s, e in regions)
                    ovB = sum(overlap_with(prev, s, e, "B") for s, e in regions)
                    scores[sp] = (ovA, ovB)

                # decide assignment: keep existing if both top-2 already mapped, else overlap-based
                if len(top) == 2:
                    sp0, sp1 = top
                    if sp0 in local_map and sp1 in local_map:
                        new_map = {sp0: local_map[sp0], sp1: local_map[sp1]}
                    else:
                        keep = scores[sp0][0] + scores[sp1][1]   # sp0->A, sp1->B
                        swap = scores[sp0][1] + scores[sp1][0]   # sp0->B, sp1->A
                        if swap > keep:
                            new_map = {sp0: "B", sp1: "A"}
                        else:
                            new_map = {sp0: "A", sp1: "B"}
                elif len(top) == 1:
                    new_map = {top[0]: local_map.get(top[0], "A")}  # keep existing or A
                else:
                    new_map = {}

                # any extra labels => Unknown (don't poison A/B)
                for sp in labels_sorted[2:]:
                    new_map[sp] = "U"

                with speaker_map_lock:
                    # stable map: keep what you already decided
                    stable = dict(speaker_map)

                    # Only force A/B when both labels appear in same run with meaningful duration (>1s)
                    # (avoids mislabeling when pyannote splits one voice into two clusters)
                    min_dur_for_lock = 1.0
                    both_substantial = (
                        len(top) == 2
                        and dur.get(top[0], 0) >= min_dur_for_lock
                        and dur.get(top[1], 0) >= min_dur_for_lock
                    )
                    have_A = any(v == "A" for v in stable.values())
                    have_B = any(v == "B" for v in stable.values())

                    for sp, ab in new_map.items():
                        if sp in stable:
                            continue

                        if both_substantial and have_A and not have_B:
                            stable[sp] = "B"
                            have_B = True
                        elif both_substantial and have_B and not have_A:
                            stable[sp] = "A"
                            have_A = True
                        else:
                            stable[sp] = ab  # fallback to computed mapping

                    speaker_map.clear()
                    speaker_map.update(stable)
                    local_map = dict(speaker_map)   # use this for building diar_segments
                    print("[DIAR DEBUG] speaker_map(after):", dict(speaker_map))

                out = []
                for s, e, sp in raw:
                    lab = local_map.get(sp, "U")
                    out.append(DiarSegment(seg_id=0, start_sec=s, end_sec=e, speaker=lab, conf=1.0))

                with diar_lock:
                    # Replace segments in the window being updated, not just append
                    window_start = start_global / SAMPLE_RATE
                    window_end = window_start + DIAR_WINDOW_SEC

                    # Remove segments that overlap the re-analyzed window (keep only fully outside)
                    diar_segments[:] = [
                        d for d in diar_segments
                        if d.end_sec <= window_start or d.start_sec >= window_end
                    ]
                    diar_segments.extend(out)
                    diar_segments.sort(key=lambda d: d.start_sec)
                    print(f"[DIAR] Added {len(out)} segments, total now: {len(diar_segments)}")

                    # Keep rolling 60s
                    cutoff = ring.now_sec() - 60.0
                    before_cutoff = len(diar_segments)
                    diar_segments[:] = [d for d in diar_segments if d.end_sec >= cutoff]
                    if len(diar_segments) < before_cutoff:
                        print(f"[DIAR] Cutoff removed {before_cutoff - len(diar_segments)} segments (cutoff={cutoff:.1f})")
                    if diar_segments:
                        print(f"[DIAR] Timeline now: {diar_segments[0].start_sec:.1f}-{diar_segments[-1].end_sec:.1f} ({len(diar_segments)} segs)")
                with last_diar_update_lock:
                    last_diar_update = time.time()
            except Exception as e:
                print("[DIAR WARN]", type(e).__name__, e)
            finally:
                diar_running.release()

    if ENABLE_DIAR and diar_pipeline is not None:
        diar_t = threading.Thread(target=diar_loop, daemon=True)
        diar_t.start()

        MAX_LABELER_TRIES = 12   # ~6s at 0.5s interval
        RETRY_BACKOFF = 0.7      # seconds

        def labeler_loop() -> None:
            """Retroactively label recent ASR segments once diarization covers them."""
            while True:
                time.sleep(0.5)

                with diar_lock:
                    dcopy = list(diar_segments)

                if not dcopy:
                    continue

                diar_start = dcopy[0].start_sec
                diar_end = dcopy[-1].end_sec
                now = time.time()

                updates = []
                with asr_hist_lock:
                    for seg in asr_history:
                        if seg["speaker"] != "U":
                            continue
                        if seg.get("_done"):
                            continue
                        if seg["end"] > diar_end:
                            continue
                        if seg["start"] < diar_start:
                            continue
                        if now < seg.get("_next_try", 0):
                            continue

                        seg["_tries"] = seg.get("_tries", 0) + 1
                        spk = best_speaker_for_interval(dcopy, seg["start"], seg["end"], min_ratio=0.05)

                        if spk != "U":
                            seg["speaker"] = spk
                            updates.append({"type": "update", "seg_id": seg["seg_id"], "speaker": spk})
                        else:
                            seg["_next_try"] = now + RETRY_BACKOFF * seg["_tries"]
                            if seg["_tries"] >= MAX_LABELER_TRIES:
                                seg["_done"] = True

                for u in updates:
                    _broadcast_segment(u)
                    print(f"[UPDATE] seg {u['seg_id']} -> {u['speaker']}")

        labeler_t = threading.Thread(target=labeler_loop, daemon=True)
        labeler_t.start()

    # ASR loop: transcribe each VAD segment once (audio supplied from ingest_loop)
    # Merge short segments (< MERGE_MIN_SEG_SECONDS) if next arrives within MERGE_GAP_MS
    def asr_loop_segments():
        pending_audio = None
        pending_start = None
        pending_last_t = 0.0
        merge_gap_sec = MERGE_GAP_MS / 1000.0
        min_seg_samples = int(MERGE_MIN_SEG_SECONDS * SAMPLE_RATE)
        min_print_samples = int(0.3 * SAMPLE_RATE)
        last_text = {"t": "", "end": 0.0}
        _first_segment_logged = False
        _low_energy_skips = 0

        def transcribe_and_print(audio: np.ndarray, _seg) -> None:
            nonlocal last_speaker, _low_energy_skips
            if len(audio) < min_print_samples:
                return

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
                if _low_energy_skips <= 3:
                    print(f"[DEBUG] Segment skipped (low energy {energy:.4f}).")
                return
            segments, _ = model.transcribe(
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
            text = " ".join((s.text or "").strip() for s in segments).strip()
            if not text:
                return
            start_sec = _seg.start_sample / SAMPLE_RATE
            end_sec = _seg.end_sample / SAMPLE_RATE
            # De-dup Whisper repeats (same text within 0.8s)
            if text == last_text["t"] and (start_sec - last_text["end"]) < 0.8:
                return
            last_text["t"] = text
            last_text["end"] = end_sec
            # Fuse with PyAnnote diar by timestamp overlap (no early-exit; assign U if can't overlap)
            speaker = "U"
            if ENABLE_DIAR and diar_pipeline is not None:
                with diar_lock:
                    copy_diar = list(diar_segments)
                if copy_diar and copy_diar[-1].end_sec >= start_sec - DIAR_LAG_TOL:
                    speaker = best_speaker_for_interval(copy_diar, start_sec, end_sec, min_ratio=0.05)
                if ENABLE_DIAR:
                    if speaker == "U":
                        speaker = "U"  # no fallback to last_speaker (prevents A smearing)
                    else:
                        with last_speaker_lock:
                            last_speaker = speaker
            # Persist structured segment for ArgueVis and broadcast to UI
            segment = {
                "seg_id": _seg.seg_id,
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "speaker": speaker,
                "text": text,
                "_tries": 0,
                "_next_try": time.time() + 1.0,  # give diar time to catch up before first retry
            }
            with asr_hist_lock:
                asr_history.append(segment)
                if len(asr_history) > 200:
                    asr_history[:] = asr_history[-200:]
            with transcript_lock:
                transcript_file.write(json.dumps({k: v for k, v in segment.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
                transcript_file.flush()
            _broadcast_segment({k: v for k, v in segment.items() if not k.startswith("_")})
            print(f"[{speaker}] {text}")

        while True:
            item = seg_q_asr.get()
            if item is None:
                # flush pending before exit
                if pending_audio is not None and len(pending_audio) >= min_print_samples:
                    transcribe_and_print(pending_audio, pending_start)
                break
            seg, seg_audio = item
            if seg_audio is None or len(seg_audio) < min_print_samples:
                continue
            if not _first_segment_logged:
                _first_segment_logged = True
                print("[INFO] First speech segment received (VAD + mic OK).")
            now_t = time.time()

            if len(seg_audio) < min_seg_samples:
                # short segment: merge with pending or start new pending
                if pending_audio is None:
                    pending_audio = seg_audio.copy()
                    pending_start = seg
                    pending_last_t = now_t
                    continue
                if now_t - pending_last_t <= merge_gap_sec:
                    pending_audio = np.concatenate([pending_audio, seg_audio])
                    pending_start.end_sample = seg.end_sample
                    pending_last_t = now_t
                    if len(pending_audio) >= min_seg_samples:
                        transcribe_and_print(pending_audio, pending_start)
                        pending_audio = None
                        pending_start = None
                else:
                    if len(pending_audio) >= min_print_samples:
                        transcribe_and_print(pending_audio, pending_start)
                    pending_audio = seg_audio.copy()
                    pending_start = seg
                    pending_last_t = now_t
                continue

            # long segment
            if pending_audio is not None:
                if now_t - pending_last_t > merge_gap_sec and len(pending_audio) >= min_print_samples:
                    transcribe_and_print(pending_audio, pending_start)
                else:
                    pending_audio = np.concatenate([pending_audio, seg_audio])
                    pending_start.end_sample = seg.end_sample
                    transcribe_and_print(pending_audio, pending_start)
                pending_audio = None
                pending_start = None
            else:
                transcribe_and_print(seg_audio, seg)

    asr_t = threading.Thread(target=asr_loop_segments, daemon=True)
    asr_t.start()

    # ---- LIVE ASR (sliding window): optional, 2x Whisper load when enabled ----
    if ENABLE_LIVE_ASR:
        last_printed = ""

        def asr_loop_live():
            nonlocal last_printed
            while True:
                time.sleep(ASR_HOP_SECONDS)
                audio, start_global = ring.get_last(ASR_WINDOW_SECONDS)
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
                if not text:
                    continue
                if text != last_printed:
                    if text.startswith(last_printed):
                        delta = text[len(last_printed):].strip()
                        if delta:
                            print(f"[live] {delta}")
                    else:
                        print(f"[live] {text}")
                    last_printed = text

        asr_live_t = threading.Thread(target=asr_loop_live, daemon=True)
        asr_live_t.start()

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

    # Helper to find headphone/headset devices
    def find_headphone_device():
        """Find input device with 'headphone' or 'headset' in name."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name_lower = dev['name'].lower()
                if 'headphone' in name_lower or 'headset' in name_lower:
                    return i
        return None

    # List available input devices (helpful for debugging)
    def list_input_devices():
        """Print all available input devices."""
        print("\n[INFO] Available input devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default_marker = " (default)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{default_marker}")
        print()

    # Optionally list devices (uncomment to see available devices)
    # list_input_devices()

    print("[INFO] Starting mic stream. Press Ctrl+C to stop.")
    try:
        # Handle device selection
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
            blocksize=FRAME_SAMPLES,  # 20ms blocks
            callback=audio_callback,
            device=dev,
        ):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        try:
            audio_frame_q.put_nowait(None)
        except queue.Full:
            pass
        try:
            seg_q_asr.put_nowait(None)
        except queue.Full:
            pass
        transcript_file.close()
    ingest_t.join(timeout=1.0)
    asr_t.join(timeout=1.0)

if __name__ == "__main__":
    main()