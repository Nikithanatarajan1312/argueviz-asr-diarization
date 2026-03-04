"""
audio_v14.py — ARgueVis Speech Pipeline
========================================
Architecture: VAD → Speaker ID (ECAPA-TDNN) → Whisper ASR → LLM repair+label → Link detection → JSONL + WebSocket

Changes from v13:

  NEW 1  — LLM transcript repair (combined with labeling, one API call)
              GPT-4o now returns both label AND cleaned_text in a single call.
              cleaned_text fixes Whisper errors: run-ons, dropped words, garbled
              phrases. Original raw_text is preserved in JSONL for debugging.
              Ollama also returns cleaned_text (quality lower but still useful).
              Patch: turn record gets "text" = cleaned_text, "raw_text" = original.

  NEW 2  — Faster quiet-room enrollment
              ENROLL_SPEECH_SEC reduced to 10s (was 15s) for quiet rooms.
              ENROLL_MIN_SEG_SEC reduced to 0.3s (was 0.5s) — shorter utterances count.
              Both changes mean enrollment completes ~2x faster in typical rooms.

  NEW 3  — Enrollment progress shows cumulative seconds in terminal
              [MIC] line now shows: enrolling 40% (4.1s/10.0s)
              Gives clear feedback that enrollment is progressing.

  NEW 4  — Auto force-enroll after timeout (no more typing 'e')
              If passive enrollment hasn't completed after 45s of wall time,
              the status broadcaster automatically force-enrolls from the ring buffer.
              Works silently — user never needs to type 'e' unless they want to reset.

  NEW 5  — Whisper 'small' model default
              WHISPER_SIZE changed from "base" to "small".
              ~2x slower (~1.5s vs ~0.8s on MacBook Air) but meaningfully better
              transcription, especially for short argument turns.
              Change back to "base" if latency is too high.

  NEW 6  — Improved LLM prompt
              Labels short/unclear turns as "other" instead of defaulting to "claim".
              Reduces false positives on filler words like "right", "okay", "yeah".

  FIX    — Removed duplicate `import sys`

All v13 fixes retained (event contract, AUs, links, turn_conf, LLM ensemble,
U-segment turn fix, VAD cap, ASR guard, vad_filter, session IDs, room prompt).

========================================
WebSocket event contract — last-write-wins per event_id:

  segment   — new transcribed segment           event_id = seg_id
              speaker patch overwrites same row  event_id = seg_id, _patch=True
  turn      — flushed speaker turn + AU fields  event_id = "turn_{id}"
              LLM label+text patch overwrites    event_id = "turn_{id}", _patch=True
  link      — AU relation (support/attack)      event_id = "link_{session}_{turn_num}"
  status    — enrollment progress, RMS, mode    event_id = "status"

Turn record fields:
  text       — raw Whisper output (until LLM patch arrives)
  cleaned_text — LLM-repaired text (in patch, replaces text)
  raw_text   — original Whisper output preserved for debugging
  label      — argument role (rule-based initially, LLM patch async)
  confidence — "high"/"medium"/"low"/"rule"
  turn_conf  — mean speaker sim score for this turn

Frontend logic: events[msg.event_id] = msg   // patches are free.

========================================
Setup:
  pip install faster-whisper speechbrain torch torchaudio sounddevice webrtcvad-wheels numpy fastapi uvicorn

LLM ensemble (optional but recommended):
  pip install openai
  brew install ollama && ollama pull llama3.2:3b && ollama serve
  export OPENAI_API_KEY=sk-...

JSONL consumer — last-write-wins per event_id:
  records = {}
  for line in open("turns_v1.jsonl"):
      r = json.loads(line)
      records[r["event_id"]] = r
"""

import asyncio
import collections
import json
import os
import re
import sys
import time
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

if sys.version_info >= (3, 12):
    raise RuntimeError(
        "Use Python 3.10/3.11 for SpeechBrain+torchaudio. "
        "Activate conda env: conda activate argueviz"
    )

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
warnings.filterwarnings("ignore", category=FutureWarning,  message=".*weights_only.*")
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
# Session ID
# =========================
SESSION_ID = int(time.time())

# =========================
# Config — overridden by prompt_room_config()
# =========================
SAMPLE_RATE        = 16000
CHANNELS           = 1
FRAME_MS           = 20
FRAME_SAMPLES      = SAMPLE_RATE * FRAME_MS // 1000
INPUT_DEVICE       = None
DROP_TIMEOUT_SEC   = 0.02
RING_SECONDS       = 90

VAD_MODE           = 1
VAD_SPEECH_PAD_MS  = 80
VAD_END_SIL_MS     = 400
VAD_MAX_SEG_SEC    = 8.0

MERGE_GAP_MS       = 120
MERGE_MIN_SEG_SEC  = 0.5

WHISPER_SIZE       = "small"   # NEW 5: better transcription quality
WHISPER_COMPUTE    = "int8"
NO_SPEECH_THRESH   = 0.6

ENROLL_SPEECH_SEC  = 10.0    # NEW 2: was 15.0
ENROLL_MIN_SEG_SEC = 0.3     # NEW 2: was 0.5
ENROLL_MIN_RMS     = 0.005

SIM_THRESHOLD         = 0.40
ANCHOR_EMA_ALPHA      = 0.05
ANCHOR_UPDATE_MIN_SIM = 0.55

MAX_PENDING_U_SEGMENTS  = 50
RECLASSIFY_BATCH_SIZE   = 20
RECLASSIFY_THROTTLE_SEC = 0.05

# --- AU linking ---
LINK_WINDOW_SEC = 20.0

# --- LLM ensemble ---
LLM_MODEL_OPENAI  = "gpt-4o"
LLM_MODEL_OLLAMA  = "llama3.2:3b"
LLM_TIMEOUT_SEC   = 8.0
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

# NEW 4: auto force-enroll after this many seconds of wall time
ENROLL_AUTO_FORCE_SEC = 45.0

# --- Output ---
TRANSCRIPT_JSONL  = "transcript_v2.jsonl"
TURNS_JSONL       = "turns_v1.jsonl"
LINKS_JSONL       = "links_v1.jsonl"
WS_PORT           = 8000


# =========================
# Room config prompt
# =========================
def prompt_room_config() -> None:
    global VAD_MODE, VAD_END_SIL_MS, VAD_SPEECH_PAD_MS
    global ENROLL_MIN_RMS, ENROLL_MIN_SEG_SEC, ENROLL_SPEECH_SEC

    print("\n[SETUP] Room environment:")
    print("  1 = Quiet      (office, bedroom, library)")
    print("  2 = Noisy      (cafe, open office, HVAC)")
    print("  3 = Very noisy (loud background, multiple people)")
    print()
    while True:
        choice = input("[SETUP] Enter 1, 2, or 3: ").strip()
        if choice == "1":
            VAD_MODE, VAD_END_SIL_MS, VAD_SPEECH_PAD_MS = 1, 400, 80
            ENROLL_MIN_RMS, ENROLL_MIN_SEG_SEC, ENROLL_SPEECH_SEC = 0.005, 0.3, 10.0  # NEW 2
            print("[SETUP] ✓ Quiet — passive enrollment, speak normally (~10s).\n"); break
        elif choice == "2":
            VAD_MODE, VAD_END_SIL_MS, VAD_SPEECH_PAD_MS = 1, 400, 80
            ENROLL_MIN_RMS, ENROLL_MIN_SEG_SEC, ENROLL_SPEECH_SEC = 0.012, 0.8, 15.0
            print("[SETUP] ✓ Noisy — speak clearly ~15s. Auto-enrolls after 45s.\n"); break
        elif choice == "3":
            VAD_MODE, VAD_END_SIL_MS, VAD_SPEECH_PAD_MS = 1, 400, 80
            ENROLL_MIN_RMS, ENROLL_MIN_SEG_SEC, ENROLL_SPEECH_SEC = 0.020, 1.0, 15.0
            print("[SETUP] ✓ Very noisy — auto-enrolls after 45s or type 'e'.\n"); break
        else:
            print("[SETUP] Please enter 1, 2, or 3.")


# =========================
# Turn + AU dataclass
# =========================
TURN_GAP_SEC = 2.0
TURN_MAX_SEC = 25.0

AU_LABELS = {"claim", "premise", "counterclaim", "rebuttal", "question", "other"}


@dataclass
class Turn:
    turn_id:    int
    speaker:    str
    start:      float
    end:        float
    text:       str
    seg_ids:    List[str]   = field(default_factory=list)
    label:      str         = "pending"
    sim_scores: List[float] = field(default_factory=list)


# =========================
# Rule-based AU classifier (instant, zero latency)
# =========================
def classify_turn_rule_based(text: str) -> str:
    t = text.lower().strip()
    if len(t.split()) < 4:
        return "other"   # NEW 6: short turns default to other, not claim
    if "?" in t:
        return "question"
    if any(w in t for w in ["because", "since", "therefore", "so ", "thus", "hence"]):
        return "premise"
    if any(w in t for w in ["but", "however", "although", "though", "yet", "despite",
                             "on the other hand", "disagree", "not true"]):
        return "counterclaim"
    if any(w in t for w in ["actually", "in fact", "that's wrong", "incorrect",
                             "not really", "i disagree"]):
        return "rebuttal"
    return "claim"


# =========================
# LLM classifiers — NEW 1: combined label + transcript repair
# =========================

# NEW 6: improved prompt — labels short/filler turns as "other"
# NEW 1: returns both label AND cleaned_text in one call
_LLM_PROMPT_SYSTEM = """\
You are an argument structure assistant for a live conversation transcript.
You receive a raw speech-to-text turn that may have transcription errors.

Return ONLY valid JSON with exactly these two keys:
  "label":        one of ["claim","premise","counterclaim","rebuttal","question","other"]
  "cleaned_text": the turn rewritten as a clean, grammatical sentence

Label definitions:
  claim        = main assertion or position
  premise      = reason or evidence supporting a claim
  counterclaim = opposing position from the other speaker's view
  rebuttal     = direct correction of what was just said
  question     = asking for information or clarification
  other        = filler, acknowledgment, unclear, or fewer than 4 meaningful words

Transcript repair rules:
  - Fix obvious speech-to-text errors (homophones, run-ons, dropped words)
  - Keep the meaning and wording as close to original as possible
  - Do NOT add new content or change the argument
  - If the turn is already clean, return it unchanged
  - If the turn is pure filler ("yeah", "okay", "right"), keep as-is

Example input:  "i think the problem is is that we don't have enough time because of the the deadline"
Example output: {"label":"premise","cleaned_text":"I think the problem is that we don't have enough time because of the deadline."}

No explanation. No extra keys. JSON only.\
"""


def _extract_json(raw: str) -> Optional[dict]:
    """
    FIX 3: robust JSON extraction from LLM output.
    1. Strip markdown code fences (Ollama often wraps in ```json ... ```)
    2. Find first '{' then walk forward counting braces to find matching '}'
       This handles nested braces correctly and ignores any preamble text.
    """
    if not raw:
        return None
    # Strip code fences
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    # Find first opening brace
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_llm_result(raw: str) -> Optional[Dict[str, str]]:
    """Parse label + cleaned_text from raw LLM output string."""
    obj = _extract_json(raw)
    if not obj:
        return None
    label        = str(obj.get("label", "")).strip().lower()
    cleaned_text = str(obj.get("cleaned_text", "")).strip()
    if label in AU_LABELS and cleaned_text:
        return {"label": label, "cleaned_text": cleaned_text}
    return None



def _classify_openai(text: str) -> Optional[Dict[str, str]]:
    """Call GPT-4o. Returns {"label": ..., "cleaned_text": ...} or None."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SEC)
        resp = client.chat.completions.create(
            model    = LLM_MODEL_OPENAI,
            messages = [
                {"role": "system", "content": _LLM_PROMPT_SYSTEM},
                {"role": "user",   "content": text},
            ],
            max_tokens  = 120,
            temperature = 0.0,
        )
        return _parse_llm_result(resp.choices[0].message.content)
    except ImportError:
        print("[LLM] openai not installed — run: pip install openai")
        return None
    except Exception as e:
        print(f"[LLM] OpenAI error: {e}")
        return None


def _classify_ollama(text: str) -> Optional[Dict[str, str]]:
    """Call local llama via Ollama REST API. Returns {"label": ..., "cleaned_text": ...} or None.
    FIX B: uses _parse_llm_result which extracts first {...} block, handling Ollama's extra text.
    """
    try:
        import urllib.request
        payload = json.dumps({
            "model":  LLM_MODEL_OLLAMA,
            "prompt": f"{_LLM_PROMPT_SYSTEM}\n\nTurn: {text}",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SEC) as r:
            result = json.loads(r.read())
        return _parse_llm_result(result.get("response", ""))
    except Exception as e:
        print(f"[LLM] Ollama error: {e}")
        return None


def classify_turn_ensemble(text: str) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Returns:
      final_label, final_cleaned_text, confidence, meta

    meta includes:
      gpt_label, gpt_cleaned_text
      llama_label, llama_cleaned_text
      rule_label
    """
    rule_label = classify_turn_rule_based(text)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_openai = ex.submit(_classify_openai, text)
        fut_ollama = ex.submit(_classify_ollama, text)
        try:
            gpt_result = fut_openai.result(timeout=LLM_TIMEOUT_SEC)
        except Exception:
            gpt_result = None
        try:
            llama_result = fut_ollama.result(timeout=LLM_TIMEOUT_SEC)
        except Exception:
            llama_result = None

    gpt_label   = gpt_result["label"] if gpt_result else None
    llama_label = llama_result["label"] if llama_result else None

    gpt_cleaned   = gpt_result["cleaned_text"] if gpt_result else None
    llama_cleaned = llama_result["cleaned_text"] if llama_result else None

    # final cleaned text preference: GPT > llama > original
    final_cleaned = gpt_cleaned or llama_cleaned or text

    # final label + confidence logic
    if gpt_label and llama_label:
        if gpt_label == llama_label:
            final_label = gpt_label
            confidence = "high"
        else:
            final_label = gpt_label  # trust GPT on disagreement
            confidence = "low"
    elif gpt_label:
        final_label = gpt_label
        confidence = "medium"
    elif llama_label:
        final_label = llama_label
        confidence = "medium"
    else:
        final_label = rule_label
        final_cleaned = text
        confidence = "rule"

    meta = {
        "rule_label": rule_label,
        "gpt_label": gpt_label,
        "gpt_cleaned_text": gpt_cleaned,
        "llama_label": llama_label,
        "llama_cleaned_text": llama_cleaned,
    }
    return final_label, final_cleaned, confidence, meta


# =========================
# AU Link heuristic
# =========================
def _make_link(
    link_id: str, src_au: str, dst_au: str, relation: str,
    confidence: float, start: float, end: float,
    deleted: bool = False,
) -> dict:
    """Build a link event dict. deleted=True emits a tombstone for UI removal."""
    return {
        "type":       "link",
        "event_id":   link_id,
        "link_id":    link_id,
        "ts":         round(time.time(), 3),
        "src_au":     src_au,
        "dst_au":     dst_au,
        "relation":   relation,
        "confidence": confidence,
        "start":      start,
        "end":        end,
        "session_id": SESSION_ID,
        "deleted":    deleted,
    }


def compute_link(
    new_au_id:   str,
    new_label:   str,
    new_speaker: str,
    new_start:   float,
    new_end:     float,
    au_history:  List[dict],
) -> Optional[dict]:
    """
    counterclaim / rebuttal → attack  → most recent AU from OTHER speaker
    premise                 → support → most recent claim AU from SAME speaker

    FIX 2: link_id includes dst_au to prevent collisions if multiple links per turn.
    """
    if new_label not in ("counterclaim", "rebuttal", "premise"):
        return None

    cutoff   = new_start - LINK_WINDOW_SEC
    turn_num = new_au_id.split("_")[-1]

    if new_label in ("counterclaim", "rebuttal"):
        for au in reversed(au_history):
            if au["speaker"] != new_speaker and au["end"] >= cutoff:
                link_id = f"link_{SESSION_ID}_{turn_num}_{au['au_id']}"  # FIX 2
                return _make_link(link_id, new_au_id, au["au_id"], "attack", 0.65, new_start, new_end)

    elif new_label == "premise":
        for au in reversed(au_history):
            if au["speaker"] == new_speaker and au["au_type"] == "claim" and au["end"] >= cutoff:
                link_id = f"link_{SESSION_ID}_{turn_num}_{au['au_id']}"  # FIX 2
                return _make_link(link_id, new_au_id, au["au_id"], "support", 0.60, new_start, new_end)

    return None


# =========================
# WebSocket state
# =========================
_ws_connections: Set[Any] = set()
_ws_lock = threading.Lock()
_ws_loop: Optional[asyncio.AbstractEventLoop] = None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _broadcast(msg: dict) -> None:
    if not HAS_FASTAPI or _ws_loop is None:
        return

    async def _send_all() -> None:
        dead = []
        with _ws_lock:
            conns = list(_ws_connections)
        for ws in conns:
            try:
                await ws.send_json(msg)
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


_broadcast_segment = _broadcast


def _broadcast_status(payload: dict) -> None:
    _broadcast({"type": "status", "event_id": "status",
                "ts": round(time.time(), 3), **payload})


if HAS_FASTAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        global _ws_loop
        _ws_loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="ARgueVis", lifespan=_lifespan)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
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
        return f"""<html><body>
        <p>ARgueVis — WebSocket: <code>ws://localhost:{WS_PORT}/ws</code></p>
        </body></html>"""


# =========================
# Data structures
# =========================
@dataclass
class SpeechSegment:
    seg_id:       str
    start_sample: int
    end_sample:   int


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
            return np.concatenate([self.buf[start_idx:].copy(),
                                   self.buf[:n - first].copy()])

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
        self._seg_counter      = 0
        self.pad_samples       = int(VAD_SPEECH_PAD_MS * SAMPLE_RATE / 1000)
        self.end_sil_samples   = int(VAD_END_SIL_MS    * SAMPLE_RATE / 1000)
        self.max_seg_samples   = int(VAD_MAX_SEG_SEC   * SAMPLE_RATE)

    def _next_seg_id(self) -> str:
        self._seg_counter += 1
        return f"{SESSION_ID}_{self._seg_counter}"

    def process_frame(self, pcm16_frame: np.ndarray, frame_start_global: int) -> Optional[SpeechSegment]:
        is_speech = self.vad.is_speech(pcm16_frame.tobytes(), SAMPLE_RATE)
        frame_end = frame_start_global + len(pcm16_frame)

        if is_speech:
            self.last_voice_sample = frame_end
            if not self.in_speech:
                self.in_speech        = True
                self.seg_start_sample = max(0, frame_start_global - self.pad_samples)

        if self.in_speech:
            seg_len       = frame_end - self.seg_start_sample
            force_close   = seg_len >= self.max_seg_samples
            silence_close = (frame_end - self.last_voice_sample) >= self.end_sil_samples

            if force_close or silence_close:
                self.in_speech = False
                end_sample     = frame_end if force_close else self.last_voice_sample + self.pad_samples
                seg_id = self._next_seg_id()
                if force_close:
                    print(f"[VAD] Hard cap — force-closing {seg_id}")
                return SpeechSegment(seg_id=seg_id,
                                     start_sample=self.seg_start_sample,
                                     end_sample=end_sample)
        return None


# =========================
# JSONL manager
# =========================
class JsonlManager:
    def __init__(self, path: str):
        self.path  = path
        self._lock = threading.Lock()
        self._fh   = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

    def patch_speaker(self, original_record: dict, new_speaker: str) -> None:
        patch            = dict(original_record)
        patch["speaker"] = new_speaker
        patch["_patch"]  = True
        patch["ts"]      = round(time.time(), 3)
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
    def __init__(self):
        print("[ENROLL] Loading SpeechBrain ECAPA-TDNN speaker encoder...")
        from speechbrain.inference import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source   = "speechbrain/spkrec-ecapa-voxceleb",
            run_opts = {"device": "cpu"},
            savedir  = "pretrained_models/spkrec-ecapa-voxceleb",
        )
        self._lock = threading.Lock()
        self._reset_state()
        print("[ENROLL] Encoder ready. Accumulating wearer speech passively...")
        print(f"[ENROLL] Auto force-enroll fires after {ENROLL_AUTO_FORCE_SEC:.0f}s. Type 'e' to force now.")

    def _reset_state(self) -> None:
        self.anchor_A:           Optional[torch.Tensor] = None
        self.enrolled:           bool   = False
        self._enroll_chunks:     List[np.ndarray] = []
        self._enroll_speech_sec: float  = 0.0
        self._enroll_done        = threading.Event()
        self._skipped_noisy:     int    = 0

    def reset(self, on_reset: Optional[Callable] = None) -> None:
        with self._lock:
            self._reset_state()
        if on_reset:
            on_reset()
        print(f"[ENROLL] ↺  Re-enrollment — need {ENROLL_SPEECH_SEC:.0f}s, RMS > {ENROLL_MIN_RMS}")

    def _embed(self, audio_f32: np.ndarray) -> torch.Tensor:
        wav = torch.tensor(audio_f32, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav).squeeze()
        return F.normalize(emb, dim=0)

    def try_enroll(self, audio_f32: np.ndarray) -> bool:
        if self.enrolled:
            return True
        dur = len(audio_f32) / SAMPLE_RATE
        if dur < ENROLL_MIN_SEG_SEC:
            return False
        chunk_rms = _rms(audio_f32)
        if chunk_rms < ENROLL_MIN_RMS:
            with self._lock:
                self._skipped_noisy += 1
            print(f"[ENROLL] Noise gate skip (rms={chunk_rms:.4f} < {ENROLL_MIN_RMS})")
            return False
        with self._lock:
            self._enroll_chunks.append(audio_f32.copy())
            self._enroll_speech_sec += dur
            if self._enroll_speech_sec >= ENROLL_SPEECH_SEC:
                combined      = np.concatenate(self._enroll_chunks)
                self.anchor_A = self._embed(combined)
                self.enrolled = True
                self._enroll_done.set()
                print(f"[ENROLL] ✓ Enrolled ({self._enroll_speech_sec:.1f}s, "
                      f"{self._skipped_noisy} skipped). Threshold={SIM_THRESHOLD}")
                return True
        return False

    def classify(self, audio_f32: np.ndarray) -> Tuple[str, float]:
        assert self.enrolled
        emb = self._embed(audio_f32)
        with self._lock:
            sim = F.cosine_similarity(emb.unsqueeze(0), self.anchor_A.unsqueeze(0)).item()
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

    @property
    def enroll_speech_sec(self) -> float:
        # NEW 3: expose for progress display
        with self._lock:
            return self._enroll_speech_sec


# =========================
# Main
# =========================
def main():
    prompt_room_config()

    print("[INFO] Loading Whisper model...")
    asr_model   = WhisperModel(WHISPER_SIZE, compute_type=WHISPER_COMPUTE)
    ring        = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    jsonl       = JsonlManager(TRANSCRIPT_JSONL)
    turns_jsonl = JsonlManager(TURNS_JSONL)
    links_jsonl = JsonlManager(LINKS_JSONL)
    seg_q: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()
    _stop = threading.Event()

    _pending_u:         Dict[str, dict] = {}
    _pending_u_order:   Deque[str]      = collections.deque()
    _pending_u_lock     = threading.Lock()
    _reclassify_running = threading.Event()

    _au_history:     List[dict] = []
    _au_history_lock = threading.Lock()

    # FIX 1: track exactly which link_id was emitted for each au_id
    # so _llm_patch can tombstone only that one, not all possible candidates
    _last_link_for_au: Dict[str, str] = {}
    _last_link_lock = threading.Lock()

    # NEW 4: track when enrollment started for auto force-enroll
    _enroll_start_time = [time.time()]
    _auto_enrolled     = [False]

    def _on_reset() -> None:
        with _pending_u_lock:
            _pending_u.clear()
            _pending_u_order.clear()
        _enroll_start_time[0] = time.time()
        _auto_enrolled[0]     = False
        print("[PATCH] Pending-U cache cleared.")

    classifier = SpeakerClassifier()

    _last_mic_print   = [0.0]
    _last_rms         = [0.0]
    _vad_in_speech    = [False]
    _asr_processing   = [False]
    _low_energy_skips = [0]
    audio_frame_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2000)

    # ---- Audio callback ----
    def audio_callback(indata, frames, time_info, status):
        x = indata[:, 0].copy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1.0, 1.0)
        rms = _rms(x)
        _last_rms[0] = rms
        if time.time() - _last_mic_print[0] >= 1.0:
            _last_mic_print[0] = time.time()
            if classifier.enrolled:
                tag = "enrolled"
            elif vad.in_speech:
                # NEW 3: show cumulative seconds collected
                secs = classifier.enroll_speech_sec
                tag  = f"enrolling {int(classifier.enrollment_progress * 100)}% ({secs:.1f}s/{ENROLL_SPEECH_SEC:.0f}s) [speaking...]"
            else:
                secs = classifier.enroll_speech_sec
                tag  = f"enrolling {int(classifier.enrollment_progress * 100)}% ({secs:.1f}s/{ENROLL_SPEECH_SEC:.0f}s)"
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
                _vad_in_speech[0] = vad.in_speech
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

    # ---- Auto force-enroll helper (shared with CLI 'e' command) ----
    def _force_enroll(source: str = "manual") -> bool:
        """Force enrollment from last 20s of ring buffer. Returns True on success."""
        now_s = ring.now_sample()
        start = max(0, now_s - int(20.0 * SAMPLE_RATE))
        audio = ring.get_range(start, now_s)
        if len(audio) < SAMPLE_RATE * 3:
            print(f"[ENROLL] Not enough audio for force-enroll ({source}).")
            return False
        # FIX 4: gate on RMS of the actual ring audio, not instantaneous mic frame.
        # Prevents enrolling on 20s of mostly silence with one loud click.
        audio_rms = _rms(audio)
        if audio_rms < ENROLL_MIN_RMS:
            print(f"[ENROLL] Ring audio too quiet for force-enroll ({source}, rms={audio_rms:.4f}).")
            return False
        print(f"[ENROLL] Force-enrolling [{source}] ({len(audio)/SAMPLE_RATE:.1f}s, rms={_rms(audio):.4f})")
        anchor = classifier._embed(audio)
        with classifier._lock:
            classifier._enroll_chunks     = [audio]
            classifier._enroll_speech_sec = len(audio) / SAMPLE_RATE
            classifier.anchor_A           = anchor
            classifier.enrolled           = True
            classifier._enroll_done.set()
        _start_reclassify()
        print(f"[ENROLL] ✓ Done [{source}]. Threshold={SIM_THRESHOLD}")
        return True

    # ---- Reclassify pending-U ----
    def _reclassify_pending_u() -> None:
        if _reclassify_running.is_set():
            return
        _reclassify_running.set()
        try:
            while classifier.enrolled:
                processed = 0
                while processed < RECLASSIFY_BATCH_SIZE:
                    with _pending_u_lock:
                        if not _pending_u_order:
                            return
                        seg_id = _pending_u_order.popleft()
                        entry  = _pending_u.pop(seg_id, None)
                    if entry is None:
                        continue
                    audio  = entry.get("audio")
                    record = entry.get("record")
                    if audio is None or len(audio) == 0 or record is None:
                        continue
                    try:
                        new_speaker, sim = classifier.classify(audio)
                    except Exception as e:
                        print(f"[PATCH] classify error {seg_id}: {e}")
                        continue
                    jsonl.patch_speaker(record, new_speaker)
                    patch_msg = {
                        "type":     "segment",
                        "event_id": seg_id,
                        "ts":       round(time.time(), 3),
                        "seg_id":   seg_id,
                        "speaker":  new_speaker,
                        "_patch":   True,
                    }
                    _broadcast(patch_msg)
                    print(f"[PATCH] {seg_id}  U → {new_speaker}  sim={sim:.3f}")
                    processed += 1
                    time.sleep(RECLASSIFY_THROTTLE_SEC)
                time.sleep(0.01)
        finally:
            _reclassify_running.clear()

    def _start_reclassify() -> None:
        if not _reclassify_running.is_set():
            threading.Thread(target=_reclassify_pending_u, daemon=True).start()

    # ---- Turn + AU buffer ----
    turn_id             = [0]
    turn_buf: Dict[str, Any] = {"active": None}
    _last_turn_activity = [time.time()]
    _last_flush_time    = [0.0]
    turn_lock           = threading.Lock()

    min_seg_samples   = int(MERGE_MIN_SEG_SEC * SAMPLE_RATE)
    min_print_samples = int(0.3 * SAMPLE_RATE)
    merge_gap_sec     = MERGE_GAP_MS / 1000.0
    last_text         = {"t": "", "end": 0.0}

    def maybe_flush_turn() -> None:
        now = time.time()
        if now - _last_flush_time[0] < 0.1:
            return
        with turn_lock:
            t: Optional[Turn] = turn_buf["active"]
            if t is None:
                return
            turn_id[0] += 1
            t.turn_id   = turn_id[0]
            t.label     = classify_turn_rule_based(t.text)
            turn_conf   = round(float(np.mean(t.sim_scores)), 3) if t.sim_scores else None
            au_id       = f"au_{t.turn_id}"

            turn_record = {
                "type":       "turn",
                "event_id":   f"turn_{t.turn_id}",
                "ts":         round(time.time(), 3),
                "turn_id":    t.turn_id,
                "speaker":    t.speaker,
                "start":      round(t.start, 2),
                "end":        round(t.end, 2),
                "text":       t.text,       # raw Whisper text until LLM patch
                "raw_text":   t.text,       # NEW 1: always preserved
                "label":      t.label,
                "au_id":      au_id,
                "au_type":    t.label,
                "turn_conf":  turn_conf,
                "confidence": "rule",
                "seg_ids":    t.seg_ids,
                "session_id": SESSION_ID,
                # Optional LLM ensemble metadata columns (initialized to None / rule)
                "gpt_label":          None,
                "gpt_cleaned_text":   None,
                "llama_label":        None,
                "llama_cleaned_text": None,
                "rule_label":         t.label,
            }
            turn_buf["active"]  = None
            _last_flush_time[0] = time.time()

        turns_jsonl.write(turn_record)
        _broadcast(turn_record)

        # Link detection
        au_entry = {
            "au_id":   au_id,
            "au_type": t.label,
            "speaker": t.speaker,
            "start":   t.start,
            "end":     t.end,
        }
        with _au_history_lock:
            _au_history.append(au_entry)
            cutoff = t.start - LINK_WINDOW_SEC
            while _au_history and _au_history[0]["end"] < cutoff:
                _au_history.pop(0)
            history_snapshot = list(_au_history[:-1])

        link = compute_link(au_id, t.label, t.speaker, t.start, t.end, history_snapshot)
        if link:
            links_jsonl.write(link)
            _broadcast(link)
            print(f"[LINK] {link['relation'].upper()}  {link['src_au']} → {link['dst_au']}")
            with _last_link_lock:                        # FIX 1: remember exactly what we emitted
                _last_link_for_au[au_id] = link["link_id"]

        # NEW 1: LLM ensemble — returns label + cleaned_text in one call
        def _llm_patch():
            llm_label, cleaned_text, confidence, meta = classify_turn_ensemble(t.text)

            # Only patch if something actually changed
            text_changed  = cleaned_text.strip() != t.text.strip()
            label_changed = llm_label != t.label

            if not text_changed and not label_changed and confidence == "rule":
                return

            patch = dict(turn_record)
            patch["label"]        = llm_label
            patch["au_type"]      = llm_label
            patch["confidence"]   = confidence
            patch["text"]         = cleaned_text    # NEW 1: repaired text
            patch["cleaned_text"] = cleaned_text    # NEW 1: explicit field
            patch["raw_text"]     = t.text          # NEW 1: original preserved
            # Per-model outputs from ensemble
            patch["gpt_label"]          = meta.get("gpt_label")
            patch["gpt_cleaned_text"]   = meta.get("gpt_cleaned_text")
            patch["llama_label"]        = meta.get("llama_label")
            patch["llama_cleaned_text"] = meta.get("llama_cleaned_text")
            patch["rule_label"]         = meta.get("rule_label")
            patch["_patch"]       = True
            patch["event_id"]     = f"turn_{t.turn_id}"   # overwrite same record
            patch["type"]         = "turn"
            patch["ts"]           = round(time.time(), 3)

            turns_jsonl.write(patch)
            _broadcast(patch)

            label_info = f"{t.label} → {llm_label}" if label_changed else f"{llm_label} (unchanged)"
            text_info  = f' | text repaired' if text_changed else ''
            print(f"[LLM] turn_{t.turn_id}  {label_info}  [{confidence}]{text_info}")

            # FIX 1+2: recompute links only if label changed.
            # Tombstone exactly the one link we know we emitted — no spam, no fake fields.
            if label_changed:
                with _au_history_lock:
                    for au in _au_history:
                        if au["au_id"] == au_id:
                            au["au_type"] = llm_label
                            break
                    history_for_link = [au for au in _au_history if au["au_id"] != au_id]

                # Tombstone the exact rule-based link we emitted (if any)
                with _last_link_lock:
                    old_link_id = _last_link_for_au.get(au_id)

                if old_link_id:
                    tomb = {
                        "type":       "link",
                        "event_id":   old_link_id,
                        "link_id":    old_link_id,
                        "deleted":    True,
                        "ts":         round(time.time(), 3),
                        "session_id": SESSION_ID,
                    }
                    links_jsonl.write(tomb)   # write for replay correctness
                    _broadcast(tomb)

                # Emit corrected link (if the new label warrants one)
                new_link = compute_link(au_id, llm_label, t.speaker, t.start, t.end, history_for_link)
                if new_link:
                    links_jsonl.write(new_link)
                    _broadcast(new_link)
                    with _last_link_lock:
                        _last_link_for_au[au_id] = new_link["link_id"]
                    print(f"[LINK] Recomputed after LLM: {new_link['relation'].upper()}  {new_link['src_au']} → {new_link['dst_au']}")

        threading.Thread(target=_llm_patch, daemon=True).start()

    # ---- ASR + classification ----
    def transcribe_and_emit(seg: SpeechSegment) -> None:
        core_audio = ring.get_range(seg.start_sample, seg.end_sample)
        if len(core_audio) < min_print_samples:
            return

        if not classifier.enrolled:
            just_enrolled = classifier.try_enroll(core_audio)
            if just_enrolled:
                speaker, sim = classifier.classify(core_audio)
                _start_reclassify()
            else:
                speaker, sim = "U", 0.0
        else:
            speaker, sim = classifier.classify(core_audio)

        tail  = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([core_audio, tail])
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak  = np.max(np.abs(audio))
        audio = audio / (peak + 1e-6)

        if float(np.std(audio)) < 1e-4:
            return
        if _rms(audio) < 0.0003:
            _low_energy_skips[0] += 1
            return

        t0 = time.time()
        _asr_processing[0] = True
        try:
            segments_out, _ = asr_model.transcribe(
                audio,
                language                    = "en",
                vad_filter                  = True,
                beam_size                   = 3,
                best_of                     = 3,
                temperature                 = 0.0,
                condition_on_previous_text  = False,
                no_speech_threshold         = NO_SPEECH_THRESH,
                log_prob_threshold          = -1.0,
                compression_ratio_threshold = 2.4,
                word_timestamps             = False,
                repetition_penalty          = 1.1,
            )
            segments_list = list(segments_out)
            text   = " ".join((s.text or "").strip() for s in segments_list).strip()
            asr_ms = (time.time() - t0) * 1000

            no_speech_probs = [s.no_speech_prob for s in segments_list if hasattr(s, "no_speech_prob")]
            avg_no_speech   = float(np.mean(no_speech_probs)) if no_speech_probs else 0.0

            if avg_no_speech > NO_SPEECH_THRESH:
                print(f"[ASR] Dropped — no_speech_prob={avg_no_speech:.2f}")
                return
            if not text:
                return

            start_sec = seg.start_sample / SAMPLE_RATE
            end_sec   = seg.end_sample   / SAMPLE_RATE

            if text == last_text["t"] and (start_sec - last_text["end"]) < 0.8:
                return
            last_text["t"]   = text
            last_text["end"] = end_sec

            sim_str = f" sim={sim:.3f}" if speaker != "U" else ""
            print(f"[{speaker}]{sim_str} [{asr_ms:.0f}ms] {text}")

            record = {
                "type":           "segment",
                "event_id":       seg.seg_id,
                "ts":             round(time.time(), 3),
                "seg_id":         seg.seg_id,
                "session_id":     SESSION_ID,
                "start":          round(start_sec, 2),
                "end":            round(end_sec, 2),
                "speaker":        speaker,
                "text":           text,
                "asr_ms":         int(asr_ms),
                "sim":            round(sim, 3) if speaker != "U" else None,
                "no_speech_prob": round(avg_no_speech, 3),
            }
            jsonl.write(record)

            if speaker == "U":
                with _pending_u_lock:
                    if len(_pending_u) >= MAX_PENDING_U_SEGMENTS:
                        evict_id = _pending_u_order.popleft()
                        _pending_u.pop(evict_id, None)
                    _pending_u[seg.seg_id] = {"audio": core_audio.copy(), "record": dict(record)}
                    _pending_u_order.append(seg.seg_id)
                # U segments do NOT flush turns

            if speaker != "U":
                flush_needed = False
                with turn_lock:
                    active = turn_buf["active"]
                    if active is not None:
                        gap = start_sec - active.end
                        if speaker != active.speaker:
                            flush_needed = True
                        elif gap > TURN_GAP_SEC:
                            flush_needed = True
                        elif (end_sec - active.start) > TURN_MAX_SEC:
                            flush_needed = True

                if flush_needed:
                    maybe_flush_turn()

                with turn_lock:
                    active = turn_buf["active"]
                    if active is None:
                        turn_buf["active"] = Turn(
                            turn_id    = 0,
                            speaker    = speaker,
                            start      = start_sec,
                            end        = end_sec,
                            text       = text,
                            seg_ids    = [seg.seg_id],
                            sim_scores = [sim],
                        )
                    else:
                        active.end = end_sec
                        active.seg_ids.append(seg.seg_id)
                        active.text = (active.text + " " + text).strip()
                        active.sim_scores.append(sim)
                    _last_turn_activity[0] = time.time()

            _broadcast(record)
        finally:
            _asr_processing[0] = False

    # ---- ASR loop ----
    def asr_loop():
        pending_seg:    Optional[SpeechSegment] = None
        pending_last_t: float = 0.0

        while True:
            item = seg_q.get()
            if item is None:
                if pending_seg is not None:
                    transcribe_and_emit(pending_seg)
                maybe_flush_turn()
                break

            seg     = item
            seg_len = seg.end_sample - seg.start_sample
            now_t   = time.time()

            if seg_len < min_print_samples:
                continue

            is_short = seg_len < min_seg_samples

            if is_short:
                if pending_seg is None:
                    pending_seg, pending_last_t = dc_replace(seg), now_t
                elif now_t - pending_last_t <= merge_gap_sec:
                    pending_seg = dc_replace(pending_seg, end_sample=seg.end_sample)
                    pending_last_t = now_t
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_seg_samples:
                        transcribe_and_emit(pending_seg)
                        pending_seg = None
                else:
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_emit(pending_seg)
                    pending_seg, pending_last_t = dc_replace(seg), now_t
                continue

            if pending_seg is not None:
                if now_t - pending_last_t > merge_gap_sec:
                    if (pending_seg.end_sample - pending_seg.start_sample) >= min_print_samples:
                        transcribe_and_emit(pending_seg)
                    pending_seg = None
                    transcribe_and_emit(seg)
                else:
                    transcribe_and_emit(dc_replace(pending_seg, end_sample=seg.end_sample))
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

    # ---- Status broadcaster + NEW 4: auto force-enroll ----
    def status_broadcast_loop():
        while not _stop.is_set():
            time.sleep(0.2)

            # Idle turn flush
            with turn_lock:
                active_turn  = turn_buf.get("active")
                last_turn_ts = _last_turn_activity[0]
            if (active_turn is not None
                    and not _asr_processing[0]
                    and (time.time() - last_turn_ts) > (TURN_GAP_SEC + 0.5)):
                maybe_flush_turn()

            # NEW 4: auto force-enroll if enrollment is stalled
            # FIX 4: RMS check is now inside _force_enroll on the actual ring audio
            if (not classifier.enrolled
                    and not _auto_enrolled[0]
                    and (time.time() - _enroll_start_time[0]) > ENROLL_AUTO_FORCE_SEC
                    and ring.now_sample() > SAMPLE_RATE * 8):
                _auto_enrolled[0] = True
                print(f"\n[ENROLL] Auto force-enroll triggered after {ENROLL_AUTO_FORCE_SEC:.0f}s")
                threading.Thread(target=_force_enroll, args=("auto",), daemon=True).start()

            # Status broadcast
            progress     = classifier.enrollment_progress
            enrolled     = classifier.enrolled
            seconds_left = 0.0 if enrolled else max(0.0, ENROLL_SPEECH_SEC * (1.0 - progress))
            mode = ("processing" if _asr_processing[0]
                    else "recording" if _vad_in_speech[0]
                    else "listening")
            _broadcast_status({
                "enrolled":       enrolled,
                "progress":       progress,
                "seconds_left":   round(seconds_left, 1),
                "enroll_sec":     round(classifier.enroll_speech_sec, 1),  # NEW 3
                "enroll_sec_target": ENROLL_SPEECH_SEC,                    # NEW 3
                "rms":            round(_last_rms[0], 4),
                "mode":           mode,
            })

    status_t = threading.Thread(target=status_broadcast_loop, daemon=True)
    status_t.start()

    # ---- CLI ----
    def cli_loop():
        print("[CLI]  r = re-enroll  |  e = force enroll  |  v = force segment  |  q = quit")
        while not _stop.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd == "r":
                    classifier.reset(on_reset=_on_reset)
                elif cmd == "e":
                    _force_enroll(source="manual")
                elif cmd == "v":
                    if vad.in_speech and vad.seg_start_sample < ring.now_sample():
                        seg_id = vad._next_seg_id()
                        seg = SpeechSegment(seg_id=seg_id,
                                            start_sample=vad.seg_start_sample,
                                            end_sample=ring.now_sample())
                        vad.in_speech = False
                        seg_q.put(seg)
                        print(f"[VAD] Force-closed {seg_id}")
                    else:
                        print("[VAD] No active segment.")
                elif cmd == "q":
                    print("[CLI] Quit.")
                    _stop.set()
                    break
                elif cmd:
                    print("[CLI]  r = re-enroll  |  e = force enroll  |  v = force segment  |  q = quit")
            except (EOFError, OSError):
                break

    cli_t = threading.Thread(target=cli_loop, daemon=True)
    cli_t.start()

    # ---- Device selection ----
    def find_device(name_hint: str) -> Optional[int]:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
                return i
        return None

    dev = INPUT_DEVICE
    if isinstance(INPUT_DEVICE, str):
        dev = find_device(INPUT_DEVICE) or sd.default.device[0]
    elif INPUT_DEVICE is None:
        dev = sd.default.device[0]

    try:
        info = sd.query_devices(dev, "input")
        print(f"[INFO] Input device [{dev}]: {info.get('name')}")
    except Exception as e:
        print(f"[WARN] Device info error: {e}")

    print("[INFO] Starting mic stream.")
    print(f"[INFO] Speak normally — first {ENROLL_SPEECH_SEC:.0f}s enrolls the wearer.")
    print(f"[INFO] Auto force-enroll fires after {ENROLL_AUTO_FORCE_SEC:.0f}s if needed.")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                            blocksize=FRAME_SAMPLES, callback=audio_callback, device=dev):
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
        try:
            maybe_flush_turn()
        except Exception:
            pass
        jsonl.close()
        turns_jsonl.close()
        links_jsonl.close()
        print(f"[INFO] Saved → {TRANSCRIPT_JSONL}, {TURNS_JSONL}, {LINKS_JSONL}")
        if _low_energy_skips[0]:
            print(f"[INFO] Low-energy skips: {_low_energy_skips[0]}")


if __name__ == "__main__":
    main()