# ARgueVis Speech Pipeline

> Real-time speech segmentation, speaker identification, and transcription, built as the foundation for live argumentation analysis and AR integration.

---

## What it does

Captures live microphone audio and produces a structured, timestamped transcript with per-speaker labels, turn boundaries, and argument role classifications, all running locally with no cloud dependencies.

```
Microphone (sounddevice)
        ↓
Ring Buffer (90 seconds circular buffer)
        ↓
WebRTC VAD (segmentation)
        ↓
Speaker Enrollment + Classification (ECAPA-TDNN)
        ↓
Whisper STT (faster-whisper, int8 CPU)
        ↓
Append-only JSONL storage
        ↓
Turn Builder
        ↓
FastAPI WebSocket server
        ↓
React Debug UI
```

**End-to-text latency: ~0.8–1.2 seconds on CPU.**

---

## Output

Two append-only JSONL files are written in real time:

**`transcript_v2.jsonl`** — one record per speech segment:
```json
{"type": "segment", "seg_id": 4, "start": 11.66, "end": 17.54,
 "speaker": "A", "text": "testing the pipeline to see if everything is working",
 "asr_ms": 457, "sim": 0.854}
```

**`turns_v1.jsonl`** — consecutive same-speaker segments grouped into turns:
```json
{"type": "turn", "turn_id": 9, "speaker": "A", "start": 50.34, "end": 54.32,
 "text": "and having a dog would make me happy.", "label": "claim", "seg_ids": [12, 13, 14]}
```

Speaker patches (U → A/B after enrollment) are written as full correction records with `"_patch": true`. Consumer always takes last record per `seg_id`.

---

## Architecture

| Stage | Tool | Why |
|---|---|---|
| Audio capture | `sounddevice` | Low-latency PortAudio bindings |
| Ring buffer | `numpy` | 90s circular buffer, thread-safe |
| VAD | `webrtcvad` | C-based, deterministic, 20ms frames |
| Speaker ID | SpeechBrain ECAPA-TDNN | State-of-the-art embeddings, CPU-feasible |
| STT | `faster-whisper` (base, int8) | Best local quality/latency tradeoff |
| Storage | Append-only JSONL | Patch-safe, last-write-wins per seg_id |
| Streaming | FastAPI WebSocket | Low-latency, AR-client compatible |
| UI | React (Vite) | Live monitoring and debug |

---

## Setup

```bash
# Create environment
conda create -n argueviz python=3.10
conda activate argueviz

# Install dependencies
pip install faster-whisper speechbrain torch torchaudio \
            sounddevice webrtcvad-wheels numpy fastapi uvicorn
```

> HuggingFace token not required — the SpeechBrain model is public.

---

## Running

**Backend:**
```bash
python audio_v10.py
```

**Frontend (optional):**
```bash
npm install
npm run dev
# open http://localhost:5173
```

**WebSocket endpoint:** `ws://localhost:8000/ws`

---

## Enrollment

The system passively enrolls the wearer's voice from the first 15 seconds of clean speech. No button press needed, just start talking.

- Segments before enrollment complete are labeled `"U"` (unknown)
- On enrollment completion, all pending `U` segments are reclassified and patched in JSONL
- Press `r` + Enter to re-enroll (e.g. wearer changes)
- Press `q` + Enter to quit

**Noisy room? Use these settings in `audio_v10.py`:**
```python
VAD_MODE          = 3       # default: 1
ENROLL_MIN_RMS    = 0.010   # default: 0.005
ENROLL_SPEECH_SEC = 20.0    # default: 15.0
```

---

## Speaker classification tuning

Observed cosine similarity ranges from the MacBook Air mic:

| Speaker | Sim range |
|---|---|
| Wearer (A) | 0.50 – 0.77 |
| Partner (B) | 0.02 – 0.09 |

```python
SIM_THRESHOLD = 0.40   # lower if wearer labeled B; raise if partner labeled A
```

---

## Argument labeling (v1) - Draft | Need to confirm

Turns are labeled with argument roles using a rule-based classifier (instant) optionally patched by an LLM ensemble (async, ~300–800ms):

| Label | Trigger |
|---|---|
| `question` | Contains `?` |
| `premise` | Contains *because, since, therefore, thus* |
| `counterclaim` | Contains *but, however, although, yet* |
| `rebuttal` | Contains *actually, in fact, that's wrong* |
| `claim` | Default |

**LLM ensemble (optional):** GPT-4o-mini + llama3.2:3b run in parallel. If they agree → `confidence: "high"`. If they disagree → rule-based tiebreaker → `confidence: "low"`. Requires `OPENAI_API_KEY` and a local Ollama instance.

```bash
# Ollama setup (one-time)
brew install ollama
ollama pull llama3.2:3b
ollama serve
```

---

## WebSocket message types

| Type | Description |
|---|---|
| `segment` | New transcribed segment |
| `segment_patch` | Speaker correction (U → A/B) |
| `turn` | Completed speaker turn with label |
| `turn_patch` | LLM label correction |
| `status` | Enrollment progress, RMS, system mode |

---

## Latency profile (MacBook Air, CPU)

| Stage | Latency |
|---|---|
| VAD silence buffer | 400 ms |
| Speaker embedding | 30–150 ms |
| Whisper STT | 300–700 ms |
| **End-to-text total** | **~0.8–1.2 s** |

---

## Next steps

What follows is an honest breakdown of what's feasible now, what's feasible with caveats, and what belongs to a later phase. Everything is designed to slot into the existing JSONL + WebSocket pipeline without architectural changes.

---

### Step 1 — LLM argument role labeling - Ready to build

**What:** Replace `"label": "pending"` on each turn with a real argument role: `claim`, `premise`, `counterclaim`, `rebuttal`, `question`, or `other`.

**How:** Two LLMs run in parallel on each turn flush. Rule-based result is written instantly; LLM patch arrives 300–800ms later.

```
Turn flushes
    ↓
Rule-based label written immediately (instant)
    ↓  (async thread)
GPT-4o-mini ──┐
               ├──► vote() ──► patch JSONL + broadcast turn_patch
llama3.2:3b ──┘
```

**Why this pair:** GPT-4o-mini is fast (~300ms), cheap (~$0.05/session), and reliable at single-label classification. llama3.2:3b runs locally via Ollama — fully offline, no cost, ~500ms on M-series. Running both in parallel and voting gives a confidence signal that's genuinely useful: high confidence when they agree, flagged for review when they don't. Neither alone is as trustworthy as the pair.

**Confidence logic:**
| Outcome | Confidence |
|---|---|
| Both agree | `high` |
| Disagree, rule-based breaks tie | `low` |
| One fails | `medium` |
| Both fail | rule-based only |

**Feasibility: high.** Single API call per turn, async, non-blocking. 

**Setup needed:**
```bash
pip install openai
brew install ollama && ollama pull llama3.2:3b && ollama serve
export OPENAI_API_KEY=your_key
```

---

### Step 2 — Premise-to-claim linking - Feasible with caveats

**What:** After labeling individual turns, detect which premises support which claims. Output a linked structure:

```json
{"turn_id": 5, "label": "premise", "supports": 3}
{"turn_id": 3, "label": "claim",   "supported_by": [5, 6]}
```

**Caveats:**
- LLM must parse structured JSON output reliably — GPT-4o-mini handles this well with `response_format: json_object`, llama3.2:3b is less consistent and needs output validation
- Links become stale if earlier turns get speaker-patched, need to re-run linking on affected window
- Latency compounds: labeling + linking = two async calls per turn, ~600ms–1.5s total. Acceptable since both are non-blocking

**Feasibility: medium.** Works well in controlled conditions. Degrades gracefully if the LLM hallucinates links, just write nothing rather than bad data.

---

### Step 3 — Argument graph construction - Feasible but stateful

```
[claim: "I need a dog"]
        ↑ supports
[premise: "dogs reduce loneliness"]
        ↑ supports
[premise: "I live alone"]

[counterclaim: "you travel too much"] ──attack──► [claim: "I need a dog"]
```
**Feasibility: medium-high** for sessions under 20 minutes. Needs explicit management for longer sessions.

**Additional dependency:**
```bash
pip install networkx
```

---

### Step 4 — AR integration - Later phase

**What:** Stream the argument graph to a Unity client running on AR glasses and render a real-time overlay.

---

### What not to build yet

| Feature | Why not yet |
|---|---|
| Multi-speaker (>2) | ECAPA enrollment is single-anchor; generalizing to N speakers requires a different classification architecture |
| Per-segment LLM labeling | Too expensive and too slow — turns are the right granularity |
| Conversation history in LLM prompt | Context grows unbounded; use sliding window instead |

---

## Zero external API calls (base mode)

The base pipeline makes **0 cloud calls**. Everything runs locally:

- No OpenAI
- No HuggingFace hosted inference
- No telemetry

LLM labeling adds API calls only if explicitly configured.

---

## Other Improvements:
1. Catching voice through multiple trials in a different voice - like siri
2. Noisy environment - accuracy improvement
