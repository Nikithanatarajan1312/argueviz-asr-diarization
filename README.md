# ARgueVis Speech Pipeline

> Real-time speech segmentation, speaker identification, transcription, and argument structure analysis — built as the foundation for live argumentation analysis and AR integration.

---

## What it does

Captures live microphone audio and produces a structured, timestamped transcript with per-speaker labels, turn boundaries, argument role classifications, LLM-repaired text, and argument links — all running locally with no required cloud dependencies.

```
Microphone (sounddevice)
        ↓
Ring Buffer (90 seconds circular buffer)
        ↓
WebRTC VAD (20ms frame segmentation)
        ↓
Speaker Enrollment + Classification (ECAPA-TDNN)
        ↓
Whisper STT (faster-whisper small, int8 CPU)
        ↓
Turn Builder + Rule-based AU Labeling (instant)
        ↓
LLM Ensemble: GPT-4o + llama3.2:3b (async patch, ~300–800ms)
  └── Argument label (claim/premise/counterclaim/rebuttal/question/other)
  └── Transcript repair (fixes Whisper errors in place)
        ↓
AU Link Detection (support/attack heuristic, tombstone-on-relabel)
        ↓
FastAPI WebSocket server
        ↓
React Debug UI
```

**End-to-text latency: ~1.2–2.0 seconds on CPU (Whisper small). Use `WHISPER_SIZE = "base"` for ~0.8–1.2s if latency is too high.**

---

## Output

Three append-only JSONL files are written in real time. All use **last-write-wins per `event_id`** — patches are written as new lines and consumers always take the last record for each `event_id`.

---

### `transcript_v2.jsonl` — one record per speech segment

```json
{
  "type": "segment",
  "event_id": "1748291234_3",
  "seg_id": "1748291234_3",
  "session_id": 1748291234,
  "start": 11.66,
  "end": 17.54,
  "speaker": "A",
  "text": "testing the pipeline to see if everything is working",
  "asr_ms": 457,
  "sim": 0.854,
  "no_speech_prob": 0.02,
  "ts": 1748291246.123
}
```

Speaker patches (U → A/B after enrollment) are written with `"_patch": true` and the same `event_id` as the original segment.

---

### `turns_v1.jsonl` — grouped speaker turns with argument labels

```json
{
  "type": "turn",
  "event_id": "turn_9",
  "turn_id": 9,
  "session_id": 1748291234,
  "speaker": "A",
  "start": 50.34,
  "end": 54.32,
  "text": "Having a dog would make me happy.",
  "raw_text": "having a dog would make me happy",
  "label": "claim",
  "au_id": "au_9",
  "au_type": "claim",
  "turn_conf": 0.71,
  "confidence": "high",
  "seg_ids": ["1748291234_12", "1748291234_13"],
  "rule_label": "claim",
  "gpt_label": "claim",
  "gpt_cleaned_text": "Having a dog would make me happy.",
  "llama_label": "claim",
  "llama_cleaned_text": "Having a dog would make me happy.",
  "ts": 1748291305.456
}
```

LLM patches arrive 300–800ms later with the same `event_id`, updating `text`, `label`, `confidence`, and per-model metadata in place.

---

### `links_v1.jsonl` — argument unit relations

```json
{
  "type": "link",
  "event_id": "link_1748291234_9_au_3",
  "link_id": "link_1748291234_9_au_3",
  "src_au": "au_9",
  "dst_au": "au_3",
  "relation": "support",
  "confidence": 0.60,
  "start": 50.34,
  "end": 54.32,
  "session_id": 1748291234,
  "deleted": false,
  "ts": 1748291305.500
}
```

If the LLM relabels a turn and the old link is no longer valid, a tombstone is written with `"deleted": true` and the same `event_id`, followed by the correct new link.

---

## WebSocket event contract

Every event carries `event_id`, `ts`, and `type`. **Frontend pattern — one line handles everything including patches:**

```javascript
const events = {};
ws.onmessage = (msg) => {
  const data = JSON.parse(msg.data);
  events[data.event_id] = data;   // last-write-wins, patches are free
  render();
};
```

### Event types

| Type | `event_id` | Description |
|---|---|---|
| `segment` | `seg_id` | New transcribed segment |
| `segment` + `_patch: true` | `seg_id` | Speaker correction (U → A/B), overwrites original |
| `turn` | `"turn_{id}"` | Completed speaker turn, rule-based label |
| `turn` + `_patch: true` | `"turn_{id}"` | LLM label + text repair, overwrites original |
| `link` | `"link_{session}_{turn}_{dst}"` | AU relation (support/attack) |
| `link` + `deleted: true` | same `link_id` | Tombstone — remove this link |
| `status` | `"status"` | Enrollment progress, RMS, mode |

---

## Architecture

| Stage | Tool | Notes |
|---|---|---|
| Audio capture | `sounddevice` | Low-latency PortAudio bindings |
| Ring buffer | `numpy` | 90s circular buffer, thread-safe |
| VAD | `webrtcvad` | C-based, deterministic, 20ms frames, 8s hard cap |
| Speaker ID | SpeechBrain ECAPA-TDNN | 512-dim embeddings, cosine sim, EMA anchor update |
| STT | `faster-whisper` (small, int8) | `vad_filter=True`, `no_speech_prob` guard |
| AU labeling | Rule-based (instant) + LLM ensemble (async) | GPT-4o + llama3.2:3b in parallel |
| Transcript repair | GPT-4o + llama3.2:3b | Same API call as labeling — zero extra latency |
| AU linking | Online heuristic | support/attack, tombstone-on-relabel, collision-safe IDs |
| Storage | Append-only JSONL | Last-write-wins per `event_id` |
| Streaming | FastAPI WebSocket | Low-latency, AR-client compatible |

---

## Setup

```bash
# Python 3.10 or 3.11 only — SpeechBrain requirement
conda create -n argueviz python=3.10
conda activate argueviz

# Core dependencies
pip install faster-whisper speechbrain torch torchaudio \
            sounddevice webrtcvad-wheels numpy fastapi uvicorn

# LLM ensemble (optional but recommended)
pip install openai
brew install ollama && ollama pull llama3.2:3b && ollama serve
export OPENAI_API_KEY=sk-...
```

> HuggingFace token not required — the SpeechBrain model is public.

---

## Running

```bash
# Backend
python audio_v14.py

# Frontend (optional)
npm install && npm run dev
# open http://localhost:5173
```

**WebSocket endpoint:** `ws://localhost:8000/ws`

---

## Enrollment

The system passively enrolls the wearer's voice. No button press needed.

- Segments before enrollment complete are labeled `"U"` (unknown)
- On enrollment, all pending `U` segments are reclassified and patched in JSONL
- Terminal shows live progress: `enrolling 60% (6.1s/10.0s)`
- **Auto force-enroll fires after 45 seconds** of wall time if passive enrollment stalls
- The force-enroll RMS-gates on the actual 20s ring buffer slice — won't enroll on silence

### CLI commands (during a session)

| Key | Action |
|---|---|
| `e` + Enter | Force-enroll immediately from last 20s of audio |
| `r` + Enter | Re-enroll (e.g. wearer changes) |
| `v` + Enter | Force-close the current VAD segment |
| `q` + Enter | Quit |

---

## Room configuration

On startup you're prompted for your environment. This sets VAD aggressiveness, RMS thresholds, and enrollment targets:

| Mode | RMS threshold | Enrollment target | Auto-enroll fallback |
|---|---|---|---|
| 1 — Quiet (office, bedroom) | 0.005 | 10s | 45s |
| 2 — Noisy (cafe, HVAC) | 0.012 | 15s | 45s |
| 3 — Very noisy | 0.020 | 15s | 45s |

---

## Speaker classification tuning

Observed cosine similarity ranges from a MacBook Air mic:

| Speaker | Sim range |
|---|---|
| Wearer (A) | 0.50 – 0.77 |
| Partner (B) | 0.02 – 0.09 |

```python
SIM_THRESHOLD = 0.40   # lower if wearer is labeled B; raise if partner is labeled A
```

`turn_conf` in each turn record is the mean cosine similarity of all A/B-labeled segments in that turn — a proxy for how confident speaker classification was.

---

## Argument labeling

Each flushed turn becomes an **Argument Unit (AU)**. Labeling happens in two stages:

**Stage 1 — Rule-based (instant):**

| Label | Trigger |
|---|---|
| `other` | Fewer than 4 words |
| `question` | Contains `?` |
| `premise` | Contains *because, since, therefore, thus, hence* |
| `counterclaim` | Contains *but, however, although, yet, despite* |
| `rebuttal` | Contains *actually, in fact, that's wrong, i disagree* |
| `claim` | Default |

**Stage 2 — LLM ensemble (async, ~300–800ms):**

GPT-4o and llama3.2:3b run in parallel, each returning `{"label": "...", "cleaned_text": "..."}` in a single call — labeling and transcript repair happen simultaneously at no extra latency cost.

| Outcome | `confidence` |
|---|---|
| Both models agree | `"high"` |
| Disagree — GPT-4o wins | `"low"` |
| Only one model responds | `"medium"` |
| Both fail | `"rule"` (rule-based only) |

The LLM patch overwrites the turn record in place (same `event_id`). Both models are fully optional.

---

## Transcript repair

In the same LLM call that classifies argument role, GPT-4o repairs Whisper errors: dropped words, run-on repetitions, homophones, missing punctuation. The repaired text replaces `text` in the turn patch. The original Whisper output is always preserved in `raw_text`.

```
raw_text:  "i think the problem is is that we don't have enough time because of the the deadline"
text:      "I think the problem is that we don't have enough time because of the deadline."
```

---

## AU link detection

After each turn flushes, a heuristic checks the last 20 seconds of AU history:

| Label | Relation | Target |
|---|---|---|
| `counterclaim` / `rebuttal` | `attack` | Most recent AU from the **other** speaker |
| `premise` | `support` | Most recent `claim` from the **same** speaker |
| `claim` / `question` / `other` | — | No link |

If the LLM relabels a turn, the old rule-based link is tombstoned (`deleted: true`) and a new correct link is emitted. Both are written to `links_v1.jsonl` and broadcast over WebSocket. Link IDs are deterministic and collision-safe: `link_{session}_{turn_num}_{dst_au_id}`.

---

## JSONL consumer pattern

```python
# Turns — last-write-wins handles all LLM patches automatically
turns = {}
for line in open("turns_v1.jsonl"):
    r = json.loads(line)
    turns[r["event_id"]] = r

# Links — handle tombstones explicitly
links = {}
for line in open("links_v1.jsonl"):
    r = json.loads(line)
    if r.get("deleted"):
        links.pop(r["event_id"], None)
    else:
        links[r["event_id"]] = r
```

---

## Latency profile (MacBook Air M2, CPU)

| Stage | Latency |
|---|---|
| VAD silence buffer | 400 ms |
| Speaker embedding | 30–150 ms |
| Whisper STT (small) | 700–1200 ms |
| **End-to-text total** | **~1.2–2.0 s** |
| LLM label + repair patch | +300–800 ms (async, non-blocking) |

---

## External API calls

**First run only (model downloads):**
- SpeechBrain ECAPA-TDNN downloads from HuggingFace (~100MB, cached to `pretrained_models/`)
- Whisper `small` downloads from HuggingFace (~500MB, cached by faster-whisper)

**Runtime — base mode (no LLM configured):**
- 0 cloud calls. All inference runs locally — VAD, speaker ID, and Whisper are fully on-device.
- No telemetry.

**Runtime — with LLM ensemble:**
- OpenAI GPT-4o: one API call per flushed turn (~$0.003–0.005/session). Requires `OPENAI_API_KEY`.
- Ollama llama3.2:3b: runs locally, 0 cloud calls. Requires a local Ollama instance.
- Both are optional — the system degrades gracefully to rule-based labeling if neither is available.

---

## Why GPT-4o and Llama 3.2

The system uses two LLMs together to improve reliability while keeping latency and cost low.

### GPT-4o (API model)
GPT-4o is used because it provides strong reasoning and good accuracy for tasks like argument role classification and transcript cleanup.  
It is relatively fast for an API model (around ~1 second response time) and has moderate cost (~$2.5 per 1M input tokens and ~$10 per 1M output tokens).  
Because of this, GPT-4o acts as the main model for deciding the final argument label.

### Llama 3.2 (3B, local model)
Llama 3.2 runs locally using Ollama, so it has **no API cost** and works offline.  
The 3B parameter version is lightweight and runs quickly on a laptop CPU, making it suitable for real-time pipelines.

### Why use both
Using both models together provides:
- **Higher reliability** — if both models agree, the result is likely correct.
- **Cost efficiency** — Llama runs locally without API charges.
- **Robustness** — if one model fails or disagrees, the system still produces a result.

The two models run in parallel, and their outputs are combined using a simple voting strategy to produce the final argument role label.

| Situation      | Result          | Confidence |
| -------------- | --------------- | ---------- |
| GPT = Llama    | use that label  | `high`     |
| GPT ≠ Llama    | use GPT         | `low`      |
| only one works | use that one    | `medium`   |
| both fail      | keep rule label | `rule`     |

After labeling, each turn becomes an argument unit, and we detect simple relationships: premises support previous claims from the same speaker, while counterclaims or rebuttals attack the other speaker’s arguments.

---

## Running the Pipeline (audio_v15.py)

### 1. Install dependencies

```bash
conda activate argueviz
pip install faster-whisper speechbrain torch torchaudio fastapi uvicorn openai python-multipart
brew install ffmpeg
```

### 2. Install Ollama + fine-tuned model

```bash
brew install ollama
ollama serve   # keep running in a separate terminal

# Option A — fine-tuned model (recommended)
python merge_and_export.py
ollama create argueviz-llama -f Modelfile

# Option B — base llama fallback
ollama pull llama3.2:3b
```

### 3. Run

```bash
conda activate argueviz
python audio_v15.py
```

Select room environment (1 = quiet, 2 = noisy, 3 = very noisy), then speak normally. The pipeline enrolls your voice in the first ~10 seconds automatically.

---

## This Week's Updates

### Fine-tuned Argument Classifier (argueviz-llama)

Fine-tuned `meta-llama/Llama-3.2-3B-Instruct` using LoRA to classify argument units in real-time conversation.

**Training data:**
- PERSUADE corpus — 2,500 argumentative essay segments (claim, premise, counterclaim, rebuttal)
- ARgueVis session turns — 64 real conversation turns from live sessions
- GPT-4o synthetic augmentation — 100 synthetic `question` examples to balance sparse label

**Final dataset:** 528 training / 75 validation examples

**Training:** Google Colab T4 GPU, LoRA (r=16, alpha=32), 4 epochs, ~20 min

**Deployment:** Merged into base model, served locally via Ollama as `argueviz-llama`

### Link Detection

Added real-time support/attack link detection between argument units:
- `counterclaim` / `rebuttal` → **attack** link to most recent opposing turn 
- `premise` → **support** link to most recent same-speaker `claim` 

---

## Preliminary Accuracy (Live Session Test)

Tested on 2-person debate (topic: university exams vs project-based evaluation):

| Component | Result |
|---|---|
| Speaker ID | ~95%  |
| Transcription | ~85%  |
| Link detection | ~80% |
| AU labels | ~65% — model over-predicts `premise`, needs more training data |

**Known issue:** Fine-tuned model tends to classify most turns as `premise`, likely due to heavy premise/evidence distribution in PERSUADE corpus. Next step: collect more balanced real-session turns and retrain.

---

## Differences from Original server.py

| | Original | This branch |
|---|---|---|
| ASR | openai-whisper base | faster-whisper small |
| Speaker ID | Energy/ZCR features | ECAPA-TDNN embeddings |
| LLM | Groq llama-3.1-8b | argueviz-llama (local) |
| AU labels | None — vague summary only | claim/premise/counterclaim/rebuttal/question/other |
| Links | None | support/attack graph |
| Real-time | REST poll | WebSocket stream |
