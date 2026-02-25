# ARgueVis Speech Pipeline

A real-time speech segmentation, speaker identification, and transcription system designed as the foundation for live argumentation analysis and future AR integration.

This system provides low-latency speech processing with structured output suitable for downstream reasoning, turn aggregation, and argument structure modeling.

---

## 1. System Overview

The ARgueVis speech pipeline performs:

- Live microphone capture
- Voice Activity Detection (VAD) segmentation
- Passive enrolled-speaker identification
- Real-time speech-to-text transcription
- Patch-safe speaker correction
- WebSocket streaming to a React monitoring UI

The entire system runs locally with zero external API calls.

---

## 2. Architecture

```
Microphone (sounddevice)
        ↓
WebRTC VAD (segmentation)
        ↓
Speaker Enrollment + Classification (ECAPA-TDNN)
        ↓
Whisper STT (faster-whisper, int8 CPU)
        ↓
Append-only JSONL storage
        ↓
FastAPI WebSocket server
        ↓
React Debug UI
```

---

## 3. Design Decisions (and Why)

### 3.1 WebRTC VAD

Tool: `webrtcvad`

Why:
- Lightweight and fast (C-based implementation)
- Reliable for conversational speech
- Deterministic segmentation boundaries
- Suitable for low-latency streaming

Configuration:
- Frame size: 20 ms
- VAD_MODE = 1 (balanced for quiet rooms)
- VAD_END_SIL_MS = 400 ms (segment finalization delay)

---

### 3.2 Speaker Identification

Tool: SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)

Why:
- State-of-the-art speaker embeddings
- Lightweight CPU inference (30–150 ms)
- Embedding-based similarity allows threshold tuning
- Supports enrollment-based personalization

Method:
- First N seconds of clean speech form anchor embedding
- Cosine similarity used for classification
- EMA-based anchor drift correction for stability

---

### 3.3 Speech-to-Text

Tool: `faster-whisper` (Whisper base, int8)

Why:
- High transcription quality
- Efficient CPU inference
- Fully local (no cloud dependency)
- Deterministic latency profile

Configuration:
- Model: `base`
- Compute: `int8`
- Beam size: 3
- Condition_on_previous_text: False (prevents compounding errors)

---

### 3.4 JSONL Storage Strategy

Append-only JSONL with full-record patch corrections.

Why:
- Prevents in-place file corruption
- Supports safe speaker reclassification
- Enables "last-write-wins" logic
- Maintains complete historical record

Patch events rewrite entire record with `_patch: true`.

---

### 3.5 WebSocket Streaming

Tool: FastAPI + WebSocket

Why:
- Low-latency event streaming
- Clean separation of backend and UI
- Future Unity / AR client compatibility
- Enables monitoring and debugging in real-time

Message types:
- `segment` (full transcript record)
- `update` (speaker correction patch)
- `status` (enrollment progress, RMS, state)
- `vad` (speech start/end)

---

## 4. Latency Measurements

Measured on CPU (MacBook Air, Whisper base int8):

| Component | Latency |
|------------|----------|
| Speaker embedding | 30–150 ms |
| Whisper STT | 300–700 ms |
| VAD silence buffer | 400 ms |
| Total end-to-text latency | ~0.8–1.2 seconds |

The system comfortably operates under 2 seconds end-to-text latency, making it suitable for live conversational analysis.

---

## 5. Current Capabilities

- Real-time speech segmentation
- Enrolled-speaker separation (A / B)
- Sub-second transcription latency
- Speaker patch correction after enrollment
- Live enrollment progress tracking
- Live RMS monitoring
- Interactive segmentation timeline UI
- Per-segment latency measurement
- Similarity score display

---

## 6. Tools and Libraries

### Audio
- sounddevice
- numpy

### Segmentation
- webrtcvad

### Speaker Identification
- speechbrain
- torch
- torchaudio

### Speech-to-Text
- faster-whisper

### Backend
- fastapi
- uvicorn

### Frontend
- React (Vite)
- WebSocket client

---

## 7. Zero External API Calls

The system currently makes:

- 0 OpenAI API calls
- 0 HuggingFace hosted calls
- 0 cloud inference calls

All computation is local.

This ensures:
- No cost
- No rate limits
- Offline capability
- Full control over latency

---

## 8. Next Phase

Planned extensions:

- Turn aggregation layer (group consecutive segments by speaker)
- Argument role detection (Claim / Evidence / Rebuttal / Question)
- LLM integration for reasoning
- Real-time argument graph construction
- Unity / AR integration layer

---

## 9. How to Run

Install dependencies:

```
pip install faster-whisper speechbrain torch torchaudio sounddevice webrtcvad-wheels numpy fastapi uvicorn
```

Run backend:

```
python audio_v10.py
```

Run frontend (React):

```
npm install
npm run dev
```

Open:

```
http://localhost:5173
```

---

## 10. Summary

The ARgueVis Speech Pipeline establishes a low-latency, structured, real-time speech processing foundation suitable for argumentation analysis and augmented reality systems.

The system is stable, modular, and designed for extension into higher-level reasoning layers.