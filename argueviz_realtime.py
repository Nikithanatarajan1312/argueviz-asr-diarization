import time
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
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
INPUT_DEVICE = 1  # set to None to use system default input


RING_SECONDS = 30                # keep last N seconds of audio in RAM
ASR_WINDOW_SECONDS = 7.0         # sliding window for transcription
ASR_HOP_SECONDS = 1.2            # how often we run ASR
ASR_MIN_FINAL_LAG = 1.4          # finalize segments older than this from "now"

VAD_MODE = 2                     # 0..3 (2 = less over-splitting, better for diar)
VAD_SPEECH_PAD_MS = 200          # pad around speech boundaries
VAD_END_SIL_MS = 700             # end segment after silence this long

# --- Merge short segments (ASR stage) ---
MERGE_GAP_MS = 300
MERGE_MAX_GAP_SAMPLES = int(MERGE_GAP_MS * SAMPLE_RATE / 1000)
MERGE_MIN_SEG_SECONDS = 1.0

# --- ASR model ---
# CPU: base + int8 is a good starting point
WHISPER_SIZE = "base"
WHISPER_COMPUTE = "int8"         # "int8" for CPU, "float16" if CUDA

# --- Live ASR (sliding window): set True for near-real-time + segment output (2x Whisper load)
ENABLE_LIVE_ASR = False

# --- Diarization (optional) ---
ENABLE_DIAR = True               # set False to skip diarization
MIN_DIAR_SEG_SECONDS = 1.2       # skip diar for very short segments (reduces flip/garbage)
SPEAKER_FLIP_MARGIN = 0.08       # higher = less flipping

# --- Diar stabilization ---
MIN_SWITCH_CONF = 0.14           # 0.12–0.18 (lower = switch more when embed says other speaker)
MIN_EMBED_SECONDS = 1.5          # don't assign speaker for < this length

# ---- Fast diar knobs ----
EMBED_EVERY_N = 2                # do a real embedding every N segments (2 = check often for correct A/B)
EMBED_ON_LONG_SEC = 1.5          # always embed if segment is this long (catch speaker turns)
AMP_THRESH = 0.012               # skip embedding if audio is too quiet (mic noise)
ENROLL_MIN_DISTINCTION = 0.75    # warn if A/B enrollment cosine sim above this (too similar)

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

# =========================
# Online 2-speaker assigner (optional)
# =========================
class OnlineTwoSpeaker:
    def __init__(self):
        self.encoder = None
        self.centroids = {"A": None, "B": None}
        self.counts = {"A": 0, "B": 0}
        self.last_speaker = None

        if ENABLE_DIAR:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                )
            except Exception as e:
                print("[WARN] Diarization disabled (SpeechBrain not available):", e)
                self.encoder = None

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    def _embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        # SpeechBrain expects torch tensor [batch, time]
        import torch
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        wav = wav.cpu()
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
        return emb

    def assign(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        audio: float32 mono segment
        returns: (speaker, conf)
        """
        emb = self._embed(audio)
        if emb is None:
            return "U", 0.0

        cA, cB = self.centroids["A"], self.centroids["B"]

        # bootstrap centroids
        if cA is None:
            self.centroids["A"] = emb
            self.counts["A"] = 1
            self.last_speaker = "A"
            return "A", 1.0
        if cB is None:
            self.centroids["B"] = emb
            self.counts["B"] = 1
            self.last_speaker = "B"
            return "B", 1.0

        sA = self._cos(emb, cA)
        sB = self._cos(emb, cB)
        winner = "A" if sA >= sB else "B"
        conf = abs(sA - sB)

        # 1) stickiness: don't switch on low confidence
        if self.last_speaker in ("A", "B") and conf < MIN_SWITCH_CONF:
            speaker = self.last_speaker
        else:
            speaker = winner

        # 2) only learn when confident (prevents centroid drift)
        if conf >= MIN_SWITCH_CONF:
            k = self.counts[speaker]
            self.centroids[speaker] = (self.centroids[speaker] * k + emb) / (k + 1)
            self.counts[speaker] = k + 1

        self.last_speaker = speaker
        return speaker, conf

    def set_centroids(self, embA: np.ndarray, embB: np.ndarray):
        self.centroids["A"] = embA
        self.centroids["B"] = embB
        self.counts["A"] = 10
        self.counts["B"] = 10
        self.last_speaker = "A"

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
def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-9))

def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def best_speaker_for_interval(diar: List[DiarSegment], s0: float, s1: float) -> str:
    if not diar:
        return "U"
    best = ("U", 0.0)
    for d in diar:
        ov = overlap(s0, s1, d.start_sec, d.end_sec)
        if ov > best[1]:
            best = (d.speaker, ov)
    return best[0]

# =========================
# Main
# =========================
def main():
    print("[INFO] Loading faster-whisper model...")
    model = WhisperModel(WHISPER_SIZE, compute_type=WHISPER_COMPUTE)

    ring = RingBuffer(RING_SECONDS, SAMPLE_RATE)
    seg_q_asr: "queue.Queue[tuple]" = queue.Queue()  # (SpeechSegment, seg_audio or None)

    vad = VadSegmenter()
    spk = OnlineTwoSpeaker()
    print("[DBG] diar enabled:", ENABLE_DIAR, "encoder loaded:", spk.encoder is not None)

    def enroll_from_mic(label: str, seconds: float = 3.0) -> Optional[np.ndarray]:
        if spk.encoder is None:
            return None

        print(f"\n[ENROLL] Speaker {label}: speak for {seconds:.1f}s (close to mic)...")
        sd.sleep(200)  # tiny pause so print flushes

        frames = int(seconds * SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
            blocking=True,
        ).reshape(-1)

        # light cleanup
        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)

        emb = spk._embed(audio)
        print(f"[ENROLL] {label} embed:", "OK" if emb is not None else "FAILED")
        return emb

    if ENABLE_DIAR and spk.encoder is not None:
        print("\n[ENROLL] Locking A/B with calibration.")
        print("[ENROLL] IMPORTANT: A and B must be different people.\n")

        embA = enroll_from_mic("A", 3.0)
        embB = enroll_from_mic("B", 3.0)

        if embA is not None and embB is not None:
            # Check that A and B are distinct (low cosine sim = different speakers)
            a_n = embA / (np.linalg.norm(embA) + 1e-9)
            b_n = embB / (np.linalg.norm(embB) + 1e-9)
            sim = float(np.dot(a_n, b_n))
            if sim > ENROLL_MIN_DISTINCTION:
                print(f"[ENROLL] WARNING: A and B embeddings very similar (cos={sim:.2f}). Use different people/voices for A vs B.\n")
            else:
                print(f"[ENROLL] A vs B distinction OK (cos={sim:.2f}).\n")
            spk.set_centroids(embA, embB)
            print("[ENROLL] Centroids set.\n")
        else:
            print("[ENROLL] Failed; using online clustering.\n")

    # audio callback -> frames
    audio_frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass

        # 1) get mono float32
        x = indata[:, 0].copy()

        # 2) sanitize + clip
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1.0, 1.0)

        # 3) Do NOT skip quiet frames here: dropping frames breaks the timeline
        #    and prevents VAD from seeing silence, so segments never end.

        # 4) float32 -> int16 PCM
        pcm = (x * 32767.0).astype(np.int16)

        # 5) push to queue
        try:
            audio_frame_q.put_nowait(pcm)
        except queue.Full:
            # drop frame if backend is behind (keeps latency bounded)
            pass

    # Consumer thread: write to ring + VAD frames + segment queue
    def ingest_loop():
        # We manage "global sample index" ourselves
        global_sample = 0
        # leftover for exact FRAME_SAMPLES chunks
        leftover = np.zeros((0,), dtype=np.int16)
        current_seg_audio = []

        while True:
            pcm = audio_frame_q.get()
            if pcm is None:
                break

            ring.append_pcm16(pcm)

            # Maintain a global sample counter (matches ring.total_written, but safer locally)
            chunk = np.concatenate([leftover, pcm])
            idx = 0
            while idx + FRAME_SAMPLES <= len(chunk):
                frame = chunk[idx:idx + FRAME_SAMPLES]
                frame_start = global_sample + idx

                is_speech = vad.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

                # if this is the start of a new speech segment, reset collector
                if (not vad.in_speech) and is_speech:
                    current_seg_audio = []

                # collect audio during the whole segment (including brief pauses)
                if vad.in_speech or is_speech:
                    current_seg_audio.append(frame.copy())

                seg = vad.process_frame(frame, frame_start)
                if seg is not None:
                    if current_seg_audio:
                        seg_audio = np.concatenate(current_seg_audio).astype(np.float32) / 32768.0

                        # add a small tail pad (helps Whisper)
                        tail = int(0.2 * SAMPLE_RATE)
                        seg_audio = np.concatenate([seg_audio, np.zeros(tail, dtype=np.float32)])

                        seg_q_asr.put((seg, seg_audio))
                    else:
                        seg_q_asr.put((seg, None))
                    current_seg_audio = []
                idx += FRAME_SAMPLES

            leftover = chunk[idx:]
            global_sample += len(pcm)

    ingest_t = threading.Thread(target=ingest_loop, daemon=True)
    ingest_t.start()

    # ASR loop: transcribe each VAD segment once (audio supplied from ingest_loop)
    # Merge short segments (< MERGE_MIN_SEG_SECONDS) if next arrives within MERGE_GAP_MS
    def asr_loop_segments():
        pending_audio = None
        pending_start = None
        pending_last_t = 0.0
        merge_gap_sec = MERGE_GAP_MS / 1000.0
        min_seg_samples = int(MERGE_MIN_SEG_SECONDS * SAMPLE_RATE)
        min_print_samples = int(0.3 * SAMPLE_RATE)

        def transcribe_and_print(audio: np.ndarray, _seg) -> None:
            if len(audio) < min_print_samples:
                return
            segments, _ = model.transcribe(
                audio,
                language="en",
                vad_filter=False,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            text = " ".join((s.text or "").strip() for s in segments).strip()
            if not text:
                return
            # --- FAST diarization ---
            speaker = "U"
            if ENABLE_DIAR and spk.encoder is not None:
                dur = len(audio) / SAMPLE_RATE
                loud = rms(audio)

                # default: stick to last speaker
                speaker = spk.last_speaker or "A"
                do_embed = False

                # embed only if:
                #  - very long segment (likely speaker turn), OR
                #  - periodic refresh, OR
                #  - if we still haven't initialized A/B
                if spk.centroids["A"] is None or spk.centroids["B"] is None:
                    do_embed = dur >= MIN_EMBED_SECONDS and loud >= AMP_THRESH
                else:
                    if dur >= EMBED_ON_LONG_SEC:
                        do_embed = True
                    elif (_seg.seg_id % EMBED_EVERY_N) == 0:
                        do_embed = True

                    # also don't waste embed on quiet junk
                    if loud < AMP_THRESH:
                        do_embed = False

                if do_embed and dur >= MIN_EMBED_SECONDS:
                    speaker, conf = spk.assign(audio)
                    # Uncomment to debug A/B: print(f"[DBG] spk={speaker} conf={conf:.3f} dur={dur:.2f}s")

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

    print("[INFO] Starting mic stream. Press Ctrl+C to stop.")
    try:
        try:
            dev = INPUT_DEVICE if INPUT_DEVICE is not None else sd.default.device[0]
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
            device=INPUT_DEVICE,
        ):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        audio_frame_q.put(None)
        seg_q_asr.put(None)

if __name__ == "__main__":
    main()