"""
modules/speech.py
Microphone listening (Whisper) and Text-to-Speech (Kokoro).
"""

import queue
import threading
import re

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

from config import (
    SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION,
    WHISPER_MODEL, GPU_DEVICE, COMPUTE_TYPE,
    TTS_VOICE, TTS_SPEED, TTS_LANG
)


# ── Text-to-Speech ─────────────────────────────────────────────────────────────

class Speaker:
    def __init__(self):
        print("[JARVIS] Loading TTS engine...")
        self.pipeline    = KPipeline(lang_code=TTS_LANG, device='cpu')
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        print("[JARVIS] TTS ready.")

    def speak(self, text: str):
        with self._lock:
            self._stop_event.clear()
            clean = self._clean(text)
            if not clean.strip():
                return
            generator = self.pipeline(clean, voice=TTS_VOICE, speed=TTS_SPEED)
            for _, _, audio in generator:
                if self._stop_event.is_set():
                    sd.stop()
                    break
                sd.play(audio, 24000)
                sd.wait()

    def stop(self):
        self._stop_event.set()
        sd.stop()

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'[*_`#~]', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ── Microphone / Speech-to-Text ───────────────────────────────────────────────

class Listener:
    def __init__(self, on_speech_callback, on_interrupt_callback=None):
        print(f"[JARVIS] Loading Whisper ({WHISPER_MODEL}, device={GPU_DEVICE})...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=GPU_DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("[JARVIS] Whisper ready.")

        self.callback           = on_speech_callback
        # ← NEW: separate callback just for interrupt word detection
        self.interrupt_callback = on_interrupt_callback

        self.audio_queue  = queue.Queue()
        self.is_listening = False

        # ← NEW: interrupt mode — mic stays open but only checks for wake word
        self.interrupt_mode = False

        self._buffer         = []
        self._silence_frames = 0
        self._speaking       = False
        self._silence_limit  = int(SILENCE_DURATION * (SAMPLE_RATE / 512))

        # ← NEW: small rolling buffer just for interrupt detection
        self._interrupt_buffer       = []
        self._interrupt_frames_limit = int(1.5 * (SAMPLE_RATE / 512))  # 1.5 seconds

    def start(self):
        self.is_listening = True
        self._worker = threading.Thread(
            target=self._transcribe_worker,
            daemon=True,
            name="Whisper-Worker"
        )
        self._worker.start()
        self._stream = sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = 1,
            blocksize  = 512,
            callback   = self._audio_callback,
            dtype      = "float32"
        )
        self._stream.start()
        print("[JARVIS] Microphone open — listening.")

    def pause(self):
        """Pause normal listening but keep mic open for interrupt detection."""
        self.is_listening   = False
        self.interrupt_mode = True   # ← switch to interrupt-only mode
        self._interrupt_buffer = []

    def resume(self):
        self.interrupt_mode  = False
        self._buffer         = []
        self._silence_frames = 0
        self._speaking       = False
        self.is_listening    = True

    def stop(self):
        self.is_listening   = False
        self.interrupt_mode = False
        self.audio_queue.put(None)
        self._stream.stop()
        self._stream.close()

    def _audio_callback(self, indata, frames, time, status):
        chunk = indata[:, 0].copy()

        # ── Interrupt mode: collect audio, check for wake word every 1.5s ──────
        if self.interrupt_mode:
            self._interrupt_buffer.append(chunk)
            if len(self._interrupt_buffer) >= self._interrupt_frames_limit:
                audio = np.concatenate(self._interrupt_buffer)
                self._interrupt_buffer = []
                # Put in queue with special flag
                self.audio_queue.put(("interrupt", audio))
            return

        # ── Normal listening mode ──────────────────────────────────────────────
        if not self.is_listening:
            return

        volume    = np.abs(chunk).mean()
        zcr       = np.sum(np.diff(np.sign(chunk)) != 0) / len(chunk)
        is_speech = volume > SILENCE_THRESHOLD and zcr > 0.04

        if is_speech:
            self._speaking       = True
            self._silence_frames = 0
            self._buffer.append(chunk)

        elif self._speaking:
            self._buffer.append(chunk)
            self._silence_frames += 1

            if self._silence_frames >= self._silence_limit:
                audio = np.concatenate(self._buffer)
                self.audio_queue.put(("speech", audio))  # ← tagged as speech
                self._buffer         = []
                self._speaking       = False
                self._silence_frames = 0

    def _transcribe_worker(self):
        while True:
            item = self.audio_queue.get()
            if item is None:
                break

            mode, audio = item  # ← unpack the tag

            segments, _ = self.model.transcribe(
                audio,
                language   = "en",
                beam_size  = 1,
                vad_filter = True,
            )
            text = " ".join(s.text for s in segments).strip()

            if not text or len(text) <= 2:
                continue

            # ── Interrupt mode: only care if "jarvis" is in the text ──────────
            if mode == "interrupt":
                if "jarvis" in text.lower() and self.interrupt_callback:
                    self.interrupt_callback()
                continue

            # ── Normal speech ─────────────────────────────────────────────────
            if not self._is_noise(text):
                self.callback(text, audio)

    @staticmethod
    def _is_noise(text: str) -> bool:
        noise_phrases = {
            "thank you", "thanks for watching", "you", "uh", "um",
            ".", "..", "...", "bye", "okay", "ok"
        }
        return text.lower().strip().strip(".").strip() in noise_phrases