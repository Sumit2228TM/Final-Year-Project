"""
speech_engine.py — Vocal & Hearing Engines
============================================
• SpeechToText: Threaded microphone capture → text via SpeechRecognition
  (Google offline / Sphinx fallback).  Emits recognized text via callback.
• TextToSpeech: Non-blocking pyttsx3 wrapper running on its own thread to
  avoid freezing the UI.

Both classes are designed to be started/stopped cleanly and share no state.
"""

from __future__ import annotations

import threading
import queue
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger("neurolink.voice_engine")

# ─── Lazy imports (fail gracefully) ────────────────────────────

def _import_sr():
    try:
        import speech_recognition as sr
        return sr
    except ImportError:
        logger.error("SpeechRecognition not installed.  pip install SpeechRecognition")
        return None


def _import_pyttsx3():
    try:
        import pyttsx3
        return pyttsx3
    except ImportError:
        logger.error("pyttsx3 not installed.  pip install pyttsx3")
        return None


# ─────────────────────────────────────────────
# Speech-to-Text (STT)
# ─────────────────────────────────────────────

class SpeechToText:
    """
    Threaded microphone listener emitting recognized text.

    Usage:
        stt = SpeechToText(on_text=lambda t: print(t))
        stt.start()
        ...
        stt.stop()

    Callback:
        on_text(text: str)        — fired every time speech is recognised
        on_partial(text: str)     — (optional) partial / interim result
        on_error(error: str)      — recognition error
    """

    def __init__(
        self,
        on_text: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        language: str = "en-US",
    ):
        self.on_text = on_text
        self.on_partial = on_partial
        self.on_error = on_error
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.language = language

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sr = _import_sr()

    # ── Public API ────────────────────────────

    def start(self):
        if self._running:
            return
        if self._sr is None:
            logger.error("Cannot start STT — SpeechRecognition missing.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="STT")
        self._thread.start()
        logger.info("SpeechToText started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("SpeechToText stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────

    def _listen_loop(self):
        sr = self._sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.energy_threshold
        recognizer.pause_threshold = self.pause_threshold
        recognizer.dynamic_energy_threshold = True

        try:
            mic = sr.Microphone()
        except Exception as exc:
            logger.error("Cannot open microphone: %s", exc)
            if self.on_error:
                self.on_error(f"Microphone error: {exc}")
            self._running = False
            return

        with mic as source:
            # Brief ambient noise calibration
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                logger.info("Ambient noise calibrated (threshold=%d)", recognizer.energy_threshold)
            except Exception:
                pass

            while self._running:
                try:
                    audio = recognizer.listen(source, timeout=2.0, phrase_time_limit=10)
                except sr.WaitTimeoutError:
                    continue
                except Exception as exc:
                    logger.warning("Listen error: %s", exc)
                    time.sleep(0.5)
                    continue

                # Recognition — try Google (online) then Sphinx (offline)
                self._recognize(recognizer, audio)

    def _recognize(self, recognizer, audio):
        sr = self._sr
        text = None

        # Attempt 1: Google (requires internet but works well)
        try:
            text = recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            # Fallback: PocketSphinx (fully offline)
            try:
                text = recognizer.recognize_sphinx(audio)
            except Exception:
                pass

        if text:
            text = text.strip()
            logger.info("STT recognised: %s", text)
            if self.on_text:
                self.on_text(text)
        else:
            if self.on_error:
                self.on_error("Could not recognise speech")


# ─────────────────────────────────────────────
# Text-to-Speech (TTS)
# ─────────────────────────────────────────────

class TextToSpeech:
    """
    Non-blocking TTS using pyttsx3 on a dedicated thread.

    The engine runs its own event loop; we feed it text via a thread-safe queue
    so the caller never blocks.

    Usage:
        tts = TextToSpeech()
        tts.start()
        tts.speak("Hello world")
        ...
        tts.stop()
    """

    def __init__(self, rate: int = 160, volume: float = 0.95):
        self._rate = rate
        self._volume = volume
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pyttsx3 = _import_pyttsx3()

    # ── Public API ────────────────────────────

    def start(self):
        if self._running:
            return
        if self._pyttsx3 is None:
            logger.error("Cannot start TTS — pyttsx3 missing.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="TTS")
        self._thread.start()
        logger.info("TextToSpeech started (rate=%d)", self._rate)

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._queue.put(None)  # sentinel to unblock
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("TextToSpeech stopped")

    def speak(self, text: str):
        """Enqueue text for speaking — returns immediately."""
        if self._running:
            self._queue.put(text)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────

    def _run(self):
        engine = self._pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                logger.debug("TTS speaking: %s", text[:60])
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS error: %s", exc)

        try:
            engine.stop()
        except Exception:
            pass
