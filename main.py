"""
main.py — NeuroLink Assist v1.0 Entry Point
=============================================
Orchestrates:
  1. Zero-login boot directly into the action UI
  2. Engine initialisation (eye, voice, comm bus)
  3. Calibration flow
  4. Mode switching (Eye Gaze / Voice / Deaf Captions)
  5. Clean shutdown

Run:
    python main.py
"""

from __future__ import annotations

import sys
import os
import logging
import time
import threading
from typing import Optional

# ── Ensure package imports work when running from project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from eye_engine import EyeEngine, GazePoint
from voice_engine import SpeechToText, TextToSpeech
from core_bridge import CrossCommBus, MessageType, CommMessage
from ui import NeuroLinkApp

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neurolink.main")


# ─────────────────────────────────────────────
# Session Controller
# ─────────────────────────────────────────────
class NeuroLinkSession:
    """
    Top-level session wiring all engines ↔ bus ↔ UI.
    """

    def __init__(self):
        # ── Qt App ──
        self.qt_app = QApplication(sys.argv)
        self.window = NeuroLinkApp()

        # ── Cross-Comm Bus ──
        self.bus = CrossCommBus()

        # ── Engines ──
        self.eye_engine: Optional[EyeEngine] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None

        # ── Flags ──
        self._calibrating = False
        self._eye_available = True
        self._voice_available = True

    # ────────────────────── Bootstrap ──────────

    def boot(self):
        """Zero-login boot: start engines and open UI."""
        logger.info("═══════════════════════════════════════════")
        logger.info("  NeuroLink Assist v1.0 — booting…")
        logger.info("═══════════════════════════════════════════")

        # Start message bus
        self.bus.start()

        # Wire bus subscribers
        self._subscribe_bus()

        # Init engines (non-blocking — each runs in its own thread)
        self._init_eye_engine()
        self._init_voice_engines()

        # Wire UI callbacks
        self._wire_ui()

        # Periodically push dwell regions to eye engine
        self._region_timer = QTimer()
        self._region_timer.timeout.connect(self._sync_dwell_regions)
        self._region_timer.start(500)

        # Show window
        self.window.showMaximized()
        self.window.signals.status_update.emit("Ready — select a mode to begin")

        logger.info("Boot complete.  UI launched.")

    def run(self) -> int:
        """Enter Qt event loop. Returns exit code."""
        self.boot()
        code = self.qt_app.exec()
        self.shutdown()
        return code

    def shutdown(self):
        """Clean shutdown of all engines."""
        logger.info("Shutting down…")
        if self.eye_engine:
            self.eye_engine.stop()
        if self.stt:
            self.stt.stop()
        if self.tts:
            self.tts.stop()
        self.bus.stop()
        logger.info("Goodbye.")

    # ────────────────────── Engine Init ────────

    def _init_eye_engine(self):
        try:
            screen = self.qt_app.primaryScreen()
            geo = screen.geometry()
            self.eye_engine = EyeEngine(
                screen_w=geo.width(),
                screen_h=geo.height(),
                on_gaze=self._on_gaze,
                on_blink=self._on_blink,
                on_dwell=self._on_dwell,
                on_frame=self._on_frame,
            )
            self.eye_engine.start()
        except Exception as exc:
            logger.warning("Eye engine init failed: %s — continuing without gaze", exc)
            self._eye_available = False
            self.window.signals.status_update.emit("Eye engine unavailable — voice/text only")

    def _init_voice_engines(self):
        # TTS (always useful)
        try:
            self.tts = TextToSpeech()
            self.tts.start()
        except Exception as exc:
            logger.warning("TTS init failed: %s", exc)

        # STT
        try:
            self.stt = SpeechToText(
                on_text=self._on_stt_text,
                on_error=self._on_stt_error,
            )
            # Don't auto-start STT — let user switch to voice mode
        except Exception as exc:
            logger.warning("STT init failed: %s", exc)
            self._voice_available = False

    # ────────────────────── Bus Subscribers ───

    def _subscribe_bus(self):
        # Emergency → speak + display
        self.bus.subscribe(
            self._handle_emergency,
            types={MessageType.EMERGENCY},
            name="emergency_handler",
        )
        # Speak requests → TTS
        self.bus.subscribe(
            self._handle_speak,
            types={MessageType.SPEAK_ALOUD},
            name="tts_handler",
        )
        # Display requests → caption area
        self.bus.subscribe(
            self._handle_display,
            types={MessageType.DISPLAY_TEXT},
            name="display_handler",
        )

    def _handle_emergency(self, msg: CommMessage):
        logger.warning("🚨 EMERGENCY: %s", msg.payload)
        # Speak the phrase
        if self.tts:
            self.tts.speak(msg.payload)
        # Display prominently
        self.window.signals.stt_text.emit(f"🚨 EMERGENCY: {msg.payload}")
        self.window.signals.status_update.emit(f"EMERGENCY: {msg.payload}")

    def _handle_speak(self, msg: CommMessage):
        if self.tts:
            self.tts.speak(msg.payload)

    def _handle_display(self, msg: CommMessage):
        self.window.signals.stt_text.emit(msg.payload)

    # ────────────────────── Eye Engine Callbacks ─

    def _on_gaze(self, gaze: GazePoint):
        self.window.signals.gaze_update.emit(gaze.x, gaze.y, gaze.confidence)

        # During calibration — check if gaze is near target
        if self._calibrating and self.eye_engine:
            overlay = self.window.calibration_overlay
            target = overlay.current_target
            if target:
                # We don't auto-capture on gaze — wait for blink (see _on_blink)
                pass

    def _on_blink(self):
        self.window.signals.blink_detected.emit()

        if self._calibrating and self.eye_engine:
            # Capture current gaze ratio for calibration
            ratio_x, ratio_y = self._get_current_ratio()
            done = self.eye_engine.calibrator.capture_point(ratio_x, ratio_y)
            if done:
                self._calibrating = False
                self.window.calibration_overlay.advance()  # hides overlay
                self.window.signals.status_update.emit("Calibration complete ✓")
                logger.info("Calibration complete!")
            else:
                self.window.calibration_overlay.advance()

    def _on_dwell(self, region_id: Optional[str], progress: float, just_triggered: bool = False):
        if region_id:
            self.window.signals.dwell_progress.emit(region_id, progress)
            if just_triggered:
                # Dwell trigger! Fired exactly ONCE upon dwell completion
                from core_bridge.cross_comm import EMERGENCY_PRESETS
                preset = next((p for p in EMERGENCY_PRESETS if p["id"] == region_id), None)
                if preset:
                    logger.info("Eye dwell activated preset: %s (%s)", region_id, preset["label"])
                    self.bus.publish_emergency(region_id)
                else:
                    logger.info("Eye dwell activated selection: %s", region_id)
                    self.bus.publish_eye_selection(region_id)
        else:
            self.window.signals.dwell_progress.emit("", 0.0)

    def _on_frame(self, frame):
        self.window.signals.frame_ready.emit(frame)

    def _get_current_ratio(self):
        """Get current pupil ratio from eye engine for calibration capture."""
        if self.eye_engine:
            return self.eye_engine.get_latest_ratio()
        return 0.5, 0.5

    # ────────────────────── Voice Callbacks ───

    def _on_stt_text(self, text: str):
        self.window.signals.stt_text.emit(text)
        # Publish to bus
        self.bus.publish_voice_command(text)
        # Check for emergency keywords
        lower = text.lower()
        keyword_map = {
            "water": "water",
            "help": "sos",
            "emergency": "sos",
            "lights": "lights_off",
            "caregiver": "caregiver",
            "call": "caregiver",
            "pain": "pain",
        }
        for kw, preset_id in keyword_map.items():
            if kw in lower:
                self.bus.publish_emergency(preset_id)
                break

    def _on_stt_error(self, error: str):
        self.window.signals.stt_error.emit(error)

    # ────────────────────── UI Wiring ─────────

    def _wire_ui(self):
        # Mode switch behaviour
        orig_switch = self.window.switch_mode

        def enhanced_switch(mode: str):
            orig_switch(mode)
            self._on_mode_switch(mode)

        self.window.switch_mode = enhanced_switch

        # Calibrate button
        self.window.calibrate_button.clicked.connect(self._start_calibration)

        # Text input → bus
        self.window.signals.comm_message.connect(self._on_ui_message)

    def _on_mode_switch(self, mode: str):
        if mode == "voice":
            if self.stt and not self.stt.is_running:
                self.stt.start()
                self.window.signals.status_update.emit("Voice mode ON — listening…")
        else:
            if self.stt and self.stt.is_running:
                self.stt.stop()
                self.window.signals.status_update.emit(f"Voice STT paused — mode: {mode}")

        if mode == "deaf":
            self.window.signals.status_update.emit("Deaf/Caption mode — live captions active")
            # Also start STT for captions even in deaf mode
            if self.stt and not self.stt.is_running:
                self.stt.start()

    def _on_ui_message(self, msg_type: str, payload: str):
        if msg_type == "TEXT_INPUT":
            self.bus.publish_text_input(payload)
            # Speak it aloud for the user
            self.bus.request_speak(payload, source="user_text")
            # Display in captions
            self.bus.request_display(f"You: {payload}", source="user_text")
        elif msg_type == "ACTION":
            self.bus.publish_emergency(payload)

    # ────────────────────── Calibration ───────

    def _start_calibration(self):
        if not self.eye_engine or not self.eye_engine.is_running:
            self.window.signals.status_update.emit("Cannot calibrate — eye engine not running")
            return
        self._calibrating = True
        self.eye_engine.calibrator.reset()
        screen = self.qt_app.primaryScreen()
        geo = screen.geometry()
        self.window.calibration_overlay.begin(geo.width(), geo.height())
        self.window.signals.status_update.emit("Calibration: look at dot and blink")
        logger.info("Calibration started")

    # ────────────────────── Dwell Sync ────────

    def _sync_dwell_regions(self):
        if self.eye_engine and self.eye_engine.is_running:
            regions = self.window.get_dwell_regions()
            self.eye_engine.set_dwell_regions(regions)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    session = NeuroLinkSession()
    sys.exit(session.run())


if __name__ == "__main__":
    main()
