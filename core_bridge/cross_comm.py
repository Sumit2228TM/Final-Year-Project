"""
cross_comm.py — The Cross-Communication Matrix
================================================
Unified message broker routing between modalities:

Inbound:   EYE_SELECTION  |  VOICE_COMMAND  |  TEXT_INPUT
Outbound:  DISPLAY_TEXT   |  SPEAK_ALOUD    |  NAVIGATE_UI

Includes emergency presets accessible via any input modality.

Design note:
    The bus uses a publish/subscribe model built on Python's queue.Queue.
    Subscribers register for specific MessageType(s) and receive CommMessage
    objects.  The bus runs a dispatcher thread that drains the inbound queue
    and fans out to all matching subscribers — this keeps publishers (eye engine,
    STT, etc.) completely decoupled from consumers (UI, TTS, etc.).
"""

from __future__ import annotations

import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("neurolink.cross_comm")


# ─────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────

class MessageType(Enum):
    """All inbound & outbound message types on the bus."""
    # Inbound (from engines)
    EYE_SELECTION = auto()
    VOICE_COMMAND = auto()
    TEXT_INPUT = auto()

    # Outbound (to consumers)
    DISPLAY_TEXT = auto()
    SPEAK_ALOUD = auto()
    NAVIGATE_UI = auto()

    # System
    EMERGENCY = auto()
    MODE_SWITCH = auto()
    CALIBRATION = auto()


# ─────────────────────────────────────────────
# Message Structure
# ─────────────────────────────────────────────

@dataclass
class CommMessage:
    """A single message on the CrossCommBus."""
    msg_type: MessageType
    payload: str = ""
    source: str = ""           # e.g. "eye_engine", "stt", "user_text"
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"CommMessage({self.msg_type.name}, "
                f"payload={self.payload!r:.40}, "
                f"source={self.source})")


# ─────────────────────────────────────────────
# Emergency Presets
# ─────────────────────────────────────────────

EMERGENCY_PRESETS: List[Dict[str, str]] = [
    {"id": "water",      "label": "I need water",      "icon": "💧"},
    {"id": "sos",        "label": "Emergency SOS",      "icon": "🚨"},
    {"id": "lights_off", "label": "Turn off lights",    "icon": "💡"},
    {"id": "caregiver",  "label": "Call caregiver",     "icon": "📞"},
    {"id": "pain",       "label": "Pain detected",      "icon": "⚠️"},
    {"id": "yes",        "label": "Yes",                "icon": "✅"},
    {"id": "no",         "label": "No",                 "icon": "❌"},
    {"id": "thank_you",  "label": "Thank you",          "icon": "🙏"},
]


# ─────────────────────────────────────────────
# Subscription Handle
# ─────────────────────────────────────────────

@dataclass
class _Subscription:
    callback: Callable[[CommMessage], None]
    types: Set[MessageType]
    name: str = ""


# ─────────────────────────────────────────────
# The Bus
# ─────────────────────────────────────────────

class CrossCommBus:
    """
    Publish/subscribe message bus for cross-modal communication.

    Publishers call `publish(msg)`.
    Subscribers call `subscribe(callback, types)` and receive matching messages.
    """

    def __init__(self):
        self._queue: queue.Queue[Optional[CommMessage]] = queue.Queue()
        self._subscribers: List[_Subscription] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._message_log: List[CommMessage] = []

    # ── Lifecycle ─────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="CommBus")
        self._thread.start()
        logger.info("CrossCommBus started")

    def stop(self):
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("CrossCommBus stopped")

    # ── Pub/Sub ───────────────────────────────

    def subscribe(
        self,
        callback: Callable[[CommMessage], None],
        types: Optional[Set[MessageType]] = None,
        name: str = "",
    ):
        """
        Register a subscriber.  If `types` is None, subscribe to ALL types.
        """
        with self._lock:
            sub = _Subscription(
                callback=callback,
                types=types or set(MessageType),
                name=name,
            )
            self._subscribers.append(sub)
        logger.info("Subscriber registered: %s (types=%s)",
                     name or "<anon>",
                     [t.name for t in sub.types] if types else "ALL")

    def publish(self, msg: CommMessage):
        """Enqueue a message for dispatch."""
        self._queue.put(msg)

    # ── Convenience publishers ────────────────

    def publish_eye_selection(self, selection_id: str):
        self.publish(CommMessage(
            msg_type=MessageType.EYE_SELECTION,
            payload=selection_id,
            source="eye_engine",
        ))

    def publish_voice_command(self, text: str):
        self.publish(CommMessage(
            msg_type=MessageType.VOICE_COMMAND,
            payload=text,
            source="stt",
        ))

    def publish_text_input(self, text: str, source: str = "keyboard"):
        self.publish(CommMessage(
            msg_type=MessageType.TEXT_INPUT,
            payload=text,
            source=source,
        ))

    def publish_emergency(self, preset_id: str):
        preset = next((p for p in EMERGENCY_PRESETS if p["id"] == preset_id), None)
        label = preset["label"] if preset else preset_id
        self.publish(CommMessage(
            msg_type=MessageType.EMERGENCY,
            payload=label,
            source="emergency",
            metadata={"preset_id": preset_id},
        ))

    def request_speak(self, text: str, source: str = "system"):
        self.publish(CommMessage(
            msg_type=MessageType.SPEAK_ALOUD,
            payload=text,
            source=source,
        ))

    def request_display(self, text: str, source: str = "system"):
        self.publish(CommMessage(
            msg_type=MessageType.DISPLAY_TEXT,
            payload=text,
            source=source,
        ))

    # ── History ───────────────────────────────

    @property
    def message_log(self) -> List[CommMessage]:
        return list(self._message_log)

    # ── Dispatcher ────────────────────────────

    def _dispatch_loop(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg is None:
                break

            self._message_log.append(msg)
            logger.debug("Bus dispatch: %s", msg)

            with self._lock:
                subs = list(self._subscribers)

            for sub in subs:
                if msg.msg_type in sub.types:
                    try:
                        sub.callback(msg)
                    except Exception:
                        logger.exception("Subscriber %s error handling %s",
                                          sub.name, msg.msg_type.name)
