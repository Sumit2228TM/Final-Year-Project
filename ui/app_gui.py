"""
app_gui.py — NeuroLink Assist High-Contrast Accessible UI (PyQt5)
===================================================================
Desktop-first interface with:
  • Zero-login boot straight into action grid
  • Three modes: Eye-Gaze, Voice Control, Deaf Captions
  • Dwell-selectable action grid with progress ring feedback
  • Emergency preset buttons (quick phrases)
  • Live caption window for hearing-impaired users
  • 5-point calibration overlay
  • Camera preview widget (for eye engine debug / confidence)
  • Dark high-contrast colour scheme (WCAG AAA compliant)

All heavy work (camera, STT, TTS) runs off the Qt main thread.
"""

from __future__ import annotations

import sys
import math
import time
import logging
from typing import Optional, Dict, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTextEdit, QFrame, QStackedWidget,
    QSizePolicy, QLineEdit, QScrollArea, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QSize, QRectF, QPropertyAnimation,
    QEasingCurve, pyqtProperty, QPointF, QPoint,
)
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPixmap, QImage, QPalette, QIcon, QFontDatabase,
)

logger = logging.getLogger("neurolink.ui")

# ─────────────────────────────────────────────
# Colour palette — high-contrast dark theme
# ─────────────────────────────────────────────
class Palette:
    BG_PRIMARY   = "#0D0D1A"   # Deep dark blue-black
    BG_SECONDARY = "#141428"   # Slightly lighter panel
    BG_CARD      = "#1C1C3A"   # Card / button default
    BG_HOVER     = "#2A2A55"   # Hover state
    BG_ACTIVE    = "#3A2A70"   # Pressed / active

    ACCENT       = "#7B61FF"   # Vivid purple accent
    ACCENT_GLOW  = "#A78BFA"   # Lighter accent for glows
    ACCENT_GREEN = "#34D399"   # Success / positive
    ACCENT_RED   = "#EF4444"   # Emergency / error
    ACCENT_AMBER = "#FBBF24"   # Warning / calibration
    ACCENT_CYAN  = "#22D3EE"   # Info / captions

    TEXT_PRIMARY  = "#F0F0FF"  # Near-white
    TEXT_SECONDARY= "#A0A0CC"  # Muted
    TEXT_DIM      = "#606088"  # Very subtle

    BORDER        = "#2E2E55"
    BORDER_FOCUS  = "#7B61FF"

    SHADOW        = "#000000"

# ─────────────────────────────────────────────
# Thread-safe signal bridge (engine → UI)
# ─────────────────────────────────────────────
class _Signals(QObject):
    gaze_update = pyqtSignal(float, float, float)       # x, y, confidence
    blink_detected = pyqtSignal()
    dwell_progress = pyqtSignal(str, float)              # region_id, progress [0-1]
    stt_text = pyqtSignal(str)                           # recognized text
    stt_error = pyqtSignal(str)
    comm_message = pyqtSignal(str, str)                  # type, payload
    calibration_next = pyqtSignal(int, str, float, float) # index, label, sx, sy
    status_update = pyqtSignal(str)                      # generic status text
    frame_ready = pyqtSignal(object)                     # numpy frame for preview


# ─────────────────────────────────────────────
# Gaze dot overlay
# ─────────────────────────────────────────────
class _GazeDot(QWidget):
    """Translucent gaze cursor drawn on top of the entire window."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._progress = 0.0
        self.setVisible(False)

    def set_progress(self, p: float):
        self._progress = max(0.0, min(1.0, p))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = QPointF(22, 22)
        # Outer progress ring
        if self._progress > 0.01:
            pen = QPen(QColor(Palette.ACCENT_GREEN), 4)
            p.setPen(pen)
            span = int(self._progress * 360 * 16)
            p.drawArc(QRectF(3, 3, 38, 38), 90 * 16, -span)
        else:
            # Subtle cyan halo when roaming
            pen = QPen(QColor(Palette.ACCENT_CYAN + "66"), 2)
            p.setPen(pen)
            p.drawEllipse(center, 18, 18)

        # Inner glowing dot
        p.setPen(Qt.NoPen)
        grad = QRadialGradient(center, 10)
        grad.setColorAt(0, QColor(Palette.ACCENT_GLOW))
        grad.setColorAt(0.7, QColor(Palette.ACCENT))
        grad.setColorAt(1, QColor(Palette.ACCENT + "44"))
        p.setBrush(QBrush(grad))
        p.drawEllipse(center, 8, 8)
        p.end()


# ─────────────────────────────────────────────
# Dwell-aware action button
# ─────────────────────────────────────────────
class ActionButton(QPushButton):
    """Big, high-contrast grid button with dwell progress ring."""

    def __init__(self, btn_id: str, label: str, icon_text: str = "", parent=None):
        super().__init__(parent)
        self.btn_id = btn_id
        self._progress = 0.0
        self._label = label
        self._icon_text = icon_text
        self.setMinimumSize(140, 110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self._style_normal())
        self.setFont(QFont("Segoe UI", 13, QFont.Bold))

    def _style_normal(self):
        return f"""
            QPushButton {{
                background: {Palette.BG_CARD};
                color: {Palette.TEXT_PRIMARY};
                border: 2px solid {Palette.BORDER};
                border-radius: 16px;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: {Palette.BG_HOVER};
                border-color: {Palette.ACCENT};
            }}
            QPushButton:pressed {{
                background: {Palette.BG_ACTIVE};
            }}
        """

    def set_dwell_progress(self, p: float):
        self._progress = max(0.0, min(1.0, p))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw icon + label manually for nicer layout
        rect = self.rect()
        icon_font = QFont("Segoe UI Emoji", 28)
        painter.setFont(icon_font)
        painter.setPen(QColor(Palette.TEXT_PRIMARY))
        icon_rect = QRectF(rect.x(), rect.y() + 10, rect.width(), 44)
        painter.drawText(icon_rect, Qt.AlignCenter, self._icon_text)

        label_font = QFont("Segoe UI", 11, QFont.DemiBold)
        painter.setFont(label_font)
        painter.setPen(QColor(Palette.TEXT_PRIMARY))
        label_rect = QRectF(rect.x(), rect.y() + 56, rect.width(), 40)
        painter.drawText(label_rect, Qt.AlignCenter, self._label)

        # Dwell progress ring
        if self._progress > 0.01:
            pen = QPen(QColor(Palette.ACCENT_GREEN), 4)
            painter.setPen(pen)
            inset = 6
            arc_rect = QRectF(inset, inset, rect.width() - 2 * inset, rect.height() - 2 * inset)
            span_angle = int(self._progress * 360 * 16)
            painter.drawArc(arc_rect, 90 * 16, -span_angle)

        painter.end()

    def get_screen_rect(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) in global screen coordinates."""
        pos = self.mapToGlobal(self.rect().topLeft())
        size = self.size()
        return (pos.x(), pos.y(), pos.x() + size.width(), pos.y() + size.height())


# ─────────────────────────────────────────────
# Calibration Overlay
# ─────────────────────────────────────────────
class _CalibrationOverlay(QWidget):
    """
    Full-screen overlay for 5-point gaze calibration.
    Shows a pulsing target dot that the user fixates on.
    """
    point_captured = pyqtSignal()       # emitted when user dwells/blinks on target
    calibration_done = pyqtSignal()

    LABELS = ["Top-Left", "Top-Right", "Center", "Bottom-Left", "Bottom-Right"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {Palette.BG_PRIMARY};")
        self._current_idx = 0
        self._targets = []
        self._pulse_phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.setInterval(30)
        self.setVisible(False)

        self._label = QLabel("Look at the pulsing dot and blink", self)
        self._label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self._label.setStyleSheet(f"color: {Palette.ACCENT_AMBER}; background: transparent;")
        self._label.setAlignment(Qt.AlignCenter)

    def begin(self, screen_w: int, screen_h: int):
        margin_x = int(screen_w * 0.1)
        margin_y = int(screen_h * 0.1)
        self._targets = [
            (margin_x, margin_y),
            (screen_w - margin_x, margin_y),
            (screen_w // 2, screen_h // 2),
            (margin_x, screen_h - margin_y),
            (screen_w - margin_x, screen_h - margin_y),
        ]
        self._current_idx = 0
        self._label.setGeometry(0, screen_h // 2 + 60, screen_w, 40)
        self.setVisible(True)
        self.raise_()
        self._timer.start()

    def advance(self):
        self._current_idx += 1
        if self._current_idx >= len(self._targets):
            self._timer.stop()
            self.setVisible(False)
            self.calibration_done.emit()
        else:
            self._update_label()

    def _update_label(self):
        if self._current_idx < len(self.LABELS):
            self._label.setText(
                f"Point {self._current_idx + 1}/5: {self.LABELS[self._current_idx]} — blink when ready"
            )

    def _animate(self):
        self._pulse_phase += 0.08
        self.update()

    @property
    def current_target(self) -> Optional[Tuple[int, int]]:
        if self._current_idx < len(self._targets):
            return self._targets[self._current_idx]
        return None

    def paintEvent(self, _):
        if not self._targets or self._current_idx >= len(self._targets):
            return
        tx, ty = self._targets[self._current_idx]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Pulse radius
        base_r = 18
        pulse = base_r + 6 * math.sin(self._pulse_phase)
        # Outer glow
        grad = QRadialGradient(QPointF(tx, ty), pulse * 2)
        grad.setColorAt(0, QColor(Palette.ACCENT_AMBER + "AA"))
        grad.setColorAt(1, QColor(Palette.ACCENT_AMBER + "00"))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(tx, ty), pulse * 2, pulse * 2)
        # Inner dot
        p.setBrush(QBrush(QColor(Palette.ACCENT_AMBER)))
        p.drawEllipse(QPointF(tx, ty), pulse, pulse)
        # Cross-hair
        pen = QPen(QColor(Palette.TEXT_PRIMARY), 1)
        p.setPen(pen)
        p.drawLine(int(tx - 30), ty, int(tx + 30), ty)
        p.drawLine(tx, int(ty - 30), tx, int(ty + 30))
        p.end()


# ─────────────────────────────────────────────
# Main Application Window
# ─────────────────────────────────────────────
class NeuroLinkApp(QMainWindow):
    """
    Primary UI container.

    Layout:
    ┌─────────────────────────────────────┐
    │  Header (title + mode switcher)     │
    ├──────────────┬──────────────────────┤
    │  Action Grid │  Side Panel          │
    │  (emergency  │  (captions / camera  │
    │   presets +  │   preview / status)  │
    │   custom)    │                      │
    ├──────────────┴──────────────────────┤
    │  Footer (status bar + text input)   │
    └─────────────────────────────────────┘
    """

    # Expose signals for engine → UI bridging
    signals = _Signals()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroLink Assist v1.0")
        self.setMinimumSize(1100, 720)

        # Apply global palette
        self._apply_dark_theme()

        # State (must be initialized before building widgets)
        self._current_mode = "eye"   # "eye" | "voice" | "deaf"
        self._action_buttons: Dict[str, ActionButton] = {}

        # Central
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        root_layout.addWidget(self._build_header())

        # Body: action grid + side panel
        body = QHBoxLayout()
        body.setContentsMargins(16, 8, 16, 8)
        body.setSpacing(12)
        self._action_grid_widget = self._build_action_grid()
        body.addWidget(self._action_grid_widget, stretch=3)
        body.addWidget(self._build_side_panel(), stretch=2)
        root_layout.addLayout(body, stretch=1)

        # Footer
        root_layout.addWidget(self._build_footer())

        # Calibration overlay (stacked on top)
        self._calibration_overlay = _CalibrationOverlay(central)

        # Gaze dot
        self._gaze_dot = _GazeDot(central)


        # Wire signals
        self._connect_signals()

    # ── Theme ─────────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {Palette.BG_PRIMARY};
            }}
            QWidget {{
                background: transparent;
                color: {Palette.TEXT_PRIMARY};
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }}
            QScrollBar:vertical {{
                background: {Palette.BG_SECONDARY};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Palette.BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    # ── Header ────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(72)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Palette.BG_SECONDARY}, stop:1 {Palette.BG_CARD});
                border-bottom: 1px solid {Palette.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        # Title
        title = QLabel("⚡ NeuroLink Assist")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet(f"color: {Palette.ACCENT_GLOW};")
        layout.addWidget(title)

        layout.addStretch()

        # Mode switcher buttons
        self._mode_buttons = {}
        for mode_id, mode_label, mode_icon in [
            ("eye", "Eye Gaze", "👁️"),
            ("voice", "Voice", "🎙️"),
            ("deaf", "Captions", "📝"),
        ]:
            btn = QPushButton(f" {mode_icon} {mode_label}")
            btn.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
            btn.setFixedHeight(42)
            btn.setMinimumWidth(130)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode_id: self.switch_mode(m))
            self._mode_buttons[mode_id] = btn
            layout.addWidget(btn)

        # Calibrate button
        self._btn_calibrate = QPushButton("🎯 Calibrate")
        self._btn_calibrate.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        self._btn_calibrate.setFixedHeight(42)
        self._btn_calibrate.setMinimumWidth(110)
        self._btn_calibrate.setCursor(Qt.PointingHandCursor)
        self._btn_calibrate.setStyleSheet(f"""
            QPushButton {{
                background: {Palette.ACCENT_AMBER}22;
                color: {Palette.ACCENT_AMBER};
                border: 2px solid {Palette.ACCENT_AMBER};
                border-radius: 10px;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: {Palette.ACCENT_AMBER}44; }}
        """)
        layout.addWidget(self._btn_calibrate)

        self._update_mode_styles()
        return header

    def _update_mode_styles(self):
        for mode_id, btn in self._mode_buttons.items():
            if mode_id == self._current_mode:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Palette.ACCENT};
                        color: {Palette.TEXT_PRIMARY};
                        border: none;
                        border-radius: 10px;
                        padding: 0 16px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Palette.BG_CARD};
                        color: {Palette.TEXT_SECONDARY};
                        border: 1px solid {Palette.BORDER};
                        border-radius: 10px;
                        padding: 0 16px;
                    }}
                    QPushButton:hover {{
                        background: {Palette.BG_HOVER};
                        color: {Palette.TEXT_PRIMARY};
                    }}
                """)

    # ── Action Grid ───────────────────────────

    def _build_action_grid(self) -> QWidget:
        from core_bridge.cross_comm import EMERGENCY_PRESETS

        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {Palette.BG_SECONDARY};
                border-radius: 16px;
                border: 1px solid {Palette.BORDER};
            }}
        """)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(16, 12, 16, 12)

        # Section label
        lbl = QLabel("Quick Actions")
        lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl.setStyleSheet(f"color: {Palette.ACCENT_GLOW};")
        outer.addWidget(lbl)

        grid = QGridLayout()
        grid.setSpacing(10)

        for idx, preset in enumerate(EMERGENCY_PRESETS):
            btn = ActionButton(
                btn_id=preset["id"],
                label=preset["label"],
                icon_text=preset["icon"],
            )
            btn.clicked.connect(lambda checked, pid=preset["id"]: self._on_action_clicked(pid))
            row, col = divmod(idx, 4)
            grid.addWidget(btn, row, col)
            self._action_buttons[preset["id"]] = btn

        outer.addLayout(grid)
        return container

    # ── Side Panel ────────────────────────────

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background: {Palette.BG_SECONDARY};
                border-radius: 16px;
                border: 1px solid {Palette.BORDER};
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)

        # Camera preview placeholder
        cam_label = QLabel("Camera Preview")
        cam_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        cam_label.setStyleSheet(f"color: {Palette.ACCENT_GLOW};")
        layout.addWidget(cam_label)

        self._camera_preview = QLabel()
        self._camera_preview.setFixedHeight(200)
        self._camera_preview.setStyleSheet(f"""
            background: {Palette.BG_PRIMARY};
            border: 1px solid {Palette.BORDER};
            border-radius: 10px;
        """)
        self._camera_preview.setAlignment(Qt.AlignCenter)
        self._camera_preview.setText("📷  Camera feed will appear here")
        self._camera_preview.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self._camera_preview)

        # Caption / transcript area
        cap_label = QLabel("Live Captions")
        cap_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        cap_label.setStyleSheet(f"color: {Palette.ACCENT_CYAN};")
        layout.addWidget(cap_label)

        self._caption_area = QTextEdit()
        self._caption_area.setReadOnly(True)
        self._caption_area.setFont(QFont("Segoe UI", 13))
        self._caption_area.setStyleSheet(f"""
            QTextEdit {{
                background: {Palette.BG_PRIMARY};
                color: {Palette.ACCENT_CYAN};
                border: 1px solid {Palette.BORDER};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        self._caption_area.setPlaceholderText("Recognised speech will appear here…")
        layout.addWidget(self._caption_area, stretch=1)

        # Status line
        self._status_label = QLabel("Status: Ready")
        self._status_label.setFont(QFont("Segoe UI", 10))
        self._status_label.setStyleSheet(f"color: {Palette.TEXT_DIM};")
        layout.addWidget(self._status_label)

        return panel

    # ── Footer ────────────────────────────────

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setStyleSheet(f"""
            QFrame {{
                background: {Palette.BG_SECONDARY};
                border-top: 1px solid {Palette.BORDER};
            }}
        """)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 8, 24, 8)

        # Text input
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Type a message to speak aloud…")
        self._text_input.setFont(QFont("Segoe UI", 13))
        self._text_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Palette.BG_PRIMARY};
                color: {Palette.TEXT_PRIMARY};
                border: 2px solid {Palette.BORDER};
                border-radius: 12px;
                padding: 8px 16px;
            }}
            QLineEdit:focus {{
                border-color: {Palette.ACCENT};
            }}
        """)
        self._text_input.returnPressed.connect(self._on_text_submit)
        layout.addWidget(self._text_input, stretch=1)

        send_btn = QPushButton("Send ➤")
        send_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        send_btn.setFixedSize(100, 42)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Palette.ACCENT};
                color: white;
                border: none;
                border-radius: 12px;
            }}
            QPushButton:hover {{ background: {Palette.ACCENT_GLOW}; }}
        """)
        send_btn.clicked.connect(self._on_text_submit)
        layout.addWidget(send_btn)

        # Gaze position readout
        self._gaze_readout = QLabel("Gaze: — , —")
        self._gaze_readout.setFont(QFont("Consolas", 9))
        self._gaze_readout.setStyleSheet(f"color: {Palette.TEXT_DIM};")
        self._gaze_readout.setFixedWidth(160)
        layout.addWidget(self._gaze_readout)

        return footer

    # ── Signal Wiring ─────────────────────────

    def _connect_signals(self):
        self.signals.gaze_update.connect(self._handle_gaze)
        self.signals.blink_detected.connect(self._handle_blink)
        self.signals.dwell_progress.connect(self._handle_dwell)
        self.signals.stt_text.connect(self._handle_stt_text)
        self.signals.stt_error.connect(self._handle_stt_error)
        self.signals.comm_message.connect(self._handle_comm_message)
        self.signals.status_update.connect(self._handle_status)
        self.signals.frame_ready.connect(self._handle_frame)

    # ── Signal Handlers ───────────────────────

    def _handle_gaze(self, x: float, y: float, confidence: float):
        self._gaze_readout.setText(f"Gaze: {x:.0f} , {y:.0f}")
        # Move gaze dot overlay
        if confidence > 0.2:
            self._gaze_dot.setVisible(True)
            global_pt = QPoint(int(x), int(y))
            local_pt = self.centralWidget().mapFromGlobal(global_pt)
            gx = local_pt.x() - self._gaze_dot.width() // 2
            gy = local_pt.y() - self._gaze_dot.height() // 2
            self._gaze_dot.move(gx, gy)
            self._gaze_dot.raise_()
        else:
            self._gaze_dot.setVisible(False)

    def _handle_blink(self):
        self._status_label.setText("Status: Blink detected! ✓")
        self._status_label.setStyleSheet(f"color: {Palette.ACCENT_GREEN};")
        QTimer.singleShot(1500, lambda: self._status_label.setStyleSheet(
            f"color: {Palette.TEXT_DIM};"
        ))

    def _handle_dwell(self, region_id: str, progress: float):
        # Update button progress ring
        for bid, btn in self._action_buttons.items():
            if bid == region_id:
                btn.set_dwell_progress(progress)
            else:
                btn.set_dwell_progress(0.0)
        # Update gaze dot ring
        self._gaze_dot.set_progress(progress)

    def _handle_stt_text(self, text: str):
        self._caption_area.append(f"🗣️  {text}")
        scrollbar = self._caption_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _handle_stt_error(self, error: str):
        self._caption_area.append(f"⚠️  {error}")

    def _handle_comm_message(self, msg_type: str, payload: str):
        self._caption_area.append(f"[{msg_type}] {payload}")

    def _handle_status(self, text: str):
        self._status_label.setText(f"Status: {text}")

    def _handle_frame(self, frame):
        """Display a camera frame in the preview widget."""
        try:
            import numpy as np
            if frame is None:
                return
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self._camera_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._camera_preview.setPixmap(pixmap)
        except Exception:
            pass

    # ── Mode Switching ────────────────────────

    def switch_mode(self, mode: str):
        self._current_mode = mode
        self._update_mode_styles()
        self.signals.status_update.emit(f"Mode → {mode.upper()}")
        logger.info("Mode switched to %s", mode)

    # ── Actions ───────────────────────────────

    def _on_action_clicked(self, preset_id: str):
        logger.info("Action clicked: %s", preset_id)
        self.signals.comm_message.emit("ACTION", preset_id)

    def _on_text_submit(self):
        text = self._text_input.text().strip()
        if text:
            self._text_input.clear()
            self.signals.comm_message.emit("TEXT_INPUT", text)

    # ── Dwell Region Export ───────────────────

    def get_dwell_regions(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Return screen-space bounding boxes for all action buttons."""
        regions = {}
        for bid, btn in self._action_buttons.items():
            regions[bid] = btn.get_screen_rect()
        return regions

    # ── Calibration ───────────────────────────

    @property
    def calibration_overlay(self) -> _CalibrationOverlay:
        return self._calibration_overlay

    @property
    def calibrate_button(self) -> QPushButton:
        return self._btn_calibrate

    # ── Resize handling for overlay ───────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._calibration_overlay.setGeometry(self.centralWidget().rect())
