"""
eye_engine.py — EyeUnlock Engine
=================================
MediaPipe Face Mesh + OpenCV pipeline for:
  • Iris landmark tracking (Left Iris [468-472], Right Iris [473-477])
  • Eye contour tracking ([33,133,159,145] left; [362,263,386,374] right)
  • 5-point gaze calibration mapping normalized pupil ratios → screen coords
  • Exponential smoothing filter for jitter reduction
  • Dwell-time trigger with configurable interval
  • Blink detection via Eye Aspect Ratio (EAR)

All capture runs in a dedicated daemon thread — never blocks the UI.
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None  # Graceful fallback — eye engine won't start but app continues

logger = logging.getLogger("neurolink.eye_engine")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# MediaPipe Face Mesh landmark indices
LEFT_EYE_CONTOUR = [33, 133, 159, 145]   # left-corner, right-corner, top, bottom
RIGHT_EYE_CONTOUR = [362, 263, 386, 374]
LEFT_IRIS = list(range(468, 473))   # 468-472
RIGHT_IRIS = list(range(473, 478))  # 473-477

# Defaults
DEFAULT_DWELL_SECONDS = 1.5
DEFAULT_BLINK_THRESHOLD_EAR = 0.21
DEFAULT_LONG_BLINK_MS = 400
DEFAULT_SMOOTHING_ALPHA = 0.35
DEFAULT_CAMERA_INDEX = 0


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class GazePoint:
    """Screen-space gaze coordinate after calibration + smoothing."""
    x: float = 0.0
    y: float = 0.0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class CalibrationPoint:
    """A single calibration reference: screen target → measured pupil ratio."""
    screen_x: float
    screen_y: float
    pupil_ratio_x: float = 0.0
    pupil_ratio_y: float = 0.0
    captured: bool = False


# ─────────────────────────────────────────────
# Blink Detector
# ─────────────────────────────────────────────

class BlinkDetector:
    """
    Detects intentional long blinks via Eye Aspect Ratio (EAR).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    When EAR drops below threshold for > blink_duration_ms → manual click.
    """

    def __init__(
        self,
        ear_threshold: float = DEFAULT_BLINK_THRESHOLD_EAR,
        long_blink_ms: float = DEFAULT_LONG_BLINK_MS,
    ):
        self.ear_threshold = ear_threshold
        self.long_blink_ms = long_blink_ms
        self._eye_closed_since: Optional[float] = None
        self._blink_triggered = False

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def compute_ear(self, landmarks: np.ndarray, eye_indices: list) -> float:
        """Compute Eye Aspect Ratio for a set of eye contour indices."""
        # eye_indices: [left_corner, right_corner, top, bottom]
        p1 = landmarks[eye_indices[0]]  # left corner
        p4 = landmarks[eye_indices[1]]  # right corner
        p2 = landmarks[eye_indices[2]]  # top
        p6 = landmarks[eye_indices[3]]  # bottom
        horizontal = self._distance(p1, p4)
        if horizontal < 1e-6:
            return 1.0  # Avoid division by zero
        vertical = self._distance(p2, p6)
        return vertical / horizontal

    def update(self, avg_ear: float) -> bool:
        """
        Feed the average EAR of both eyes. Returns True once per long-blink event.
        """
        now = time.time()
        if avg_ear < self.ear_threshold:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
                self._blink_triggered = False
            elif (
                not self._blink_triggered
                and (now - self._eye_closed_since) * 1000 >= self.long_blink_ms
            ):
                self._blink_triggered = True
                return True
        else:
            self._eye_closed_since = None
            self._blink_triggered = False
        return False


# ─────────────────────────────────────────────
# Gaze Calibrator
# ─────────────────────────────────────────────

class GazeCalibrator:
    """
    5-point calibration mapping normalized pupil-to-eye-corner ratios
    to screen coordinates using least-squares affine transform.

    Calibration points:
        TL  ·  ·  TR
         ·  C  ·
        BL  ·  ·  BR
    """

    POINT_LABELS = ["Top-Left", "Top-Right", "Center", "Bottom-Left", "Bottom-Right"]

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        margin_x = int(screen_w * 0.1)
        margin_y = int(screen_h * 0.1)
        self.targets: List[CalibrationPoint] = [
            CalibrationPoint(margin_x, margin_y),                          # TL
            CalibrationPoint(screen_w - margin_x, margin_y),               # TR
            CalibrationPoint(screen_w // 2, screen_h // 2),                # C
            CalibrationPoint(margin_x, screen_h - margin_y),               # BL
            CalibrationPoint(screen_w - margin_x, screen_h - margin_y),    # BR
        ]
        self._transform_x: Optional[np.ndarray] = None  # [a, b, c] for x = a*rx + b*ry + c
        self._transform_y: Optional[np.ndarray] = None
        self.calibrated = False

    @property
    def current_index(self) -> int:
        """Index of the next un-captured calibration point."""
        for i, cp in enumerate(self.targets):
            if not cp.captured:
                return i
        return len(self.targets)

    @property
    def is_complete(self) -> bool:
        return all(cp.captured for cp in self.targets)

    def capture_point(self, ratio_x: float, ratio_y: float) -> bool:
        """Capture the pupil ratio for the current calibration target. Returns True if all done."""
        idx = self.current_index
        if idx >= len(self.targets):
            return True
        self.targets[idx].pupil_ratio_x = ratio_x
        self.targets[idx].pupil_ratio_y = ratio_y
        self.targets[idx].captured = True
        logger.info("Calibration point %d (%s) captured: ratio=(%.4f, %.4f)",
                     idx, self.POINT_LABELS[idx], ratio_x, ratio_y)
        if self.is_complete:
            self._solve_transform()
            return True
        return False

    def _solve_transform(self):
        """Fit affine transform via least-squares: screen = A * [ratio_x, ratio_y, 1]^T."""
        A = np.array([
            [cp.pupil_ratio_x, cp.pupil_ratio_y, 1.0]
            for cp in self.targets
        ])
        bx = np.array([cp.screen_x for cp in self.targets])
        by = np.array([cp.screen_y for cp in self.targets])
        self._transform_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        self._transform_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        self.calibrated = True
        logger.info("Gaze calibration complete. Transform X=%s  Y=%s",
                     self._transform_x, self._transform_y)

    def map_to_screen(self, ratio_x: float, ratio_y: float) -> Tuple[float, float]:
        """Map pupil ratio to screen coordinates using calibrated transform."""
        if not self.calibrated:
            # Fallback: linear mapping over [0,1] range
            return ratio_x * self.screen_w, ratio_y * self.screen_h
        v = np.array([ratio_x, ratio_y, 1.0])
        sx = float(np.dot(self._transform_x, v))
        sy = float(np.dot(self._transform_y, v))
        # Clamp to screen bounds
        sx = max(0.0, min(self.screen_w, sx))
        sy = max(0.0, min(self.screen_h, sy))
        return sx, sy

    def reset(self):
        """Reset calibration to re-do the process."""
        for cp in self.targets:
            cp.captured = False
            cp.pupil_ratio_x = 0.0
            cp.pupil_ratio_y = 0.0
        self._transform_x = None
        self._transform_y = None
        self.calibrated = False


# ─────────────────────────────────────────────
# Exponential Smoothing Filter
# ─────────────────────────────────────────────

class _GazeSmoother:
    """Simple exponential moving average filter to reduce jitter."""

    def __init__(self, alpha: float = DEFAULT_SMOOTHING_ALPHA):
        self.alpha = alpha
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            self._sx = self.alpha * x + (1 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1 - self.alpha) * self._sy
        return self._sx, self._sy

    def reset(self):
        self._sx = self._sy = None


# ─────────────────────────────────────────────
# Dwell Timer
# ─────────────────────────────────────────────

class _DwellTimer:
    """Tracks how long gaze dwells inside a rectangular region."""

    def __init__(self, dwell_seconds: float = DEFAULT_DWELL_SECONDS, cooldown_seconds: float = 2.0):
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._region_id: Optional[str] = None
        self._enter_time: Optional[float] = None
        self._triggered = False
        self._last_trigger_time: float = 0.0

    def update(self, region_id: Optional[str]) -> Tuple[Optional[str], float, bool]:
        """
        Call each frame with the region_id the gaze is currently inside,
        or None if not inside any region.
        Returns (current_region_id | None, progress 0..1, just_triggered_bool).
        """
        now = time.time()
        if region_id is None or region_id != self._region_id:
            # Left previous region or entered a new one
            self._region_id = region_id
            self._enter_time = now if region_id else None
            self._triggered = False
            return region_id, 0.0, False

        if self._triggered:
            # Prevent rapid re-firing while gaze remains on the same button
            if now - self._last_trigger_time >= self.cooldown_seconds:
                self._triggered = False
                self._enter_time = now
                return region_id, 0.0, False
            return region_id, 1.0, False

        elapsed = now - self._enter_time
        progress = min(1.0, elapsed / self.dwell_seconds)
        if progress >= 1.0:
            self._triggered = True
            self._last_trigger_time = now
            return region_id, 1.0, True

        return region_id, progress, False

    def reset(self):
        self._region_id = None
        self._enter_time = None
        self._triggered = False
        self._last_trigger_time = 0.0


# ─────────────────────────────────────────────
# Main Eye Engine
# ─────────────────────────────────────────────

class EyeEngine:
    """
    Threaded eye-tracking engine.

    Usage:
        engine = EyeEngine(on_gaze=..., on_blink=..., on_dwell=...)
        engine.start()
        ...
        engine.stop()

    Callbacks:
        on_gaze(GazePoint)                   — called each frame with smoothed gaze
        on_blink()                           — called when a long blink is detected
        on_dwell(region_id, progress, just_triggered) — called when dwell updates
    """

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        dwell_seconds: float = DEFAULT_DWELL_SECONDS,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        screen_w: int = 1920,
        screen_h: int = 1080,
        on_gaze: Optional[Callable[[GazePoint], None]] = None,
        on_blink: Optional[Callable[[], None]] = None,
        on_dwell: Optional[Callable[[Optional[str], float, bool], None]] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.camera_index = camera_index
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.on_gaze = on_gaze
        self.on_blink = on_blink
        self.on_dwell = on_dwell
        self.on_frame = on_frame

        self.calibrator = GazeCalibrator(screen_w=screen_w, screen_h=screen_h)
        self.blink_detector = BlinkDetector()
        self._smoother = _GazeSmoother(alpha=smoothing_alpha)
        self._dwell = _DwellTimer(dwell_seconds=dwell_seconds)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

        # Dwell target regions: {id: (x1, y1, x2, y2)} in screen coordinates
        self._dwell_regions: dict[str, Tuple[float, float, float, float]] = {}

        # Latest raw gaze and ratio (for external queries / calibration)
        self._latest_gaze: Optional[GazePoint] = None
        self._latest_ratio: Tuple[float, float] = (0.5, 0.5)

        # MediaPipe Face Mesh instance (lazy init)
        self._face_mesh = None

    # ── Public API ─────────────────────────────

    def start(self):
        """Start the eye-tracking capture loop in a background thread."""
        if self._running:
            logger.warning("EyeEngine already running.")
            return
        if mp is None:
            logger.error("mediapipe is not installed — eye engine cannot start.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="EyeEngine")
        self._thread.start()
        logger.info("EyeEngine started (camera=%d)", self.camera_index)

    def stop(self):
        """Stop the capture loop and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("EyeEngine stopped.")

    def set_dwell_regions(self, regions: dict[str, Tuple[float, float, float, float]]):
        """Update the set of UI regions to watch for dwell events."""
        with self._lock:
            self._dwell_regions = dict(regions)

    @property
    def latest_gaze(self) -> Optional[GazePoint]:
        return self._latest_gaze

    @property
    def latest_ratio(self) -> Tuple[float, float]:
        return self._latest_ratio

    def get_latest_ratio(self) -> Tuple[float, float]:
        """Return the most recent normalized (ratio_x, ratio_y)."""
        return self._latest_ratio

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal pipeline ──────────────────────

    @staticmethod
    def _use_tasks_api() -> bool:
        """Detect which MediaPipe API is available at runtime."""
        return (mp is not None) and (not hasattr(mp, 'solutions'))

    def _init_face_mesh(self):
        """Initialize MediaPipe Face Mesh / Face Landmarker depending on API version."""
        if mp is None:
            logger.warning("MediaPipe is not available; eye tracking will run without landmark detection.")
            self._face_mesh = None
            self._face_landmarker = None
            return

        if self._use_tasks_api():
            # MediaPipe >= 1.0 — use tasks API with FaceLandmarker
            self._init_face_landmarker_tasks()
        else:
            # MediaPipe <= 0.10.x — use solutions API
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # needed for iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_landmarker = None
        logger.info("MediaPipe initialized (tasks_api=%s)", self._use_tasks_api())

    def _init_face_landmarker_tasks(self):
        """Initialize using mediapipe.tasks.vision.FaceLandmarker API (>= 1.0)."""
        import mediapipe.tasks as mpt
        import urllib.request
        import os

        # Download model if not cached
        model_dir = os.path.join(os.path.dirname(__file__), "_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker_v2_with_blendshapes.task")

        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            logger.info("Downloading face landmarker model → %s", model_path)
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                logger.error("Failed to download face landmarker model: %s", e)
                self._face_landmarker = None
                return

        BaseOptions = mpt.BaseOptions
        FaceLandmarker = mpt.vision.FaceLandmarker
        FaceLandmarkerOptions = mpt.vision.FaceLandmarkerOptions
        VisionRunningMode = mpt.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_landmarker = FaceLandmarker.create_from_options(options)
        self._face_mesh = None
        self._frame_ts_ms = 0

    def _capture_loop(self):
        """Main capture loop running in its own thread."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Cannot open camera %d — aborting eye engine.", self.camera_index)
            self._running = False
            return

        self._init_face_mesh()

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Frame grab failed — retrying…")
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)  # Mirror for natural interaction
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                landmarks = self._detect_landmarks(rgb)

                if landmarks is not None:
                    self._process_landmarks(landmarks, frame)
                else:
                    # No face detected — fallback: emit last known gaze with low confidence
                    if self._latest_gaze:
                        self._latest_gaze.confidence = 0.0

                # Emit frame for debug / UI overlay
                if self.on_frame:
                    self.on_frame(frame)

                # ~30 FPS cap
                time.sleep(0.016)
        except Exception:
            logger.exception("EyeEngine capture loop crashed")
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None
            if self._face_mesh:
                self._face_mesh.close()
                self._face_mesh = None
            if hasattr(self, '_face_landmarker') and self._face_landmarker:
                self._face_landmarker.close()
                self._face_landmarker = None
            self._running = False

    def _detect_landmarks(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Run face detection and return normalized landmarks array, or None.
        Supports both mediapipe solutions and tasks APIs.
        """
        if self._use_tasks_api() and hasattr(self, '_face_landmarker') and self._face_landmarker:
            # Tasks API path
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_ts_ms += 33  # ~30fps
            result = self._face_landmarker.detect_for_video(mp_image, self._frame_ts_ms)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                face = result.face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face],
                    dtype=np.float64,
                )
                return landmarks
            return None
        elif self._face_mesh:
            # Solutions API path
            results = self._face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face.landmark],
                    dtype=np.float64,
                )
                return landmarks
            return None
        return None

    def _process_landmarks(self, landmarks: np.ndarray, frame: np.ndarray):
        """Extract gaze ratios, run blink detection, dwell logic, and emit events."""
        # ── Pupil-to-eye-corner ratio ──
        ratio_x, ratio_y = self._compute_gaze_ratio(landmarks)
        self._latest_ratio = (ratio_x, ratio_y)

        # ── Map to screen via calibration ──
        sx, sy = self.calibrator.map_to_screen(ratio_x, ratio_y)

        # ── Smooth ──
        sx, sy = self._smoother.smooth(sx, sy)

        gaze = GazePoint(x=sx, y=sy, confidence=1.0)
        self._latest_gaze = gaze

        if self.on_gaze:
            self.on_gaze(gaze)

        # ── Blink detection ──
        ear_left = self.blink_detector.compute_ear(landmarks, LEFT_EYE_CONTOUR)
        ear_right = self.blink_detector.compute_ear(landmarks, RIGHT_EYE_CONTOUR)
        avg_ear = (ear_left + ear_right) / 2.0

        if self.blink_detector.update(avg_ear):
            logger.debug("Long blink detected!")
            if self.on_blink:
                self.on_blink()

        # ── Dwell detection ──
        with self._lock:
            regions = dict(self._dwell_regions)

        hit_region = None
        for rid, (x1, y1, x2, y2) in regions.items():
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                hit_region = rid
                break

        active_region, progress, just_triggered = self._dwell.update(hit_region)
        if self.on_dwell:
            self.on_dwell(active_region, progress, just_triggered)

        # ── Draw tracking landmarks on frame for visual preview ──
        try:
            h, w = frame.shape[:2]
            # Left iris (green dot)
            lx, ly = int(landmarks[468][0] * w), int(landmarks[468][1] * h)
            cv2.circle(frame, (lx, ly), 4, (0, 255, 128), -1)
            # Right iris (green dot)
            rx, ry = int(landmarks[473][0] * w), int(landmarks[473][1] * h)
            cv2.circle(frame, (rx, ry), 4, (0, 255, 128), -1)
            # Nose tip (cyan dot)
            nx, ny = int(landmarks[1][0] * w), int(landmarks[1][1] * h)
            cv2.circle(frame, (nx, ny), 3, (255, 200, 0), -1)
        except Exception:
            pass

    def _compute_gaze_ratio(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute normalized pupil position + head pose.
        Returns (ratio_x, ratio_y) in [0, 1] range mapped across full screen breadth.
        """
        try:
            # ── 1. Left Eye Iris Ratio (mirrored frame: left eye is screen-left) ──
            # Outer corner: 33, Inner corner: 133, Iris center: 468
            left_min_x = min(landmarks[33][0], landmarks[133][0])
            left_max_x = max(landmarks[33][0], landmarks[133][0])
            left_span_x = left_max_x - left_min_x
            left_ratio_x = (landmarks[468][0] - left_min_x) / max(left_span_x, 1e-5)

            # Top eyelid: 159, Bottom eyelid: 145
            left_min_y = min(landmarks[159][1], landmarks[145][1])
            left_max_y = max(landmarks[159][1], landmarks[145][1])
            left_span_y = left_max_y - left_min_y
            left_ratio_y = (landmarks[468][1] - left_min_y) / max(left_span_y, 1e-5)

            # ── 2. Right Eye Iris Ratio (mirrored frame: right eye is screen-right) ──
            # Inner corner: 362, Outer corner: 263, Iris center: 473
            right_min_x = min(landmarks[362][0], landmarks[263][0])
            right_max_x = max(landmarks[362][0], landmarks[263][0])
            right_span_x = right_max_x - right_min_x
            right_ratio_x = (landmarks[473][0] - right_min_x) / max(right_span_x, 1e-5)

            # Top eyelid: 386, Bottom eyelid: 374
            right_min_y = min(landmarks[386][1], landmarks[374][1])
            right_max_y = max(landmarks[386][1], landmarks[374][1])
            right_span_y = right_max_y - right_min_y
            right_ratio_y = (landmarks[473][1] - right_min_y) / max(right_span_y, 1e-5)

            # Combine eye ratios
            eye_ratio_x = (left_ratio_x + right_ratio_x) / 2.0
            eye_ratio_y = (left_ratio_y + right_ratio_y) / 2.0

            # ── 3. Head Pose (Natural Yaw & Pitch Assistance) ──
            # Nose tip (1), left cheek (234), right cheek (454)
            face_min_x = min(landmarks[234][0], landmarks[454][0])
            face_max_x = max(landmarks[234][0], landmarks[454][0])
            face_span_x = face_max_x - face_min_x
            head_yaw = (landmarks[1][0] - face_min_x) / max(face_span_x, 1e-5)

            # Forehead (10), chin (152), nose tip (1)
            face_min_y = min(landmarks[10][1], landmarks[152][1])
            face_max_y = max(landmarks[10][1], landmarks[152][1])
            face_span_y = face_max_y - face_min_y
            head_pitch = (landmarks[1][1] - face_min_y) / max(face_span_y, 1e-5)

            # Blend: 60% eye gaze + 40% head pose
            raw_x = 0.60 * eye_ratio_x + 0.40 * head_yaw
            raw_y = 0.60 * eye_ratio_y + 0.40 * head_pitch

            # ── 4. Dynamic Sensitivity Amplification ──
            # Human eye+head movement in front of a monitor naturally covers ~[0.38, 0.62].
            # An amplification gain of 2.8 maps this comfortably across [0.0, 1.0] of the screen.
            GAIN_X = 2.8
            GAIN_Y = 2.8
            ratio_x = 0.5 + (raw_x - 0.5) * GAIN_X
            ratio_y = 0.5 + (raw_y - 0.5) * GAIN_Y

            # Clamp to [0, 1]
            ratio_x = float(max(0.0, min(1.0, ratio_x)))
            ratio_y = float(max(0.0, min(1.0, ratio_y)))

            return ratio_x, ratio_y
        except Exception:
            return 0.5, 0.5
