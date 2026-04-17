from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import mss
import numpy as np
from PIL import Image

from src.openai_utils import OpenAIWorkflowError, detect_focus_concept


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_LIDS = (159, 145)
RIGHT_EYE_LIDS = (386, 374)
NOSE_TIP = 1


@dataclass
class CalibrationSample:
    features: np.ndarray
    target: tuple[float, float]


@dataclass
class RuntimeState:
    running: bool = False
    calibrated: bool = False
    calibration_samples: int = 0
    screen_size: tuple[int, int] = (0, 0)
    focus_point: tuple[int, int] | None = None
    focus_box: tuple[int, int, int, int] | None = None
    concept_label: str = "waiting for label"
    last_error: str | None = None
    debug_line: str = "Camera idle."
    screen_preview: np.ndarray | None = None
    crop_preview: np.ndarray | None = None
    camera_preview: np.ndarray | None = None
    last_calibration_target: tuple[float, float] | None = None
    label_updated_at: float | None = None
    capture_fps: float = 0.0


@dataclass
class TrackerConfig:
    camera_index: int = 0
    crop_size: int = 240
    preview_width: int = 1040
    recognition_interval: float = 2.2
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"


class EyeFocusRuntime:
    """Background webcam and screen-processing loop for the Streamlit app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._label_thread: threading.Thread | None = None
        self._state = RuntimeState()
        self._config = TrackerConfig()
        self._samples: list[CalibrationSample] = []
        self._projection_x: np.ndarray | None = None
        self._projection_y: np.ndarray | None = None
        self._latest_features: np.ndarray | None = None
        self._smoothed_focus: np.ndarray | None = None
        self._last_label_request = 0.0
        self._label_inflight = False

    def configure(
        self,
        *,
        camera_index: int,
        crop_size: int,
        recognition_interval: float,
        openai_api_key: str,
        openai_model: str,
    ) -> None:
        with self._lock:
            self._config.camera_index = camera_index
            self._config.crop_size = int(np.clip(crop_size, 80, 420))
            self._config.recognition_interval = max(0.8, recognition_interval)
            self._config.openai_api_key = openai_api_key.strip()
            self._config.openai_model = openai_model.strip() or "gpt-4.1-mini"

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._state.running = True
                return
            self._stop_event.clear()
            self._state.running = True
            self._state.last_error = None
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._state.running = False

    def clear_calibration(self) -> None:
        with self._lock:
            self._samples.clear()
            self._projection_x = None
            self._projection_y = None
            self._smoothed_focus = None
            self._state.calibrated = False
            self._state.calibration_samples = 0
            self._state.debug_line = "Calibration cleared."

    def add_calibration_sample(self, target_x: float, target_y: float) -> bool:
        with self._lock:
            if self._latest_features is None:
                self._state.last_error = "No eye landmarks available yet. Start the tracker and face the camera."
                return False
            self._samples.append(
                CalibrationSample(
                    features=self._latest_features.copy(),
                    target=(float(target_x), float(target_y)),
                )
            )
            self._state.last_calibration_target = (float(target_x), float(target_y))
            self._fit_calibration_locked()
            return True

    def get_state(self) -> RuntimeState:
        with self._lock:
            return RuntimeState(
                running=self._state.running,
                calibrated=self._state.calibrated,
                calibration_samples=self._state.calibration_samples,
                screen_size=self._state.screen_size,
                focus_point=self._state.focus_point,
                focus_box=self._state.focus_box,
                concept_label=self._state.concept_label,
                last_error=self._state.last_error,
                debug_line=self._state.debug_line,
                screen_preview=self._state.screen_preview.copy()
                if self._state.screen_preview is not None
                else None,
                crop_preview=self._state.crop_preview.copy() if self._state.crop_preview is not None else None,
                camera_preview=self._state.camera_preview.copy()
                if self._state.camera_preview is not None
                else None,
                last_calibration_target=self._state.last_calibration_target,
                label_updated_at=self._state.label_updated_at,
                capture_fps=self._state.capture_fps,
            )

    def _fit_calibration_locked(self) -> None:
        self._state.calibration_samples = len(self._samples)
        if len(self._samples) < 5:
            self._state.calibrated = False
            self._state.debug_line = (
                f"Collected {len(self._samples)} calibration points. Five or more are required."
            )
            return

        matrix = np.vstack([sample.features for sample in self._samples])
        targets_x = np.array([sample.target[0] for sample in self._samples], dtype=np.float64)
        targets_y = np.array([sample.target[1] for sample in self._samples], dtype=np.float64)
        self._projection_x, *_ = np.linalg.lstsq(matrix, targets_x, rcond=None)
        self._projection_y, *_ = np.linalg.lstsq(matrix, targets_y, rcond=None)
        self._state.calibrated = True
        self._state.debug_line = f"Calibration ready with {len(self._samples)} samples."

    def _capture_loop(self) -> None:
        with self._lock:
            camera_index = self._config.camera_index
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            with self._lock:
                self._state.running = False
                self._state.last_error = f"Could not open camera index {camera_index}."
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_counter = 0
        fps_started = time.time()

        try:
            with mss.mss() as screen_capture:
                monitor = screen_capture.monitors[1]
                screen_size = (int(monitor["width"]), int(monitor["height"]))
                with self._lock:
                    self._state.screen_size = screen_size

                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        with self._lock:
                            self._state.last_error = "Camera frame capture failed."
                        time.sleep(0.05)
                        continue

                    camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(camera_rgb)
                    screen_rgb = np.array(screen_capture.grab(monitor))[:, :, :3]
                    screen_rgb = cv2.cvtColor(screen_rgb, cv2.COLOR_BGR2RGB)

                    focus_point = None
                    focus_box = None
                    crop_preview = None
                    debug_line = "Face not detected."

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        features = self._extract_features(landmarks)
                        with self._lock:
                            self._latest_features = features
                            self._state.last_error = None
                        focus_point = self._predict_focus(features, screen_size)
                        debug_line = self._describe_features(features)
                        camera_rgb = self._annotate_camera(camera_rgb, landmarks)

                        if focus_point is not None:
                            focus_box = self._build_focus_box(focus_point, screen_size)
                            crop_preview = self._extract_crop(screen_rgb, focus_box)
                            screen_rgb = self._annotate_screen(screen_rgb, focus_box, focus_point)
                            self._maybe_request_label(crop_preview)
                    else:
                        camera_rgb = cv2.flip(camera_rgb, 1)

                    preview = self._resize_for_preview(screen_rgb, self._config.preview_width)
                    frame_counter += 1
                    elapsed = max(time.time() - fps_started, 0.001)
                    if elapsed >= 1.0:
                        fps = frame_counter / elapsed
                        frame_counter = 0
                        fps_started = time.time()
                    else:
                        fps = self._state.capture_fps

                    with self._lock:
                        self._state.running = True
                        self._state.focus_point = focus_point
                        self._state.focus_box = focus_box
                        self._state.screen_preview = preview
                        self._state.crop_preview = crop_preview
                        self._state.camera_preview = camera_rgb
                        self._state.debug_line = debug_line
                        self._state.capture_fps = fps

                    time.sleep(0.03)
        except Exception as exc:
            with self._lock:
                self._state.last_error = (
                    "Capture loop failed. On macOS, confirm Camera and Screen Recording permissions. "
                    f"Details: {exc}"
                )
        finally:
            cap.release()
            with self._lock:
                self._state.running = False

    def _extract_features(self, landmarks: list) -> np.ndarray:
        left_iris = np.mean([[landmarks[i].x, landmarks[i].y] for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([[landmarks[i].x, landmarks[i].y] for i in RIGHT_IRIS], axis=0)

        left_eye_x = self._relative_axis(
            left_iris[0],
            landmarks[LEFT_EYE_CORNERS[0]].x,
            landmarks[LEFT_EYE_CORNERS[1]].x,
        )
        right_eye_x = self._relative_axis(
            right_iris[0],
            landmarks[RIGHT_EYE_CORNERS[0]].x,
            landmarks[RIGHT_EYE_CORNERS[1]].x,
        )
        left_eye_y = self._relative_axis(
            left_iris[1],
            landmarks[LEFT_EYE_LIDS[0]].y,
            landmarks[LEFT_EYE_LIDS[1]].y,
        )
        right_eye_y = self._relative_axis(
            right_iris[1],
            landmarks[RIGHT_EYE_LIDS[0]].y,
            landmarks[RIGHT_EYE_LIDS[1]].y,
        )
        nose = landmarks[NOSE_TIP]

        avg_eye_x = float((left_eye_x + right_eye_x) / 2.0)
        avg_eye_y = float((left_eye_y + right_eye_y) / 2.0)
        return np.array([avg_eye_x, avg_eye_y, nose.x, nose.y, 1.0], dtype=np.float64)

    def _predict_focus(self, features: np.ndarray, screen_size: tuple[int, int]) -> tuple[int, int]:
        width, height = screen_size

        with self._lock:
            projection_x = self._projection_x.copy() if self._projection_x is not None else None
            projection_y = self._projection_y.copy() if self._projection_y is not None else None
            calibrated = self._state.calibrated

        if calibrated and projection_x is not None and projection_y is not None:
            normalized = np.array(
                [
                    float(np.dot(features, projection_x)),
                    float(np.dot(features, projection_y)),
                ],
                dtype=np.float64,
            )
        else:
            normalized = np.array(
                [
                    1.15 - features[0],
                    features[1] * 1.25 - 0.12,
                ],
                dtype=np.float64,
            )

        normalized = np.clip(normalized, 0.0, 1.0)

        if self._smoothed_focus is None:
            self._smoothed_focus = normalized
        else:
            self._smoothed_focus = 0.72 * self._smoothed_focus + 0.28 * normalized

        return (
            int(self._smoothed_focus[0] * width),
            int(self._smoothed_focus[1] * height),
        )

    def _build_focus_box(
        self,
        focus_point: tuple[int, int],
        screen_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = screen_size
        half = self._config.crop_size // 2
        center_x, center_y = focus_point
        left = int(np.clip(center_x - half, 0, max(width - self._config.crop_size, 0)))
        top = int(np.clip(center_y - half, 0, max(height - self._config.crop_size, 0)))
        right = min(left + self._config.crop_size, width)
        bottom = min(top + self._config.crop_size, height)
        return left, top, right, bottom

    def _extract_crop(self, screen_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = box
        return screen_rgb[top:bottom, left:right].copy()

    def _maybe_request_label(self, crop_rgb: np.ndarray | None) -> None:
        if crop_rgb is None:
            return

        with self._lock:
            api_key = self._config.openai_api_key
            model = self._config.openai_model
            interval = self._config.recognition_interval
            label_inflight = self._label_inflight

        if not api_key:
            with self._lock:
                self._state.concept_label = "set OPENAI_API_KEY to label crops"
            return

        now = time.time()
        if label_inflight or now - self._last_label_request < interval:
            return

        image = Image.fromarray(crop_rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        payload = buffer.getvalue()

        with self._lock:
            self._label_inflight = True
            self._last_label_request = now
        self._label_thread = threading.Thread(
            target=self._label_crop,
            args=(payload, api_key, model),
            daemon=True,
        )
        self._label_thread.start()

    def _label_crop(self, image_bytes: bytes, api_key: str, model: str) -> None:
        try:
            label = detect_focus_concept(
                image_bytes=image_bytes,
                mime_type="image/jpeg",
                api_key=api_key,
                model=model,
            )
            with self._lock:
                self._state.concept_label = label
                self._state.label_updated_at = time.time()
        except OpenAIWorkflowError as exc:
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            with self._lock:
                self._label_inflight = False

    def _annotate_screen(
        self,
        screen_rgb: np.ndarray,
        box: tuple[int, int, int, int],
        focus_point: tuple[int, int],
    ) -> np.ndarray:
        canvas = screen_rgb.copy()
        left, top, right, bottom = box
        cv2.rectangle(canvas, (left, top), (right, bottom), (255, 70, 70), 4)
        cv2.circle(canvas, focus_point, 7, (255, 70, 70), -1)
        cv2.circle(canvas, focus_point, 15, (255, 240, 240), 2)

        with self._lock:
            label = self._state.concept_label
        anchor_x = max(10, min(left, canvas.shape[1] - 280))
        anchor_y = top - 18 if top > 34 else bottom + 36
        cv2.rectangle(
            canvas,
            (anchor_x - 8, anchor_y - 28),
            (min(anchor_x + 260, canvas.shape[1] - 10), anchor_y + 8),
            (15, 20, 30),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (anchor_x, anchor_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _annotate_camera(self, camera_rgb: np.ndarray, landmarks: list) -> np.ndarray:
        canvas = cv2.flip(camera_rgb.copy(), 1)
        points = []
        for index in LEFT_IRIS + RIGHT_IRIS:
            point = landmarks[index]
            points.append((1.0 - point.x, point.y))

        for x_norm, y_norm in points:
            x = int(x_norm * canvas.shape[1])
            y = int(y_norm * canvas.shape[0])
            cv2.circle(canvas, (x, y), 3, (255, 80, 80), -1)

        cv2.putText(
            canvas,
            "iris landmarks",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _resize_for_preview(image: np.ndarray, target_width: int) -> np.ndarray:
        height, width = image.shape[:2]
        if width <= target_width:
            return image.copy()
        ratio = target_width / float(width)
        resized = cv2.resize(image, (target_width, int(height * ratio)), interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def _relative_axis(value: float, start: float, end: float) -> float:
        low = min(start, end)
        high = max(start, end)
        span = max(high - low, 1e-6)
        return float(np.clip((value - low) / span, 0.0, 1.0))

    @staticmethod
    def _describe_features(features: np.ndarray) -> str:
        return (
            f"eye x {features[0]:.2f} | eye y {features[1]:.2f} | "
            f"nose x {features[2]:.2f} | nose y {features[3]:.2f}"
        )
