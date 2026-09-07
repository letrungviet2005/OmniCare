"""OpenCV webcam input for the local OmniCare monitoring pipeline."""

from __future__ import annotations

import math
import queue
import threading
import time

from .video_input import VideoFrame


class CameraInputError(RuntimeError):
    """Raised when a local webcam cannot provide frames."""


class CameraInput:
    """Own one OpenCV VideoCapture and a monotonic camera-session timeline."""

    def __init__(self, camera_index=0, clock=time.monotonic):
        if isinstance(camera_index, bool):
            raise CameraInputError("Camera index must be a non-negative integer.")
        try:
            self.index = int(camera_index)
        except (TypeError, ValueError) as exc:
            raise CameraInputError(
                "Camera index must be a non-negative integer."
            ) from exc
        if self.index < 0:
            raise CameraInputError("Camera index must be a non-negative integer.")
        self._clock = clock
        self._capture = None
        self._started_at = None
        self._frame_index = 0
        self._cv2_module = None
        self._frames = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._capture_error = None
        self._capture_thread = None

    @property
    def source_id(self):
        return f"camera:{self.index}"

    @property
    def started_at(self):
        if self._started_at is None:
            raise CameraInputError("Camera must be opened before reading its timeline.")
        return self._started_at

    @property
    def cv2(self):
        if self._cv2_module is None:
            try:
                import cv2
            except ImportError as exc:
                raise CameraInputError(
                    "OpenCV is required for webcam monitoring. Install the pipeline requirements."
                ) from exc
            self._cv2_module = cv2
        return self._cv2_module

    def open(self):
        if self._capture is not None:
            return self
        capture = self.cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraInputError(
                f"Could not open camera index {self.index}. Check camera access and availability."
            )
        self._capture = capture
        self._started_at = self._clock()
        self._frame_index = 0
        self._capture_error = None
        self._stop_event.clear()
        try:
            while True:
                self._frames.get_nowait()
        except queue.Empty:
            pass
        set_property = getattr(capture, "set", None)
        buffer_property = getattr(self.cv2, "CAP_PROP_BUFFERSIZE", None)
        if callable(set_property) and buffer_property is not None:
            try:
                set_property(buffer_property, 1)
            except Exception:
                pass
        self._capture_thread = threading.Thread(
            target=self._capture_latest,
            name=f"omnicare-camera-{self.index}",
            daemon=True,
        )
        self._capture_thread.start()
        return self

    def _capture_latest(self):
        while not self._stop_event.is_set():
            ok, image = self._capture.read()
            if not ok or image is None:
                if not self._stop_event.is_set():
                    self._capture_error = (
                        f"Camera index {self.index} stopped providing frames."
                    )
                return
            timestamp = max(0.0, self._clock() - self._started_at)
            frame = VideoFrame(self._frame_index, timestamp, image)
            self._frame_index += 1
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    pass

    @property
    def reported_fps(self):
        if self._capture is None:
            return 0.0
        fps = float(self._capture.get(self.cv2.CAP_PROP_FPS))
        return fps if math.isfinite(fps) and fps > 0 else 0.0

    def read(self):
        if self._capture is None or self._started_at is None:
            raise CameraInputError("Camera must be opened before reading frames.")
        while True:
            try:
                return self._frames.get(timeout=0.5)
            except queue.Empty:
                if self._capture_error is not None:
                    raise CameraInputError(self._capture_error)
                if self._stop_event.is_set():
                    raise CameraInputError(
                        f"Camera index {self.index} has been stopped."
                    )

    def release(self):
        self._stop_event.set()
        capture = self._capture
        thread = self._capture_thread
        if thread is not None:
            thread.join(timeout=1)
        if capture is not None:
            capture.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)
        self._capture = None
        self._capture_thread = None
        self._started_at = None
