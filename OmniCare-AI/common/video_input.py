"""One canonical video input for all local AI modules.

The class deliberately shares a *path*, not decoded frames.  Each consumer can
open that one immutable media file in the way it needs, without creating media
copies.  Frame timestamps are derived from the frame index and source FPS so
all event timestamps belong to the same video timeline.
"""

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".mkv", ".avi"}


class VideoInputError(RuntimeError):
    """Raised when the selected shared video cannot be used."""


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int

    def to_dict(self):
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp: float
    image: object


class VideoInput:
    """Validated, canonical reference to the one video under analysis."""

    def __init__(self, video_path):
        self.path = Path(video_path).expanduser().resolve()

    @property
    def source_id(self):
        return str(self.path)

    def validate_file(self):
        if not self.path.is_file():
            raise VideoInputError(f"Video file does not exist: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
            raise VideoInputError(
                f"Unsupported video format '{self.path.suffix}'. Supported formats: {supported}"
            )
        return self

    @staticmethod
    def _opencv():
        try:
            import cv2
        except ImportError as exc:
            raise VideoInputError(
                "OpenCV is required to read video frames. Install the pipeline requirements."
            ) from exc
        return cv2

    def metadata(self):
        self.validate_file()
        cv2 = self._opencv()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"OpenCV could not open video: {self.path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if fps <= 0 or fps != fps:
            raise VideoInputError(f"Video reports an invalid FPS value: {self.path}")
        return VideoMetadata(
            fps=fps,
            frame_count=max(0, frame_count),
            duration_seconds=max(0.0, frame_count / fps),
            width=max(0, width),
            height=max(0, height),
        )

    def iter_frames(self):
        """Yield decoded frames with timestamps on the source video timeline."""
        self.validate_file()
        cv2 = self._opencv()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"OpenCV could not open video: {self.path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps != fps:
                raise VideoInputError(f"Video reports an invalid FPS value: {self.path}")
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                yield VideoFrame(
                    index=frame_index,
                    timestamp=frame_index / fps,
                    image=frame,
                )
                frame_index += 1
        finally:
            capture.release()
