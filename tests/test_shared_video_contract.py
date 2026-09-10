"""Fast wiring tests; they do not need OpenCV, FFmpeg, or model weights."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from common.video_input import VideoMetadata
from fall_detection.pipeline import (
    FALL_EVENT_BBOX_RATIO_THRESHOLD,
    FALL_EVENT_REARM_SECONDS,
    FALL_EVENT_SCORE_THRESHOLD,
    FallAnalysis,
    FallEvent,
    FallPipeline,
)
from voice_detection.audio.extractor import AudioExtractionError, AudioExtractor
from voice_detection.pipeline import VoiceAnalysis, VoicePipeline
from voice_detection.speech.recognizer import TranscriptSegment


def load_runner():
    spec = importlib.util.spec_from_file_location("run_ai_for_test", ROOT / "run_ai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedVideoContractTest(unittest.TestCase):
    def test_all_modules_receive_the_same_canonical_video(self):
        runner = load_runner()
        received = {}

        class FakeExtractor:
            def validate_video(self, video_path):
                received["extractor_video"] = str(video_path)
                return video_path

            def extract(self, video_path, wav_path):
                received["extract_video"] = str(video_path)
                Path(wav_path).write_bytes(b"temporary test audio")
                return wav_path

        class FakeVideoInput:
            def __init__(self, video_path):
                self.path = Path(video_path).expanduser().resolve()

            @property
            def source_id(self):
                return str(self.path)

            def validate_file(self):
                if not self.path.is_file():
                    raise AssertionError("Test video placeholder was not created")
                return self

            def metadata(self):
                return VideoMetadata(30.0, 303, 10.1, 720, 1280)

        class FakeVoicePipeline:
            def __init__(self, **_kwargs):
                pass

            def analyze(self, source, wav_path):
                received["voice_video"] = source.source_id
                received["voice_audio"] = str(wav_path)
                segment = TranscriptSegment(155.0, 158.0, "Toi bi nga")
                return VoiceAnalysis(
                    source.source_id,
                    [segment],
                    [],
                    {"score": 85, "level": "EMERGENCY"},
                )

        class FakeFallPipeline:
            def analyze(self, source):
                received["fall_video"] = source.source_id
                return FallAnalysis(source.source_id, 1, [])

        original_extractor = runner.AudioExtractor
        original_video_input = runner.VideoInput
        original_voice = runner.VoicePipeline
        original_fall = runner.FallPipeline
        runner.AudioExtractor = FakeExtractor
        runner.VideoInput = FakeVideoInput
        runner.VoicePipeline = FakeVoicePipeline
        runner.FallPipeline = FakeFallPipeline
        try:
            with tempfile.TemporaryDirectory() as directory:
                video_path = Path(directory) / "shared.mp4"
                video_path.write_bytes(b"test video placeholder")
                result = runner.run_pipeline(
                    SimpleNamespace(
                        video=video_path,
                        model="tiny",
                        device="cpu",
                        compute_type=None,
                        language="vi",
                        threshold=78,
                    )
                )
        finally:
            runner.AudioExtractor = original_extractor
            runner.VideoInput = original_video_input
            runner.VoicePipeline = original_voice
            runner.FallPipeline = original_fall

        source_id = result["input_video"]
        self.assertEqual(received["extractor_video"], source_id)
        self.assertEqual(received["extract_video"], source_id)
        self.assertEqual(received["voice_video"], source_id)
        self.assertEqual(received["fall_video"], source_id)
        self.assertEqual(result["modules"]["conversation"]["input_video"], source_id)
        self.assertEqual(result["modules"]["conversation"]["events"][0]["start"], 155.0)
        self.assertEqual(result["modules"]["conversation"]["events"][0]["end"], 158.0)
        self.assertEqual(result["modules"]["conversation"]["events"][0]["timestamp"], 155.0)
        self.assertEqual(result["duration"], 10.1)
        self.assertEqual(result["events"][0]["source"], "conversation")
        self.assertEqual(result["events"][0]["type"], "CONVERSATION_SEGMENT")
        self.assertEqual(result["events"][0]["timestamp"], 155.0)

    def test_voice_events_only_include_detected_emergency_phrases(self):
        class FakeRecognizer:
            def __init__(self, **_kwargs):
                pass

            def transcribe(self, _audio_path, language):
                self.language = language
                return [
                    TranscriptSegment(0.5, 1.5, "Hôm nay trời đẹp"),
                    TranscriptSegment(2.5, 3.5, "Làm ơn giúp tôi"),
                ]

        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "shared.wav"
            wav_path.write_bytes(b"test audio placeholder")
            with patch("voice_detection.pipeline.SpeechRecognizer", FakeRecognizer):
                result = VoicePipeline(language="vi").analyze(
                    SimpleNamespace(source_id="shared.mp4"), wav_path
                )

        self.assertEqual(len(result.transcript_segments), 2)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].timestamp, 2.5)
        self.assertEqual(result.events[0].type, "HELP_REQUEST")
        self.assertEqual(result.events[0].text, "Làm ơn giúp tôi")

    def test_audio_extraction_timeout_is_reported_and_partial_output_removed(self):
        extractor = AudioExtractor(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shared.wav"
            with patch(
                "voice_detection.audio.extractor.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffmpeg", 120),
            ):
                with self.assertRaisesRegex(
                    AudioExtractionError, "audio extraction timed out"
                ):
                    extractor.extract("shared.mp4", output)
            self.assertFalse(output.exists())

    def test_video_without_audio_has_component_specific_error(self):
        extractor = AudioExtractor(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "silent.mp4"
            video_path.write_bytes(b"test video placeholder")
            with patch.object(extractor, "has_audio_stream", return_value=False):
                with self.assertRaisesRegex(
                    AudioExtractionError,
                    "Voice Detection and Conversation require an audio stream",
                ):
                    extractor.validate_video(video_path)

    def test_fall_event_exposes_video_timestamp_and_type(self):
        event = FallEvent(3.42, 90, 81.0, 0.91).to_dict()
        self.assertEqual(event["timestamp"], 3.42)
        self.assertEqual(event["type"], "FALL_DETECTED")
        self.assertEqual(event["confidence"], 0.91)

    def test_scored_horizontal_fall_candidate_generates_one_timed_event(self):
        import numpy as np

        pipeline = FallPipeline.__new__(FallPipeline)
        pipeline._previous_status = "NORMAL"
        pipeline._fall_event_emitted = False
        pipeline.person_detector = SimpleNamespace(
            detect=lambda _frame: [{"bbox": (0, 0, 130, 100), "conf": 0.91}]
        )
        pose_landmarks = SimpleNamespace(
            landmark=[SimpleNamespace(x=0.4, y=0.7), SimpleNamespace(x=0.6, y=0.7)]
        )
        pipeline.pose_detector = SimpleNamespace(
            detect=lambda _frame: SimpleNamespace(pose_landmarks=pose_landmarks),
            drawer=SimpleNamespace(draw_landmarks=lambda *_args: None),
            mp_pose=SimpleNamespace(
                POSE_CONNECTIONS=(),
                PoseLandmark=SimpleNamespace(LEFT_HIP=0, RIGHT_HIP=1),
            ),
        )
        pipeline.motion_analyzer = SimpleNamespace(
            update=lambda _x, _y: {"distance": 0.01, "velocity": 0.01, "still": False}
        )
        pipeline.fall_detector = SimpleNamespace(
            body_angle=lambda _landmarks: 132.0,
            detect=lambda _angle, _bbox, _motion: {
                "score": 70,
                "state": "FALLING",
                "angle_speed": 37.0,
                "ratio": 1.16,
            },
        )
        frame = np.zeros((100, 130, 3), dtype=np.uint8)

        first = pipeline.process_frame(frame, 177 / 30)
        second = pipeline.process_frame(frame, 178 / 30)

        self.assertIsNotNone(first.event)
        self.assertEqual(first.event.timestamp, 5.9)
        self.assertEqual(first.event.trigger, "score_and_horizontal_bbox")
        self.assertIsNone(second.event)

    def test_fall_fallback_and_rearm_configuration(self):
        self.assertEqual(FALL_EVENT_SCORE_THRESHOLD, 70)
        self.assertEqual(FALL_EVENT_BBOX_RATIO_THRESHOLD, 1.15)
        self.assertEqual(FALL_EVENT_REARM_SECONDS, 4.0)

    def test_legacy_terminal_fall_state_generates_one_timed_event(self):
        import numpy as np

        pipeline = FallPipeline.__new__(FallPipeline)
        pipeline._previous_status = "LYING"
        pipeline._fall_event_emitted = False
        pipeline.person_detector = SimpleNamespace(
            detect=lambda _frame: [{"bbox": (0, 0, 90, 120), "conf": 0.88}]
        )
        pose_landmarks = SimpleNamespace(
            landmark=[SimpleNamespace(x=0.4, y=0.7), SimpleNamespace(x=0.6, y=0.7)]
        )
        pipeline.pose_detector = SimpleNamespace(
            detect=lambda _frame: SimpleNamespace(pose_landmarks=pose_landmarks),
            drawer=SimpleNamespace(draw_landmarks=lambda *_args: None),
            mp_pose=SimpleNamespace(
                POSE_CONNECTIONS=(),
                PoseLandmark=SimpleNamespace(LEFT_HIP=0, RIGHT_HIP=1),
            ),
        )
        pipeline.motion_analyzer = SimpleNamespace(
            update=lambda _x, _y: {"distance": 0.0, "velocity": 0.0, "still": True}
        )
        pipeline.fall_detector = SimpleNamespace(
            body_angle=lambda _landmarks: 78.0,
            detect=lambda _angle, _bbox, _motion: {
                "score": 55,
                "state": "FALL",
                "angle_speed": 0.0,
                "ratio": 0.75,
            },
        )
        frame = np.zeros((120, 90, 3), dtype=np.uint8)

        first = pipeline.process_frame(frame, 12.5)
        second = pipeline.process_frame(frame, 12.6)

        self.assertEqual(first.status, "FALL")
        self.assertIsNotNone(first.event)
        self.assertEqual(first.event.timestamp, 12.5)
        self.assertEqual(first.event.trigger, "temporal_state")
        self.assertIsNone(second.event)

    def test_fall_event_rearms_after_four_observed_normal_seconds(self):
        import numpy as np

        detector_results = iter(
            [
                {"score": 55, "state": "FALL", "angle_speed": 0.0, "ratio": 0.75},
                {"score": 55, "state": "FALL", "angle_speed": 0.0, "ratio": 0.75},
                {"score": 0, "state": "NORMAL", "angle_speed": 0.0, "ratio": 0.75},
                {"score": 0, "state": "NORMAL", "angle_speed": 0.0, "ratio": 0.75},
                {"score": 0, "state": "NORMAL", "angle_speed": 0.0, "ratio": 0.75},
                {"score": 55, "state": "FALL", "angle_speed": 0.0, "ratio": 0.75},
            ]
        )
        pipeline = FallPipeline.__new__(FallPipeline)
        pipeline._previous_status = "NORMAL"
        pipeline._fall_event_emitted = False
        pipeline._normal_since = None
        pipeline.person_detector = SimpleNamespace(
            detect=lambda _frame: [{"bbox": (0, 0, 90, 120), "conf": 0.88}]
        )
        pose_landmarks = SimpleNamespace(
            landmark=[SimpleNamespace(x=0.4, y=0.7), SimpleNamespace(x=0.6, y=0.7)]
        )
        pipeline.pose_detector = SimpleNamespace(
            detect=lambda _frame: SimpleNamespace(pose_landmarks=pose_landmarks),
            drawer=SimpleNamespace(draw_landmarks=lambda *_args: None),
            mp_pose=SimpleNamespace(
                POSE_CONNECTIONS=(),
                PoseLandmark=SimpleNamespace(LEFT_HIP=0, RIGHT_HIP=1),
            ),
        )
        pipeline.motion_analyzer = SimpleNamespace(
            update=lambda _x, _y: {"distance": 0.0, "velocity": 0.0, "still": True}
        )
        pipeline.fall_detector = SimpleNamespace(
            body_angle=lambda _landmarks: 78.0,
            detect=lambda _angle, _bbox, _motion: next(detector_results),
        )
        frame = np.zeros((120, 90, 3), dtype=np.uint8)

        results = [
            pipeline.process_frame(frame, timestamp)
            for timestamp in (10.0, 10.1, 20.0, 23.9, 24.0, 25.0)
        ]

        self.assertIsNotNone(results[0].event)
        self.assertTrue(all(result.event is None for result in results[1:5]))
        self.assertIsNotNone(results[5].event)
        self.assertEqual(results[5].event.timestamp, 25.0)


if __name__ == "__main__":
    unittest.main()
