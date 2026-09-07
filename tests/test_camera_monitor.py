"""Webcam monitoring tests that do not require physical camera hardware."""

import contextlib
import io
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
for import_root in (ROOT, AI_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_ai
from common.camera_input import CameraInput, CameraInputError
from common.video_input import VideoFrame
from fall_detection.pipeline import FallEvent
from voice_detection.pipeline import VoicePipeline
from voice_detection.microphone_monitor import MicrophoneMonitor
from voice_detection.speech_recognizer import TranscriptSegment


class FakeImage:
    shape = (480, 640, 3)


class FakeCv2:
    CAP_PROP_FPS = 5
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self, quit_after=1):
        self.quit_after = quit_after
        self.wait_calls = 0
        self.destroyed = False
        self.frames_shown = 0

    def rectangle(self, *_args):
        pass

    def putText(self, *_args):
        pass

    def imshow(self, _title, _frame):
        self.frames_shown += 1

    def waitKey(self, _delay):
        self.wait_calls += 1
        return ord("q") if self.wait_calls >= self.quit_after else -1

    def destroyAllWindows(self):
        self.destroyed = True


class FakeCameraInput:
    instance = None

    def __init__(self, camera_index):
        self.index = camera_index
        self.source_id = f"camera:{camera_index}"
        self.cv2 = FakeCv2(quit_after=2)
        self.opened = False
        self.released = False
        self.read_count = 0
        self.started_at = 100.0
        FakeCameraInput.instance = self

    def open(self):
        self.opened = True
        return self

    def read(self):
        timestamp = float(self.read_count + 1)
        frame = VideoFrame(self.read_count, timestamp, FakeImage())
        self.read_count += 1
        return frame

    def release(self):
        self.released = True


class RepeatingFallPipeline:
    instances = 0

    def __init__(self):
        RepeatingFallPipeline.instances += 1

    def process_frame(self, _frame, timestamp):
        # The repeated stable event simulates multiple frames from one fall.
        event = FallEvent(1.0, 85, 132.0, 0.9, "temporal_state")
        return SimpleNamespace(timestamp=timestamp, status="FALL DETECTED", event=event)


def camera_args(*, ingest=False, token=None):
    return SimpleNamespace(
        camera=0,
        monitor=True,
        ingest=ingest,
        token=token,
        backend_url="http://localhost:3001",
        fusion_window=10.0,
        output=None,
        audio=False,
        audio_language="auto",
        audio_threshold=70,
        model="small",
        device="cpu",
        compute_type="int8",
        threshold=78,
    )


class CameraArgumentTest(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["run_ai.py", *arguments]):
            return run_ai.parse_args()

    def test_camera_argument_is_parsed(self):
        args = self.parse("--camera", "0", "--monitor")
        self.assertEqual(args.camera, 0)
        self.assertIsNone(args.video)
        self.assertTrue(args.monitor)

    def test_audio_argument_is_available_for_camera_mode(self):
        args = self.parse("--camera", "0", "--monitor", "--audio")
        self.assertTrue(args.audio)
        self.assertEqual(args.audio_language, "auto")
        self.assertEqual(args.audio_threshold, 70)
        self.assertEqual(run_ai.REALTIME_WHISPER_MODEL, "base")

    def test_conversation_requires_camera_audio_mode(self):
        args = self.parse(
            "--camera", "0", "--monitor", "--audio", "--conversation"
        )
        self.assertTrue(args.conversation)
        for invalid in (
            ("--camera", "0", "--monitor", "--conversation"),
            ("--video", "Videos/fall.mp4", "--conversation"),
        ):
            with self.subTest(arguments=invalid):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.parse(*invalid)

    def test_video_and_camera_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(
                    "--video",
                    "Videos/fall.mp4",
                    "--camera",
                    "0",
                    "--monitor",
                )

    def test_camera_requires_monitor_mode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--camera", "0")

    def test_existing_video_mode_arguments_are_preserved(self):
        args = self.parse("--video", "Videos/fall.mp4")
        self.assertEqual(args.video, "Videos/fall.mp4")
        self.assertIsNone(args.camera)
        self.assertFalse(args.monitor)

    def test_audio_is_rejected_for_file_video_mode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--video", "Videos/fall.mp4", "--audio")


class CameraInputTest(unittest.TestCase):
    def test_camera_initialization_failure_is_clear_and_releases_capture(self):
        capture = SimpleNamespace(
            isOpened=lambda: False,
            release=lambda: setattr(capture, "released", True),
            released=False,
        )
        fake_cv2 = SimpleNamespace(VideoCapture=lambda _index: capture)
        camera = CameraInput(0)
        camera._cv2_module = fake_cv2

        with self.assertRaisesRegex(CameraInputError, "Could not open camera index 0"):
            camera.open()

        self.assertTrue(capture.released)

    def test_camera_input_keeps_only_latest_captured_frame(self):
        finished = threading.Event()

        class FakeCapture:
            def __init__(self):
                self.images = ["frame-1", "frame-2", "frame-3"]
                self.properties = []
                self.released = False

            def isOpened(self):
                return True

            def set(self, key, value):
                self.properties.append((key, value))
                return True

            def get(self, _key):
                return 30.0

            def read(self):
                if self.images:
                    return True, self.images.pop(0)
                finished.set()
                return False, None

            def release(self):
                self.released = True

        capture = FakeCapture()
        fake_cv2 = SimpleNamespace(
            CAP_PROP_BUFFERSIZE=38,
            CAP_PROP_FPS=5,
            VideoCapture=lambda _index: capture,
        )
        camera = CameraInput(0)
        camera._cv2_module = fake_cv2
        camera.open()
        self.assertTrue(finished.wait(timeout=1))

        latest = camera.read()
        camera.release()

        self.assertEqual(latest.index, 2)
        self.assertEqual(latest.image, "frame-3")
        self.assertIn((38, 1), capture.properties)
        self.assertTrue(capture.released)

    def test_microphone_audio_queue_is_bounded(self):
        import numpy as np

        worker = MicrophoneMonitor(
            timeline_origin=0.0,
            device="cpu",
            compute_type="int8",
            max_audio_blocks=2,
            clock=lambda: 1.0,
            sounddevice_module=SimpleNamespace(),
        )
        block = np.zeros((1600, 1), dtype=np.float32)
        for _ in range(10):
            worker._audio_callback(block, 1600, None, None)

        self.assertLessEqual(worker._audio.qsize(), 2)
        self.assertEqual(worker._events.maxsize, 16)

    def test_realtime_microphone_defaults_are_low_latency_and_bounded(self):
        worker = MicrophoneMonitor(
            timeline_origin=0.0,
            device="cpu",
            compute_type="int8",
            sounddevice_module=SimpleNamespace(),
        )
        self.assertEqual(worker._decoder.model_name, "base")
        self.assertEqual(worker.window_seconds, 2.5)
        self.assertEqual(worker._audio.maxsize, 128)
        self.assertEqual(worker._decoder.cpu_threads, 1)
        self.assertEqual(worker._decoder.beam_size, 1)
        self.assertEqual(worker.min_speech_rms, 0.004)

    def test_realtime_microphone_skips_only_low_energy_windows(self):
        import numpy as np

        worker = MicrophoneMonitor(
            timeline_origin=0.0,
            device="cpu",
            compute_type="int8",
            sounddevice_module=SimpleNamespace(),
        )
        self.assertFalse(
            worker._contains_speech_energy(np.zeros(40000, dtype=np.float32))
        )
        self.assertTrue(
            worker._contains_speech_energy(
                np.full(40000, 0.01, dtype=np.float32)
            )
        )


class CameraMonitoringTest(unittest.TestCase):
    def setUp(self):
        RepeatingFallPipeline.instances = 0

    def test_camera_reuses_one_model_and_stops_on_q(self):
        with patch.object(run_ai, "CameraInput", FakeCameraInput), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_camera_monitor(camera_args())

        camera = FakeCameraInput.instance
        self.assertEqual(code, 0)
        self.assertEqual(RepeatingFallPipeline.instances, 1)
        self.assertEqual(camera.read_count, 2)
        self.assertTrue(camera.released)
        self.assertTrue(camera.cv2.destroyed)
        self.assertEqual(camera.cv2.frames_shown, 2)
        self.assertIn("Risk: HIGH (75)", stdout.getvalue())
        self.assertIn("Alert: FALL_ALERT", stdout.getvalue())
        self.assertIn("Events detected: 1", stdout.getvalue())

    def test_repeated_fall_event_is_ingested_only_once(self):
        ingestion = {
            "analysisId": "AN-CAMERA-1",
            "duplicate": False,
            "alertCreated": True,
        }
        with patch.object(run_ai, "CameraInput", FakeCameraInput), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ), patch.object(run_ai, "ingest_result", return_value=ingestion) as ingest:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_camera_monitor(
                    camera_args(ingest=True, token="jwt")
                )

        self.assertEqual(code, 0)
        ingest.assert_called_once()
        sent_result = ingest.call_args.args[0]
        self.assertEqual(sent_result["risk_engine"]["level"], "HIGH")
        self.assertEqual(sent_result["alert_engine"]["type"], "FALL_ALERT")
        self.assertEqual(len(sent_result["event_engine"]["timeline"]), 1)
        self.assertIn("Backend ingestion requests: 1", stdout.getvalue())

    def test_backend_ingestion_does_not_pause_camera_frames(self):
        ingestion_started = threading.Event()
        release_ingestion = threading.Event()
        outcome = {}

        def blocking_ingestion(*_args):
            ingestion_started.set()
            release_ingestion.wait(timeout=2)
            return {
                "analysisId": "AN-CAMERA-ASYNC",
                "duplicate": False,
                "alertCreated": True,
            }

        def run_camera():
            outcome["code"] = run_ai.run_camera_monitor(
                camera_args(ingest=True, token="jwt")
            )

        with patch.object(run_ai, "CameraInput", FakeCameraInput), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ), patch.object(run_ai, "ingest_result", side_effect=blocking_ingestion):
            with contextlib.redirect_stdout(io.StringIO()):
                runner = threading.Thread(target=run_camera)
                runner.start()
                self.assertTrue(ingestion_started.wait(timeout=1))
                self.assertEqual(FakeCameraInput.instance.read_count, 2)
                release_ingestion.set()
                runner.join(timeout=2)

        self.assertFalse(runner.is_alive())
        self.assertEqual(outcome["code"], 0)

    def test_camera_loop_does_not_wait_for_gemini(self):
        request_started = threading.Event()
        release_request = threading.Event()
        second_frame_read = threading.Event()
        outcome = {}

        class SignalingCamera(FakeCameraInput):
            def read(self):
                frame = super().read()
                if self.read_count == 2:
                    second_frame_read.set()
                return frame

        class OneTranscriptMicrophone:
            def __init__(self, **_kwargs):
                self.sent = False

            def start(self):
                return self

            def poll(self):
                if self.sent:
                    return []
                self.sent = True
                return [
                    (
                        "result",
                        SimpleNamespace(
                            segments=[
                                TranscriptSegment(1.0, 1.8, "Hello there")
                            ]
                        ),
                    )
                ]

            def stop(self):
                pass

        class SlowConversation:
            model = "gemini-2.5-flash"

            def respond(self, _text):
                request_started.set()
                release_request.wait(timeout=2)
                return "Chào bác ạ."

            def close(self):
                pass

        conversation_worker = run_ai.ConversationWorker(
            conversation=SlowConversation(), settle_seconds=0
        )
        args = camera_args()
        args.audio = True
        args.conversation = True

        def run_camera():
            outcome["code"] = run_ai.run_camera_monitor(args)

        with patch.object(run_ai, "CameraInput", SignalingCamera), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ), patch.object(
            run_ai, "MicrophoneMonitor", OneTranscriptMicrophone
        ), patch.object(
            run_ai, "ConversationWorker", return_value=conversation_worker
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                runner = threading.Thread(target=run_camera)
                runner.start()
                self.assertTrue(request_started.wait(timeout=1))
                self.assertTrue(second_frame_read.wait(timeout=1))
                release_request.set()
                runner.join(timeout=2)

        self.assertFalse(runner.is_alive())
        self.assertEqual(outcome["code"], 0)

    def test_camera_ingestion_is_optional(self):
        with patch.object(run_ai, "CameraInput", FakeCameraInput), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ), patch.object(run_ai, "ingest_result") as ingest:
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_ai.run_camera_monitor(camera_args())

        self.assertEqual(code, 0)
        ingest.assert_not_called()

    def test_camera_ctrl_c_releases_preview_resources(self):
        class InterruptedCamera(FakeCameraInput):
            def read(self):
                raise KeyboardInterrupt

        with patch.object(run_ai, "CameraInput", InterruptedCamera), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_camera_monitor(camera_args())

        camera = InterruptedCamera.instance
        self.assertEqual(code, 130)
        self.assertTrue(camera.released)
        self.assertTrue(camera.cv2.destroyed)
        self.assertIn("Monitoring stopped.", stdout.getvalue())

    def test_microphone_failure_continues_camera_only(self):
        class UnavailableMicrophone:
            instance = None

            def __init__(self, **_kwargs):
                self.messages = [("error", "Microphone is unavailable")]
                self.stopped = False
                UnavailableMicrophone.instance = self

            def start(self):
                return self

            def poll(self):
                messages, self.messages = self.messages, []
                return messages

            def stop(self):
                self.stopped = True

        args = camera_args()
        args.audio = True
        with patch.object(run_ai, "CameraInput", FakeCameraInput), patch.object(
            run_ai, "FallPipeline", RepeatingFallPipeline
        ), patch.object(run_ai, "MicrophoneMonitor", UnavailableMicrophone):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_camera_monitor(args)

        self.assertEqual(code, 0)
        self.assertTrue(UnavailableMicrophone.instance.stopped)
        self.assertEqual(FakeCameraInput.instance.read_count, 2)
        self.assertIn("Monitoring stopped.", stdout.getvalue())


class RealtimeVoiceIntegrationTest(unittest.TestCase):
    def test_existing_voice_matcher_converts_emergency_phrases(self):
        pipeline = VoicePipeline(language="auto", threshold=78)
        for phrase in ("Help", "Help me", "Emergency", "Call my family"):
            with self.subTest(phrase=phrase):
                events = pipeline.events_from_segments(
                    [TranscriptSegment(12.4, 13.0, phrase)]
                )
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].type, "HELP_REQUEST")
                self.assertEqual(events[0].timestamp, 12.4)

    def test_normal_speech_does_not_create_voice_event(self):
        pipeline = VoicePipeline(language="auto", threshold=78)
        events = pipeline.events_from_segments(
            [TranscriptSegment(2.0, 3.0, "The weather is pleasant today")]
        )
        self.assertEqual(events, [])

    def test_fall_and_help_use_existing_event_fusion(self):
        session = run_ai.MonitoringSession(fusion_window_seconds=10)
        help_event = VoicePipeline(language="auto").events_from_segments(
            [TranscriptSegment(12.0, 12.8, "Help")]
        )[0].to_dict()
        help_event["source"] = "voice_detection"
        fall_event = dict(FALL_EVENT_FOR_FUSION)

        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events([fall_event])
            updates = session.observe_events([help_event])

        self.assertIn("COMBINED_EMERGENCY", [event.type for event in updates])
        self.assertEqual(session.latest_risk_level, "CRITICAL")
        self.assertEqual(session.latest_risk_score, 100)
        self.assertEqual(session.latest_alert_type, "COMBINED_EMERGENCY_ALERT")


FALL_EVENT_FOR_FUSION = {
    "timestamp": 10.0,
    "source": "fall_detection",
    "type": "FALL_DETECTED",
    "confidence": 0.9,
    "score": 85,
}


if __name__ == "__main__":
    unittest.main()
