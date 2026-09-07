"""Focused tests for the optional continuous monitoring orchestration."""

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ai


FALL_EVENT = {
    "timestamp": 5.9,
    "source": "fall_detection",
    "type": "FALL_DETECTED",
    "confidence": 0.9,
    "score": 85,
}


def make_result(events):
    event_result, risk, alert = run_ai.evaluate_events(events, 10.0)
    return {
        "video": "Videos/fall.mp4",
        "input_video": "Videos/fall.mp4",
        "duration": 10.1,
        "metadata": {
            "fps": 30.0,
            "frame_count": 302,
            "duration_seconds": 10.1,
            "width": 720,
            "height": 1280,
        },
        "events": copy.deepcopy(events),
        "event_engine": event_result.to_dict(),
        "risk_engine": risk.to_dict(),
        "alert_engine": alert.to_dict(),
        "modules": {
            "fall_detection": {"frames_analyzed": 302, "events": []},
            "voice_detection": {"events": []},
            "conversation": {"events": []},
        },
    }


def monitor_args(*, ingest=False, token=None):
    return SimpleNamespace(
        ingest=ingest,
        output=None,
        token=token,
        backend_url="http://localhost:3001",
        fusion_window=10.0,
    )


class RunAiMonitorTest(unittest.TestCase):
    def test_processing_fps_is_reported_periodically(self):
        session = run_ai.MonitoringSession()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            session.report_processing_fps(9.9, 13.2)
            session.report_processing_fps(10.0, 13.2)
            session.report_processing_fps(10.1, 13.4)
        self.assertEqual(output.getvalue().count("[MONITOR] FPS:"), 1)
        self.assertIn("[MONITOR] FPS: 13.2", output.getvalue())

    def test_current_risk_recovers_without_deleting_history(self):
        session = run_ai.MonitoringSession(risk_recovery_seconds=30)
        fall = dict(FALL_EVENT, timestamp=10.0)
        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events([fall])

        self.assertEqual(session.latest_risk_level, "HIGH")
        session.advance_time(20.0)
        self.assertEqual(session.latest_risk_level, "HIGH")
        session.advance_time(39.9)
        self.assertEqual(session.latest_risk_level, "HIGH")
        with contextlib.redirect_stdout(io.StringIO()) as recovery_output:
            session.advance_time(40.0)
        self.assertEqual(session.latest_risk_level, "NORMAL")
        self.assertIn(
            "[MONITOR] 00:40.000 | status=NORMAL",
            recovery_output.getvalue(),
        )
        self.assertEqual(len(session.historical_events), 1)
        self.assertEqual(session.alerts_generated, 1)

        help_event = {
            "timestamp": 45.0,
            "source": "voice_detection",
            "type": "HELP_REQUEST",
            "confidence": 0.95,
            "text": "Help",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events([help_event])
        self.assertEqual(session.latest_risk_level, "HIGH")

    def test_new_concerning_event_extends_risk_recovery(self):
        session = run_ai.MonitoringSession(risk_recovery_seconds=30)
        first_event = dict(FALL_EVENT, timestamp=10.0)
        second_event = {
            "timestamp": 25.0,
            "source": "voice_detection",
            "type": "HELP_REQUEST",
            "confidence": 0.95,
            "text": "Help",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events([first_event])
            session.observe_events([second_event])

        session.advance_time(54.9)
        self.assertEqual(session.latest_risk_level, "HIGH")
        session.advance_time(55.0)
        self.assertEqual(session.latest_risk_level, "NORMAL")
        self.assertEqual(len(session.historical_events), 2)

    def test_delayed_audio_event_activates_risk_when_worker_returns(self):
        session = run_ai.MonitoringSession(risk_recovery_seconds=30)
        session.advance_time(50.0)
        delayed_help = {
            "timestamp": 20.0,
            "source": "voice_detection",
            "type": "HELP_REQUEST",
            "confidence": 0.95,
            "text": "Help",
        }

        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events([delayed_help])

        self.assertEqual(session.latest_risk_level, "HIGH")
        session.advance_time(50.0)
        self.assertEqual(session.latest_risk_level, "NORMAL")

    def test_monitor_processes_complete_source_once(self):
        result = make_result([FALL_EVENT])
        calls = []

        def fake_pipeline(_args, monitor):
            calls.append(monitor)
            monitor.observe_events([FALL_EVENT])
            return result

        with patch.object(run_ai, "run_pipeline", side_effect=fake_pipeline):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args())

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn('"frame_count": 302', stdout.getvalue())
        self.assertIn("Monitoring completed: source ended.", stdout.getvalue())
        self.assertIn("Events detected: 1", stdout.getvalue())

    def test_monitor_observation_does_not_change_existing_event_output(self):
        original = make_result([FALL_EVENT])
        unchanged = copy.deepcopy(original)
        session = run_ai.MonitoringSession()

        with contextlib.redirect_stdout(io.StringIO()):
            session.observe_events(original["events"])

        self.assertEqual(original, unchanged)

    def test_same_event_is_published_once_and_ingested_in_one_request(self):
        result = make_result([FALL_EVENT, copy.deepcopy(FALL_EVENT)])

        def fake_pipeline(_args, monitor):
            monitor.observe_events([FALL_EVENT, FALL_EVENT])
            return result

        ingestion = {
            "analysisId": "AN-1",
            "duplicate": False,
            "alertCreated": True,
        }
        with patch.object(run_ai, "run_pipeline", side_effect=fake_pipeline), patch.object(
            run_ai, "ingest_result", return_value=ingestion
        ) as ingest:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args(ingest=True, token="jwt"))

        self.assertEqual(code, 0)
        ingest.assert_called_once()
        posted_result = ingest.call_args.args[0]
        self.assertEqual(len(posted_result["event_engine"]["timeline"]), 1)
        self.assertIn("Events detected: 1", stdout.getvalue())
        self.assertIn("Backend ingestion requests: 1", stdout.getvalue())

    def test_normal_frames_do_not_generate_backend_requests(self):
        result = make_result([])

        def fake_pipeline(_args, monitor):
            monitor.observe_fall_frame(
                SimpleNamespace(timestamp=4.0, event=None)
            )
            monitor.observe_fall_frame(
                SimpleNamespace(timestamp=8.0, event=None)
            )
            return result

        with patch.object(run_ai, "run_pipeline", side_effect=fake_pipeline), patch.object(
            run_ai, "ingest_result"
        ) as ingest:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args(ingest=True, token="jwt"))

        self.assertEqual(code, 0)
        ingest.assert_not_called()
        self.assertIn("[MONITOR] 00:04.000 | status=NORMAL", stdout.getvalue())
        self.assertIn("No concerning events; backend ingestion skipped", stdout.getvalue())

    def test_ingestion_is_optional_in_monitor_mode(self):
        result = make_result([FALL_EVENT])

        def fake_pipeline(_args, monitor):
            monitor.observe_events([FALL_EVENT])
            return result

        with patch.object(run_ai, "run_pipeline", side_effect=fake_pipeline), patch.object(
            run_ai, "ingest_result"
        ) as ingest:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args())

        self.assertEqual(code, 0)
        ingest.assert_not_called()
        self.assertIn("Backend: DISABLED", stdout.getvalue())

    def test_monitor_ingestion_without_token_fails_before_http(self):
        result = make_result([FALL_EVENT])

        def fake_pipeline(_args, monitor):
            monitor.observe_events([FALL_EVENT])
            return result

        with patch.object(run_ai, "run_pipeline", side_effect=fake_pipeline), patch(
            "run_ai.urllib.request.urlopen"
        ) as urlopen:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args(ingest=True))

        self.assertEqual(code, 1)
        urlopen.assert_not_called()
        self.assertIn("Backend ingestion requests: 0", stdout.getvalue())

    def test_ctrl_c_stops_cleanly(self):
        def interrupted(_args, monitor):
            monitor.observe_fall_frame(SimpleNamespace(timestamp=4.0, event=None))
            raise KeyboardInterrupt

        with patch.object(run_ai, "run_pipeline", side_effect=interrupted):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = run_ai.run_monitor(monitor_args())

        self.assertEqual(code, 130)
        self.assertIn("Monitoring stopped by user.", stdout.getvalue())
        self.assertIn("Events detected: 0", stdout.getvalue())
        self.assertIn("Alerts generated: 0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
