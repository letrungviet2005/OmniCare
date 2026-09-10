"""Focused producer tests for Daily Care persistence inputs."""

import json
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
for import_root in (ROOT, AI_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_ai
from fall_detection.activity_session_producer import FallActivitySessionProducer
from voice_detection.conversation.worker import ConversationResponse


SESSION_START = datetime(2026, 9, 9, 8, 0, 0, tzinfo=timezone.utc)


def frame(timestamp, status, *, person=1, pose="Detected"):
    return SimpleNamespace(
        timestamp=timestamp,
        status=status,
        info={"Person": person, "Pose": pose},
    )


def event(event_id, timestamp, event_type="FALL_DETECTED"):
    return SimpleNamespace(event_id=event_id, timestamp=timestamp, type=event_type)


class FakeHttpResponse:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FallActivitySessionProducerTest(unittest.TestCase):
    def producer(self):
        return FallActivitySessionProducer(
            "CAM-DEMO-002", "MONITOR-1", SESSION_START
        )

    def test_only_an_emitted_fall_event_opens_a_session(self):
        producer = self.producer()
        self.assertEqual(producer.observe(frame(5, "LYING"), []), [])
        self.assertEqual(
            producer.observe(frame(6, "NORMAL"), [event("help-1", 6, "HELP_REQUEST")]),
            [],
        )

        updates = producer.observe(
            frame(10, "LYING"), [event("fall-1", 10)]
        )
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].activity_type, "FALLEN")
        self.assertEqual(updates[0].started_at.isoformat(), "2026-09-09T08:00:10+00:00")
        self.assertIsNone(updates[0].ended_at)

    def test_duplicate_fall_event_does_not_create_duplicate_session(self):
        producer = self.producer()
        producer.observe(frame(10, "LYING"), [event("fall-1", 10)])
        self.assertEqual(
            producer.observe(frame(11, "FALL DETECTED"), [event("fall-1", 10)]),
            [],
        )

    def test_session_closes_only_after_four_seconds_of_reliable_normal(self):
        producer = self.producer()
        producer.observe(frame(10, "LYING"), [event("fall-1", 10)])
        self.assertEqual(producer.observe(frame(12, "NORMAL"), []), [])
        self.assertEqual(producer.observe(frame(16, "NORMAL", pose="Not Found"), []), [])
        self.assertEqual(producer.observe(frame(20, "NORMAL"), []), [])
        closed = producer.observe(frame(24, "NORMAL"), [])

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].ended_at.isoformat(), "2026-09-09T08:00:20+00:00")
        self.assertEqual(closed[0].duration_seconds, 10)
        self.assertIsNone(producer.active_session)

    def test_second_fall_after_recovery_creates_a_new_session(self):
        producer = self.producer()
        first = producer.observe(frame(10, "LYING"), [event("fall-1", 10)])[0]
        producer.observe(frame(12, "NORMAL"), [])
        producer.observe(frame(16, "NORMAL"), [])

        second = producer.observe(frame(30, "LYING"), [event("fall-2", 30)])[0]
        self.assertNotEqual(first.activity_session_id, second.activity_session_id)
        self.assertEqual(second.started_at.isoformat(), "2026-09-09T08:00:30+00:00")


class PersistenceWorkerTest(unittest.TestCase):
    def test_activity_http_uses_jwt_and_exact_utc_payload(self):
        captured = {}

        def urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHttpResponse({
                "success": True,
                "activitySessionId": "ACT-1",
                "created": True,
                "updated": False,
                "duplicate": False,
            })

        payload = {
            "activitySessionId": "ACT-1",
            "cameraId": "CAM-DEMO-002",
            "activityType": "FALLEN",
            "startedAt": "2026-09-09T08:00:10Z",
            "endedAt": None,
            "durationSeconds": None,
        }
        secret = "private-jwt"
        with patch.object(run_ai.urllib.request, "urlopen", side_effect=urlopen):
            response = run_ai.ingest_activity_session(
                payload, "http://backend:3001", secret
            )

        self.assertEqual(captured["url"], "http://backend:3001/api/v1/ai/activity-sessions")
        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertEqual(captured["payload"], payload)
        self.assertNotIn(secret, json.dumps(response))

    def test_activity_worker_does_not_block_camera_caller(self):
        request_started = threading.Event()
        release_request = threading.Event()

        def blocked_ingestion(payload, _url, _token):
            request_started.set()
            release_request.wait(timeout=1)
            return {
                "success": True,
                "activitySessionId": payload["activitySessionId"],
                "created": True,
                "updated": False,
                "duplicate": False,
            }

        worker = run_ai.ActivityPersistenceWorker(
            "http://backend:3001", "jwt-token"
        )
        with patch.object(
            run_ai, "ingest_activity_session", side_effect=blocked_ingestion
        ):
            worker.start()
            started = time.perf_counter()
            self.assertTrue(worker.submit({"activitySessionId": "ACT-1"}))
            self.assertLess(time.perf_counter() - started, 0.05)
            self.assertTrue(request_started.wait(timeout=1))
            release_request.set()
            worker.stop()

    def test_conversation_exchanges_reuse_one_monitoring_session(self):
        captured = []

        def persistence(payload, _url, _token):
            captured.append(payload)
            return {
                "success": True,
                "conversationId": payload["conversationId"],
                "messageCount": len(captured) * 2,
                "duplicate": False,
            }

        worker = run_ai.ConversationPersistenceWorker(
            "http://backend:3001",
            "jwt-token",
            "CONV-MONITOR-1",
            "CAM-DEMO-002",
            SESSION_START,
        )
        responses = [
            ConversationResponse(2.0, "Xin chào", "Chào bác ạ.", 0.1, "GREETING"),
            ConversationResponse(8.0, "Tôi đã ăn cơm", "Dạ, tốt quá bác ạ.", 0.1, "DAILY_ACTIVITY"),
        ]
        with patch.object(run_ai, "ingest_conversation", side_effect=persistence):
            worker.start()
            for response in responses:
                self.assertTrue(worker.submit(response))
            worker.stop()

        self.assertEqual(len(captured), 2)
        self.assertEqual({item["conversationId"] for item in captured}, {"CONV-MONITOR-1"})
        self.assertEqual(len({item["exchangeId"] for item in captured}), 2)
        self.assertEqual(
            [item["intent"] for item in captured],
            ["GREETING", "DAILY_ACTIVITY"],
        )


if __name__ == "__main__":
    unittest.main()
