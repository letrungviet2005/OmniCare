"""Focused tests for live occurrence time and conversation persistence bridging."""

import json
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
for import_root in (ROOT, AI_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_ai
from event_engine import EventEngine
from voice_detection.conversation.worker import ConversationResponse


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


class DailyCareBridgeTest(unittest.TestCase):
    def test_live_event_occurrence_is_derived_from_session_clock(self):
        event = EventEngine().process(
            [{
                "timestamp": 5.9,
                "source": "fall_detection",
                "type": "FALL_DETECTED",
                "confidence": 0.9,
                "score": 85,
            }]
        ).timeline[0]
        result = run_ai.camera_event_result(
            SimpleNamespace(source_id="camera:1"),
            SimpleNamespace(
                timestamp=6.0,
                image=SimpleNamespace(shape=(480, 640, 3)),
            ),
            30,
            20.0,
            event,
            10.0,
            "CAM-DEMO-002",
            datetime(2026, 9, 9, 8, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(result["events"][0]["timestamp"], 5.9)
        self.assertEqual(result["events"][0]["occurredAt"], "2026-09-09T08:00:05.900000Z")
        self.assertEqual(
            run_ai.ingestion_payload(result)["events"][0]["occurredAt"],
            "2026-09-09T08:00:05.900000Z",
        )

    def test_conversation_worker_builds_timestamped_payload_without_blocking(self):
        request_started = threading.Event()
        release_request = threading.Event()
        captured = {}

        def blocked_ingestion(payload, _url, _token):
            captured.update(payload)
            request_started.set()
            release_request.wait(timeout=1)
            return {
                "success": True,
                "conversationId": payload["conversationId"],
                "messageCount": 2,
                "duplicate": False,
            }

        worker = run_ai.ConversationPersistenceWorker(
            "http://backend:3001",
            "jwt-token",
            "CONV-1",
            "CAM-DEMO-002",
            datetime(2026, 9, 9, 8, 0, 0, tzinfo=timezone.utc),
        )
        with patch.object(run_ai, "ingest_conversation", side_effect=blocked_ingestion):
            worker.start()
            started = time.perf_counter()
            accepted = worker.submit(
                ConversationResponse(
                    12.4,
                    "Hôm nay bác thấy khỏe",
                    "Dạ, cháu rất vui khi nghe vậy ạ.",
                    0.3,
                    "HEALTH_COMPLAINT",
                )
            )
            self.assertTrue(accepted)
            self.assertLess(time.perf_counter() - started, 0.05)
            self.assertTrue(request_started.wait(timeout=1))
            release_request.set()
            worker.stop()

        self.assertEqual(captured["cameraId"], "CAM-DEMO-002")
        self.assertEqual(captured["intent"], "HEALTH_COMPLAINT")
        self.assertEqual(captured["userOccurredAt"], "2026-09-09T08:00:12.400000Z")
        self.assertEqual(captured["metadata"]["relativeTimestamp"], 12.4)
        self.assertTrue(captured["exchangeId"])

    def test_conversation_http_uses_existing_jwt_without_exposing_it(self):
        captured = {}

        def urlopen(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHttpResponse({
                "success": True,
                "conversationId": "CONV-1",
                "messageCount": 2,
                "duplicate": False,
            })

        payload = {
            "conversationId": "CONV-1",
            "exchangeId": "EX-1",
            "cameraId": "CAM-DEMO-002",
            "userText": "Xin chào",
            "assistantText": "Chào bác ạ.",
            "userOccurredAt": "2026-09-09T08:00:00Z",
            "assistantOccurredAt": "2026-09-09T08:00:01Z",
        }
        secret = "never-log-this-jwt"
        with patch.object(run_ai.urllib.request, "urlopen", side_effect=urlopen):
            response = run_ai.ingest_conversation(
                payload, "http://backend:3001", secret
            )
        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertEqual(captured["payload"], payload)
        self.assertNotIn(secret, json.dumps(response))

    def test_completed_conversation_is_forwarded_to_persistence_worker(self):
        response = ConversationResponse(
            2.0,
            "Xin chào bác",
            "Chào bác ạ.",
            0.2,
            "GREETING",
        )
        persistence = SimpleNamespace(submit=Mock(return_value=True))
        run_ai.drain_conversation_results(
            SimpleNamespace(poll=lambda: [response]),
            persistence_worker=persistence,
        )
        persistence.submit.assert_called_once_with(response)


if __name__ == "__main__":
    unittest.main()
