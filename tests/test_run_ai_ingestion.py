"""Focused tests for optional run_ai.py backend ingestion."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ai


RESULT = {
    "video": "Videos/fall.mp4",
    "duration": 10.1,
    "events": [{"eventId": "event-1", "timestamp": 5.9, "source": "fall_detection", "type": "FALL_DETECTED", "confidence": 0.9, "payload": {}}],
    "event_engine": {"timeline": [], "fused_events": [], "summary": {}},
    "risk_engine": {"level": "HIGH", "score": 75, "reasons": ["Fall detected"], "related_event_ids": ["event-1"]},
    "alert_engine": {"should_alert": True, "priority": "HIGH", "type": "FALL_ALERT", "message": "Fall detected", "timestamp": 5.9, "related_event_ids": ["event-1"], "risk_level": "HIGH", "risk_score": 75, "reasons": ["Fall detected"]},
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self.payload

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RunAiIngestionTest(unittest.TestCase):
    def test_ingest_disabled_makes_no_http_request(self):
        args = SimpleNamespace(ingest=False, output=None, token=None, backend_url="http://localhost:3001")
        with patch.object(run_ai, "parse_args", return_value=args), patch.object(run_ai, "run_pipeline", return_value=RESULT), patch.object(run_ai, "ingest_result") as ingest:
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_ai.main()
        self.assertEqual(code, 0)
        ingest.assert_not_called()

    def test_successful_post_returns_summary(self):
        response = FakeResponse({"success": True, "analysisId": "AN-1", "duplicate": False, "alertCreated": True})
        with patch("run_ai.urllib.request.urlopen", return_value=response):
            summary = run_ai.ingest_result(RESULT, "http://backend:3001", "jwt-token")
        self.assertEqual(summary, {"analysisId": "AN-1", "duplicate": False, "alertCreated": True})

    def test_missing_jwt_fails_before_http(self):
        with patch("run_ai.urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(run_ai.BackendIngestionError, "requires a JWT"):
                run_ai.ingest_result(RESULT, "http://backend:3001", None)
        urlopen.assert_not_called()

    def test_connection_failure_is_reported(self):
        with patch("run_ai.urllib.request.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(run_ai.BackendIngestionError, "connection failed"):
                run_ai.ingest_result(RESULT, "http://backend:3001", "jwt-token")

    def test_http_4xx_and_5xx_are_reported(self):
        for status in (400, 401, 403, 500):
            with self.subTest(status=status):
                error = HTTPError("http://backend:3001/api/v1/ai/events", status, "failure", {}, io.BytesIO())
                with patch("run_ai.urllib.request.urlopen", side_effect=error):
                    with self.assertRaisesRegex(run_ai.BackendIngestionError, f"HTTP {status}"):
                        run_ai.ingest_result(RESULT, "http://backend:3001", "jwt-token")

    def test_duplicate_response_is_preserved(self):
        response = FakeResponse({"success": True, "analysisId": "AN-1", "duplicate": True, "alertCreated": True})
        with patch("run_ai.urllib.request.urlopen", return_value=response):
            summary = run_ai.ingest_result(RESULT, "http://backend:3001", "jwt-token")
        self.assertTrue(summary["duplicate"])

    def test_authorization_header_is_sent_without_logging_token(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse({"success": True, "analysisId": "AN-1", "duplicate": False, "alertCreated": True})

        secret = "do-not-print-this-token"
        with patch("run_ai.urllib.request.urlopen", side_effect=fake_urlopen):
            summary = run_ai.ingest_result(RESULT, "http://backend:3001", secret)
        self.assertEqual(captured["request"].get_header("Authorization"), f"Bearer {secret}")
        self.assertEqual(captured["request"].get_header("Content-type"), "application/json")
        self.assertNotIn(secret, json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
