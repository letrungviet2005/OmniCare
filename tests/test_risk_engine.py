"""Unit tests for deterministic MVP risk assessment."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "OmniCare-BE"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from risk_engine import RiskEngine


def event(event_id, event_type, timestamp=1.0):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "source": "test",
        "type": event_type,
        "confidence": 0.9,
    }


class RiskEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()

    def assess(self, timeline=None, fused_events=None):
        return self.engine.assess(
            {"timeline": timeline or [], "fused_events": fused_events or []}
        ).to_dict()

    def test_empty_events_are_normal(self):
        self.assertEqual(
            self.assess(),
            {"level": "NORMAL", "score": 0, "reasons": [], "related_event_ids": []},
        )

    def test_conversation_only_is_normal(self):
        result = self.assess([event("conversation-1", "CONVERSATION_SEGMENT")])
        self.assertEqual(result["level"], "NORMAL")
        self.assertEqual(result["score"], 0)

    def test_fall_is_high(self):
        result = self.assess([event("fall-1", "FALL_DETECTED")])
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(result["score"], 75)
        self.assertIn("Fall detected", result["reasons"])

    def test_help_request_is_high(self):
        result = self.assess([event("help-1", "HELP_REQUEST")])
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(result["score"], 75)

    def test_emergency_voice_is_high(self):
        result = self.assess([event("voice-1", "EMERGENCY_VOICE")])
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(result["score"], 75)

    def test_combined_emergency_is_critical(self):
        fused = {
            "eventId": "combined-1",
            "timestamp": 1.0,
            "source": "event_engine",
            "type": "COMBINED_EMERGENCY",
            "confidence": 0.95,
            "payload": {"relatedEventIds": ["fall-1", "voice-1"]},
        }
        result = self.assess(
            [event("fall-1", "FALL_DETECTED"), event("voice-1", "HELP_REQUEST")],
            [fused],
        )
        self.assertEqual(result["level"], "CRITICAL")
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["related_event_ids"], ["fall-1", "voice-1"])

    def test_multiple_events_choose_highest_risk(self):
        result = self.assess(
            [
                event("conversation-1", "CONVERSATION_SEGMENT"),
                event("fall-1", "FALL_DETECTED"),
                event("voice-1", "EMERGENCY_VOICE"),
            ]
        )
        self.assertEqual(result["level"], "HIGH")
        self.assertEqual(result["score"], 75)
        self.assertEqual(result["related_event_ids"], ["fall-1", "voice-1"])

    def test_related_event_ids_are_preserved(self):
        result = self.assess([event("fall-42", "FALL_DETECTED")])
        self.assertEqual(result["related_event_ids"], ["fall-42"])

    def test_non_normal_reasons_are_explainable(self):
        result = self.assess([event("help-9", "HELP_REQUEST")])
        self.assertTrue(result["reasons"])
        self.assertTrue(all(isinstance(reason, str) and reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
