"""Unit tests for deterministic Alert Engine decisions."""

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "OmniCare-BE"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from alert_engine import AlertEngine


def event(event_id, event_type, timestamp=1.0):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "source": "test",
        "type": event_type,
        "confidence": 0.9,
    }


def risk(level, score, reasons=None, related_ids=None):
    return {
        "level": level,
        "score": score,
        "reasons": reasons or [],
        "related_event_ids": related_ids or [],
    }


class AlertEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine()

    def decide(self, risk_result, timeline=None, fused_events=None):
        return self.engine.decide(
            risk_result,
            {"timeline": timeline or [], "fused_events": fused_events or []},
        ).to_dict()

    def test_normal_risk_does_not_alert(self):
        result = self.decide(risk("NORMAL", 0))
        self.assertFalse(result["should_alert"])
        self.assertEqual(result["priority"], "NONE")
        self.assertEqual(result["type"], "NONE")
        self.assertIsNone(result["message"])
        self.assertIsNone(result["timestamp"])
        self.assertEqual(result["related_event_ids"], [])

    def test_low_risk_does_not_alert(self):
        result = self.decide(risk("LOW", 25, ["Low priority signal"], ["low-1"]))
        self.assertFalse(result["should_alert"])
        self.assertEqual(result["priority"], "LOW")
        self.assertEqual(result["type"], "NONE")
        self.assertEqual(result["related_event_ids"], ["low-1"])

    def test_medium_risk_alerts_with_generic_type(self):
        result = self.decide(risk("MEDIUM", 50, ["Medium priority signal"], ["medium-1"]))
        self.assertTrue(result["should_alert"])
        self.assertEqual(result["priority"], "MEDIUM")
        self.assertEqual(result["type"], "RISK_ALERT")
        self.assertEqual(result["message"], "Medium risk event detected")

    def test_high_fall_alert(self):
        timeline = [event("fall-1", "FALL_DETECTED", 5.9)]
        result = self.decide(
            risk("HIGH", 75, ["Fall detected"], ["fall-1"]), timeline
        )
        self.assertTrue(result["should_alert"])
        self.assertEqual(result["priority"], "HIGH")
        self.assertEqual(result["type"], "FALL_ALERT")
        self.assertEqual(result["message"], "Fall detected")
        self.assertEqual(result["timestamp"], 5.9)
        self.assertEqual(result["related_event_ids"], ["fall-1"])

    def test_high_help_request_alert(self):
        timeline = [event("help-1", "HELP_REQUEST", 4.8)]
        result = self.decide(risk("HIGH", 75, ["Help request detected"], ["help-1"]), timeline)
        self.assertEqual(result["type"], "HELP_ALERT")
        self.assertEqual(result["message"], "Help request detected")

    def test_high_emergency_voice_alert(self):
        timeline = [event("voice-1", "EMERGENCY_VOICE", 4.8)]
        result = self.decide(risk("HIGH", 75, ["Emergency voice detected"], ["voice-1"]), timeline)
        self.assertEqual(result["type"], "EMERGENCY_VOICE_ALERT")
        self.assertEqual(result["message"], "Emergency voice detected")

    def test_critical_combined_alert_uses_fused_timestamp(self):
        timeline = [event("fall-1", "FALL_DETECTED", 5.9), event("voice-1", "HELP_REQUEST", 8.0)]
        fused = [
            {
                "eventId": "combined-1",
                "timestamp": 5.9,
                "source": "event_engine",
                "type": "COMBINED_EMERGENCY",
                "confidence": 0.95,
                "payload": {"relatedEventIds": ["fall-1", "voice-1"]},
            }
        ]
        result = self.decide(
            risk("CRITICAL", 100, ["Fall detected with emergency voice event"], ["fall-1", "voice-1"]),
            timeline,
            fused,
        )
        self.assertTrue(result["should_alert"])
        self.assertEqual(result["priority"], "CRITICAL")
        self.assertEqual(result["type"], "COMBINED_EMERGENCY_ALERT")
        self.assertEqual(result["message"], "Fall and emergency voice detected")
        self.assertEqual(result["timestamp"], 5.9)
        self.assertEqual(result["related_event_ids"], ["fall-1", "voice-1"])

    def test_multiple_related_event_ids_are_preserved(self):
        timeline = [event("fall-1", "FALL_DETECTED", 5.9), event("voice-1", "EMERGENCY_VOICE", 6.2)]
        ids = ["fall-1", "voice-1"]
        result = self.decide(risk("HIGH", 75, ["Multiple concerning events"], ids), timeline)
        self.assertEqual(result["related_event_ids"], ids)

    def test_multiple_related_events_use_earliest_timestamp(self):
        timeline = [event("fall-1", "FALL_DETECTED", 5.9), event("voice-1", "HELP_REQUEST", 4.8)]
        result = self.decide(risk("HIGH", 75, ["Multiple concerning events"], ["fall-1", "voice-1"]), timeline)
        self.assertEqual(result["timestamp"], 4.8)

    def test_missing_specific_event_uses_generic_alert(self):
        result = self.decide(risk("HIGH", 75, ["High risk event"], ["missing-1"]))
        self.assertEqual(result["type"], "RISK_ALERT")
        self.assertEqual(result["message"], "High risk event detected")
        self.assertIsNone(result["timestamp"])

    def test_output_is_deterministic(self):
        timeline = [event("fall-1", "FALL_DETECTED", 5.9)]
        risk_result = risk("HIGH", 75, ["Fall detected"], ["fall-1"])
        self.assertEqual(self.decide(risk_result, timeline), self.decide(risk_result, timeline))

    def test_input_is_not_mutated(self):
        risk_result = risk("HIGH", 75, ["Fall detected"], ["fall-1"])
        event_result = {"timeline": [event("fall-1", "FALL_DETECTED", 5.9)], "fused_events": []}
        original_risk = copy.deepcopy(risk_result)
        original_events = copy.deepcopy(event_result)
        self.engine.decide(risk_result, event_result)
        self.assertEqual(risk_result, original_risk)
        self.assertEqual(event_result, original_events)


if __name__ == "__main__":
    unittest.main()
