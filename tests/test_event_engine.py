"""Unit tests for the deterministic structured-event engine."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "OmniCare-BE"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from event_engine import EventEngine, EventEngineError


class EventEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = EventEngine(fusion_window_seconds=5)

    def test_normalization_adds_event_id_and_preserves_payload(self):
        result = self.engine.process(
            [
                {
                    "timestamp": 5.9,
                    "source": "fall_detection",
                    "type": "FALL_DETECTED",
                    "confidence": 0.85,
                    "score": 85,
                }
            ]
        ).to_dict()
        event = result["timeline"][0]
        self.assertTrue(event["eventId"].startswith("event-"))
        self.assertEqual(event["timestamp"], 5.9)
        self.assertEqual(event["payload"]["score"], 85)

    def test_timestamp_ordering_is_stable(self):
        result = self.engine.process(
            [
                {"eventId": "late", "timestamp": 4, "source": "x", "type": "OTHER"},
                {"eventId": "early", "timestamp": 1, "source": "x", "type": "OTHER"},
            ]
        )
        self.assertEqual([event.event_id for event in result.timeline], ["early", "late"])

    def test_exact_duplicate_is_removed(self):
        event = {"eventId": "same", "timestamp": 2, "source": "voice_detection", "type": "HELP_REQUEST", "confidence": 0.9}
        result = self.engine.process([event, dict(event)])
        self.assertEqual(len(result.timeline), 1)

    def test_fall_event_is_counted(self):
        result = self.engine.process(
            [{"timestamp": 5.9, "source": "fall_detection", "type": "FALL_DETECTED", "confidence": 0.51}]
        )
        self.assertEqual(result.summary["total_events"], 1)
        self.assertEqual(result.summary["fall_events"], 1)

    def test_emergency_voice_event_is_counted(self):
        result = self.engine.process(
            [{"timestamp": 6, "source": "voice_detection", "type": "EMERGENCY_VOICE", "confidence": 0.94, "text": "Help"}]
        )
        self.assertEqual(result.summary["emergency_events"], 1)

    def test_fall_and_emergency_voice_are_fused(self):
        result = self.engine.process(
            [
                {"eventId": "fall", "timestamp": 5.9, "source": "fall_detection", "type": "FALL_DETECTED", "confidence": 0.85},
                {"eventId": "voice", "timestamp": 8, "source": "voice_detection", "type": "HELP_REQUEST", "confidence": 0.94},
            ]
        )
        self.assertEqual(len(result.fused_events), 1)
        fused = result.fused_events[0]
        self.assertEqual(fused.type, "COMBINED_EMERGENCY")
        self.assertEqual(fused.timestamp, 5.9)
        self.assertEqual(fused.payload["relatedEventIds"], ["fall", "voice"])
        self.assertEqual(fused.confidence, 0.94)

    def test_events_outside_fusion_window_are_not_fused(self):
        result = self.engine.process(
            [
                {"eventId": "fall", "timestamp": 5.9, "source": "fall_detection", "type": "FALL_DETECTED", "confidence": 0.85},
                {"eventId": "voice", "timestamp": 11, "source": "voice_detection", "type": "HELP_REQUEST", "confidence": 0.94},
            ]
        )
        self.assertEqual(result.fused_events, [])

    def test_empty_event_list(self):
        result = self.engine.process([]).to_dict()
        self.assertEqual(result, {"timeline": [], "fused_events": [], "summary": {"total_events": 0, "fall_events": 0, "emergency_events": 0, "fused_events": 0}})

    def test_invalid_event_is_rejected(self):
        with self.assertRaises(EventEngineError):
            self.engine.process([{"timestamp": -1, "source": "x", "type": "OTHER"}])


if __name__ == "__main__":
    unittest.main()
