"""Deterministic tests for Event Engine to Conversation context adaptation."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from voice_detection.conversation import (
    ConversationContextBuilder,
    ConversationManager,
    GeminiProvider,
)
from voice_detection.conversation.llm_provider import LLMMessage


def event(event_id, timestamp, event_type, payload=None, source="test"):
    return {
        "eventId": event_id,
        "timestamp": timestamp,
        "source": source,
        "type": event_type,
        "confidence": 0.9,
        "payload": payload or {},
    }


class ContextBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = ConversationContextBuilder()

    def build(self, timeline=(), fused=(), **kwargs):
        return self.builder.build(
            event_result={"timeline": timeline, "fused_events": fused},
            **kwargs,
        )

    def test_empty_context(self):
        context = self.build(current_timestamp=10)
        self.assertEqual(context.recent_events, ())
        self.assertIsNone(context.latest_event)
        self.assertIsNone(context.active_risk_signal)

    def test_single_relevant_event(self):
        context = self.build(
            [event("fall-1", 10, "FALL_DETECTED")],
            current_timestamp=11,
        )
        self.assertEqual(len(context.recent_events), 1)
        self.assertEqual(context.latest_event.type, "FALL_DETECTED")

    def test_events_are_sorted_chronologically(self):
        context = self.build(
            [
                event("voice-1", 14, "HELP_REQUEST"),
                event("fall-1", 10, "FALL_DETECTED"),
                event("talk-1", 12, "CONVERSATION_SEGMENT"),
            ],
            current_timestamp=15,
        )
        self.assertEqual(
            [item.timestamp for item in context.recent_events],
            [10, 12, 14],
        )
        self.assertEqual(context.latest_event.type, "HELP_REQUEST")

    def test_old_events_are_excluded(self):
        builder = ConversationContextBuilder(max_age_seconds=20)
        context = builder.build(
            event_result={
                "timeline": [
                    event("old", 79.9, "FALL_DETECTED"),
                    event("edge", 80, "HELP_REQUEST"),
                ],
                "fused_events": [],
            },
            current_timestamp=100,
        )
        self.assertEqual(
            [item.type for item in context.recent_events],
            ["HELP_REQUEST"],
        )

    def test_future_events_are_excluded(self):
        context = self.build(
            [
                event("current", 10, "FALL_DETECTED"),
                event("future", 11, "HELP_REQUEST"),
            ],
            current_timestamp=10,
        )
        self.assertEqual(
            [item.type for item in context.recent_events],
            ["FALL_DETECTED"],
        )

    def test_event_count_is_bounded_to_most_recent(self):
        builder = ConversationContextBuilder(max_events=3)
        context = builder.build(
            event_result={
                "timeline": [
                    event(f"event-{index}", index, "CONVERSATION_SEGMENT")
                    for index in range(10)
                ],
                "fused_events": [],
            },
            current_timestamp=10,
        )
        self.assertEqual(
            [item.timestamp for item in context.recent_events],
            [7, 8, 9],
        )

    def test_payload_text_is_bounded(self):
        builder = ConversationContextBuilder(max_payload_chars=30)
        context = builder.build(
            event_result={
                "timeline": [
                    event(
                        "talk-1",
                        10,
                        "CONVERSATION_SEGMENT",
                        {"text": "x" * 5000},
                    )
                ],
                "fused_events": [],
            },
            current_timestamp=10,
        )
        self.assertLessEqual(len(context.latest_event.details), 30)

    def test_irrelevant_and_unknown_events_are_ignored(self):
        context = self.build(
            [
                event("other", 1, "OTHER"),
                event("future", 2, "UNSUPPORTED_EVENT"),
                event("fall", 3, "FALL_DETECTED"),
            ],
            current_timestamp=3,
        )
        self.assertEqual(
            [item.type for item in context.recent_events],
            ["FALL_DETECTED"],
        )

    def test_supported_conversation_and_emergency_events(self):
        timeline = [
            event("talk", 1, "CONVERSATION_SEGMENT", {"text": "Xin chào"}),
            event("help", 2, "HELP_REQUEST", {"text": "Giúp tôi"}),
            event("voice", 3, "EMERGENCY_VOICE"),
            event("fall", 4, "FALL_DETECTED"),
        ]
        fused = [event("combined", 5, "COMBINED_EMERGENCY")]
        context = self.build(timeline, fused, current_timestamp=5)
        self.assertEqual(
            [item.type for item in context.recent_events],
            [
                "CONVERSATION_SEGMENT",
                "HELP_REQUEST",
                "EMERGENCY_VOICE",
                "FALL_DETECTED",
                "COMBINED_EMERGENCY",
            ],
        )

    def test_normal_or_missing_risk_has_no_active_signal(self):
        missing = self.build(current_timestamp=0)
        normal = self.build(
            risk_result={"level": "NORMAL", "score": 0},
            current_timestamp=0,
        )
        self.assertIsNone(missing.active_risk_signal)
        self.assertIsNone(normal.active_risk_signal)

    def test_existing_risk_and_alert_values_are_preserved(self):
        context = self.build(
            [event("fall", 10, "FALL_DETECTED")],
            risk_result={
                "level": "HIGH",
                "score": 75,
                "reasons": ["Fall detected"],
            },
            alert_result={
                "should_alert": True,
                "priority": "HIGH",
                "type": "FALL_ALERT",
            },
            current_timestamp=10,
        )
        self.assertEqual(context.active_risk_signal["level"], "HIGH")
        self.assertEqual(context.active_risk_signal["score"], 75)
        self.assertEqual(
            context.active_risk_signal["reasons"], ("Fall detected",)
        )
        self.assertTrue(context.active_risk_signal["alert_generated"])
        self.assertEqual(
            context.active_risk_signal["alert_type"], "FALL_ALERT"
        )

    def test_conversation_history_is_bounded(self):
        builder = ConversationContextBuilder(max_conversation_turns=2)
        history = tuple(
            LLMMessage("user" if index % 2 == 0 else "model", str(index))
            for index in range(10)
        )
        context = builder.build(recent_conversation=history)
        self.assertEqual(
            [message.text for message in context.recent_conversation],
            ["6", "7", "8", "9"],
        )

    def test_malformed_event_is_ignored_without_failure(self):
        context = self.build(
            [None, "bad", {}, {"timestamp": "bad", "type": "FALL_DETECTED"}],
            current_timestamp=10,
        )
        self.assertEqual(context.recent_events, ())

    def test_prompt_is_compact_and_omits_internal_ids(self):
        context = self.build(
            [event("internal-secret-id", 10, "FALL_DETECTED")],
            risk_result={"level": "HIGH", "score": 75},
            alert_result={
                "should_alert": True,
                "priority": "HIGH",
                "type": "FALL_ALERT",
            },
            current_timestamp=10,
        )
        rendered = context.to_prompt_text()
        self.assertIn("sự cố ngã", rendered)
        self.assertIn("HIGH (75)", rendered)
        self.assertNotIn("internal-secret-id", rendered)


class CapturingProvider:
    name = "test"
    model = "test"

    def __init__(self):
        self.contexts = []

    def generate_response(self, context):
        self.contexts.append(context)
        return "Dạ, bác nghỉ ngơi một chút nhé."

    def close(self):
        pass


class ConversationContextIntegrationTest(unittest.TestCase):
    def test_manager_passes_built_context_to_provider(self):
        provider = CapturingProvider()
        manager = ConversationManager(provider=provider)
        response = manager.respond(
            "Tôi thấy hơi đau",
            context_data={
                "event_engine": {
                    "timeline": [event("fall", 10, "FALL_DETECTED")],
                    "fused_events": [],
                },
                "risk_engine": {"level": "HIGH", "score": 75},
                "alert_engine": {
                    "should_alert": True,
                    "priority": "HIGH",
                    "type": "FALL_ALERT",
                },
                "current_timestamp": 15,
            },
        )
        self.assertEqual(response, "Dạ, bác nghỉ ngơi một chút nhé.")
        context = provider.contexts[0]
        self.assertEqual(context.intent, "HEALTH_COMPLAINT")
        self.assertEqual(
            context.conversation_context.latest_event.type,
            "FALL_DETECTED",
        )
        self.assertEqual(
            context.conversation_context.active_risk_signal["level"],
            "HIGH",
        )

    def test_context_builder_failure_continues_without_context(self):
        class BrokenBuilder:
            def build(self, **_kwargs):
                raise RuntimeError("broken context")

        provider = CapturingProvider()
        manager = ConversationManager(
            provider=provider,
            context_builder=BrokenBuilder(),
        )
        with self.assertLogs("omnicare_ai.conversation", level="WARNING"):
            response = manager.respond(
                "Xin chào",
                context_data={"event_engine": {}},
            )
        self.assertEqual(response, "Dạ, bác nghỉ ngơi một chút nhé.")
        self.assertIsNone(provider.contexts[0].conversation_context)

    def test_gemini_serializes_compact_human_readable_context(self):
        generate = Mock(
            return_value=SimpleNamespace(text="Dạ, bác nghỉ ngơi nhé.")
        )
        sdk_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate)
        )
        manager = ConversationManager(
            provider=GeminiProvider(client=sdk_client)
        )
        manager.respond(
            "Tôi thấy hơi đau",
            context_data={
                "event_engine": {
                    "timeline": [
                        event(
                            "internal-event-id",
                            10,
                            "FALL_DETECTED",
                            {"large": "x" * 5000},
                        )
                    ],
                    "fused_events": [],
                },
                "risk_engine": {"level": "HIGH", "score": 75},
                "alert_engine": {
                    "should_alert": True,
                    "priority": "HIGH",
                    "type": "FALL_ALERT",
                },
                "current_timestamp": 15,
            },
        )
        prompt = generate.call_args.kwargs["contents"][-1]["parts"][0]["text"]
        self.assertIn("sự cố ngã", prompt)
        self.assertIn("HIGH (75)", prompt)
        self.assertIn("Lời bác vừa nói: Tôi thấy hơi đau", prompt)
        self.assertNotIn("internal-event-id", prompt)
        self.assertNotIn("x" * 100, prompt)


if __name__ == "__main__":
    unittest.main()
