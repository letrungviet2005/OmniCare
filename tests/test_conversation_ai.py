"""Focused tests for Gemini conversation without real API requests."""

import contextlib
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
for import_root in (ROOT, AI_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_ai
from voice_detection.conversation import (
    DEFAULT_FALLBACK_RESPONSE,
    ConversationAI,
    ConversationIntent,
    ConversationManager,
    ConversationProviderConfigurationError,
    GeminiProvider,
    GeminiProviderError,
    LLMContext,
    LLMProvider,
    LLMProviderError,
    classify_intent,
    create_llm_provider,
)
from voice_detection.conversation.worker import ConversationWorker
from voice_detection.pipeline import VoicePipeline
from voice_detection.speech.recognizer import TranscriptSegment


class StubProvider:
    name = "test"
    model = "gemini-2.5-flash"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []
        self.closed = False

    def generate_response(self, context):
        self.calls.append(context)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if self.responses else ""

    def close(self):
        self.closed = True


class ConversationAITest(unittest.TestCase):
    def test_legacy_conversation_ai_facade_uses_configured_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            conversation = ConversationAI(api_key="test-key")
        self.assertIsInstance(conversation, ConversationManager)
        self.assertIsInstance(conversation.provider, GeminiProvider)

    def test_missing_api_key_returns_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            conversation = ConversationManager(provider=GeminiProvider())
            with self.assertLogs("omnicare_ai.conversation", level="WARNING") as logs:
                response = conversation.respond("I feel tired")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)
        self.assertEqual(conversation.history, ())
        rendered = " ".join(logs.output)
        self.assertIn("[missing_credentials]", rendered)
        self.assertIn("request was not sent", rendered)
        self.assertNotIn("Bearer", rendered)

    def test_successful_gemini_response(self):
        provider = StubProvider(["  Bác đã nghỉ ngơi chưa ạ?  "])
        conversation = ConversationManager(provider=provider)

        response = conversation.respond("I feel tired today")

        self.assertEqual(response, "Bác đã nghỉ ngơi chưa ạ?")
        self.assertEqual(len(provider.calls), 1)
        prompt = provider.calls[0].system_instruction
        self.assertIn("Luôn trả lời hoàn toàn bằng tiếng Việt", prompt)
        self.assertIn('người dùng là "bác"', prompt)
        self.assertIn("Không tiết lộ", prompt)
        self.assertIn("không chẩn đoán", prompt)

    def test_empty_gemini_response_uses_fallback(self):
        conversation = ConversationManager(provider=StubProvider(["  "]))
        with self.assertLogs("omnicare_ai.conversation", level="WARNING"):
            response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)

    def test_network_failure_uses_fallback(self):
        conversation = ConversationManager(
            provider=StubProvider(error=TimeoutError("secret endpoint"))
        )
        with self.assertLogs("omnicare_ai.conversation", level="WARNING") as logs:
            response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)
        self.assertNotIn("secret endpoint", " ".join(logs.output))

    def test_gemini_client_failure_uses_fallback(self):
        conversation = ConversationManager(
            provider=StubProvider(
                error=LLMProviderError("rate limited", code="rate_limit")
            )
        )
        response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)

    def test_conversation_history_is_sent_as_recent_context(self):
        provider = StubProvider(["First answer", "Second answer"])
        conversation = ConversationManager(provider=provider)
        conversation.respond("First question")
        conversation.respond("Second question")

        roles = [item.role for item in provider.calls[1].recent_conversation]
        self.assertEqual(roles, ["user", "model"])
        self.assertEqual(provider.calls[1].user_text, "Second question")
        self.assertEqual(len(conversation.history), 2)

    def test_history_limit_keeps_only_recent_turns(self):
        provider = StubProvider([f"answer-{index}" for index in range(4)])
        conversation = ConversationManager(provider=provider, history_limit=2)
        for index in range(4):
            conversation.respond(f"question-{index}")
        self.assertEqual(
            [turn.user for turn in conversation.history],
            ["question-2", "question-3"],
        )

    def test_reset_clears_history(self):
        conversation = ConversationManager(
            provider=StubProvider(["Answer"])
        )
        conversation.respond("Question")
        conversation.reset()
        self.assertEqual(conversation.history, ())

    def test_official_client_contract_extracts_clean_text(self):
        generate_content = Mock(
            return_value=SimpleNamespace(text="  Chào bác ạ.\nBác khỏe không?  ")
        )
        generate_content_stream = Mock()
        sdk_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=generate_content,
                generate_content_stream=generate_content_stream,
            )
        )
        provider = GeminiProvider(client=sdk_client, model="gemini-2.5-flash")
        self.assertIsInstance(provider, LLMProvider)

        from voice_detection.conversation.llm_provider import LLMContext

        response = provider.generate_response(
            LLMContext("GREETING", (), "Hi", "System prompt")
        )

        self.assertEqual(response, "Chào bác ạ. Bác khỏe không?")
        self.assertEqual(
            generate_content.call_args.kwargs["model"],
            "gemini-2.5-flash",
        )
        config = generate_content.call_args.kwargs["config"]
        self.assertEqual(
            config["automatic_function_calling"],
            {"disable": True},
        )
        generate_content_stream.assert_not_called()

    def test_model_can_be_configured_from_environment(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-gemini-model"}):
            provider = GeminiProvider(api_key="not-logged")
        self.assertEqual(provider.model, "custom-gemini-model")

    def test_gemini_failures_have_sanitized_diagnostic_categories(self):
        class HttpFailure(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code

        cases = (
            (HttpFailure(401, "secret-key-value"), "authentication"),
            (HttpFailure(404, "model not found: internal-resource"), "model_not_found"),
            (HttpFailure(429, "quota project-secret"), "rate_limit"),
            (TimeoutError("secret endpoint timed out"), "timeout"),
            (ConnectionError("secret hostname connection refused"), "network"),
            (HttpFailure(500, "private upstream body"), "provider_unavailable"),
            (HttpFailure(400, "private request payload"), "request_rejected"),
        )
        for failure, expected in cases:
            with self.subTest(expected=expected):
                generate = Mock(side_effect=failure)
                provider = GeminiProvider(
                    client=SimpleNamespace(
                        models=SimpleNamespace(generate_content=generate)
                    )
                )
                with self.assertRaises(GeminiProviderError) as raised:
                    provider.generate_response(
                        LLMContext("UNKNOWN", (), "Xin chào", "System prompt")
                    )
                self.assertEqual(raised.exception.code, expected)
                self.assertNotIn("secret", str(raised.exception).casefold())
                self.assertNotIn("private", str(raised.exception).casefold())

    def test_malformed_gemini_response_has_its_own_category(self):
        provider = GeminiProvider(
            client=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=Mock(return_value=SimpleNamespace(text=None))
                )
            )
        )
        with self.assertRaises(GeminiProviderError) as raised:
            provider.generate_response(
                LLMContext("UNKNOWN", (), "Xin chào", "System prompt")
            )
        self.assertEqual(raised.exception.code, "malformed_response")

    def test_manager_logs_category_but_not_raw_provider_error(self):
        provider = GeminiProvider(
            client=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=Mock(
                        side_effect=TimeoutError("https://secret-host.invalid/key-value")
                    )
                )
            )
        )
        manager = ConversationManager(provider=provider)
        with self.assertLogs("omnicare_ai.conversation", level="WARNING") as logs:
            response = manager.respond("Xin chào")
        rendered = " ".join(logs.output)
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)
        self.assertIn("[timeout]", rendered)
        self.assertNotIn("secret-host", rendered)
        self.assertNotIn("key-value", rendered)

    def test_manager_passes_structured_intent_context(self):
        provider = StubProvider(["Dạ, bác nghỉ ngơi nhé."])
        manager = ConversationManager(provider=provider)
        manager.respond("Hôm nay tôi hơi mệt")
        context = provider.calls[0]
        self.assertEqual(context.intent, "HEALTH_COMPLAINT")
        self.assertEqual(context.user_text, "Hôm nay tôi hơi mệt")
        self.assertEqual(context.recent_conversation, ())


class IntentClassifierTest(unittest.TestCase):
    def test_supported_intents(self):
        examples = {
            "Chào bác": ConversationIntent.GREETING,
            "Hôm nay tôi hơi mệt": ConversationIntent.HEALTH_COMPLAINT,
            "Tôi muốn gọi cho con": ConversationIntent.FAMILY_QUERY,
            "Hôm nay tôi đã ăn cơm rồi": ConversationIntent.DAILY_ACTIVITY,
            "Tôi nhờ cháu mở cửa giúp": ConversationIntent.REQUEST_HELP,
            "Tôi thích nghe nhạc cổ": ConversationIntent.UNKNOWN,
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(classify_intent(text), expected)


class ProviderFactoryTest(unittest.TestCase):
    def test_provider_selection_defaults_to_gemini(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = create_llm_provider(api_key="test-key")
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.name, "gemini")

    def test_invalid_provider_configuration_is_clear(self):
        with patch.dict(
            os.environ,
            {"CONVERSATION_PROVIDER": "unsupported"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationProviderConfigurationError,
                "Unsupported CONVERSATION_PROVIDER",
            ):
                create_llm_provider()


class ConversationWorkerTest(unittest.TestCase):
    def test_duplicate_transcript_is_submitted_once(self):
        conversation = ConversationManager(
            provider=StubProvider(["Response"])
        )
        worker = ConversationWorker(conversation=conversation).start()
        self.assertTrue(worker.submit(12.4, "I feel tired"))
        self.assertFalse(worker.submit(12.4, "  I FEEL   TIRED "))
        worker.stop()

        results = worker.poll()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].response_text, "Response")

    def test_fragments_produce_one_complete_response(self):
        provider = StubProvider(["Chào bác ạ."])
        conversation = ConversationManager(provider=provider)
        worker = ConversationWorker(
            conversation=conversation,
            settle_seconds=0.01,
        )
        self.assertTrue(worker.submit(10.0, "Oh, I'm"))
        self.assertTrue(worker.submit(10.8, "I'm sorry"))
        worker.start()
        worker.stop()

        results = worker.poll()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].user_text, "Oh, I'm sorry")
        self.assertEqual(results[0].response_text, "Chào bác ạ.")
        self.assertEqual(len(provider.calls), 1)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_ai.drain_conversation_results(
                SimpleNamespace(poll=lambda: results)
            )
        self.assertEqual(output.getvalue().count("[CONVERSATION] User:"), 1)
        self.assertEqual(output.getvalue().count("[CONVERSATION] OmniCare:"), 1)

    def test_worker_submit_does_not_wait_for_gemini(self):
        request_started = threading.Event()
        release_request = threading.Event()

        class SlowConversation:
            model = "gemini-2.5-flash"

            def respond(self, _text, context_data=None):
                request_started.set()
                release_request.wait(timeout=2)
                return "Dạ, cháu đang lắng nghe bác ạ."

            def close(self):
                pass

        worker = ConversationWorker(
            conversation=SlowConversation(),
            settle_seconds=0,
        ).start()
        started = time.perf_counter()
        submitted = worker.submit(12.4, "Hello")
        submit_seconds = time.perf_counter() - started

        self.assertTrue(submitted)
        self.assertLess(submit_seconds, 0.05)
        self.assertTrue(request_started.wait(timeout=1))
        release_request.set()
        worker.stop()

    def test_normal_transcript_is_queued_for_conversation(self):
        transcript = TranscriptSegment(12.4, 13.2, "I feel tired today")
        audio_worker = SimpleNamespace(
            poll=lambda: [("result", SimpleNamespace(segments=[transcript]))]
        )
        conversation_worker = SimpleNamespace(submit=Mock(return_value=True))
        session = run_ai.MonitoringSession()

        with contextlib.redirect_stdout(io.StringIO()):
            _, updates = run_ai.drain_microphone_events(
                audio_worker,
                VoicePipeline(language="auto", threshold=70),
                session,
                "ON",
                conversation_worker,
            )

        self.assertEqual(updates, [])
        call = conversation_worker.submit.call_args
        self.assertEqual(call.args[:2], (12.4, "I feel tired today"))
        self.assertEqual(
            call.args[2]["risk_engine"]["level"],
            "NORMAL",
        )

    def test_emergency_transcript_bypasses_conversation(self):
        transcript = TranscriptSegment(12.4, 13.2, "Help me")
        audio_worker = SimpleNamespace(
            poll=lambda: [("result", SimpleNamespace(segments=[transcript]))]
        )
        conversation_worker = SimpleNamespace(submit=Mock(return_value=True))
        session = run_ai.MonitoringSession()

        with contextlib.redirect_stdout(io.StringIO()):
            _, updates = run_ai.drain_microphone_events(
                audio_worker,
                VoicePipeline(language="auto", threshold=70),
                session,
                "ON",
                conversation_worker,
            )

        conversation_worker.submit.assert_not_called()
        self.assertIn("HELP_REQUEST", [event.type for event in updates])
        self.assertEqual(session.latest_risk_level, "HIGH")


if __name__ == "__main__":
    unittest.main()
