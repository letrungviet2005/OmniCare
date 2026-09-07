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
from conversation import (
    DEFAULT_FALLBACK_RESPONSE,
    ConversationAI,
    GeminiClient,
    GeminiClientError,
)
from conversation.worker import ConversationWorker
from voice_detection.pipeline import VoicePipeline
from voice_detection.speech_recognizer import TranscriptSegment


class StubConversationClient:
    model = "gemini-2.5-flash"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []
        self.closed = False

    def generate(self, contents, system_instruction):
        self.calls.append((contents, system_instruction))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if self.responses else ""

    def close(self):
        self.closed = True


class ConversationAITest(unittest.TestCase):
    def test_missing_api_key_returns_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            conversation = ConversationAI()
            with self.assertLogs("omnicare_ai.conversation", level="WARNING"):
                response = conversation.respond("I feel tired")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)
        self.assertEqual(conversation.history, ())

    def test_successful_gemini_response(self):
        client = StubConversationClient(["  Bác đã nghỉ ngơi chưa ạ?  "])
        conversation = ConversationAI(client=client)

        response = conversation.respond("I feel tired today")

        self.assertEqual(response, "Bác đã nghỉ ngơi chưa ạ?")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Luôn trả lời hoàn toàn bằng tiếng Việt", client.calls[0][1])
        self.assertIn('người dùng là "bác"', client.calls[0][1])
        self.assertIn("Không tiết lộ", client.calls[0][1])
        self.assertIn("không chẩn đoán", client.calls[0][1])

    def test_empty_gemini_response_uses_fallback(self):
        conversation = ConversationAI(client=StubConversationClient(["  "]))
        with self.assertLogs("omnicare_ai.conversation", level="WARNING"):
            response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)

    def test_network_failure_uses_fallback(self):
        conversation = ConversationAI(
            client=StubConversationClient(error=TimeoutError("secret endpoint"))
        )
        with self.assertLogs("omnicare_ai.conversation", level="WARNING") as logs:
            response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)
        self.assertNotIn("secret endpoint", " ".join(logs.output))

    def test_gemini_client_failure_uses_fallback(self):
        conversation = ConversationAI(
            client=StubConversationClient(error=GeminiClientError("rate limited"))
        )
        response = conversation.respond("Hello")
        self.assertEqual(response, DEFAULT_FALLBACK_RESPONSE)

    def test_conversation_history_is_sent_as_recent_context(self):
        client = StubConversationClient(["First answer", "Second answer"])
        conversation = ConversationAI(client=client)
        conversation.respond("First question")
        conversation.respond("Second question")

        roles = [item["role"] for item in client.calls[1][0]]
        self.assertEqual(roles, ["user", "model", "user"])
        self.assertEqual(len(conversation.history), 2)

    def test_history_limit_keeps_only_recent_turns(self):
        client = StubConversationClient([f"answer-{index}" for index in range(4)])
        conversation = ConversationAI(client=client, history_limit=2)
        for index in range(4):
            conversation.respond(f"question-{index}")
        self.assertEqual(
            [turn.user for turn in conversation.history],
            ["question-2", "question-3"],
        )

    def test_reset_clears_history(self):
        conversation = ConversationAI(
            client=StubConversationClient(["Answer"])
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
        client = GeminiClient(client=sdk_client, model="gemini-2.5-flash")

        response = client.generate(
            [{"role": "user", "parts": [{"text": "Hi"}]}],
            "System prompt",
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
            client = GeminiClient(api_key="not-logged")
        self.assertEqual(client.model, "custom-gemini-model")


class ConversationWorkerTest(unittest.TestCase):
    def test_duplicate_transcript_is_submitted_once(self):
        conversation = ConversationAI(
            client=StubConversationClient(["Response"])
        )
        worker = ConversationWorker(conversation=conversation).start()
        self.assertTrue(worker.submit(12.4, "I feel tired"))
        self.assertFalse(worker.submit(12.4, "  I FEEL   TIRED "))
        worker.stop()

        results = worker.poll()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].response_text, "Response")

    def test_fragments_produce_one_complete_response(self):
        client = StubConversationClient(["Chào bác ạ."])
        conversation = ConversationAI(client=client)
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
        self.assertEqual(len(client.calls), 1)

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

            def respond(self, _text):
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
        conversation_worker.submit.assert_called_once_with(
            12.4, "I feel tired today"
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
