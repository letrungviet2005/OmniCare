"""Focused tests for realtime utterance recognition and asynchronous TTS."""

import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
for import_root in (ROOT, AI_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_ai
from voice_detection.conversation.worker import ConversationResponse
from voice_detection.audio.capture import MicrophoneMonitor
from voice_detection.audio.gate import AudioInputGate
from voice_detection.audio.vad import UtteranceVAD
from voice_detection.config.settings import VoiceSettings, microphone_device
from voice_detection.models.schemas import AudioChunk
from voice_detection.pipeline import VoicePipeline
from voice_detection.speech.postprocessor import (
    TranscriptDeduplicator,
    TranscriptQualityGate,
    correct_transcript,
)
from voice_detection.speech.recognizer import SpeechRecognizer, TranscriptSegment
from voice_detection.speech.realtime import RealtimeTranscriber
from voice_detection.tts.worker import TTSWorker
from voice_detection.tts.synthesizer import WindowsSapiTTS


class RealtimeConfigurationTest(unittest.TestCase):
    def test_vietnamese_is_default_realtime_language(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = VoiceSettings.from_environment()
        self.assertEqual(settings.language, "vi")
        self.assertEqual(settings.realtime_model, "small")

    def test_explicit_auto_language_remains_available(self):
        with patch.dict(os.environ, {"WHISPER_LANGUAGE": "auto"}, clear=True):
            self.assertEqual(VoiceSettings.from_environment().language, "auto")

    def test_microphone_device_accepts_index_or_name(self):
        self.assertEqual(microphone_device(" 2 "), 2)
        self.assertEqual(microphone_device("Microphone Array"), "Microphone Array")
        self.assertIsNone(microphone_device(""))

    def test_cuda_runtime_selects_float16_when_dependencies_are_ready(self):
        with patch.object(
            SpeechRecognizer, "_cuda_diagnostic", return_value=(True, None)
        ):
            recognizer = SpeechRecognizer(device="auto")
        self.assertEqual((recognizer.device, recognizer.compute_type), ("cuda", "float16"))
        self.assertIsNone(recognizer.runtime_reason)

    def test_cuda_failure_reports_reason_and_falls_back_to_cpu(self):
        with patch.object(
            SpeechRecognizer,
            "_cuda_diagnostic",
            return_value=(False, "cublas64_12.dll missing"),
        ):
            recognizer = SpeechRecognizer(device="auto")
        self.assertEqual((recognizer.device, recognizer.compute_type), ("cpu", "int8"))
        self.assertIn("cublas64_12.dll", recognizer.runtime_reason)

    def test_realtime_transcription_forces_task_and_vietnamese(self):
        calls = []

        class Model:
            def transcribe(self, _samples, **kwargs):
                calls.append(kwargs)
                segment = SimpleNamespace(start=0.0, end=0.8, text="Xin chào", words=[])
                info = SimpleNamespace(language="vi", language_probability=1.0)
                return [segment], info

        decoder = RealtimeTranscriber(".", device="cpu", language="vi")
        segments, language, _ = decoder._transcribe_samples(
            Model(), np.ones(16000, dtype=np.float32), 12.0
        )
        self.assertEqual(language, "vi")
        self.assertEqual(segments[0].start, 12.0)
        self.assertEqual(calls[0]["language"], "vi")
        self.assertEqual(calls[0]["task"], "transcribe")
        self.assertEqual(calls[0]["beam_size"], 3)

    def test_completed_microphone_utterance_skips_second_vad_pass(self):
        calls = []

        class Model:
            def transcribe(self, _samples, **kwargs):
                calls.append(kwargs)
                return [], SimpleNamespace(language="vi", language_probability=1.0)

        decoder = RealtimeTranscriber(".", device="cpu", language="vi")
        decoder._transcribe_samples(
            Model(),
            np.ones(16000, dtype=np.float32),
            0.0,
            presegmented=True,
        )
        self.assertFalse(calls[0]["vad_filter"])
        self.assertIsNone(calls[0]["vad_parameters"])

    def test_completed_microphone_utterance_retains_one_conversation_segment(self):
        class Model:
            def transcribe(self, _samples, **_kwargs):
                return [
                    SimpleNamespace(start=0.0, end=3.8, text="Hôm nay bác cảm thấy"),
                    SimpleNamespace(start=3.8, end=5.2, text="khá khỏe."),
                ], SimpleNamespace(language="vi", language_probability=1.0)

        decoder = RealtimeTranscriber(".", device="cpu", language="vi")
        segments, _, _ = decoder._transcribe_samples(
            Model(),
            np.ones(16000, dtype=np.float32),
            10.0,
            presegmented=True,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Hôm nay bác cảm thấy khá khỏe.")
        self.assertEqual((segments[0].start, segments[0].end), (10.0, 15.2))

    def test_cuda_model_initialization_retries_once_on_cpu(self):
        with patch.object(
            SpeechRecognizer, "_cuda_diagnostic", return_value=(True, None)
        ):
            decoder = RealtimeTranscriber(".", device="auto", language="vi")
        cpu_model = object()
        with patch.object(
            decoder,
            "_load_model",
            side_effect=[RuntimeError("cublas failed"), cpu_model],
        ), patch.object(decoder, "_warm_model") as warm:
            model, reason = decoder.load_ready_model()
        self.assertIs(model, cpu_model)
        self.assertEqual((decoder.device, decoder.compute_type), ("cpu", "int8"))
        self.assertIn("cublas failed", reason)
        warm.assert_called_once_with(cpu_model)


class MicrophoneAndVadTest(unittest.TestCase):
    def test_microphone_reports_selected_device_and_preferred_rate(self):
        class InputOutputPair:
            def __getitem__(self, index):
                return (4, 9)[index]

        sounddevice = SimpleNamespace(
            default=SimpleNamespace(device=InputOutputPair()),
            query_devices=Mock(
                return_value={
                    "name": "Microphone Array",
                    "max_input_channels": 2,
                    "default_samplerate": 44100,
                }
            ),
            check_input_settings=Mock(),
        )
        monitor = MicrophoneMonitor(
            timeline_origin=0,
            device="cpu",
            sounddevice_module=sounddevice,
        )
        info = monitor._resolve_microphone(sounddevice)
        self.assertEqual(info.device_id, 4)
        self.assertEqual(info.name, "Microphone Array")
        self.assertEqual(info.capture_sample_rate, 16000)
        self.assertEqual(info.channels, 1)

    def test_microphone_falls_back_to_native_rate(self):
        checks = Mock(side_effect=[RuntimeError("unsupported"), None])
        sounddevice = SimpleNamespace(
            default=SimpleNamespace(device=(3, 8)),
            query_devices=Mock(
                return_value={
                    "name": "Native microphone",
                    "max_input_channels": 1,
                    "default_samplerate": 48000,
                }
            ),
            check_input_settings=checks,
        )
        monitor = MicrophoneMonitor(
            timeline_origin=0,
            device="cpu",
            sounddevice_module=sounddevice,
        )
        info = monitor._resolve_microphone(sounddevice)
        self.assertEqual(info.capture_sample_rate, 48000)
        self.assertEqual(info.whisper_sample_rate, 16000)

    def test_vad_emits_one_utterance_after_end_silence(self):
        vad = UtteranceVAD(
            sample_rate=1000,
            speech_start_rms=0.01,
            silence_seconds=0.3,
            minimum_speech_seconds=0.2,
            maximum_utterance_seconds=3.0,
            pre_roll_seconds=0.1,
        )
        chunks = []
        timestamp = 0.0
        for amplitude in [0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]:
            chunks.append(
                AudioChunk(
                    timestamp,
                    np.full(100, amplitude, dtype=np.float32),
                    1000,
                )
            )
            timestamp += 0.1
        results = [vad.accept(chunk) for chunk in chunks]
        utterances = [result for result in results if result is not None]
        self.assertEqual(len(utterances), 1)
        self.assertAlmostEqual(utterances[0].start, 0.0)
        self.assertGreaterEqual(utterances[0].end, 0.7)

    def test_vad_keeps_natural_pause_inside_one_utterance(self):
        vad = UtteranceVAD(
            sample_rate=1000,
            speech_start_rms=0.01,
            silence_seconds=0.9,
            minimum_speech_seconds=0.45,
            maximum_utterance_seconds=8.0,
            pre_roll_seconds=0.35,
        )
        timestamp = 0.0
        results = []
        amplitudes = [0.1] * 5 + [0.0] * 6 + [0.1] * 5 + [0.0] * 9
        for amplitude in amplitudes:
            result = vad.accept(
                AudioChunk(
                    timestamp,
                    np.full(100, amplitude, dtype=np.float32),
                    1000,
                )
            )
            if result is not None:
                results.append(result)
            timestamp += 0.1
        self.assertEqual(len(results), 1)
        self.assertLessEqual(results[0].start, 0.0)
        self.assertGreaterEqual(results[0].end, 2.5)

    def test_vad_rejects_very_short_noise_burst(self):
        vad = UtteranceVAD(
            sample_rate=1000,
            speech_start_rms=0.01,
            silence_seconds=0.9,
            minimum_speech_seconds=0.45,
        )
        timestamp = 0.0
        emitted = []
        for amplitude in [0.1] * 3 + [0.0] * 9:
            result = vad.accept(
                AudioChunk(
                    timestamp,
                    np.full(100, amplitude, dtype=np.float32),
                    1000,
                )
            )
            if result is not None:
                emitted.append(result)
            timestamp += 0.1
        self.assertEqual(emitted, [])

    def test_normal_speech_is_not_an_emergency(self):
        events = VoicePipeline(language="vi", threshold=70).events_from_segments(
            [TranscriptSegment(1.0, 2.0, "Hôm nay thời tiết rất đẹp")]
        )
        self.assertEqual(events, [])

    def test_transcript_deduplication_is_bounded_and_time_aware(self):
        deduplicator = TranscriptDeduplicator(max_entries=2, duplicate_seconds=5)
        self.assertFalse(deduplicator.is_duplicate("Xin chào bác", 1.0))
        self.assertTrue(deduplicator.is_duplicate(" XIN CHÀO  BÁC! ", 2.0))
        self.assertFalse(deduplicator.is_duplicate("Xin chào bác", 7.1))
        self.assertLessEqual(len(deduplicator._recent), 2)


class EmergencyQualityGateTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = VoicePipeline(language="vi", threshold=70)

    def events(self, text, duration=1.5):
        return self.pipeline.events_from_segments(
            [TranscriptSegment(10.0, 10.0 + duration, text)]
        )

    def test_vietnamese_greeting_is_not_emergency(self):
        self.assertEqual(self.events("Xin chào cháu."), [])

    def test_health_complaint_is_not_emergency(self):
        self.assertEqual(self.events("Bác vẫn còn hơi đau."), [])

    def test_clear_fall_request_is_emergency(self):
        events = self.events("Cứu tôi, tôi bị ngã.")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "HELP_REQUEST")

    def test_clear_elderly_fall_phrase_is_emergency(self):
        events = self.events("Bác bị ngã.")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "HELP_REQUEST")

    def test_clear_immobility_and_breathing_request_is_emergency(self):
        events = self.events("Tôi không đứng dậy được, khó thở.", duration=2.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "HELP_REQUEST")

    def test_clear_family_call_request_is_emergency(self):
        events = self.events("Làm ơn gọi con tôi.")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "HELP_REQUEST")

    def test_repetitive_hallucination_is_rejected_before_emergency(self):
        text = "giúp ơn giúp, giữ ơn, giải chạy, giảm ơn."
        assessment = TranscriptQualityGate().assess(text, 2.0)
        self.assertFalse(assessment.accepted)
        self.assertIn("fragmented repetitive clauses", assessment.reasons)
        self.assertEqual(self.events(text, duration=2.0), [])

    def test_known_whisper_subtitle_hallucination_is_rejected(self):
        assessment = TranscriptQualityGate().assess("Hẹn gặp lại các bạn", 2.0)
        self.assertFalse(assessment.accepted)
        self.assertIn("known subtitle hallucination pattern", assessment.reasons)
        segment = TranscriptSegment(1.0, 3.0, "Hẹn gặp lại các bạn")
        self.assertEqual(self.pipeline.accepted_segments([segment]), [])
        self.assertEqual(self.pipeline.events_from_segments([segment]), [])

        legitimate_context = TranscriptQualityGate().assess(
            "Ngày mai bác sẽ hẹn gặp lại các bạn cũ", 3.0
        )
        self.assertTrue(legitimate_context.accepted)

    def test_short_inaccurate_greeting_is_not_emergency(self):
        self.assertEqual(self.events("chào chó"), [])

    def test_observed_vietnamese_typos_are_corrected_contextually(self):
        examples = {
            "xin chào chào": "xin chào cháu",
            "chào chào bác": "cháu chào bác",
            "chào chó": "chào cháu",
            "hôm nay bắt cảm thấy khá khỏi": "hôm nay bác cảm thấy khá khỏe",
            "bác muốn gọi cho con giái": "bác muốn gọi cho con gái",
            "cháu lội cho con gái của bác đừng khân": (
                "cháu gọi cho con gái của bác được không"
            ),
        }
        for observed, expected in examples.items():
            with self.subTest(observed=observed):
                self.assertEqual(correct_transcript(observed, "vi"), expected)
                self.assertEqual(correct_transcript(observed, "en"), observed)

    def test_single_generic_word_is_rejected_without_blacklist(self):
        assessment = TranscriptQualityGate().assess("giúp", 0.5)
        self.assertFalse(assessment.accepted)
        self.assertEqual(self.events("giúp", duration=0.5), [])

    def test_tiny_non_emergency_fragment_is_not_sent_to_conversation(self):
        assessment = TranscriptQualityGate().assess("khó ai", 0.8)
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.allow_conversation)

    def test_tiny_fragment_does_not_reach_conversation_worker(self):
        segment = TranscriptSegment(10.0, 10.8, "khó ai")
        audio_worker = SimpleNamespace(
            poll=lambda: [
                (
                    "result",
                    SimpleNamespace(
                        segments=[segment],
                        processing_seconds=0.2,
                        language="vi",
                    ),
                )
            ]
        )
        conversation_worker = SimpleNamespace(submit=Mock(return_value=True))
        run_ai.drain_microphone_events(
            audio_worker,
            self.pipeline,
            run_ai.MonitoringSession(),
            "ON",
            conversation_worker,
        )
        conversation_worker.submit.assert_not_called()

    def test_short_clear_emergency_still_reaches_emergency_detector(self):
        events = self.events("Cứu tôi", duration=0.8)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "HELP_REQUEST")

    def test_rejected_transcript_reaches_neither_emergency_nor_conversation(self):
        segment = TranscriptSegment(
            10.0,
            12.0,
            "giúp ơn giúp, giữ ơn, giải chạy, giảm ơn.",
        )
        audio_worker = SimpleNamespace(
            poll=lambda: [
                (
                    "result",
                    SimpleNamespace(
                        segments=[segment],
                        processing_seconds=0.2,
                        language="vi",
                    ),
                )
            ]
        )
        conversation_worker = SimpleNamespace(submit=Mock(return_value=True))
        session = run_ai.MonitoringSession()
        with self.assertLogs("omnicare_ai.voice_detection", level="WARNING"):
            _, updates = run_ai.drain_microphone_events(
                audio_worker,
                self.pipeline,
                session,
                "ON",
                conversation_worker,
            )
        self.assertEqual(updates, [])
        self.assertEqual(session.latest_risk_level, "NORMAL")
        conversation_worker.submit.assert_not_called()


class AsyncTtsTest(unittest.TestCase):
    @staticmethod
    def wait_for(worker, kind, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in worker.poll():
                if event[0] == kind:
                    return event[1]
            time.sleep(0.01)
        raise AssertionError(f"TTS event {kind!r} was not emitted")

    def test_tts_queue_synthesizes_and_plays_in_background(self):
        provider = SimpleNamespace(
            synthesize=Mock(return_value=b"RIFF-wave"),
            close=Mock(),
        )
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player, max_pending=2).start()
        self.assertTrue(worker.submit("Chào bác ạ."))
        result = self.wait_for(worker, "result")
        worker.stop()
        provider.synthesize.assert_called_once_with("Chào bác ạ.")
        player.play.assert_called_once_with(b"RIFF-wave")
        self.assertEqual(result.text, "Chào bác ạ.")

    def test_tts_active_blocks_microphone_pcm_from_stt(self):
        gate = AudioInputGate()
        release_playback = threading.Event()
        playback_started = threading.Event()
        provider = SimpleNamespace(
            synthesize=Mock(return_value=b"RIFF-wave"),
            close=Mock(),
        )

        def play(_audio):
            playback_started.set()
            release_playback.wait(timeout=1)

        player = SimpleNamespace(play=play, stop=lambda: release_playback.set())
        monitor = MicrophoneMonitor(
            timeline_origin=0,
            device="cpu",
            input_gate=gate,
            clock=lambda: 1.0,
            sounddevice_module=SimpleNamespace(),
        )
        worker = TTSWorker(provider, player=player, input_gate=gate).start()
        self.assertTrue(worker.submit("Cứu tôi"))
        self.assertTrue(playback_started.wait(timeout=1))
        self.assertTrue(gate.blocked)

        loud_tts = np.full((1600, 1), 0.2, dtype=np.float32)
        for _ in range(10):
            monitor._audio_callback(loud_tts, 1600, None, None)
        self.assertEqual(monitor._audio.qsize(), 0)
        self.assertEqual(monitor.poll(), [])

        release_playback.set()
        self.wait_for(worker, "result")
        deadline = time.monotonic() + 1
        while gate.blocked and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(gate.blocked)
        monitor._audio_callback(loud_tts, 1600, None, None)
        self.assertEqual(monitor._audio.qsize(), 1)
        worker.stop()

    def test_tts_failure_always_reopens_microphone(self):
        gate = AudioInputGate()
        provider = SimpleNamespace(
            synthesize=Mock(side_effect=RuntimeError("synthesis failed")),
            close=Mock(),
        )
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player, input_gate=gate).start()
        self.assertTrue(worker.submit("Chào bác"))
        warning = self.wait_for(worker, "warning")
        self.assertIn("synthesis failed", warning)
        deadline = time.monotonic() + 1
        while gate.blocked and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(gate.blocked)
        worker.stop()

    def test_tts_cancellation_reopens_microphone(self):
        gate = AudioInputGate()
        synthesis_started = threading.Event()
        cancelled = threading.Event()

        class CancelledProvider:
            def synthesize(self, _text):
                synthesis_started.set()
                cancelled.wait(timeout=1)
                raise RuntimeError("cancelled")

            def close(self):
                cancelled.set()

        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(
            CancelledProvider(), player=player, input_gate=gate
        ).start()
        self.assertTrue(worker.submit("Chào bác"))
        self.assertTrue(synthesis_started.wait(timeout=1))
        self.assertTrue(gate.blocked)
        worker.stop()
        self.assertFalse(gate.blocked)

    def test_tts_playback_produces_no_help_request_or_alert(self):
        gate = AudioInputGate()
        monitor = MicrophoneMonitor(
            timeline_origin=0,
            device="cpu",
            input_gate=gate,
            clock=lambda: 1.0,
            sounddevice_module=SimpleNamespace(),
        )
        gate.block()
        try:
            tts_audio = np.full((1600, 1), 0.3, dtype=np.float32)
            for _ in range(20):
                monitor._audio_callback(tts_audio, 1600, None, None)
            session = run_ai.MonitoringSession()
            _, updates = run_ai.drain_microphone_events(
                monitor,
                VoicePipeline(language="vi", threshold=70),
                session,
                "ON",
            )
        finally:
            gate.unblock()
        self.assertEqual(updates, [])
        self.assertEqual(session.events_detected, 0)
        self.assertEqual(session.alerts_generated, 0)
        self.assertEqual(session.latest_risk_level, "NORMAL")

    def test_tts_failure_is_reported_without_crashing_worker(self):
        provider = SimpleNamespace(
            synthesize=Mock(side_effect=RuntimeError("speaker unavailable")),
            close=Mock(),
        )
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player).start()
        self.assertTrue(worker.submit("Chào bác"))
        warning = self.wait_for(worker, "warning")
        worker.stop()
        self.assertIn("speaker unavailable", warning)
        player.play.assert_not_called()

    def test_tts_submission_does_not_wait_for_synthesis_or_playback(self):
        release = threading.Event()

        def blocked_synthesis(_text):
            release.wait(timeout=1)
            return b"RIFF-wave"

        provider = SimpleNamespace(synthesize=blocked_synthesis, close=Mock())
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player).start()
        started = time.perf_counter()
        self.assertTrue(worker.submit("Bác nghỉ ngơi nhé."))
        self.assertLess(time.perf_counter() - started, 0.05)
        release.set()
        self.wait_for(worker, "result")
        worker.stop()

    def test_pending_tts_queue_keeps_only_latest_response(self):
        first_started = threading.Event()
        release_first = threading.Event()
        synthesized = []

        def synthesize(text):
            synthesized.append(text)
            if len(synthesized) == 1:
                first_started.set()
                release_first.wait(timeout=1)
            return b"RIFF-wave"

        provider = SimpleNamespace(synthesize=synthesize, close=Mock())
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player, max_pending=3).start()
        self.assertEqual(worker._requests.maxsize, 1)
        self.assertTrue(worker.submit("Response 1"))
        self.assertTrue(first_started.wait(timeout=1))
        self.assertTrue(worker.submit("Response 2"))
        self.assertTrue(worker.submit("Response 3"))
        release_first.set()
        deadline = time.monotonic() + 1
        while len(synthesized) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.stop()
        self.assertEqual(synthesized, ["Response 1", "Response 3"])

    def test_vietnamese_sapi_voice_is_preferred_by_culture(self):
        provider = WindowsSapiTTS(
            powershell_path="pwsh.exe",
            installed_voices=[
                {"name": "English Voice", "culture": "en-US", "enabled": True},
                {"name": "Vietnamese Voice", "culture": "vi-VN", "enabled": True},
            ],
        )
        self.assertEqual(provider.selected_voice, "Vietnamese Voice")
        self.assertEqual(provider.selected_voice_culture, "vi-VN")
        self.assertIsNone(provider.voice_warning)

    def test_missing_vietnamese_sapi_voice_has_clear_warning(self):
        provider = WindowsSapiTTS(
            powershell_path="pwsh.exe",
            installed_voices=[
                {"name": "English Voice", "culture": "en-US", "enabled": True}
            ],
        )
        self.assertEqual(provider.selected_voice, "English Voice")
        self.assertIn("No installed Vietnamese SAPI voice", provider.voice_warning)

    def test_conversation_response_is_forwarded_to_tts(self):
        tts = SimpleNamespace(submit=Mock(return_value=True))
        response = ConversationResponse(
            1.0,
            "Xin chào",
            "Chào bác ạ.",
            0.2,
        )
        worker = SimpleNamespace(poll=lambda: [response])
        with patch("builtins.print"):
            run_ai.drain_conversation_results(worker, tts)
        tts.submit.assert_called_once_with("Chào bác ạ.")

    def test_stop_releases_player_and_provider(self):
        provider = SimpleNamespace(synthesize=Mock(), close=Mock())
        player = SimpleNamespace(play=Mock(), stop=Mock())
        worker = TTSWorker(provider, player=player).start()
        worker.stop()
        player.stop.assert_called_once()
        provider.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
