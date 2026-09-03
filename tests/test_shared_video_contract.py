"""Fast wiring tests; they do not need OpenCV, FFmpeg, or model weights."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "OmniCare-AI"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from fall_detection.pipeline import FallAnalysis
from voice_detection.pipeline import VoiceAnalysis
from voice_detection.speech_recognizer import TranscriptSegment


def load_runner():
    spec = importlib.util.spec_from_file_location("run_ai_for_test", ROOT / "run_ai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedVideoContractTest(unittest.TestCase):
    def test_all_modules_receive_the_same_canonical_video(self):
        runner = load_runner()
        received = {}

        class FakeExtractor:
            def validate_video(self, video_path):
                received["extractor_video"] = str(video_path)
                return video_path

            def extract(self, video_path, wav_path):
                received["extract_video"] = str(video_path)
                Path(wav_path).write_bytes(b"temporary test audio")
                return wav_path

        class FakeVoicePipeline:
            def __init__(self, **_kwargs):
                pass

            def analyze(self, source, wav_path):
                received["voice_video"] = source.source_id
                received["voice_audio"] = str(wav_path)
                segment = TranscriptSegment(155.0, 158.0, "Toi bi nga")
                return VoiceAnalysis(
                    source.source_id,
                    [segment],
                    [],
                    {"score": 85, "level": "EMERGENCY"},
                )

        class FakeFallPipeline:
            def analyze(self, source):
                received["fall_video"] = source.source_id
                return FallAnalysis(source.source_id, 1, [])

        original_extractor = runner.AudioExtractor
        original_voice = runner.VoicePipeline
        original_fall = runner.FallPipeline
        runner.AudioExtractor = FakeExtractor
        runner.VoicePipeline = FakeVoicePipeline
        runner.FallPipeline = FakeFallPipeline
        try:
            with tempfile.TemporaryDirectory() as directory:
                video_path = Path(directory) / "shared.mp4"
                video_path.write_bytes(b"test video placeholder")
                result = runner.run_pipeline(
                    SimpleNamespace(
                        video=video_path,
                        model="tiny",
                        device="cpu",
                        compute_type=None,
                        language="vi",
                        threshold=78,
                    )
                )
        finally:
            runner.AudioExtractor = original_extractor
            runner.VoicePipeline = original_voice
            runner.FallPipeline = original_fall

        source_id = result["input_video"]
        self.assertEqual(received["extractor_video"], source_id)
        self.assertEqual(received["extract_video"], source_id)
        self.assertEqual(received["voice_video"], source_id)
        self.assertEqual(received["fall_video"], source_id)
        self.assertEqual(result["modules"]["conversation"]["input_video"], source_id)
        self.assertEqual(result["modules"]["conversation"]["events"][0]["start"], 155.0)
        self.assertEqual(result["modules"]["conversation"]["events"][0]["end"], 158.0)


if __name__ == "__main__":
    unittest.main()
