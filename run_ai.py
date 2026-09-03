"""Run Fall Detection, Voice Detection, and Conversation on one video file."""

import argparse
import json
import sys
import tempfile
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    # Keep Vietnamese/Korean transcripts readable on Windows consoles that
    # otherwise default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


WORKSPACE_ROOT = Path(__file__).resolve().parent
AI_ROOT = WORKSPACE_ROOT / "OmniCare-AI"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from common.video_input import VideoInput, VideoInputError
from conversation.pipeline import ConversationPipeline
from fall_detection.pipeline import FallPipeline, FallPipelineError
from voice_detection.audio_extractor import AudioExtractionError, AudioExtractor
from voice_detection.pipeline import VoicePipeline, VoicePipelineError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze one video with OmniCare Fall, Voice, and Conversation modules."
    )
    parser.add_argument("--video", required=True, help="Shared source video, e.g. Videos/test_video.mp4")
    parser.add_argument("--model", default="small", help="Faster-Whisper model name (default: small)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default=None, help="Optional CTranslate2 compute type")
    parser.add_argument("--language", choices=("auto", "vi", "ko"), default="vi")
    parser.add_argument("--threshold", type=float, default=78, help="Voice emergency phrase threshold")
    parser.add_argument("--output", help="Optional JSON result path; stdout is always written")
    return parser.parse_args()


def run_pipeline(args):
    source = VideoInput(args.video).validate_file()
    # `AudioExtractor` receives exactly source.path. It creates one temporary
    # WAV only; no video is copied and no module accepts another media input.
    extractor = AudioExtractor()
    extractor.validate_video(source.path)
    with tempfile.TemporaryDirectory(prefix="omnicare-ai-") as temp_directory:
        wav_path = Path(temp_directory) / f"{source.path.stem}.wav"
        extractor.extract(source.path, wav_path)

        voice = VoicePipeline(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            threshold=args.threshold,
        ).analyze(source, wav_path)
        conversation = ConversationPipeline().analyze(source, voice.transcript_segments)
        fall = FallPipeline().analyze(source)

    return {
        "input_video": source.source_id,
        "modules": {
            "fall_detection": fall.to_dict(),
            "voice_detection": voice.to_dict(),
            "conversation": conversation.to_dict(),
        },
    }


def main():
    args = parse_args()
    try:
        result = run_pipeline(args)
    except (VideoInputError, AudioExtractionError, VoicePipelineError, FallPipelineError) as exc:
        raise SystemExit(f"OmniCare AI pipeline error: {exc}") from exc

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
