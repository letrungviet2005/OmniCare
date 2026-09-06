"""Run Fall Detection, Voice Detection, and Conversation on one video file."""

import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    # Keep Vietnamese/Korean transcripts readable on Windows consoles that
    # otherwise default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


WORKSPACE_ROOT = Path(__file__).resolve().parent
AI_ROOT = WORKSPACE_ROOT / "OmniCare-AI"
BACKEND_ROOT = WORKSPACE_ROOT / "OmniCare-BE"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from common.video_input import VideoInput, VideoInputError
from conversation.pipeline import ConversationPipeline, ConversationPipelineError
from alert_engine import AlertEngine, AlertEngineError
from event_engine import EventEngine, EventEngineError
from fall_detection.pipeline import FallPipeline, FallPipelineError
from risk_engine import RiskEngine, RiskEngineError
from voice_detection.audio_extractor import AudioExtractionError, AudioExtractor
from voice_detection.pipeline import VoicePipeline, VoicePipelineError


LOGGER = logging.getLogger("omnicare_ai")
DEFAULT_BACKEND_URL = "http://localhost:3001"


class BackendIngestionError(RuntimeError):
    """Raised when the optional backend ingestion cannot be completed."""


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
    parser.add_argument("--fusion-window", type=float, default=10.0, help="Event Engine fall/voice fusion window in seconds (default: 10)")
    parser.add_argument("--ingest", action="store_true", help="POST the final structured result to Spring Boot")
    parser.add_argument("--backend-url", default=os.getenv("OMNICARE_BACKEND_URL", DEFAULT_BACKEND_URL), help="Spring Boot base URL")
    parser.add_argument("--token", default=os.getenv("OMNICARE_JWT_TOKEN"), help="Existing JWT for optional ingestion")
    parser.add_argument("--output", help="Optional JSON result path; stdout is always written")
    return parser.parse_args()


def configure_logging():
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def aggregate_events(fall, voice, conversation):
    events = []
    for event in fall.events:
        normalized = event.to_dict()
        normalized["source"] = "fall_detection"
        events.append(normalized)
    for event in voice.events:
        normalized = event.to_dict()
        normalized["source"] = "voice_detection"
        events.append(normalized)
    for event in conversation.events:
        normalized = event.to_dict()
        normalized["source"] = "conversation"
        events.append(normalized)
    return sorted(events, key=lambda event: (event["timestamp"], event["source"]))


def ingestion_payload(result):
    """Adapt the existing final result to the backend's ingestion DTO."""
    return {
        "source": result["video"],
        "duration": result["duration"],
        # Event Engine timeline is the existing normalized event contract and
        # contains the deterministic eventId required by the backend DTO.
        "events": result["event_engine"]["timeline"],
        "event_engine": result["event_engine"],
        "risk_engine": result["risk_engine"],
        "alert_engine": result["alert_engine"],
    }


def ingest_result(result, backend_url, token, timeout=15):
    if not token:
        raise BackendIngestionError(
            "--ingest requires a JWT. Pass --token or set OMNICARE_JWT_TOKEN."
        )
    if not backend_url:
        raise BackendIngestionError("Backend URL cannot be empty.")

    request = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/v1/ai/events",
        data=json.dumps(ingestion_payload(result), ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            status = getattr(response, "status", response.getcode())
    except urllib.error.HTTPError as exc:
        raise BackendIngestionError(f"Backend ingestion failed with HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise BackendIngestionError(f"Backend ingestion connection failed: {reason}") from exc

    if status < 200 or status >= 300:
        raise BackendIngestionError(f"Backend ingestion failed with HTTP {status}.")
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise BackendIngestionError("Backend returned a malformed JSON response.") from exc
    if not isinstance(parsed, dict) or parsed.get("success") is not True:
        raise BackendIngestionError("Backend returned an invalid ingestion response.")
    analysis_id = parsed.get("analysisId")
    if not isinstance(analysis_id, str) or not analysis_id:
        raise BackendIngestionError("Backend response did not contain analysisId.")
    if not isinstance(parsed.get("duplicate"), bool) or not isinstance(parsed.get("alertCreated"), bool):
        raise BackendIngestionError("Backend response did not contain duplicate/alertCreated flags.")
    return {
        "analysisId": analysis_id,
        "duplicate": parsed["duplicate"],
        "alertCreated": parsed["alertCreated"],
    }


def run_pipeline(args):
    source = VideoInput(args.video).validate_file()
    LOGGER.info("Input video: %s", source.path)
    metadata = source.metadata()
    LOGGER.info(
        "Video stream: OK (%.2f FPS, %dx%d, %.2fs)",
        metadata.fps,
        metadata.width,
        metadata.height,
        metadata.duration_seconds,
    )

    # `AudioExtractor` receives exactly source.path. It creates one temporary
    # WAV only; no video is copied and no module accepts another media input.
    extractor = AudioExtractor()
    extractor.validate_video(source.path)
    LOGGER.info("Audio stream: OK")

    LOGGER.info("Running Fall Detection...")
    try:
        fall = FallPipeline().analyze(source)
    except FallPipelineError:
        raise
    except Exception as exc:
        raise FallPipelineError(f"Fall Detection failed: {exc}") from exc
    LOGGER.info("Fall Detection completed (%d frames).", fall.frames_analyzed)

    with tempfile.TemporaryDirectory(prefix="omnicare-ai-") as temp_directory:
        wav_path = Path(temp_directory) / f"{source.path.stem}.wav"
        LOGGER.info("Extracting audio from the shared input video...")
        extractor.extract(source.path, wav_path)
        LOGGER.info("Audio extraction completed.")

        LOGGER.info("Running Voice Detection...")
        try:
            voice = VoicePipeline(
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                threshold=args.threshold,
            ).analyze(source, wav_path)
        except VoicePipelineError:
            raise
        except Exception as exc:
            raise VoicePipelineError(f"Voice Detection failed: {exc}") from exc
        LOGGER.info(
            "Voice Detection completed (%d transcript segments, %d emergency events).",
            len(voice.transcript_segments),
            len(voice.events),
        )

        LOGGER.info("Running Conversation...")
        try:
            conversation = ConversationPipeline().analyze(
                source, voice.transcript_segments
            )
        except Exception as exc:
            raise ConversationPipelineError(
                f"Conversation failed: {exc}"
            ) from exc
        LOGGER.info("Conversation completed (%d segments).", len(conversation.events))

    structured_events = aggregate_events(fall, voice, conversation)
    try:
        event_engine = EventEngine(
            fusion_window_seconds=getattr(args, "fusion_window", 10.0)
        ).process(structured_events)
    except EventEngineError as exc:
        raise EventEngineError(f"Event Engine failed: {exc}") from exc
    LOGGER.info("Running Risk Engine...")
    try:
        risk_assessment = RiskEngine().assess(event_engine)
    except RiskEngineError as exc:
        raise RiskEngineError(f"Risk Engine failed: {exc}") from exc
    LOGGER.info("Risk Engine completed (%s).", risk_assessment.level)
    LOGGER.info("Running Alert Engine...")
    try:
        alert_decision = AlertEngine().decide(risk_assessment, event_engine)
    except AlertEngineError as exc:
        raise AlertEngineError(f"Alert Engine failed: {exc}") from exc
    LOGGER.info(
        "Alert Engine completed (%s, %s).",
        alert_decision.priority,
        alert_decision.type,
    )

    return {
        "video": source.source_id,
        "input_video": source.source_id,
        "duration": metadata.duration_seconds,
        "metadata": metadata.to_dict(),
        "events": structured_events,
        "event_engine": event_engine.to_dict(),
        "risk_engine": risk_assessment.to_dict(),
        "alert_engine": alert_decision.to_dict(),
        "modules": {
            "fall_detection": fall.to_dict(),
            "voice_detection": voice.to_dict(),
            "conversation": conversation.to_dict(),
        },
    }


def main():
    configure_logging()
    args = parse_args()
    try:
        result = run_pipeline(args)
    except (
        VideoInputError,
        AudioExtractionError,
        VoicePipelineError,
        FallPipelineError,
        ConversationPipelineError,
        EventEngineError,
        RiskEngineError,
        AlertEngineError,
    ) as exc:
        LOGGER.error("OmniCare AI pipeline error: %s", exc)
        return 1

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.ingest:
        try:
            ingestion = ingest_result(result, args.backend_url, args.token)
        except BackendIngestionError as exc:
            LOGGER.error("Backend ingestion failed: %s", exc)
            return 1
        print("Backend ingestion: SUCCESS")
        print(f"analysisId: {ingestion['analysisId']}")
        print(f"duplicate: {str(ingestion['duplicate']).lower()}")
        print(f"alertCreated: {str(ingestion['alertCreated']).lower()}")
    LOGGER.info("Pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
