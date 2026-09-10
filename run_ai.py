"""Run Fall Detection, Voice Detection, and Conversation on one video file."""

import argparse
import copy
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
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

from common.camera_input import CameraInput, CameraInputError
from common.video_input import VideoInput, VideoInputError
from alert_engine import AlertEngine, AlertEngineError
from event_engine import EventEngine, EventEngineError
from fall_detection.activity_session_producer import FallActivitySessionProducer
from fall_detection.pipeline import FallPipeline, FallPipelineError
from risk_engine import RiskEngine, RiskEngineError
from voice_detection.audio.capture import MicrophoneMonitor
from voice_detection.audio.gate import AudioInputGate
from voice_detection.audio.extractor import AudioExtractionError, AudioExtractor
from voice_detection.config.settings import VoiceSettings, microphone_device
from voice_detection.conversation.pipeline import (
    ConversationPipeline,
    ConversationPipelineError,
)
from voice_detection.conversation.provider_factory import (
    ConversationProviderConfigurationError,
)
from voice_detection.conversation.worker import ConversationWorker
from voice_detection.pipeline import VoicePipeline, VoicePipelineError
from voice_detection.tts import TTSProviderError, TTSWorker, create_tts_provider


LOGGER = logging.getLogger("omnicare_ai")
DEFAULT_BACKEND_URL = "http://localhost:3001"
MONITOR_HEARTBEAT_SECONDS = 4.0
MONITOR_FPS_LOG_SECONDS = 10.0
RISK_RECOVERY_SECONDS = 30.0
VOICE_SETTINGS = VoiceSettings.from_environment()
REALTIME_WHISPER_MODEL = VOICE_SETTINGS.realtime_model
REALTIME_VOICE_THRESHOLD = 70
MEANINGFUL_EVENT_TYPES = frozenset(
    {
        "FALL_DETECTED",
        "HELP_REQUEST",
        "EMERGENCY_VOICE",
        "COMBINED_EMERGENCY",
    }
)


class BackendIngestionError(RuntimeError):
    """Raised when the optional backend ingestion cannot be completed."""


class CameraIngestionWorker:
    """Run bounded backend requests without pausing the live frame loop."""

    def __init__(self, backend_url, token, max_pending=8):
        self.backend_url = backend_url
        self.token = token
        self._requests = queue.Queue(maxsize=max(1, int(max_pending)))
        self._results = queue.Queue(maxsize=max(2, int(max_pending) * 2))
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="omnicare-backend-ingestion",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit(self, result):
        try:
            self._requests.put_nowait(result)
            return True
        except queue.Full:
            return False

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _emit(self, item):
        try:
            self._results.put_nowait(item)
        except queue.Full:
            try:
                self._results.get_nowait()
                self._results.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _run(self):
        while True:
            result = self._requests.get()
            if result is None:
                return
            try:
                response = ingest_result(result, self.backend_url, self.token)
            except BackendIngestionError as exc:
                self._emit(("error", str(exc)))
            else:
                self._emit(("success", response))

    def stop(self):
        if self._thread is None:
            return
        self._requests.put(None)
        self._thread.join(timeout=20)


class ConversationPersistenceWorker:
    """Persist completed exchanges without blocking camera, audio, or LLM work."""

    def __init__(self, backend_url, token, conversation_id, camera_id, session_started_at, max_pending=8):
        self.backend_url = backend_url
        self.token = token
        self.conversation_id = conversation_id
        self.camera_id = camera_id
        self.session_started_at = session_started_at
        self._requests = queue.Queue(maxsize=max(1, int(max_pending)))
        self._results = queue.Queue(maxsize=max(2, int(max_pending) * 2))
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="omnicare-conversation-persistence",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit(self, result):
        user_occurred_at = self.session_started_at + timedelta(
            seconds=max(0.0, float(result.timestamp))
        )
        exchange_id = ConversationWorker.transcript_id(result.timestamp, result.user_text)
        payload = {
            "conversationId": self.conversation_id,
            "exchangeId": exchange_id,
            "cameraId": self.camera_id,
            "userText": result.user_text,
            "assistantText": result.response_text,
            "intent": getattr(result, "intent", None),
            "userOccurredAt": utc_text(user_occurred_at),
            "assistantOccurredAt": utc_text(datetime.now(timezone.utc)),
            "metadata": {
                "source": "realtime_microphone",
                "relativeTimestamp": float(result.timestamp),
            },
        }
        try:
            self._requests.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _emit(self, item):
        try:
            self._results.put_nowait(item)
        except queue.Full:
            try:
                self._results.get_nowait()
                self._results.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _run(self):
        while True:
            payload = self._requests.get()
            if payload is None:
                return
            try:
                response = ingest_conversation(payload, self.backend_url, self.token)
            except BackendIngestionError as exc:
                self._emit(("error", str(exc)))
            else:
                self._emit(("success", response))

    def stop(self):
        if self._thread is None:
            return
        self._requests.put(None)
        self._thread.join(timeout=20)


class ActivityPersistenceWorker:
    """Persist bounded activity updates without pausing camera inference."""

    def __init__(self, backend_url, token, max_pending=8):
        self.backend_url = backend_url
        self.token = token
        self._requests = queue.Queue(maxsize=max(1, int(max_pending)))
        self._results = queue.Queue(maxsize=max(2, int(max_pending) * 2))
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="omnicare-activity-persistence",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit(self, payload):
        try:
            self._requests.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _emit(self, item):
        try:
            self._results.put_nowait(item)
        except queue.Full:
            try:
                self._results.get_nowait()
                self._results.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _run(self):
        while True:
            payload = self._requests.get()
            if payload is None:
                return
            try:
                response = ingest_activity_session(
                    payload, self.backend_url, self.token
                )
            except BackendIngestionError as exc:
                self._emit(("error", str(exc)))
            else:
                self._emit(("success", response))

    def stop(self):
        if self._thread is None:
            return
        self._requests.put(None)
        self._thread.join(timeout=20)


def utc_text(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OmniCare AI on one video or a local webcam."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Shared source video, e.g. Videos/test_video.mp4")
    source.add_argument("--camera", type=camera_index, help="OpenCV camera index, e.g. 0")
    parser.add_argument(
        "--camera-id",
        help="Persisted OmniCare camera ID associated with --camera, e.g. CAM-DEMO-002",
    )
    parser.add_argument("--model", default="small", help="Faster-Whisper model name (default: small)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=VOICE_SETTINGS.device)
    parser.add_argument("--compute-type", default=VOICE_SETTINGS.compute_type, help="Optional CTranslate2 compute type")
    parser.add_argument("--language", choices=("auto", "vi", "ko"), default="vi")
    parser.add_argument("--threshold", type=float, default=78, help="Voice emergency phrase threshold")
    parser.add_argument("--fusion-window", type=float, default=10.0, help="Event Engine fall/voice fusion window in seconds (default: 10)")
    parser.add_argument("--monitor", action="store_true", help="Continuously monitor the source timeline until it ends")
    parser.add_argument("--audio", action="store_true", help="Enable microphone monitoring in camera mode")
    parser.add_argument("--audio-language", choices=("auto", "vi", "ko", "en"), default=VOICE_SETTINGS.language, help="Realtime microphone language (default: vi)")
    parser.add_argument("--microphone-device", type=microphone_device, default=VOICE_SETTINGS.microphone_device, help="Optional sounddevice input index or device name")
    parser.add_argument("--audio-threshold", type=float, default=REALTIME_VOICE_THRESHOLD, help="Realtime emergency phrase threshold (default: 70)")
    parser.add_argument("--conversation", action="store_true", help="Enable Gemini responses for normal microphone transcripts")
    parser.add_argument("--tts", action="store_true", help="Speak Conversation responses on a background TTS worker")
    parser.add_argument("--tts-provider", default=os.getenv("TTS_PROVIDER", "windows-sapi"), help="TTS provider (default: windows-sapi)")
    parser.add_argument("--tts-voice", default=os.getenv("OMNICARE_TTS_VOICE"), help="Optional installed Windows SAPI voice name")
    parser.add_argument("--ingest", action="store_true", help="POST the final structured result to Spring Boot")
    parser.add_argument("--backend-url", default=os.getenv("OMNICARE_BACKEND_URL", DEFAULT_BACKEND_URL), help="Spring Boot base URL")
    parser.add_argument("--token", default=os.getenv("OMNICARE_JWT_TOKEN"), help="Existing JWT for optional ingestion")
    parser.add_argument("--output", help="Optional JSON result path; stdout is always written")
    args = parser.parse_args()
    if args.camera is not None and not args.monitor:
        parser.error("--camera requires --monitor")
    if args.camera_id is not None:
        args.camera_id = args.camera_id.strip()
        if args.camera is None:
            parser.error("--camera-id requires --camera")
        if not args.camera_id:
            parser.error("--camera-id cannot be empty")
    if args.audio and args.camera is None:
        parser.error("--audio requires --camera")
    if args.conversation and (args.camera is None or not args.audio):
        parser.error("--conversation requires --camera and --audio")
    if args.tts and not args.conversation:
        parser.error("--tts requires --conversation")
    return args


def camera_index(value):
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "camera index must be a non-negative integer"
        ) from exc
    if index < 0 or str(index) != str(value).strip():
        raise argparse.ArgumentTypeError(
            "camera index must be a non-negative integer"
        )
    return index


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


def evaluate_events(structured_events, fusion_window_seconds=10.0):
    """Run the existing deterministic Event, Risk, and Alert Engine chain."""
    try:
        event_engine = EventEngine(
            fusion_window_seconds=fusion_window_seconds
        ).process(structured_events)
    except EventEngineError as exc:
        raise EventEngineError(f"Event Engine failed: {exc}") from exc

    try:
        risk_assessment = RiskEngine().assess(event_engine)
    except RiskEngineError as exc:
        raise RiskEngineError(f"Risk Engine failed: {exc}") from exc

    try:
        alert_decision = AlertEngine().decide(risk_assessment, event_engine)
    except AlertEngineError as exc:
        raise AlertEngineError(f"Alert Engine failed: {exc}") from exc
    return event_engine, risk_assessment, alert_decision


def format_monitor_timestamp(timestamp):
    timestamp = max(0.0, float(timestamp))
    minutes = int(timestamp // 60)
    seconds = timestamp - (minutes * 60)
    return f"{minutes:02d}:{seconds:06.3f}"


class MonitoringSession:
    """Observe one pipeline pass and publish only meaningful timeline changes."""

    def __init__(
        self,
        fusion_window_seconds=10.0,
        ingest_enabled=False,
        heartbeat_seconds=MONITOR_HEARTBEAT_SECONDS,
        risk_recovery_seconds=RISK_RECOVERY_SECONDS,
    ):
        self.fusion_window_seconds = float(fusion_window_seconds)
        self.ingest_enabled = bool(ingest_enabled)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.risk_recovery_seconds = float(risk_recovery_seconds)
        if self.risk_recovery_seconds <= 0:
            raise ValueError("risk_recovery_seconds must be positive")
        self._next_heartbeat = self.heartbeat_seconds
        self._next_fps_report = MONITOR_FPS_LOG_SECONDS
        self._structured_events = []
        self._fused_events = []
        self._published_event_ids = set()
        self._alert_signatures = set()
        self._backend_requests = 0
        self._current_timestamp = 0.0
        self._last_concerning_timestamp = None
        self.latest_risk_level = "NORMAL"
        self.latest_risk_score = 0
        self.latest_event_type = None
        self.latest_alert_type = "NONE"
        self.latest_voice_text = None
        self.should_alert = False
        self._current_risk = {
            "level": "NORMAL",
            "score": 0,
            "reasons": [],
            "related_event_ids": [],
        }
        self._current_alert = {
            "should_alert": False,
            "priority": "NONE",
            "type": "NONE",
            "message": None,
            "timestamp": None,
            "related_event_ids": [],
            "risk_level": "NORMAL",
            "risk_score": 0,
            "reasons": [],
        }

    @property
    def events_detected(self):
        return len(self._published_event_ids)

    @property
    def alerts_generated(self):
        return len(self._alert_signatures)

    @property
    def backend_requests(self):
        return self._backend_requests

    @property
    def has_meaningful_events(self):
        return bool(self._published_event_ids)

    def has_published(self, event_id):
        return str(event_id) in self._published_event_ids

    def _evaluate_new_event(self, event_timestamp):
        cutoff = self._current_timestamp - self.risk_recovery_seconds
        active_events = [
            event
            for event in self._structured_events
            if float(event["timestamp"]) > cutoff
            or abs(float(event["timestamp"]) - event_timestamp)
            <= self.fusion_window_seconds
        ]
        return evaluate_events(active_events, self.fusion_window_seconds)

    @property
    def historical_events(self):
        return tuple(self._structured_events)

    def conversation_context_snapshot(self):
        """Copy only a small engine-output slice for the background builder."""
        return {
            "event_engine": {
                "timeline": copy.deepcopy(self._structured_events[-16:]),
                "fused_events": copy.deepcopy(self._fused_events[-16:]),
            },
            "risk_engine": copy.deepcopy(self._current_risk),
            "alert_engine": copy.deepcopy(self._current_alert),
            "current_timestamp": self._current_timestamp,
        }

    def advance_time(self, timestamp):
        """Advance current state without removing historical engine events."""
        self._current_timestamp = max(self._current_timestamp, float(timestamp))
        if self._last_concerning_timestamp is None:
            return False
        if (
            self._current_timestamp - self._last_concerning_timestamp
            < self.risk_recovery_seconds
        ):
            return False
        self.latest_risk_level = "NORMAL"
        self.latest_risk_score = 0
        self.latest_alert_type = "NONE"
        self.latest_voice_text = None
        self.should_alert = False
        self._current_risk = {
            "level": "NORMAL",
            "score": 0,
            "reasons": [],
            "related_event_ids": [],
        }
        self._current_alert = {
            "should_alert": False,
            "priority": "NONE",
            "type": "NONE",
            "message": None,
            "timestamp": None,
            "related_event_ids": [],
            "risk_level": "NORMAL",
            "risk_score": 0,
            "reasons": [],
        }
        self._last_concerning_timestamp = None
        print(
            f"[MONITOR] {format_monitor_timestamp(self._current_timestamp)} "
            "| status=NORMAL",
            flush=True,
        )
        return True

    def report_processing_fps(self, timestamp, fps):
        if float(timestamp) < self._next_fps_report:
            return
        print(f"[MONITOR] FPS: {float(fps):.1f}", flush=True)
        while float(timestamp) >= self._next_fps_report:
            self._next_fps_report += MONITOR_FPS_LOG_SECONDS

    def observe_fall_frame(self, frame_result):
        timestamp = float(frame_result.timestamp)
        while timestamp >= self._next_heartbeat:
            recovered = self.advance_time(self._next_heartbeat)
            if not recovered:
                print(
                    f"[MONITOR] {format_monitor_timestamp(self._next_heartbeat)} "
                    f"| status={self.latest_risk_level}",
                    flush=True,
                )
            self._next_heartbeat += self.heartbeat_seconds
        self.advance_time(timestamp)

        if frame_result.event is not None:
            event = frame_result.event.to_dict()
            event["source"] = "fall_detection"
            return self.observe_events([event])
        return []

    def observe_events(self, events):
        published = []
        for event in sorted(events, key=lambda item: (item["timestamp"], item.get("source", ""))):
            self.advance_time(float(event["timestamp"]))
            event_result = EventEngine(
                fusion_window_seconds=self.fusion_window_seconds
            ).process([*self._structured_events, event])
            self._structured_events = [item.to_dict() for item in event_result.timeline]
            self._fused_events = [
                item.to_dict() for item in event_result.fused_events
            ]
            _, risk, alert = self._evaluate_new_event(float(event["timestamp"]))

            candidates = [
                item
                for item in [*event_result.timeline, *event_result.fused_events]
                if item.type in MEANINGFUL_EVENT_TYPES
                and item.event_id not in self._published_event_ids
            ]
            for candidate in candidates:
                # A timeline re-evaluation can expose old entries again. IDs are
                # stable, so publishing each ID once prevents duplicate alerts.
                self._published_event_ids.add(candidate.event_id)
                if alert.should_alert:
                    signature = (
                        alert.type,
                        alert.timestamp,
                        tuple(alert.related_event_ids),
                    )
                    self._alert_signatures.add(signature)
                self.latest_risk_level = risk.level
                self.latest_risk_score = risk.score
                self.latest_event_type = candidate.type
                self.latest_alert_type = alert.type
                self.should_alert = alert.should_alert
                self._current_risk = risk.to_dict()
                self._current_alert = alert.to_dict()
                self._last_concerning_timestamp = max(
                    candidate.timestamp,
                    self._last_concerning_timestamp
                    if self._last_concerning_timestamp is not None
                    else candidate.timestamp,
                )
                self._print_event(candidate, risk, alert)
                published.append(candidate)
        return published

    def _print_event(self, event, risk, alert):
        print(f"[MONITOR] {format_monitor_timestamp(event.timestamp)}", flush=True)
        print(f"Event: {event.type}", flush=True)
        print(f"Risk: {risk.level} ({risk.score})", flush=True)
        print(f"Alert: {alert.type}", flush=True)
        print(
            f"Backend: {'PENDING' if self.ingest_enabled else 'DISABLED'}",
            flush=True,
        )

    def mark_backend_request(self):
        self._backend_requests += 1

    def print_completion(self, interrupted=False):
        if interrupted:
            print("Monitoring stopped by user.", flush=True)
        else:
            print("Monitoring completed: source ended.", flush=True)
        print(f"Events detected: {self.events_detected}", flush=True)
        print(f"Alerts generated: {self.alerts_generated}", flush=True)
        print(f"Backend ingestion requests: {self.backend_requests}", flush=True)


def ingestion_payload(result):
    """Adapt the existing final result to the backend's ingestion DTO."""
    result_events = result.get("events")
    if not (
        isinstance(result_events, list)
        and all(
            isinstance(event, dict) and event.get("eventId")
            for event in result_events
        )
    ):
        result_events = result["event_engine"]["timeline"]
    return {
        "source": result["video"],
        "duration": result["duration"],
        # Live event batches are already normalized. One-shot results retain
        # their raw module events and therefore use Event Engine's timeline.
        "events": result_events,
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


def ingest_conversation(payload, backend_url, token, timeout=15):
    if not token:
        raise BackendIngestionError("Conversation persistence requires a JWT.")
    if not backend_url:
        raise BackendIngestionError("Backend URL cannot be empty.")
    request = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/v1/ai/conversations",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
        raise BackendIngestionError(
            f"Conversation persistence failed with HTTP {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise BackendIngestionError(
            f"Conversation persistence connection failed: {reason}"
        ) from exc
    if status < 200 or status >= 300:
        raise BackendIngestionError(
            f"Conversation persistence failed with HTTP {status}."
        )
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise BackendIngestionError(
            "Backend returned malformed conversation persistence JSON."
        ) from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("success") is not True
        or not isinstance(parsed.get("conversationId"), str)
        or not isinstance(parsed.get("duplicate"), bool)
    ):
        raise BackendIngestionError(
            "Backend returned an invalid conversation persistence response."
        )
    return parsed


def ingest_activity_session(payload, backend_url, token, timeout=15):
    if not token:
        raise BackendIngestionError("Activity persistence requires a JWT.")
    if not backend_url:
        raise BackendIngestionError("Backend URL cannot be empty.")
    request = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/v1/ai/activity-sessions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
        raise BackendIngestionError(
            f"Activity persistence failed with HTTP {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise BackendIngestionError(
            f"Activity persistence connection failed: {reason}"
        ) from exc
    if status < 200 or status >= 300:
        raise BackendIngestionError(
            f"Activity persistence failed with HTTP {status}."
        )
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise BackendIngestionError(
            "Backend returned malformed activity persistence JSON."
        ) from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("success") is not True
        or not isinstance(parsed.get("activitySessionId"), str)
        or not isinstance(parsed.get("duplicate"), bool)
    ):
        raise BackendIngestionError(
            "Backend returned an invalid activity persistence response."
        )
    return parsed


def run_pipeline(args, monitor=None):
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
        fall_pipeline = FallPipeline()
        if monitor is None:
            fall = fall_pipeline.analyze(source)
        else:
            fall = fall_pipeline.analyze(source, on_frame=monitor.observe_fall_frame)
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
    if monitor is not None:
        # Fall events were observed at frame time. This pass adds timestamped
        # voice/conversation output and safely ignores the already-seen fall ID.
        monitor.observe_events(structured_events)

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


def render_result(result, args):
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        write_result_file(result, args.output)


def write_result_file(result, output_path):
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_monitor(args):
    session = MonitoringSession(
        fusion_window_seconds=getattr(args, "fusion_window", 10.0),
        ingest_enabled=args.ingest,
    )
    LOGGER.info("Monitoring one continuous pass of the shared source video...")
    try:
        result = run_pipeline(args, monitor=session)
    except KeyboardInterrupt:
        session.print_completion(interrupted=True)
        return 130

    # Retain the same structured JSON contract as one-shot mode.
    render_result(result, args)

    ingestion_failed = False
    if args.ingest and session.has_meaningful_events:
        if args.token and args.backend_url:
            session.mark_backend_request()
        try:
            ingestion = ingest_result(result, args.backend_url, args.token)
        except BackendIngestionError as exc:
            LOGGER.error("Backend ingestion failed: %s", exc)
            ingestion_failed = True
        else:
            print("Backend ingestion: SUCCESS")
            print(f"analysisId: {ingestion['analysisId']}")
            print(f"duplicate: {str(ingestion['duplicate']).lower()}")
            print(f"alertCreated: {str(ingestion['alertCreated']).lower()}")
    elif args.ingest:
        print("[MONITOR] No concerning events; backend ingestion skipped.")

    session.print_completion()
    return 1 if ingestion_failed else 0


def camera_event_result(
    source,
    camera_frame,
    frames_analyzed,
    observed_fps,
    event,
    fusion_window,
    camera_id=None,
    session_started_at=None,
):
    """Build the existing structured analysis contract for one new camera event."""
    normalized = event.to_dict()
    if camera_id:
        payload = normalized.get("payload")
        payload = dict(payload) if isinstance(payload, dict) else {}
        payload["cameraId"] = camera_id
        normalized["payload"] = payload
    if session_started_at is not None:
        normalized["occurredAt"] = utc_text(
            session_started_at + timedelta(seconds=max(0.0, float(event.timestamp)))
        )
    if event.type == "COMBINED_EMERGENCY":
        event_engine_data = {
            "timeline": [],
            "fused_events": [normalized],
            "summary": {
                "total_events": 0,
                "fall_events": 0,
                "emergency_events": 0,
                "fused_events": 1,
            },
        }
        risk = RiskEngine().assess(event_engine_data)
        alert = AlertEngine().decide(risk, event_engine_data)
    else:
        event_engine, risk, alert = evaluate_events([normalized], fusion_window)
        event_engine_data = event_engine.to_dict()
    height, width = camera_frame.image.shape[:2]
    duration = float(camera_frame.timestamp)
    fall_events = [normalized] if event.source == "fall_detection" else []
    voice_events = [normalized] if event.source == "voice_detection" else []
    voice_risk = normalized.get("payload", {}).get("voice_risk")
    if not isinstance(voice_risk, dict):
        voice_risk = {
            "score": 0,
            "level": "NORMAL",
            "matched_phrases": [],
            "reason": "No voice event in this batch",
            "matches": [],
        }
    return {
        "video": source.source_id,
        "input_video": source.source_id,
        "duration": duration,
        "metadata": {
            "fps": float(observed_fps),
            "frame_count": int(frames_analyzed),
            "duration_seconds": duration,
            "width": int(width),
            "height": int(height),
        },
        "events": [normalized],
        "event_engine": event_engine_data,
        "risk_engine": risk.to_dict(),
        "alert_engine": alert.to_dict(),
        "modules": {
            "fall_detection": {
                "input_video": source.source_id,
                "frames_analyzed": int(frames_analyzed),
                "events": fall_events,
                "diagnostics": {},
            },
            "voice_detection": {
                "input_video": source.source_id,
                "transcript_segments": [],
                "events": voice_events,
                "risk": voice_risk,
            },
            "conversation": {
                "input_video": source.source_id,
                "events": [],
            },
        },
    }


def drain_microphone_events(
    worker,
    voice_pipeline,
    session,
    audio_status,
    conversation_worker=None,
):
    """Move bounded worker results onto the camera thread and shared engines."""
    updates = []
    try:
        messages = worker.poll()
    except Exception as exc:
        LOGGER.warning("Realtime audio disabled: %s", exc)
        return "OFF", updates

    for kind, payload in messages:
        if kind == "loading":
            LOGGER.debug("%s", payload)
        elif kind == "microphone":
            LOGGER.info(
                "[MIC] READY | device=%s (%s) | rate=%d Hz | channels=%d",
                payload.name,
                payload.device_id,
                payload.capture_sample_rate,
                payload.channels,
            )
        elif kind == "runtime":
            LOGGER.info("[WHISPER] %s", payload)
        elif kind == "ready":
            audio_status = "ON"
        elif kind in ("warning", "error"):
            LOGGER.warning("Realtime audio: %s", payload)
            if kind == "error":
                audio_status = "OFF"
        elif kind == "result":
            LOGGER.debug(
                "[WHISPER] Utterance transcription: %.2fs (%s)",
                getattr(payload, "processing_seconds", 0.0),
                getattr(payload, "language", "unknown"),
            )
            quality_segments = voice_pipeline.accepted_segments(
                payload.segments, log_rejections=True
            )
            voice_risk = voice_pipeline.detector.analyze(quality_segments).to_dict()
            for segment in quality_segments:
                voice_events = voice_pipeline.events_from_segments([segment])
                if voice_events:
                    for voice_event in voice_events:
                        raw_event = voice_event.to_dict()
                        raw_event["source"] = "voice_detection"
                        raw_event["voice_risk"] = voice_risk
                        normalized = EventEngine(
                            fusion_window_seconds=session.fusion_window_seconds
                        ).process([raw_event]).timeline[0]
                        if session.has_published(normalized.event_id):
                            continue
                        session.latest_voice_text = voice_event.text
                        print(
                            f"[AUDIO] {format_monitor_timestamp(voice_event.timestamp)} "
                            f"| transcript={json.dumps(voice_event.text, ensure_ascii=False)}",
                            flush=True,
                        )
                        updates.extend(session.observe_events([raw_event]))
                    continue

                assessment = voice_pipeline.assess_segment_quality(segment)
                if not assessment.allow_conversation:
                    LOGGER.debug(
                        "Conversation skipped a short transcript (%d words).",
                        assessment.word_count,
                    )
                    continue
                if conversation_worker is not None and conversation_worker.submit(
                    segment.start,
                    segment.text,
                    session.conversation_context_snapshot(),
                ):
                    print(
                        f"[AUDIO] {format_monitor_timestamp(segment.start)} "
                        f"| transcript={json.dumps(segment.text, ensure_ascii=False)}",
                        flush=True,
                    )
    return audio_status, updates


def drain_conversation_results(worker, tts_worker=None, persistence_worker=None):
    if worker is None:
        return
    for result in worker.poll():
        print(
            f"[CONVERSATION] User: {result.user_text}",
            flush=True,
        )
        print(
            f"[CONVERSATION] OmniCare: {result.response_text}",
            flush=True,
        )
        LOGGER.debug(
            "Conversation response latency: %.2fs.",
            getattr(result, "processing_seconds", 0.0),
        )
        if tts_worker is not None and not tts_worker.submit(result.response_text):
            LOGGER.warning("TTS queue is full; response will remain text-only.")
        if persistence_worker is not None and not persistence_worker.submit(result):
            LOGGER.warning("Conversation persistence queue is full; exchange was not stored.")


def drain_tts_results(worker):
    if worker is None:
        return
    for kind, payload in worker.poll():
        if kind == "warning":
            LOGGER.warning("%s", payload)
        elif kind == "speaking":
            print("[TTS] Speaking response...", flush=True)
        elif kind == "result":
            LOGGER.debug(
                "TTS completed: queue %.2fs, synthesis %.2fs, audio %s, "
                "playback %.2fs, total %.2fs.",
                payload.queue_wait_seconds,
                payload.synthesis_seconds,
                (
                    f"{payload.audio_duration_seconds:.2f}s"
                    if payload.audio_duration_seconds is not None
                    else "unknown"
                ),
                payload.playback_seconds,
                payload.total_seconds,
            )


def drain_ingestion_results(worker):
    if worker is None:
        return
    for kind, payload in worker.poll():
        if kind == "error":
            LOGGER.error("Backend ingestion failed: %s", payload)
            continue
        print("Backend ingestion: SUCCESS", flush=True)
        print(f"analysisId: {payload['analysisId']}", flush=True)
        print(f"duplicate: {str(payload['duplicate']).lower()}", flush=True)
        print(
            f"alertCreated: {str(payload['alertCreated']).lower()}",
            flush=True,
        )


def drain_conversation_persistence_results(worker):
    if worker is None:
        return
    for kind, payload in worker.poll():
        if kind == "error":
            LOGGER.warning("Conversation persistence failed: %s", payload)
        else:
            LOGGER.debug(
                "Conversation persisted: %s (%d messages, duplicate=%s).",
                payload.get("conversationId"),
                payload.get("messageCount", 0),
                payload.get("duplicate", False),
            )


def drain_activity_persistence_results(worker):
    if worker is None:
        return
    for kind, payload in worker.poll():
        if kind == "error":
            LOGGER.warning("Activity persistence failed: %s", payload)
        else:
            LOGGER.debug(
                "Activity session persisted: %s (created=%s, updated=%s, duplicate=%s).",
                payload.get("activitySessionId"),
                payload.get("created", False),
                payload.get("updated", False),
                payload.get("duplicate", False),
            )


def draw_camera_overlay(
    cv2,
    frame,
    frame_result,
    session,
    observed_fps,
    audio_status="OFF",
):
    """Draw a compact status panel without changing Fall Detection output."""
    panel_width = min(max(1, frame.shape[1] - 20), 460)
    panel_height = min(max(1, frame.shape[0] - 20), 190)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (18, 18, 18), -1)
    alerting = session.should_alert
    color = (60, 60, 255) if alerting else (70, 220, 120)
    status = "ALERT" if alerting else frame_result.status
    risk = session.latest_risk_level
    if session.latest_risk_score:
        risk = f"{risk} ({session.latest_risk_score})"
    lines = (
        "OMNICARE MONITORING",
        f"Status: {status}",
        f"Risk: {risk}",
        f"Event: {session.latest_event_type or 'None'}",
        f"Alert: {session.latest_alert_type}",
        f"Audio: {audio_status}",
        *(
            (f"Voice: {session.latest_voice_text[:36]}",)
            if session.latest_voice_text
            else ()
        ),
        f"FPS: {observed_fps:.1f}",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 34 + (index * 21)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color if index == 0 else (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def run_camera_monitor(args):
    """Monitor one webcam session with one persistent FallPipeline instance."""
    if args.ingest and not args.token:
        raise BackendIngestionError(
            "--ingest requires a JWT. Pass --token or set OMNICARE_JWT_TOKEN."
        )
    if args.ingest and not args.backend_url:
        raise BackendIngestionError("Backend URL cannot be empty.")

    source = CameraInput(args.camera)
    cv2 = source.cv2
    session = MonitoringSession(
        fusion_window_seconds=getattr(args, "fusion_window", 10.0),
        ingest_enabled=args.ingest,
    )
    frames_analyzed = 0
    interrupted = False
    audio_worker = None
    ingestion_worker = None
    conversation_worker = None
    conversation_persistence_worker = None
    activity_producer = None
    activity_persistence_worker = None
    tts_worker = None
    audio_input_gate = AudioInputGate()
    voice_pipeline = None
    session_started_at = None
    audio_status = "OFF"
    LOGGER.info("Opening camera index %d...", args.camera)
    if getattr(args, "camera_id", None):
        LOGGER.info("Persisted camera identity: %s", args.camera_id)
    try:
        source.open()
        session_started_at = datetime.now(timezone.utc)
        LOGGER.info("Camera stream: OK. Press q or Ctrl+C to stop.")
        fall_pipeline = FallPipeline()
        processing_started = time.monotonic()
        if args.ingest:
            ingestion_worker = CameraIngestionWorker(
                args.backend_url, args.token
            ).start()
            if getattr(args, "camera_id", None):
                monitoring_session_id = "MON-" + uuid.uuid4().hex
                activity_producer = FallActivitySessionProducer(
                    args.camera_id,
                    monitoring_session_id,
                    session_started_at,
                )
                activity_persistence_worker = ActivityPersistenceWorker(
                    args.backend_url, args.token
                ).start()
            else:
                LOGGER.warning(
                    "Activity persistence disabled: --camera-id is required "
                    "to resolve the elderly person."
                )
        if getattr(args, "conversation", False):
            try:
                conversation_worker = ConversationWorker().start()
            except ConversationProviderConfigurationError as exc:
                LOGGER.warning(
                    "Conversation AI disabled: %s Continuing without LLM responses.",
                    exc,
                )
            else:
                provider = getattr(
                    conversation_worker.conversation, "provider", None
                )
                LOGGER.info(
                    "Conversation AI enabled (%s/%s).",
                    getattr(provider, "name", "injected"),
                    conversation_worker.conversation.model,
                )
                if args.ingest:
                    if getattr(args, "camera_id", None):
                        conversation_persistence_worker = ConversationPersistenceWorker(
                            args.backend_url,
                            args.token,
                            "CONV-" + uuid.uuid4().hex,
                            args.camera_id,
                            session_started_at,
                        ).start()
                    else:
                        LOGGER.warning(
                            "Conversation persistence disabled: --camera-id is required "
                            "to resolve the elderly person."
                        )
                if getattr(args, "tts", False):
                    try:
                        tts_provider = create_tts_provider(
                                getattr(args, "tts_provider", None),
                                getattr(args, "tts_voice", None),
                            )
                        selected_voice = getattr(tts_provider, "selected_voice", None)
                        selected_culture = getattr(
                            tts_provider, "selected_voice_culture", None
                        )
                        if selected_voice:
                            LOGGER.info(
                                "[TTS] Vietnamese voice: %s (%s).",
                                selected_voice,
                                selected_culture or "unknown language",
                            )
                        voice_warning = getattr(tts_provider, "voice_warning", None)
                        if voice_warning:
                            LOGGER.warning("[TTS] %s", voice_warning)
                        tts_worker = TTSWorker(
                            tts_provider,
                            input_gate=audio_input_gate,
                        ).start()
                    except Exception as exc:
                        LOGGER.warning(
                            "TTS disabled: %s Conversation will remain text-only.",
                            exc,
                        )
        if getattr(args, "audio", False):
            try:
                voice_pipeline = VoicePipeline(
                    model_name=REALTIME_WHISPER_MODEL,
                    device=getattr(args, "device", "auto"),
                    compute_type=getattr(args, "compute_type", None),
                    language=getattr(args, "audio_language", "vi"),
                    threshold=getattr(
                        args, "audio_threshold", REALTIME_VOICE_THRESHOLD
                    ),
                )
                audio_worker = MicrophoneMonitor(
                    timeline_origin=source.started_at,
                    model_name=REALTIME_WHISPER_MODEL,
                    device=getattr(args, "device", "auto"),
                    compute_type=getattr(args, "compute_type", None),
                    language=getattr(args, "audio_language", "vi"),
                    microphone_device=getattr(args, "microphone_device", None),
                    input_gate=audio_input_gate,
                ).start()
                audio_status = "STARTING"
            except Exception as exc:
                LOGGER.warning(
                    "Realtime audio unavailable; continuing camera-only: %s",
                    exc,
                )
                audio_worker = None
                voice_pipeline = None
                audio_status = "OFF"

        while True:
            camera_frame = source.read()
            frame_result = fall_pipeline.process_frame(
                camera_frame.image, camera_frame.timestamp
            )
            frames_analyzed += 1
            updates = session.observe_fall_frame(frame_result)
            if activity_producer is not None:
                for activity_update in activity_producer.observe(
                    frame_result, updates
                ):
                    if not activity_persistence_worker.submit(
                        activity_update.to_dict()
                    ):
                        LOGGER.warning(
                            "Activity persistence queue is full; update was not stored."
                        )
            if audio_worker is not None and voice_pipeline is not None:
                audio_status, audio_updates = drain_microphone_events(
                    audio_worker,
                    voice_pipeline,
                    session,
                    audio_status,
                    conversation_worker,
                )
                updates.extend(audio_updates)
            drain_conversation_results(
                conversation_worker,
                tts_worker,
                conversation_persistence_worker,
            )
            drain_tts_results(tts_worker)
            drain_ingestion_results(ingestion_worker)
            drain_conversation_persistence_results(conversation_persistence_worker)
            drain_activity_persistence_results(activity_persistence_worker)
            processing_elapsed = max(time.monotonic() - processing_started, 1e-9)
            observed_fps = frames_analyzed / processing_elapsed
            session.report_processing_fps(camera_frame.timestamp, observed_fps)

            try:
                draw_camera_overlay(
                    cv2,
                    camera_frame.image,
                    frame_result,
                    session,
                    observed_fps,
                    audio_status,
                )
                cv2.imshow("OmniCare Monitoring", camera_frame.image)
                key = cv2.waitKey(1) & 0xFF
            except Exception as exc:
                raise CameraInputError(f"Camera preview failed: {exc}") from exc

            for event in updates:
                result = camera_event_result(
                    source,
                    camera_frame,
                    frames_analyzed,
                    observed_fps,
                    event,
                    getattr(args, "fusion_window", 10.0),
                    getattr(args, "camera_id", None),
                    session_started_at,
                )
                if args.output:
                    write_result_file(result, args.output)
                if args.ingest:
                    if ingestion_worker.submit(result):
                        session.mark_backend_request()
                    else:
                        LOGGER.error(
                            "Backend ingestion queue is full; event was not submitted."
                        )

            if key == ord("q"):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if audio_worker is not None:
            try:
                audio_worker.stop()
            except Exception as exc:
                LOGGER.warning("Could not stop realtime audio cleanly: %s", exc)
        if conversation_worker is not None:
            conversation_worker.stop()
            drain_conversation_results(
                conversation_worker,
                tts_worker,
                conversation_persistence_worker,
            )
        if conversation_persistence_worker is not None:
            conversation_persistence_worker.stop()
            drain_conversation_persistence_results(conversation_persistence_worker)
        if activity_persistence_worker is not None:
            activity_persistence_worker.stop()
            drain_activity_persistence_results(activity_persistence_worker)
        if ingestion_worker is not None:
            ingestion_worker.stop()
            drain_ingestion_results(ingestion_worker)
        if tts_worker is not None:
            tts_worker.stop()
            drain_tts_results(tts_worker)
        source.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            LOGGER.warning("Could not close the OpenCV preview window cleanly.")
        print("Monitoring stopped.", flush=True)
        print(f"Events detected: {session.events_detected}", flush=True)
        print(f"Alerts generated: {session.alerts_generated}", flush=True)
        print(
            f"Backend ingestion requests: {session.backend_requests}",
            flush=True,
        )
    return 130 if interrupted else 0


def main():
    configure_logging()
    args = parse_args()
    try:
        if getattr(args, "camera", None) is not None:
            return run_camera_monitor(args)
        if getattr(args, "monitor", False):
            return run_monitor(args)
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
        CameraInputError,
        BackendIngestionError,
    ) as exc:
        LOGGER.error("OmniCare AI pipeline error: %s", exc)
        return 1

    render_result(result, args)
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
