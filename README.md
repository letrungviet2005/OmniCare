# OmniCare

OmniCare is an elderly-care monitoring MVP that turns camera and microphone input into deterministic events, risk assessments, alerts, and bounded companion conversations.

## Current Architecture

```text
Camera frames -> Fall Detection -----------+
                                             |
Microphone -> Whisper -> Emergency matcher -+-> Event Engine -> Risk Engine -> Alert Engine
                    |
                    +-> normal transcript -> ConversationWorker -> LLMProvider (Gemini by default)
                                                |
                                                +-> optional TTS

Structured outputs -> authenticated REST ingestion -> Spring Boot -> MariaDB -> Frontend polling
                                         |
                                         +-> Daily Care persistence producers
                                                   |
                                                   +-> DailyCareAggregationService -> DailyCareReport
                                                                                       |
                                                                                       +-> per-elder Daily Care view
```

The camera loop, realtime STT, Conversation, TTS, event ingestion, and Daily Care persistence use bounded background workers where network or model work could block frame processing. Emergency matching remains authoritative and runs before normal Conversation handling.

## Implemented

- Shared-video and live-camera Fall Detection using the existing YOLO, MediaPipe, motion, and temporal state machine.
- File and realtime Faster-Whisper transcription with emergency phrase detection.
- Provider-independent Vietnamese Conversation with bounded memory, intent classification, context building, Gemini fallback handling, and optional TTS.
- Deterministic Event, Risk, and Alert Engines.
- Optional authenticated backend ingestion; default local runs do not require the backend.
- MariaDB production persistence, JWT tenant isolation, customer-owned elders, and multiple cameras.
- Canonical `occurredAt` UTC instants for live AI events while retaining relative `sourceTimestamp`.
- Per-elder, tenant-scoped date-range APIs for events, risks, alerts, conversations, activity sessions, and sleep sessions.
- Deterministic per-elder Daily Care aggregation over persisted data for a bounded UTC range.
- Authenticated family-facing Daily Care view with local-day selection and deterministic backend data only.

## Daily Care Producers

Conversation persistence is active for camera monitoring when `--ingest`, `--camera-id`, `--audio`, and `--conversation` are supplied. One `Conversation` is reused for that monitoring session. Each completed exchange asynchronously stores deterministic user and assistant message IDs, UTC timestamps, and the existing intent. Prompts, credentials, and provider internals are not persisted.

Activity persistence currently produces only `FALLEN` sessions from trustworthy existing signals:

- a session opens only when Fall Detection emits `FALL_DETECTED`;
- repeated copies of the same normalized event do not create another session;
- it closes only after four continuous seconds of `NORMAL` with a valid person and pose;
- start/end instants use the camera session UTC clock, and duration is derived from those instants;
- the backend resolves customer and elder ownership from JWT plus the persisted camera.

No `ACTIVE` or `INACTIVE` duration is produced because the current detector does not provide a trustworthy activity-state signal. `sleep_sessions` remains empty because there is no reliable sleep producer; inactivity and legacy `Elder.sleep` snapshots are not treated as sleep.

`DailyCareAggregationService` reads only tenant/elder/range-filtered records. It counts distinct persisted fall and emergency event identities, preserves Risk Engine scores, counts persisted alerts and conversation roles, and aggregates only explicit activity/sleep sessions. Session duration is clipped to the requested range and is returned only when every included session has a known end and duration.

`DailyCareReport` contains elder identity and period, event/fall/emergency/alert counts, highest persisted risk, risk and alert records, structured important incidents, conversation/message counts, activity and sleep summaries, and deterministic observation codes. Missing duration or risk data is represented as JSON `null`, not a fabricated zero.

The frontend route `/elder-management/{id}/daily-care` consumes this report through the existing authenticated Axios client. It displays overview counts, highest persisted risk, important incidents, activity/sleep sessions, conversation counts, and observation codes. A selected local calendar day is converted to an exact UTC `[from,to)` range. Missing activity or sleep duration is shown as unavailable rather than zero.

## Backend Boundaries

Production uses MariaDB/MySQL; H2 is test-only. Existing contracts remain in place, with these producer ingestion endpoints:

- `POST /api/v1/ai/events`
- `POST /api/v1/ai/conversations`
- `POST /api/v1/ai/activity-sessions`

Daily Care reads are authenticated under:

```text
GET /api/customers/me/elderly/{id}/events
GET /api/customers/me/elderly/{id}/risks
GET /api/customers/me/elderly/{id}/alerts
GET /api/customers/me/elderly/{id}/conversations
GET /api/customers/me/elderly/{id}/conversation-messages
GET /api/customers/me/elderly/{id}/activity-sessions
GET /api/customers/me/elderly/{id}/sleep-sessions
GET /api/customers/me/elderly/{id}/daily-care
```

All date-range endpoints require `from` and `to` UTC instants, use an inclusive start and exclusive end, and accept at most 31 days. Customer ownership is derived from the authenticated user; client-supplied customer IDs are not authorization inputs.

## Run

```powershell
python run_ai.py --video Videos/fall.mp4
python run_ai.py --video Videos/fall.mp4 --monitor
python run_ai.py --camera 1 --monitor
python run_ai.py --camera 1 --camera-id CAM-DEMO-002 --monitor --audio --conversation
python run_ai.py --camera 1 --camera-id CAM-DEMO-002 --monitor --audio --conversation --ingest
```

Backend ingestion uses `OMNICARE_BACKEND_URL` (default `http://localhost:3001`) and `OMNICARE_JWT_TOKEN`; tokens are never logged. Gemini responses use `GEMINI_API_KEY`, with `GEMINI_MODEL` defaulting to `gemini-2.5-flash`; the existing safe fallback remains available when Gemini is unavailable.

## Verification

Focused Daily Care checks:

```powershell
python -m pytest -q tests/test_daily_care_producers.py tests/test_daily_care_bridge.py tests/test_camera_monitor.py
cd OmniCare-BE
mvn -q "-Dtest=DailyCareAggregationIntegrationTest,DailyCareIntelligenceIntegrationTest,TenantPersistenceIntegrationTest" test
cd ../OmniCare-FE
npm run build
```

The stable video regression remains approximately 302 analyzed frames, `FALL_DETECTED` near 5.9 seconds, risk `HIGH / 75`, and alert `FALL_ALERT`.

## Known Limitations

- An open `FALLEN` activity session remains open if the detector never observes a reliable recovery or monitoring stops first; no end time is fabricated.
- General activity and sleep duration need dedicated trustworthy signals before producers can be enabled.
- Conversation persistence requires a persisted camera identity so the backend can resolve the elder and tenant.
- Daily Care reports are computed on request and are not historical snapshots.
- LLM-generated Daily Care summaries are not implemented.
- The Daily Care screen presents deterministic records only; it does not generate narrative care advice.

## Recommended Next Step

Run an authenticated browser smoke test against MariaDB data produced by a real monitoring session. Once that user-facing contract is verified, an optional AI summary writer can consume `DailyCareReport` without changing its deterministic source data.
