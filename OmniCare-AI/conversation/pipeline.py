"""Conversation events derived from the shared video's timestamped transcript."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConversationEvent:
    start: float
    end: float
    transcript: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ConversationAnalysis:
    input_video: str
    events: list

    def to_dict(self):
        return {
            "input_video": self.input_video,
            "events": [event.to_dict() for event in self.events],
        }


class ConversationPipeline:
    """Reuse the one shared STT pass instead of accepting a second media input."""

    def analyze(self, video_input, transcript_segments):
        events = [
            ConversationEvent(
                start=float(segment.start),
                end=float(segment.end),
                transcript=segment.text,
            )
            for segment in transcript_segments
            if segment.text.strip()
        ]
        return ConversationAnalysis(video_input.source_id, events)
