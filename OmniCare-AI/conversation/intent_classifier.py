"""Deterministic conversation intent hints; not an emergency detector."""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class ConversationIntent(str, Enum):
    GREETING = "GREETING"
    HEALTH_COMPLAINT = "HEALTH_COMPLAINT"
    FAMILY_QUERY = "FAMILY_QUERY"
    DAILY_ACTIVITY = "DAILY_ACTIVITY"
    REQUEST_HELP = "REQUEST_HELP"
    UNKNOWN = "UNKNOWN"


def _normalize(text):
    value = unicodedata.normalize("NFKD", str(text).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace("đ", "d")
    return re.sub(r"[^a-z0-9\s]", " ", value)


_PATTERNS = (
    (
        ConversationIntent.HEALTH_COMPLAINT,
        r"\b(met|dau|chong mat|kho chiu|khong khoe|yeu|mat ngu|buon non|tired|pain|unwell)\b",
    ),
    (
        ConversationIntent.REQUEST_HELP,
        r"\b(giup|ho tro|nho|can giup|lam on|help|assist)\b",
    ),
    (
        ConversationIntent.FAMILY_QUERY,
        r"\b(con|chau|gia dinh|nguoi nha|nguoi than|bo|me|family|daughter|son)\b",
    ),
    (
        ConversationIntent.DAILY_ACTIVITY,
        r"\b(an com|an sang|an trua|an toi|uong|ngu|di bo|tap the duc|xem tivi|daily|walk|exercise|ate|sleep)\b",
    ),
    (
        ConversationIntent.GREETING,
        r"\b(xin chao|chao|alo|hello|hi|good morning|good evening)\b",
    ),
)


def classify_intent(text):
    normalized = re.sub(r"\s+", " ", _normalize(text)).strip()
    if not normalized:
        return ConversationIntent.UNKNOWN
    for intent, pattern in _PATTERNS:
        if re.search(pattern, normalized):
            return intent
    return ConversationIntent.UNKNOWN
