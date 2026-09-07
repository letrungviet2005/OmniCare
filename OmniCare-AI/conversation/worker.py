"""Bounded background worker for non-blocking conversation responses."""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from .conversation_ai import ConversationAI


@dataclass(frozen=True)
class ConversationResponse:
    timestamp: float
    user_text: str
    response_text: str


class ConversationWorker:
    def __init__(
        self,
        conversation=None,
        max_pending=8,
        recent_limit=64,
        settle_seconds=0.35,
    ):
        self.conversation = conversation or ConversationAI()
        self._requests = queue.Queue(maxsize=max(1, int(max_pending)))
        self._results = queue.Queue(maxsize=max(2, int(max_pending) * 2))
        self._recent_ids = deque(maxlen=max(1, int(recent_limit)))
        self._recent_id_set = set()
        self._thread = None
        self.settle_seconds = max(0.0, float(settle_seconds))

    @staticmethod
    def transcript_id(timestamp, text):
        normalized = re.sub(r"\s+", " ", str(text)).strip().casefold()
        body = f"{float(timestamp):.3f}\n{normalized}".encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="omnicare-gemini-conversation",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit(self, timestamp, text):
        clean_text = str(text or "").strip()
        if not clean_text:
            return False
        transcript_id = self.transcript_id(timestamp, clean_text)
        if transcript_id in self._recent_id_set:
            return False
        try:
            self._requests.put_nowait((float(timestamp), clean_text))
        except queue.Full:
            return False
        if len(self._recent_ids) == self._recent_ids.maxlen:
            expired = self._recent_ids.popleft()
            self._recent_id_set.discard(expired)
        self._recent_ids.append(transcript_id)
        self._recent_id_set.add(transcript_id)
        return True

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _emit(self, result):
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
                self._results.put_nowait(result)
            except (queue.Empty, queue.Full):
                pass

    def _run(self):
        while True:
            item = self._requests.get()
            if item is None:
                return
            timestamp, user_text = item
            fragments = [user_text]
            stop_after_response = False
            deadline = time.monotonic() + self.settle_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = self._requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is None:
                    stop_after_response = True
                    break
                next_timestamp, next_text = next_item
                timestamp = min(timestamp, next_timestamp)
                fragments.append(next_text)
            user_text = self._merge_fragments(fragments)
            response_text = self.conversation.respond(user_text)
            self._emit(
                ConversationResponse(timestamp, user_text, response_text)
            )
            if stop_after_response:
                return

    @staticmethod
    def _merge_fragments(fragments):
        merged = []
        for fragment in fragments:
            words = str(fragment).strip().split()
            if not words:
                continue
            overlap = 0
            max_overlap = min(len(merged), len(words))
            for size in range(max_overlap, 0, -1):
                left = [re.sub(r"\W+", "", word).casefold() for word in merged[-size:]]
                right = [re.sub(r"\W+", "", word).casefold() for word in words[:size]]
                if left == right:
                    overlap = size
                    break
            merged.extend(words[overlap:])
        return " ".join(merged)

    def stop(self):
        if self._thread is None:
            self.conversation.close()
            return
        self._requests.put(None)
        self._thread.join(timeout=20)
        self.conversation.close()
