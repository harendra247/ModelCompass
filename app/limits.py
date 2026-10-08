"""In-memory abuse and spend controls. Per process on purpose: see README trade-offs."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from .config import Settings


class LimitError(Exception):
    def __init__(self, code: str, message: str, status: int, retry_after: int | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after


class Limiter:
    def __init__(self, settings: Settings, clock=time.time):
        self.s = settings
        self.clock = clock
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self.active: dict[str, list[float]] = defaultdict(list)  # expiry timestamps (self-healing slots)
        self._day = self._today()
        self.spend_usd = 0.0

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self):
        if self._today() != self._day:
            self._day, self.spend_usd = self._today(), 0.0

    def check_spend(self):
        self._roll_day()
        if self.spend_usd >= self.s.daily_spend_ceiling_usd:
            raise LimitError("daily_budget_reached",
                             "The demo's daily budget is used up. Try again tomorrow or run it with your own key.", 503)

    def check_rate(self, ip: str, bucket: str = "compare"):
        now = self.clock()
        q = self.hits[(ip, bucket)]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.s.compares_per_hour:
            retry = int(3600 - (now - q[0])) + 1
            raise LimitError("rate_limited",
                             f"Limit of {self.s.compares_per_hour} per hour reached. Try again in {retry // 60 + 1} min.", 429, retry)
        q.append(now)

    def acquire(self, ip: str, hold_s: float) -> None:
        now = self.clock()
        slots = [t for t in self.active[ip] if t > now]
        if len(slots) >= self.s.concurrent_per_ip:
            self.active[ip] = slots
            raise LimitError("too_many_concurrent", "Too many comparisons running at once. Wait for one to finish.", 429, 5)
        slots.append(now + hold_s)
        self.active[ip] = slots

    def release(self, ip: str) -> None:
        if self.active[ip]:
            self.active[ip].pop(0)

    def add_spend(self, usd: float | None):
        if usd:
            self._roll_day()
            self.spend_usd += usd
