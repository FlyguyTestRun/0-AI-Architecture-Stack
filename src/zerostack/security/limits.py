"""Rate limiting.

An unauthenticated or thinly authenticated endpoint that runs a model on every
call is a denial of wallet as much as a denial of service, because each request
costs real compute. A limiter is the cheapest control that bounds both.

This is a token bucket rather than a fixed window. A fixed window lets a caller
send its whole allowance in the last second of one window and again in the first
second of the next, delivering twice the intended rate at the boundary. A bucket
refills continuously, so the average rate holds while still allowing a short
burst, which is what real clients actually do.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class RateLimitExceeded(RuntimeError):
    """The caller has spent its allowance."""

    def __init__(self, message: str, retry_after_seconds: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(0.0, retry_after_seconds)


@dataclass
class Bucket:
    """One caller's allowance."""

    capacity: float
    refill_per_second: float
    tokens: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now

    def take(self, amount: float = 1.0) -> tuple[bool, float]:
        """Try to spend. Returns whether it succeeded and the wait if it did not."""
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        if self.refill_per_second <= 0:
            return False, float("inf")
        shortfall = amount - self.tokens
        return False, shortfall / self.refill_per_second


class RateLimiter:
    """Per key token buckets.

    Buckets are created on first use and evicted once idle, so a deployment that
    sees many short lived keys does not accumulate one bucket per key forever.
    """

    def __init__(
        self,
        requests_per_minute: int = 0,
        burst: int = 0,
        idle_eviction_seconds: float = 900.0,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        # A burst equal to the per minute rate lets a client issue its whole
        # minute at once, which is usually what a page load or a batch does.
        self.burst = burst or requests_per_minute
        self.idle_eviction_seconds = idle_eviction_seconds
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, key: str, requests_per_minute: int = 0) -> None:
        """Spend one unit for ``key``, or raise.

        ``requests_per_minute`` overrides the deployment default for a single
        caller, so one principal can be given a higher ceiling without changing
        anyone else's.
        """
        rate = requests_per_minute or self.requests_per_minute
        if rate <= 0:
            return

        capacity = float(requests_per_minute or self.burst)
        with self._lock:
            self._evict_idle()
            bucket = self._buckets.get(key)
            if bucket is None:
                # A new caller starts full, so its first request is never
                # rejected for having no history.
                bucket = Bucket(
                    capacity=capacity,
                    refill_per_second=rate / 60.0,
                    tokens=capacity,
                )
                self._buckets[key] = bucket
            allowed, wait = bucket.take()

        if not allowed:
            raise RateLimitExceeded(
                f"rate limit of {int(rate)} requests per minute exceeded",
                retry_after_seconds=wait,
            )

    def _evict_idle(self) -> None:
        if not self.idle_eviction_seconds:
            return
        cutoff = time.monotonic() - self.idle_eviction_seconds
        for key in [k for k, b in self._buckets.items() if b.updated_at < cutoff]:
            del self._buckets[key]

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "requests_per_minute": self.requests_per_minute,
                "burst": self.burst,
                "tracked_callers": len(self._buckets),
            }
