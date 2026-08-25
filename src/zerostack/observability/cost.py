"""Token accounting, spend estimation and a budget guard.

A locally hosted model has no invoice, which makes it easy to build a system
with no idea what it costs and then discover the number only after switching to
a hosted model. Accounting for tokens from the start means the switch reveals a
figure that was already being tracked rather than a surprise.

Prices are expressed per million tokens and are configuration, not constants.
They change often, and a hardcoded price becomes wrong silently, so the table is
a starting point that a deployment is expected to set for its own contract.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Per million tokens, as (prompt, completion). A locally hosted model has no per
# token price, so it is zero and the accounting still records the volume.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "local": (0.0, 0.0),
    "offline": (0.0, 0.0),
}

_WORD = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Approximate a token count without a tokenizer.

    Real tokenizers are model specific and pulling one in for accounting alone
    would add a dependency to the offline path. Counting word-like runs and
    punctuation lands within roughly ten percent for English prose, which is
    accurate enough for budgeting and trend lines, and it is labelled an estimate
    everywhere it surfaces so nobody mistakes it for billing truth.
    """
    if not text:
        return 0
    return max(1, len(_WORD.findall(text)))


@dataclass
class Usage:
    """Token and spend accounting for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    estimated: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "estimated": self.estimated,
        }


class BudgetExceeded(RuntimeError):
    """Raised when a call would take a window past its configured ceiling."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class CostTracker:
    """Accumulates usage and refuses calls past a daily ceiling.

    The ceiling is checked before a call rather than after, because a limit that
    only reports the overspend afterwards is a report, not a limit.

    The budget is daily in fact and not only in name: the counters roll over at
    the UTC day boundary. Without that a deployment that sets a budget spends it
    once and then refuses every request forever, because nothing would ever
    lower the running total again. A ceiling that never reopens is an outage
    with a schedule, which is worse than having no ceiling at all.
    """

    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    daily_token_budget: int = 0
    daily_cost_budget_usd: float = 0.0

    tokens_used: int = 0
    cost_used_usd: float = 0.0
    calls: int = 0

    # Injectable so a test can cross a day boundary without waiting for one.
    clock: Callable[[], datetime] = _utc_now
    window_day: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.window_day:
            self.window_day = self.clock().date().isoformat()

    def _roll_if_new_day(self) -> None:
        """Start a fresh window when the calendar day has turned.

        The caller must hold the lock: this is a check followed by a write, and
        two request threads arriving at midnight would otherwise both observe
        the old day and one would reset the other's freshly recorded usage.
        """
        today = self.clock().date().isoformat()
        if today != self.window_day:
            self.window_day = today
            self.tokens_used = 0
            self.cost_used_usd = 0.0
            self.calls = 0

    def price_for(self, model: str) -> tuple[float, float]:
        """Look up a price, falling back to the longest matching prefix.

        Model names carry version suffixes that change more often than pricing,
        so an exact table would go stale on every point release.
        """
        if model in self.prices:
            return self.prices[model]
        matches = [key for key in self.prices if model.startswith(key)]
        if matches:
            return self.prices[max(matches, key=len)]
        return (0.0, 0.0)

    def estimate(self, prompt: str, completion: str, model: str) -> Usage:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(completion)
        return self.price(prompt_tokens, completion_tokens, model, estimated=True)

    def price(
        self, prompt_tokens: int, completion_tokens: int, model: str, estimated: bool = False
    ) -> Usage:
        prompt_price, completion_price = self.price_for(model)
        cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            model=model,
            estimated=estimated,
        )

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._roll_if_new_day()
            self.tokens_used += usage.total_tokens
            self.cost_used_usd += usage.cost_usd
            self.calls += 1

    def check(self, projected_tokens: int = 0, projected_cost_usd: float = 0.0) -> None:
        """Raise if this call would breach a configured ceiling."""
        with self._lock:
            self._roll_if_new_day()
        if self.daily_token_budget and (
            self.tokens_used + projected_tokens > self.daily_token_budget
        ):
            raise BudgetExceeded(
                f"token budget exhausted: {self.tokens_used} of {self.daily_token_budget} used"
            )
        if self.daily_cost_budget_usd and (
            self.cost_used_usd + projected_cost_usd > self.daily_cost_budget_usd
        ):
            raise BudgetExceeded(
                f"cost budget exhausted: {self.cost_used_usd:.4f} of "
                f"{self.daily_cost_budget_usd:.4f} USD used"
            )

    def reset(self) -> None:
        with self._lock:
            self.tokens_used = 0
            self.cost_used_usd = 0.0
            self.calls = 0
            self.window_day = self.clock().date().isoformat()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._roll_if_new_day()
        remaining_tokens = (
            max(0, self.daily_token_budget - self.tokens_used) if self.daily_token_budget else None
        )
        remaining_cost = (
            round(max(0.0, self.daily_cost_budget_usd - self.cost_used_usd), 6)
            if self.daily_cost_budget_usd
            else None
        )
        return {
            "calls": self.calls,
            "tokens_used": self.tokens_used,
            "cost_used_usd": round(self.cost_used_usd, 6),
            "daily_token_budget": self.daily_token_budget or None,
            "daily_cost_budget_usd": self.daily_cost_budget_usd or None,
            "tokens_remaining": remaining_tokens,
            "cost_remaining_usd": remaining_cost,
            "window_day": self.window_day,
        }
