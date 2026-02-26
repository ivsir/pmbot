"""Correlation Monitor — tracks co-movement between positions."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class CorrelationResult:
    market_a: str
    market_b: str
    correlation: float
    window_minutes: int
    sample_count: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def is_high(self) -> bool:
        return abs(self.correlation) >= get_settings().correlation_threshold


class CorrelationMonitor:
    """Tracks price co-movement across active positions.

    For BTC 5-min markets, multiple markets at different strikes
    on the same underlying will be naturally correlated.
    This monitor detects when correlation is too high, signaling
    concentration risk.
    """

    def __init__(self, window_minutes: int = 30) -> None:
        self._settings = get_settings()
        self._window_minutes = window_minutes
        self._price_series: dict[str, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._correlation_matrix: dict[tuple[str, str], float] = {}

    def record_price(self, market_id: str, price: float) -> None:
        """Record a price observation for a market."""
        ts = int(time.time() * 1000)
        self._price_series[market_id].append((ts, price))

    def compute_correlations(
        self, market_ids: list[str]
    ) -> list[CorrelationResult]:
        """Compute pairwise correlations between all active markets."""
        results: list[CorrelationResult] = []
        cutoff_ms = int(time.time() * 1000) - (self._window_minutes * 60_000)

        for i, m_a in enumerate(market_ids):
            for m_b in market_ids[i + 1 :]:
                corr = self._pairwise_correlation(m_a, m_b, cutoff_ms)
                if corr is not None:
                    result = CorrelationResult(
                        market_a=m_a,
                        market_b=m_b,
                        correlation=corr.correlation,
                        window_minutes=self._window_minutes,
                        sample_count=corr.sample_count,
                    )
                    self._correlation_matrix[(m_a, m_b)] = corr.correlation
                    results.append(result)

                    if result.is_high:
                        logger.warning(
                            "correlation.high",
                            market_a=m_a,
                            market_b=m_b,
                            corr=round(corr.correlation, 3),
                        )

        return results

    def get_max_correlation(self, market_id: str) -> float:
        """Get the maximum absolute correlation of a market with any other."""
        max_corr = 0.0
        for (a, b), corr in self._correlation_matrix.items():
            if a == market_id or b == market_id:
                max_corr = max(max_corr, abs(corr))
        return max_corr

    def would_increase_concentration(
        self, new_market_id: str, existing_markets: list[str]
    ) -> bool:
        """Check if adding a new market would create excessive correlation."""
        threshold = self._settings.correlation_threshold
        for m in existing_markets:
            corr = self._correlation_matrix.get(
                (min(new_market_id, m), max(new_market_id, m)), 0
            )
            if abs(corr) >= threshold:
                return True
        return False

    def _pairwise_correlation(
        self, m_a: str, m_b: str, cutoff_ms: int
    ) -> CorrelationResult | None:
        series_a = [
            (ts, p)
            for ts, p in self._price_series.get(m_a, [])
            if ts >= cutoff_ms
        ]
        series_b = [
            (ts, p)
            for ts, p in self._price_series.get(m_b, [])
            if ts >= cutoff_ms
        ]

        if len(series_a) < 10 or len(series_b) < 10:
            return None

        # Align on nearest timestamps
        prices_a = np.array([p for _, p in series_a])
        prices_b = np.array([p for _, p in series_b])

        # Truncate to same length
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[:min_len]
        prices_b = prices_b[:min_len]

        # Compute returns
        if min_len < 3:
            return None
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        # Handle zero variance
        if np.std(returns_a) == 0 or np.std(returns_b) == 0:
            corr_val = 0.0
        else:
            corr_matrix = np.corrcoef(returns_a, returns_b)
            corr_val = float(corr_matrix[0, 1])

        return CorrelationResult(
            market_a=m_a,
            market_b=m_b,
            correlation=round(corr_val, 4),
            window_minutes=self._window_minutes,
            sample_count=min_len,
        )
