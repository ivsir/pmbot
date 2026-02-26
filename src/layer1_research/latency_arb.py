"""Latency Arb Detector — exploits 500ms Polymarket lag behind CEX spot."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick
from src.layer0_ingestion.polymarket_client import OrderBook

logger = structlog.get_logger(__name__)


@dataclass
class LatencySignal:
    market_id: str
    cex_move_pct: float
    pm_stale_price: float
    cex_current_price: float
    estimated_lag_ms: int
    direction: str  # "BUY_YES" (price going up) or "BUY_NO" (price going down)
    confidence: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def expected_edge_pct(self) -> float:
        return abs(self.cex_move_pct) * self.confidence


class LatencyArbDetector:
    """Detects when CEX price has moved significantly but Polymarket hasn't caught up.

    Uses real timestamps from CEX server messages and local receive times
    to measure actual latency, rather than assuming a fixed 500ms lag.
    """

    def __init__(self, lookback_ticks: int = 100) -> None:
        self._settings = get_settings()
        self._cex_history: deque[CEXTick] = deque(maxlen=lookback_ticks)
        self._pm_last_receive: dict[str, int] = {}  # market_id → local_receive_ms
        self._pm_last_server: dict[str, int] = {}   # market_id → server timestamp_ms
        self._lag_estimates: deque[int] = deque(maxlen=500)
        self._cex_transmission: deque[int] = deque(maxlen=500)  # CEX network delay
        self._pm_transmission: deque[int] = deque(maxlen=500)   # PM network delay
        self._signals: list[LatencySignal] = []

    def record_cex_tick(self, tick: CEXTick) -> None:
        self._cex_history.append(tick)
        # Track CEX transmission delay when server timestamp is available
        if tick.local_receive_ms > 0 and tick.timestamp_ms > 0:
            delay = tick.local_receive_ms - tick.timestamp_ms
            if 0 <= delay < 5000:  # sanity check: ignore clock skew > 5s
                self._cex_transmission.append(delay)

    def record_pm_update(
        self, market_id: str, local_receive_ms: int, server_ts: int = 0
    ) -> None:
        self._pm_last_receive[market_id] = local_receive_ms
        if server_ts > 0:
            self._pm_last_server[market_id] = server_ts
            delay = local_receive_ms - server_ts
            if 0 <= delay < 5000:
                self._pm_transmission.append(delay)

    def detect(
        self,
        orderbook: OrderBook,
        cex_tick: CEXTick,
        strike_price: float,
    ) -> LatencySignal | None:
        """Check if CEX has moved but PM is lagging behind."""
        if len(self._cex_history) < 5:
            return None

        now_ms = int(time.time() * 1000)

        # Calculate recent CEX price momentum (last 500ms window)
        recent_ticks = [
            t
            for t in self._cex_history
            if now_ms - t.local_receive_ms < 500
            if t.local_receive_ms > 0
        ]
        # Fall back to timestamp_ms if local_receive_ms not populated
        if len(recent_ticks) < 2:
            recent_ticks = [
                t
                for t in self._cex_history
                if now_ms - t.timestamp_ms < 500
            ]
        if len(recent_ticks) < 2:
            return None

        old_price = recent_ticks[0].mid
        new_price = recent_ticks[-1].mid
        if old_price == 0:
            return None

        cex_move_pct = (new_price - old_price) / old_price * 100

        # Need a meaningful move (>0.05% in 500ms is significant for BTC)
        if abs(cex_move_pct) < 0.05:
            return None

        # Measure real PM staleness using local receive timestamps
        pm_last_recv = self._pm_last_receive.get(orderbook.market_id, 0)
        if pm_last_recv > 0:
            measured_lag = now_ms - pm_last_recv
        elif orderbook.local_receive_ms > 0:
            measured_lag = now_ms - orderbook.local_receive_ms
        else:
            # No real measurement available — use conservative estimate
            measured_lag = now_ms - orderbook.timestamp_ms if orderbook.timestamp_ms > 0 else 500

        self._lag_estimates.append(measured_lag)

        # Only signal if PM appears stale (lag > 200ms)
        if measured_lag < 200:
            return None

        # Confidence based on move size, lag, and measurement quality
        has_real_timestamps = pm_last_recv > 0 or orderbook.local_receive_ms > 0
        measurement_bonus = 1.0 if has_real_timestamps else 0.7

        lag_factor = min(measured_lag / 500, 1.0)
        move_factor = min(abs(cex_move_pct) / 0.2, 1.0)
        confidence = lag_factor * move_factor * 0.85 * measurement_bonus

        if confidence < 0.3:
            return None

        direction = "BUY_YES" if cex_move_pct > 0 else "BUY_NO"

        signal = LatencySignal(
            market_id=orderbook.market_id,
            cex_move_pct=cex_move_pct,
            pm_stale_price=orderbook.mid_price,
            cex_current_price=new_price,
            estimated_lag_ms=measured_lag,
            direction=direction,
            confidence=confidence,
        )

        self._signals.append(signal)
        if len(self._signals) > 1000:
            self._signals = self._signals[-500:]

        logger.info(
            "latency_arb_detected",
            market=orderbook.market_id,
            cex_move=round(cex_move_pct, 4),
            measured_lag_ms=measured_lag,
            has_real_ts=has_real_timestamps,
            avg_cex_delay=round(self.avg_cex_transmission_ms, 1),
            avg_pm_delay=round(self.avg_pm_transmission_ms, 1),
            confidence=round(confidence, 3),
            direction=direction,
        )

        return signal

    @property
    def avg_lag_ms(self) -> float:
        return float(np.mean(list(self._lag_estimates))) if self._lag_estimates else 0.0

    @property
    def avg_cex_transmission_ms(self) -> float:
        return float(np.mean(list(self._cex_transmission))) if self._cex_transmission else 0.0

    @property
    def avg_pm_transmission_ms(self) -> float:
        return float(np.mean(list(self._pm_transmission))) if self._pm_transmission else 0.0

    def get_recent_signals(self, n: int = 20) -> list[LatencySignal]:
        return self._signals[-n:]
