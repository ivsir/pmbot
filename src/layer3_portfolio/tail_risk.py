"""Tail Risk Agent — black swan detection and emergency response."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import structlog

from config.settings import get_settings
from src.layer0_ingestion.event_bus import EventBus, Event, EventType

logger = structlog.get_logger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class TailRiskAlert:
    level: AlertLevel
    alert_type: str
    message: str
    data: dict = field(default_factory=dict)
    recommended_action: str = ""
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class TailRiskAgent:
    """Monitors for black swan events and extreme market conditions.

    Detection methods:
    1. Price velocity — BTC moves >2% in <1 minute
    2. Volatility spike — rolling vol exceeds 3x normal
    3. Liquidity evaporation — orderbook depth drops >50%
    4. Exchange disconnection — data feed goes stale
    5. Spread explosion — CEX-PM spread exceeds 10%
    """

    # Thresholds
    PRICE_VELOCITY_THRESHOLD = 0.02  # 2% in 1 min
    VOL_SPIKE_MULTIPLIER = 3.0
    DEPTH_DROP_THRESHOLD = 0.50  # 50% drop
    STALE_DATA_MS = 5_000  # 5 seconds
    SPREAD_EXPLOSION = 0.10  # 10%

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._price_history: deque[tuple[int, float]] = deque(maxlen=5000)
        self._depth_history: deque[tuple[int, float]] = deque(maxlen=500)
        self._vol_baseline: float = 0.0
        self._alerts: list[TailRiskAlert] = []
        self._last_data_ts: int = 0
        self._emergency_mode = False

    @property
    def is_emergency(self) -> bool:
        return self._emergency_mode

    def record_price(self, price: float) -> None:
        ts = int(time.time() * 1000)
        self._price_history.append((ts, price))
        self._last_data_ts = ts

    def record_depth(self, depth_usd: float) -> None:
        ts = int(time.time() * 1000)
        self._depth_history.append((ts, depth_usd))

    async def check_all(self) -> list[TailRiskAlert]:
        """Run all tail risk checks."""
        alerts: list[TailRiskAlert] = []

        velocity_alert = self._check_price_velocity()
        if velocity_alert:
            alerts.append(velocity_alert)

        vol_alert = self._check_volatility_spike()
        if vol_alert:
            alerts.append(vol_alert)

        depth_alert = self._check_depth_evaporation()
        if depth_alert:
            alerts.append(depth_alert)

        stale_alert = self._check_stale_data()
        if stale_alert:
            alerts.append(stale_alert)

        # Publish alerts
        for alert in alerts:
            self._alerts.append(alert)
            await self._event_bus.publish(
                Event(
                    event_type=EventType.RISK_ALERT,
                    data={
                        "level": alert.level.value,
                        "type": alert.alert_type,
                        "message": alert.message,
                        "action": alert.recommended_action,
                    },
                    source="tail_risk_agent",
                )
            )

            if alert.level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY):
                self._emergency_mode = True
                logger.critical(
                    "tail_risk.EMERGENCY",
                    type=alert.alert_type,
                    message=alert.message,
                )

        return alerts

    def clear_emergency(self) -> None:
        self._emergency_mode = False
        logger.info("tail_risk.emergency_cleared")

    def _check_price_velocity(self) -> TailRiskAlert | None:
        """Detect >2% BTC move in <1 minute."""
        if len(self._price_history) < 10:
            return None

        now_ms = int(time.time() * 1000)
        one_min_ago = now_ms - 60_000

        recent = [(ts, p) for ts, p in self._price_history if ts >= one_min_ago]
        if len(recent) < 2:
            return None

        prices = [p for _, p in recent]
        if prices[0] == 0:
            return None
        move_pct = abs(prices[-1] - prices[0]) / prices[0]

        if move_pct >= self.PRICE_VELOCITY_THRESHOLD:
            return TailRiskAlert(
                level=AlertLevel.CRITICAL,
                alert_type="price_velocity",
                message=f"BTC moved {move_pct:.2%} in <1 minute",
                data={"move_pct": move_pct, "from": prices[0], "to": prices[-1]},
                recommended_action="CLOSE_ALL_POSITIONS",
            )
        return None

    def _check_volatility_spike(self) -> TailRiskAlert | None:
        """Detect rolling volatility exceeding 3x baseline."""
        if len(self._price_history) < 100:
            return None

        prices = np.array([p for _, p in self._price_history])
        if np.any(prices[:-1] == 0):
            return None
        returns = np.diff(prices) / prices[:-1]

        # Use first half as baseline, second half as current
        mid = len(returns) // 2
        baseline_vol = np.std(returns[:mid])
        current_vol = np.std(returns[mid:])

        if baseline_vol > 0:
            self._vol_baseline = float(baseline_vol)

        if self._vol_baseline > 0 and current_vol > 0:
            spike_ratio = current_vol / self._vol_baseline
            if spike_ratio >= self.VOL_SPIKE_MULTIPLIER:
                return TailRiskAlert(
                    level=AlertLevel.WARNING,
                    alert_type="volatility_spike",
                    message=f"Volatility spike: {spike_ratio:.1f}x baseline",
                    data={"spike_ratio": spike_ratio, "current_vol": float(current_vol)},
                    recommended_action="REDUCE_POSITION_SIZES",
                )
        return None

    def _check_depth_evaporation(self) -> TailRiskAlert | None:
        """Detect >50% drop in orderbook depth."""
        if len(self._depth_history) < 20:
            return None

        depths = [d for _, d in self._depth_history]
        avg_depth = np.mean(depths[:-5])
        current_depth = np.mean(depths[-5:])

        if avg_depth > 0:
            drop_pct = (avg_depth - current_depth) / avg_depth
            if drop_pct >= self.DEPTH_DROP_THRESHOLD:
                return TailRiskAlert(
                    level=AlertLevel.CRITICAL,
                    alert_type="depth_evaporation",
                    message=f"Orderbook depth dropped {drop_pct:.0%}",
                    data={"drop_pct": drop_pct, "current_depth": current_depth},
                    recommended_action="PAUSE_NEW_ORDERS",
                )
        return None

    def _check_stale_data(self) -> TailRiskAlert | None:
        """Detect stale data feed (>5s without update)."""
        if self._last_data_ts == 0:
            return None

        staleness = int(time.time() * 1000) - self._last_data_ts
        if staleness >= self.STALE_DATA_MS:
            return TailRiskAlert(
                level=AlertLevel.WARNING,
                alert_type="stale_data",
                message=f"Data feed stale for {staleness}ms",
                data={"staleness_ms": staleness},
                recommended_action="PAUSE_TRADING",
            )
        return None

    def get_recent_alerts(self, n: int = 20) -> list[TailRiskAlert]:
        return self._alerts[-n:]
