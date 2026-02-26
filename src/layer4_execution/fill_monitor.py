"""Fill Monitor — tracks order fills and slippage."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class FillStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class FillResult:
    order_id: str
    position_id: str
    status: FillStatus
    requested_price: float
    fill_price: float
    requested_size: float
    filled_size: float
    slippage_bps: float
    fill_time_ms: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def fill_ratio(self) -> float:
        if self.requested_size == 0:
            return 0.0
        return self.filled_size / self.requested_size

    @property
    def is_complete(self) -> bool:
        return self.status in (
            FillStatus.FILLED,
            FillStatus.CANCELLED,
            FillStatus.EXPIRED,
            FillStatus.FAILED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "position_id": self.position_id,
            "status": self.status.value,
            "requested_price": self.requested_price,
            "fill_price": self.fill_price,
            "requested_size": self.requested_size,
            "filled_size": self.filled_size,
            "slippage_bps": self.slippage_bps,
            "fill_time_ms": self.fill_time_ms,
        }


class FillMonitor:
    """Tracks order execution quality — slippage, fill rates, latency.

    Monitors:
    1. Slippage vs expected (from sniper estimate)
    2. Fill time (target <500ms)
    3. Partial fills and their handling
    4. Historical execution quality metrics
    """

    MAX_FILL_WAIT_MS = 5_000  # 5 seconds before giving up

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, Any]] = {}
        self._fills: deque[FillResult] = deque(maxlen=5000)

    def track_order(
        self,
        order_id: str,
        position_id: str,
        requested_price: float,
        requested_size: float,
    ) -> None:
        """Start tracking a new order."""
        self._pending[order_id] = {
            "position_id": position_id,
            "requested_price": requested_price,
            "requested_size": requested_size,
            "submitted_at_ms": int(time.time() * 1000),
        }
        logger.debug(
            "fill_monitor.tracking",
            order_id=order_id,
            price=requested_price,
            size=requested_size,
        )

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        filled_size: float,
        status: FillStatus = FillStatus.FILLED,
    ) -> FillResult | None:
        """Record a fill event for a tracked order."""
        pending = self._pending.pop(order_id, None)
        if not pending:
            logger.warning("fill_monitor.unknown_order", order_id=order_id)
            return None

        fill_time_ms = int(time.time() * 1000) - pending["submitted_at_ms"]
        requested_price = pending["requested_price"]

        # Calculate slippage
        if requested_price > 0:
            slippage_bps = (
                abs(fill_price - requested_price) / requested_price * 10_000
            )
        else:
            slippage_bps = 0.0

        result = FillResult(
            order_id=order_id,
            position_id=pending["position_id"],
            status=status,
            requested_price=requested_price,
            fill_price=fill_price,
            requested_size=pending["requested_size"],
            filled_size=filled_size,
            slippage_bps=round(slippage_bps, 2),
            fill_time_ms=fill_time_ms,
        )

        self._fills.append(result)

        logger.info(
            "fill_monitor.filled",
            order_id=order_id,
            slippage_bps=round(slippage_bps, 2),
            fill_time_ms=fill_time_ms,
            status=status.value,
        )

        return result

    def check_expired(self) -> list[str]:
        """Check for orders that have exceeded max wait time."""
        now_ms = int(time.time() * 1000)
        expired: list[str] = []
        for order_id, info in list(self._pending.items()):
            if now_ms - info["submitted_at_ms"] > self.MAX_FILL_WAIT_MS:
                expired.append(order_id)
                self.record_fill(order_id, 0, 0, FillStatus.EXPIRED)
        return expired

    # ── Metrics ──

    @property
    def avg_slippage_bps(self) -> float:
        filled = [f for f in self._fills if f.status == FillStatus.FILLED]
        if not filled:
            return 0.0
        return float(np.mean([f.slippage_bps for f in filled]))

    @property
    def avg_fill_time_ms(self) -> float:
        filled = [f for f in self._fills if f.status == FillStatus.FILLED]
        if not filled:
            return 0.0
        return float(np.mean([f.fill_time_ms for f in filled]))

    @property
    def fill_rate(self) -> float:
        if not self._fills:
            return 0.0
        filled = sum(1 for f in self._fills if f.status == FillStatus.FILLED)
        return filled / len(self._fills)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_fills": len(self._fills),
            "pending_orders": len(self._pending),
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "avg_fill_time_ms": round(self.avg_fill_time_ms, 1),
            "fill_rate": round(self.fill_rate, 3),
        }

    def get_recent_fills(self, n: int = 20) -> list[FillResult]:
        return list(self._fills)[-n:]
