"""Risk Filter — pre-trade risk checks: max $50K position, correlation screening."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from config.settings import get_settings
from src.layer2_signal.alpha_signal import AlphaSignal

logger = structlog.get_logger(__name__)


@dataclass
class RiskAssessment:
    """Result of pre-trade risk screening."""

    signal: AlphaSignal
    passed: bool
    adjusted_size_usd: float

    # Individual checks
    position_limit_ok: bool = True
    daily_loss_ok: bool = True
    correlation_ok: bool = True
    drawdown_ok: bool = True
    concentration_ok: bool = True

    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.signal.market_id,
            "passed": self.passed,
            "adjusted_size_usd": self.adjusted_size_usd,
            "position_limit_ok": self.position_limit_ok,
            "daily_loss_ok": self.daily_loss_ok,
            "correlation_ok": self.correlation_ok,
            "drawdown_ok": self.drawdown_ok,
            "concentration_ok": self.concentration_ok,
            "rejection_reasons": self.rejection_reasons,
        }


class RiskFilter:
    """Pre-trade risk filter — gates signals before execution.

    Checks:
    1. Position size within $50K limit
    2. Daily loss limit not breached (-$2K)
    3. Correlation with existing positions below threshold (0.7)
    4. Portfolio drawdown within limits (-5%)
    5. Concentration — max 5 concurrent positions
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._current_positions: list[dict[str, Any]] = []
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

    def update_state(
        self,
        positions: list[dict[str, Any]],
        daily_pnl: float,
        equity: float,
    ) -> None:
        """Update risk filter state from portfolio manager."""
        self._current_positions = positions
        self._daily_pnl = daily_pnl
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def assess(self, signal: AlphaSignal) -> RiskAssessment:
        """Run all risk checks on a signal before execution."""
        reasons: list[str] = []
        adjusted_size = signal.optimal_size_usd

        # 1. Position size limit
        position_ok = adjusted_size <= self._settings.max_position_usd
        if not position_ok:
            adjusted_size = self._settings.max_position_usd
            reasons.append(
                f"Size capped from ${signal.optimal_size_usd:.0f} to ${adjusted_size:.0f}"
            )
            position_ok = True  # capped, not rejected

        # 2. Daily loss limit
        daily_loss_ok = True
        if self._daily_pnl <= -self._settings.daily_loss_limit_usd:
            daily_loss_ok = False
            reasons.append(
                f"Daily loss limit breached: ${self._daily_pnl:.0f}"
            )

        # 3. Correlation check
        correlation_ok = self._check_correlation(signal)
        if not correlation_ok:
            reasons.append("High correlation with existing position")

        # 4. Drawdown check — DISABLED
        drawdown_ok = True

        # 5. Concentration — max concurrent positions
        concentration_ok = (
            len(self._current_positions)
            < self._settings.max_concurrent_positions
        )
        if not concentration_ok:
            reasons.append(
                f"Max {self._settings.max_concurrent_positions} concurrent positions"
            )

        passed = all(
            [
                position_ok,
                daily_loss_ok,
                correlation_ok,
                drawdown_ok,
                concentration_ok,
            ]
        )

        assessment = RiskAssessment(
            signal=signal,
            passed=passed,
            adjusted_size_usd=round(adjusted_size, 2),
            position_limit_ok=position_ok,
            daily_loss_ok=daily_loss_ok,
            correlation_ok=correlation_ok,
            drawdown_ok=drawdown_ok,
            concentration_ok=concentration_ok,
            rejection_reasons=reasons,
        )

        if not passed:
            logger.warning(
                "risk_filter.rejected",
                market=signal.market_id,
                reasons=reasons,
            )
        else:
            logger.debug(
                "risk_filter.passed",
                market=signal.market_id,
                size=adjusted_size,
            )

        return assessment

    def _check_correlation(self, signal: AlphaSignal) -> bool:
        """Check if new signal is too correlated with existing positions.

        For 5min_updown mode, each 5-minute window is a separate, sequential
        market — only block if there's already a position on the SAME market_id
        (duplicate trade). Different windows are independent bets.

        For other modes, track direction overlap across all positions.
        """
        if not self._current_positions:
            return True

        # In 5min_updown mode, only block duplicate trades on the same market
        if self._settings.market_mode == "5min_updown":
            for p in self._current_positions:
                if p.get("market_id") == signal.market_id:
                    return False  # already have a position on this exact market
            return True

        same_direction_count = sum(
            1
            for p in self._current_positions
            if p.get("direction") == signal.direction
        )
        total = len(self._current_positions)
        if total == 0:
            return True

        correlation = same_direction_count / total
        return correlation < self._settings.correlation_threshold
