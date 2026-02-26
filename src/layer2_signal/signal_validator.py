"""Signal Validator — final gate: Edge >2%, Confidence >60% → TRADE or SKIP."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from config.settings import get_settings
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer1_research.research_synthesis import ResearchOutput
from src.layer2_signal.alpha_signal import AlphaSignal, AlphaSignalGenerator
from src.layer2_signal.backtester import Backtester
from src.layer2_signal.risk_filter import RiskFilter, RiskAssessment

logger = structlog.get_logger(__name__)


@dataclass
class ValidatedSignal:
    """Final validated signal — ready for execution."""

    action: str  # "TRADE" or "SKIP"
    signal: AlphaSignal
    risk: RiskAssessment
    backtest_valid: bool
    final_size_usd: float
    reason: str = ""
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def is_trade(self) -> bool:
        return self.action == "TRADE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "market_id": self.signal.market_id,
            "direction": self.signal.direction,
            "final_size_usd": self.final_size_usd,
            "edge_pct": self.signal.edge_pct,
            "win_probability": self.signal.win_probability,
            "entry_price": self.signal.entry_price,
            "backtest_valid": self.backtest_valid,
            "risk_passed": self.risk.passed,
            "reason": self.reason,
            "timestamp_ms": self.timestamp_ms,
        }


class SignalValidator:
    """Final validation gate for trade signals.

    Requirements to TRADE:
    - Edge > 2%
    - Confidence > 60%
    - Alpha signal says ENTRY
    - Risk filter passes
    - Backtester validates (>70% win rate over 30 days)
    """

    def __init__(
        self,
        event_bus: EventBus,
        alpha_gen: AlphaSignalGenerator | None = None,
        backtester: Backtester | None = None,
        risk_filter: RiskFilter | None = None,
    ) -> None:
        self._settings = get_settings()
        self._event_bus = event_bus
        self._alpha_gen = alpha_gen or AlphaSignalGenerator()
        self._backtester = backtester or Backtester()
        self._risk_filter = risk_filter or RiskFilter()
        self._history: list[ValidatedSignal] = []

    @property
    def alpha_generator(self) -> AlphaSignalGenerator:
        return self._alpha_gen

    @property
    def backtester(self) -> Backtester:
        return self._backtester

    @property
    def risk_filter(self) -> RiskFilter:
        return self._risk_filter

    async def validate(self, research: ResearchOutput) -> ValidatedSignal:
        """Full validation pipeline: research → alpha → risk → backtest → TRADE/SKIP."""

        # Step 1: Generate alpha signal with Kelly sizing
        alpha = self._alpha_gen.generate(research)

        # Step 2: Check alpha entry decision
        if not alpha.entry:
            return self._skip(alpha, "Alpha signal: no entry (edge/confidence too low)")

        # Step 3: Edge threshold check
        if alpha.edge_pct < self._settings.min_edge_pct * 100:
            return self._skip(alpha, f"Edge {alpha.edge_pct:.2f}% below {self._settings.min_edge_pct * 100}% threshold")

        # Step 4: Confidence threshold check
        if alpha.win_probability < self._settings.min_confidence:
            return self._skip(
                alpha,
                f"Confidence {alpha.win_probability:.2%} below {self._settings.min_confidence:.0%} threshold",
            )

        # Step 5: Risk filter
        risk = self._risk_filter.assess(alpha)
        if not risk.passed:
            return self._skip(
                alpha,
                f"Risk filter rejected: {', '.join(risk.rejection_reasons)}",
                risk=risk,
            )

        # Step 6: Backtest validation
        bt_valid = self._backtester.is_strategy_valid
        if not bt_valid:
            result = self._backtester.latest_result
            wr = result.win_rate if result else 0
            return self._skip(
                alpha,
                f"Backtest invalid: win rate {wr:.1%} < 70% target",
                risk=risk,
                bt_valid=False,
            )

        # ✓ All checks passed → TRADE
        final_size = risk.adjusted_size_usd
        validated = ValidatedSignal(
            action="TRADE",
            signal=alpha,
            risk=risk,
            backtest_valid=bt_valid,
            final_size_usd=final_size,
            reason="All validation checks passed",
        )

        self._history.append(validated)

        # Publish trade signal
        await self._event_bus.publish(
            Event(
                event_type=EventType.TRADE_SIGNAL,
                data=validated.to_dict(),
                source="signal_validator",
            )
        )

        logger.info(
            "signal_validator.TRADE",
            market=alpha.market_id,
            direction=alpha.direction,
            size=final_size,
            edge=alpha.edge_pct,
            confidence=alpha.win_probability,
        )

        return validated

    def _skip(
        self,
        alpha: AlphaSignal,
        reason: str,
        risk: RiskAssessment | None = None,
        bt_valid: bool = True,
    ) -> ValidatedSignal:
        if risk is None:
            risk = RiskAssessment(
                signal=alpha, passed=False, adjusted_size_usd=0
            )
        validated = ValidatedSignal(
            action="SKIP",
            signal=alpha,
            risk=risk,
            backtest_valid=bt_valid,
            final_size_usd=0,
            reason=reason,
        )
        self._history.append(validated)
        if len(self._history) > 5000:
            self._history = self._history[-2500:]

        logger.debug(
            "signal_validator.SKIP",
            market=alpha.market_id,
            reason=reason,
        )
        return validated

    def get_trade_rate(self) -> float:
        if not self._history:
            return 0.0
        trades = sum(1 for v in self._history if v.is_trade)
        return trades / len(self._history)

    def get_recent(self, n: int = 50) -> list[ValidatedSignal]:
        return self._history[-n:]
