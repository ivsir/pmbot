"""Alpha Signal Generator — produces entry signals with Kelly sizing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from config.settings import get_settings
from src.layer1_research.research_synthesis import ResearchOutput

logger = structlog.get_logger(__name__)


@dataclass
class AlphaSignal:
    """Actionable trade signal with position sizing."""

    market_id: str
    direction: str  # "BUY_YES" or "BUY_NO"
    entry: bool  # YES/NO decision
    edge_pct: float
    win_probability: float
    kelly_fraction: float
    optimal_size_usd: float
    entry_price: float
    expected_profit_usd: float
    expected_payout: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # Source data
    research: ResearchOutput | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "direction": self.direction,
            "entry": self.entry,
            "edge_pct": self.edge_pct,
            "win_probability": self.win_probability,
            "kelly_fraction": self.kelly_fraction,
            "optimal_size_usd": self.optimal_size_usd,
            "entry_price": self.entry_price,
            "expected_profit_usd": self.expected_profit_usd,
            "expected_payout": self.expected_payout,
            "timestamp_ms": self.timestamp_ms,
        }


class AlphaSignalGenerator:
    """Converts ResearchOutput into actionable AlphaSignals with Kelly sizing.

    Kelly criterion: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = net odds (payout ratio)
    """

    def __init__(self, wallet_balance_fn=None) -> None:
        self._settings = get_settings()
        self._kelly_cap = self._settings.kelly_fraction
        self._max_position = self._settings.max_position_usd
        self._wallet_balance_fn = wallet_balance_fn
        self._signals: list[AlphaSignal] = []

    def generate(self, research: ResearchOutput) -> AlphaSignal:
        """Generate an alpha signal from research output."""
        # Use fair probability directly — Bayesian combined_probability dampens
        # displacement signal (~0.63 → 0.50), making Kelly=0.
        # MomentumSignal has .fair_up_prob; SpreadOpportunity has .fair_prob.
        raw_prob = None
        if research.spread_opp:
            raw_prob = getattr(research.spread_opp, 'fair_up_prob', None) \
                    or getattr(research.spread_opp, 'fair_prob', None)
        if raw_prob is not None:
            win_prob = raw_prob if research.direction == "BUY_YES" else 1.0 - raw_prob
        else:
            win_prob = research.combined_probability
        edge_pct = research.edge_pct

        # Entry price: use market price for Kelly sizing.
        # The execution layer (sniper) determines the actual order price.
        if research.direction == "BUY_YES" and research.spread_opp:
            entry_price = research.spread_opp.pm_yes_price
        elif research.direction == "BUY_NO" and research.spread_opp:
            entry_price = research.spread_opp.pm_no_price
        else:
            entry_price = 0.50  # default mid

        # Payout on binary market: if you buy YES at price p, payout = 1/p - 1
        if entry_price > 0 and entry_price < 1:
            payout_ratio = (1.0 / entry_price) - 1.0
        else:
            payout_ratio = 1.0

        # Kelly criterion
        q = 1.0 - win_prob
        kelly_raw = (win_prob * payout_ratio - q) / payout_ratio
        kelly_raw = max(0.0, kelly_raw)

        # Cap Kelly fraction (fractional Kelly for safety)
        kelly_fraction = min(kelly_raw, self._kelly_cap)

        # Optimal size: Kelly fraction of wallet (proportional to edge)
        bankroll = self._max_position
        if self._wallet_balance_fn:
            try:
                live_balance = self._wallet_balance_fn()
                if live_balance and live_balance > 0:
                    bankroll = live_balance
            except Exception:
                pass
        optimal_size = max(bankroll * kelly_fraction, 0.10)

        # Respect max safe order from liquidity analysis
        if research.max_safe_size_usd > 0:
            optimal_size = min(optimal_size, research.max_safe_size_usd)

        # Expected profit
        expected_profit = optimal_size * payout_ratio * win_prob - optimal_size * q

        # PM price quality filter: only enter when our token is priced 5-80¢.
        # Below 5¢ means extreme disagreement (likely bad signal).
        # Above 80¢ means the market already agrees (no edge left).
        price_quality_ok = 0.05 <= entry_price <= 0.80

        # Entry decision — Kelly is the gatekeeper for positive EV.
        # Size is capped by MAX_BID_USD in risk_filter, so no overbet guard needed.
        entry = (
            kelly_fraction > 0.005  # minimum kelly threshold
            and edge_pct >= self._settings.min_edge_pct * 100
            and optimal_size >= 0.10  # minimum $0.10 trade
            and price_quality_ok  # PM price in acceptable range
        )

        signal = AlphaSignal(
            market_id=research.market_id,
            direction=research.direction,
            entry=entry,
            edge_pct=edge_pct,
            win_probability=round(win_prob, 4),
            kelly_fraction=round(kelly_fraction, 6),
            optimal_size_usd=round(optimal_size, 2),
            entry_price=round(entry_price, 4),
            expected_profit_usd=round(expected_profit, 2),
            expected_payout=round(payout_ratio, 4),
            research=research,
        )

        self._signals.append(signal)
        if len(self._signals) > 2000:
            self._signals = self._signals[-1000:]

        # Track which signals contributed (for debugging signal quality)
        signals_active = []
        if research.spread_score > 0.4:
            signals_active.append("momentum")
        if research.latency_score > 0.4:
            signals_active.append("latency")
        if research.liquidity_score > 0.4:
            signals_active.append("liquidity")
        signal_count = len(signals_active)

        if entry:
            logger.info(
                "alpha_signal.entry",
                market=research.market_id,
                direction=research.direction,
                edge=round(edge_pct, 3),
                kelly=round(kelly_fraction, 5),
                size_usd=round(optimal_size, 2),
                expected_profit=round(expected_profit, 2),
                signals="+".join(signals_active) or "none",
                signal_count=signal_count,
                confidence=round(research.confidence, 3),
            )
        else:
            reason = "criteria_not_met"
            if not price_quality_ok:
                reason = f"pm_price_out_of_range({entry_price:.2f})"
            logger.debug(
                "alpha_signal.skip",
                market=research.market_id,
                reason=reason,
                win_prob=round(win_prob, 3),
                edge=round(edge_pct, 3),
                kelly=round(kelly_fraction, 6),
                size=round(optimal_size, 2),
                entry_price=round(entry_price, 4),
                payout=round(payout_ratio, 4),
                signals="+".join(signals_active) or "none",
            )

        return signal

    def get_recent_signals(self, n: int = 50) -> list[AlphaSignal]:
        return self._signals[-n:]

    @property
    def entry_rate(self) -> float:
        if not self._signals:
            return 0.0
        entries = sum(1 for s in self._signals if s.entry)
        return entries / len(self._signals)
