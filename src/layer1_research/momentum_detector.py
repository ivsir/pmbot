"""Momentum Detector — estimates P(BTC Up) for 5-min Up/Down markets."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook

logger = structlog.get_logger(__name__)


@dataclass
class MomentumSignal:
    """Output of the momentum detector — duck-type compatible with SpreadOpportunity.

    Downstream consumers (ResearchSynthesis, AlphaSignalGenerator) access
    .market_id, .direction, .spread_pct, .pm_yes_price, .pm_no_price, .edge,
    and .is_actionable — all of which are provided here.
    """

    market_id: str
    pm_up_price: float  # Polymarket "Up" token best ask
    pm_down_price: float  # Polymarket "Down" token implied price
    cex_price: float
    cex_source: CEXFeed
    implied_up_prob: float  # PM mid price for Up token
    fair_up_prob: float  # Our momentum-derived P(Up)
    spread_pct: float  # |fair - implied| * 100
    direction: str  # "BUY_YES" (bet Up) or "BUY_NO" (bet Down)

    # Momentum components
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_5m: float = 0.0
    obi: float = 0.0  # Order Book Imbalance (-1 to +1)
    zscore: float = 0.0

    window_start_ms: int = 0
    window_end_ms: int = 0
    seconds_until_start: float = 0.0
    seconds_until_end: float = 0.0

    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def edge(self) -> float:
        return abs(self.fair_up_prob - self.implied_up_prob)

    @property
    def is_actionable(self) -> bool:
        return self.spread_pct >= get_settings().min_edge_pct * 100

    # ── SpreadOpportunity compatibility ──
    @property
    def pm_yes_price(self) -> float:
        return self.pm_up_price

    @property
    def pm_no_price(self) -> float:
        return self.pm_down_price


class MomentumDetector:
    """Estimates P(BTC Up in window) using CEX price momentum.

    Model: Weighted momentum returns → z-score → sigmoid → P(Up)

    Weights:
        1-minute return: 0.5 (strongest short-term predictor)
        3-minute return: 0.3 (trend persistence)
        5-minute return: 0.2 (longer context)

    Calibration (sensitivity=50):
        z-score +0.5 → P(Up) ~62%
        z-score +1.0 → P(Up) ~73%
        z-score  0.0 → P(Up) = 50% (no edge)
    """

    W_1M = 0.4
    W_3M = 0.25
    W_5M = 0.15
    W_OBI = 0.20

    def __init__(self) -> None:
        self._settings = get_settings()
        self._sensitivity = self._settings.momentum_sigmoid_sensitivity
        self._min_spread_pct = self._settings.min_edge_pct * 100
        self._opportunities: list[MomentumSignal] = []
        self._return_history: deque[float] = deque(maxlen=500)

    def detect(
        self,
        orderbook: OrderBook,
        cex_tick: CEXTick,
        price_history: list[CEXTick],
        window_start_ms: int = 0,
        window_end_ms: int = 0,
    ) -> MomentumSignal | None:
        """Detect momentum edge between CEX trend and PM Up/Down price.

        Args:
            orderbook: PM orderbook for the Up token
            cex_tick: Latest CEX tick
            price_history: Recent CEX ticks for momentum calculation
            window_start_ms: When the 5-min window starts
            window_end_ms: When the 5-min window ends
        """
        now_ms = int(time.time() * 1000)

        pm_up_price = orderbook.best_ask
        pm_down_price = 1.0 - orderbook.best_bid
        implied_up_prob = orderbook.mid_price

        # Calculate momentum returns
        ret_1m = self._calculate_return(price_history, 60_000)
        ret_3m = self._calculate_return(price_history, 180_000)
        ret_5m = self._calculate_return(price_history, 300_000)

        if ret_1m is None:
            return None  # insufficient history

        self._return_history.append(ret_1m)

        # Order Book Imbalance (OBI): bid-ask volume imbalance
        obi = self._compute_obi(orderbook)

        # Z-score: volatility-adjusted momentum
        vol = self._rolling_volatility()
        if vol < 1e-8:
            vol = 0.001

        raw_momentum = (
            self.W_1M * (ret_1m or 0.0)
            + self.W_3M * (ret_3m or 0.0)
            + self.W_5M * (ret_5m or 0.0)
            + self.W_OBI * obi * vol  # scale OBI to return-like magnitude
        )
        zscore = raw_momentum / vol

        # Early-entry time factor: backtest shows 62.6% win rate at T+0.
        # We trade in the first 120s when PM prices are near 50/50.
        time_factor = 0.7  # default
        if window_end_ms > 0:
            secs_until_end = (window_end_ms - now_ms) / 1000
            secs_into_window = 300 - secs_until_end
            if secs_into_window <= 30:
                time_factor = 0.8  # first 30s — freshest prices
            elif secs_into_window <= 60:
                time_factor = 0.75
            elif secs_into_window <= 120:
                time_factor = 0.65
            else:
                time_factor = 0.5  # >120s — PM has absorbed

        adjusted_zscore = zscore * time_factor

        # Sigmoid → P(Up)
        fair_up_prob = 1.0 / (1.0 + math.exp(-self._sensitivity * adjusted_zscore))
        fair_up_prob = max(0.01, min(0.99, fair_up_prob))

        spread_pct = (fair_up_prob - implied_up_prob) * 100

        if abs(spread_pct) < self._min_spread_pct:
            return None

        direction = "BUY_YES" if spread_pct > 0 else "BUY_NO"

        signal = MomentumSignal(
            market_id=orderbook.market_id,
            pm_up_price=pm_up_price,
            pm_down_price=pm_down_price,
            cex_price=cex_tick.mid,
            cex_source=cex_tick.exchange,
            implied_up_prob=implied_up_prob,
            fair_up_prob=fair_up_prob,
            spread_pct=abs(spread_pct),
            direction=direction,
            return_1m=ret_1m or 0.0,
            return_3m=ret_3m or 0.0,
            return_5m=ret_5m or 0.0,
            obi=obi,
            zscore=adjusted_zscore,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            seconds_until_start=max(0, (window_start_ms - now_ms) / 1000) if window_start_ms else 0,
            seconds_until_end=max(0, (window_end_ms - now_ms) / 1000) if window_end_ms else 0,
        )

        self._opportunities.append(signal)
        if len(self._opportunities) > 1000:
            self._opportunities = self._opportunities[-500:]

        logger.info(
            "momentum_detected",
            market=orderbook.market_id[:20] + "...",
            spread_pct=round(abs(spread_pct), 3),
            direction=direction,
            fair_up=round(fair_up_prob, 4),
            pm_mid=round(implied_up_prob, 4),
            zscore=round(adjusted_zscore, 3),
            ret_1m=round(ret_1m or 0, 6),
        )

        return signal

    @staticmethod
    def _compute_obi(orderbook: OrderBook) -> float:
        """Compute Order Book Imbalance: (bid_depth - ask_depth) / total.

        OBI explains ~65% of short-interval price variance (R² = 0.65).
        Positive OBI (more bids) suggests upward pressure → higher P(Up).
        """
        bid_depth = sum(level.price * level.size for level in orderbook.bids)
        ask_depth = sum(level.price * level.size for level in orderbook.asks)
        total = bid_depth + ask_depth
        if total < 1e-6:
            return 0.0
        return (bid_depth - ask_depth) / total

    @staticmethod
    def _calculate_return(
        history: list[CEXTick], lookback_ms: int
    ) -> float | None:
        """Calculate price return over the last `lookback_ms` milliseconds."""
        if len(history) < 2:
            return None

        now_ms = history[-1].timestamp_ms
        cutoff = now_ms - lookback_ms

        # Find oldest tick near our lookback target
        old_ticks = [t for t in history if t.timestamp_ms <= cutoff + 5000]
        if not old_ticks:
            if len(history) >= 5:
                old_price = history[0].mid
            else:
                return None
        else:
            old_price = old_ticks[-1].mid

        new_price = history[-1].mid
        if old_price == 0:
            return None

        return (new_price - old_price) / old_price

    def _rolling_volatility(self) -> float:
        """Standard deviation of recent 1-minute returns."""
        if len(self._return_history) < 5:
            return 0.001
        return float(np.std(list(self._return_history)))

    def get_recent_opportunities(self, n: int = 20) -> list[MomentumSignal]:
        return self._opportunities[-n:]

    @property
    def hit_rate(self) -> float:
        if not self._opportunities:
            return 0.0
        actionable = sum(1 for o in self._opportunities if o.is_actionable)
        return actionable / len(self._opportunities)
