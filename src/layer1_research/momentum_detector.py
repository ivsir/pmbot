"""Displacement Detector — estimates P(BTC Up) for 5-min Up/Down markets.

Uses BTC price displacement from window open with three confirmation filters:
  1. Velocity confirmation — price must be moving in displacement direction
  2. Volatility-normalized threshold — adaptive to market regime
  3. PM OBI gating — Polymarket order book must not disagree

When an ML model is available (models/displacement_model.joblib), uses it
instead of the sigmoid for P(Up) estimation. Falls back to sigmoid otherwise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook
from src.layer1_research.displacement_predictor import DisplacementPredictor
from src.layer1_research.feature_engine import FeatureEngine, RollingCandleBuffer

logger = structlog.get_logger(__name__)


@dataclass
class MomentumSignal:
    """Output of the displacement detector — duck-type compatible with SpreadOpportunity.

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
    fair_up_prob: float  # Displacement-derived P(Up)
    spread_pct: float  # |fair - implied| * 100
    direction: str  # "BUY_YES" (bet Up) or "BUY_NO" (bet Down)

    # Displacement components
    displacement_pct: float = 0.0
    velocity_pct: float = 0.0  # 15s price velocity
    z_displacement: float = 0.0  # volatility-normalized displacement
    pm_obi: float = 0.0  # Polymarket order book imbalance [-1, +1]

    window_start_ms: int = 0
    window_end_ms: int = 0
    seconds_until_start: float = 0.0
    seconds_until_end: float = 0.0
    ml_features: np.ndarray | None = None  # 24-feature vector for trade logging

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
    """Estimates P(BTC Up in window) using displacement from window open.

    Model: displacement_pct → sigmoid(scale=10) → P(Up)
    Filters: velocity confirmation, volatility normalization, PM OBI gating

    Calibration (scale=10):
        displacement +0.02% → P(Up) ~55%
        displacement +0.05% → P(Up) ~62%
        displacement +0.10% → P(Up) ~73%
        displacement  0.00% → P(Up) = 50% (no edge)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._scale = self._settings.displacement_sigmoid_scale
        self._min_displacement = self._settings.min_displacement_pct
        self._velocity_lookback_s = self._settings.velocity_lookback_s
        self._require_velocity_confirm = self._settings.require_velocity_confirm
        self._require_vol_normalization = self._settings.require_vol_normalization
        self._min_z_displacement = self._settings.min_z_displacement
        self._require_obi_confirm = self._settings.require_obi_confirm
        self._obi_contra_limit = self._settings.obi_contra_limit
        self._opportunities: list[MomentumSignal] = []
        self._predictor = DisplacementPredictor()
        self._candle_buffer = RollingCandleBuffer()

    def update_candle_buffer(self, tick: CEXTick) -> None:
        """Ingest a CEX tick to maintain rolling 1-min candle buffer for ML features."""
        self._candle_buffer.update(tick)

    def detect(
        self,
        orderbook: OrderBook,
        cex_tick: CEXTick,
        price_history: list[CEXTick],
        window_start_ms: int = 0,
        window_end_ms: int = 0,
    ) -> MomentumSignal | None:
        """Detect displacement edge between CEX price movement and PM Up/Down price.

        Args:
            orderbook: PM orderbook for the Up token
            cex_tick: Latest CEX tick
            price_history: Recent CEX ticks for window open price lookup
            window_start_ms: When the 5-min window starts
            window_end_ms: When the 5-min window ends
        """
        now_ms = int(time.time() * 1000)

        pm_up_price = orderbook.best_ask
        pm_down_price = 1.0 - orderbook.best_bid
        implied_up_prob = orderbook.mid_price

        # Find window open price from CEX history
        window_open_price = self._get_window_open_price(price_history, window_start_ms)
        if window_open_price is None or window_open_price <= 0:
            return None

        # Displacement: how far has BTC moved from window open?
        current_price = cex_tick.mid
        displacement_pct = (current_price - window_open_price) / window_open_price * 100

        # Filter: skip noise-level displacement
        if abs(displacement_pct) < self._min_displacement:
            return None

        # ── Filter 1: Velocity confirmation ──
        velocity_pct = self._compute_velocity(price_history, cex_tick, self._velocity_lookback_s)
        if self._require_velocity_confirm:
            # Displacement and velocity must agree in sign
            if displacement_pct > 0 and velocity_pct < 0:
                logger.debug("displacement_skip.velocity_contra", disp=round(displacement_pct, 4), vel=round(velocity_pct, 4))
                return None
            if displacement_pct < 0 and velocity_pct > 0:
                logger.debug("displacement_skip.velocity_contra", disp=round(displacement_pct, 4), vel=round(velocity_pct, 4))
                return None

        # ── Filter 2: Volatility-normalized threshold ──
        rolling_stdev = self._compute_rolling_stdev(price_history)
        z_displacement = displacement_pct / rolling_stdev if rolling_stdev > 1e-8 else 0.0
        if self._require_vol_normalization:
            if abs(z_displacement) < self._min_z_displacement:
                logger.debug("displacement_skip.low_z", z=round(z_displacement, 3), stdev=round(rolling_stdev, 5))
                return None

        # ── Filter 3: PM OBI gating ──
        pm_obi = self._compute_pm_obi(orderbook)
        if self._require_obi_confirm:
            # Don't trade when PM order book is heavily against our direction
            if displacement_pct > 0 and pm_obi < -self._obi_contra_limit:
                logger.debug("displacement_skip.obi_contra", disp=round(displacement_pct, 4), obi=round(pm_obi, 3))
                return None
            if displacement_pct < 0 and pm_obi > self._obi_contra_limit:
                logger.debug("displacement_skip.obi_contra", disp=round(displacement_pct, 4), obi=round(pm_obi, 3))
                return None

        # P(Up) — ML model if available, sigmoid fallback
        features = FeatureEngine.compute_from_ticks(
            price_history=price_history,
            cex_tick=cex_tick,
            window_start_ms=window_start_ms,
            window_open_price=window_open_price,
            candle_buffer=self._candle_buffer,
        )
        fair_up_prob = self._predictor.predict(features, displacement_pct)
        fair_up_prob = max(0.01, min(0.99, fair_up_prob))

        # Direction MUST follow displacement — never bet against the observed movement.
        # Old logic used spread sign (model vs PM), which could produce BUY_YES on negative
        # displacement when PM overreacted, causing systematic losses.
        direction = "BUY_YES" if displacement_pct > 0 else "BUY_NO"

        # Edge: how much PM underprices our direction
        if direction == "BUY_YES":
            spread_pct = (fair_up_prob - implied_up_prob) * 100
        else:
            spread_pct = ((1.0 - fair_up_prob) - (1.0 - implied_up_prob)) * 100

        # Only trade when PM hasn't already fully priced in the displacement
        if spread_pct < self._settings.min_edge_pct * 100:
            return None

        signal = MomentumSignal(
            market_id=orderbook.market_id,
            pm_up_price=pm_up_price,
            pm_down_price=pm_down_price,
            cex_price=current_price,
            cex_source=cex_tick.exchange,
            implied_up_prob=implied_up_prob,
            fair_up_prob=fair_up_prob,
            spread_pct=abs(spread_pct),
            direction=direction,
            displacement_pct=displacement_pct,
            velocity_pct=velocity_pct,
            z_displacement=z_displacement,
            pm_obi=pm_obi,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            seconds_until_start=max(0, (window_start_ms - now_ms) / 1000) if window_start_ms else 0,
            seconds_until_end=max(0, (window_end_ms - now_ms) / 1000) if window_end_ms else 0,
            ml_features=features,
        )

        self._opportunities.append(signal)
        if len(self._opportunities) > 1000:
            self._opportunities = self._opportunities[-500:]

        logger.info(
            "displacement_detected",
            market=orderbook.market_id[:20] + "...",
            spread_pct=round(abs(spread_pct), 3),
            direction=direction,
            fair_up=round(fair_up_prob, 4),
            pm_mid=round(implied_up_prob, 4),
            displacement=round(displacement_pct, 4),
            velocity=round(velocity_pct, 4),
            z_disp=round(z_displacement, 2),
            obi=round(pm_obi, 3),
            window_open=round(window_open_price, 2),
            cex_now=round(current_price, 2),
        )

        return signal

    @staticmethod
    def _compute_velocity(
        price_history: list[CEXTick], current_tick: CEXTick, lookback_s: float = 15.0
    ) -> float:
        """Compute price velocity over the last lookback_s seconds.

        Returns percentage change: (price_now - price_Ns_ago) / price_Ns_ago * 100
        Positive = price rising, negative = price falling.
        """
        if not price_history:
            return 0.0
        target_ms = current_tick.timestamp_ms - int(lookback_s * 1000)
        # Find the tick closest to lookback_s ago
        candidates = [t for t in price_history if t.timestamp_ms <= target_ms]
        if not candidates:
            # All ticks are newer than lookback — use oldest available
            oldest = min(price_history, key=lambda t: t.timestamp_ms)
            if oldest.mid > 0:
                return (current_tick.mid - oldest.mid) / oldest.mid * 100
            return 0.0
        ref_tick = max(candidates, key=lambda t: t.timestamp_ms)  # closest to target
        if ref_tick.mid > 0:
            return (current_tick.mid - ref_tick.mid) / ref_tick.mid * 100
        return 0.0

    @staticmethod
    def _compute_rolling_stdev(price_history: list[CEXTick]) -> float:
        """Compute rolling standard deviation of 1-tick returns over price_history.

        Returns stdev in percentage terms (same units as displacement_pct).
        """
        if len(price_history) < 10:
            return 0.01  # default small stdev — makes z-score large (conservative)
        prices = [t.mid for t in price_history if t.mid > 0]
        if len(prices) < 10:
            return 0.01
        # Compute returns as pct
        returns = []
        for i in range(1, len(prices)):
            r = (prices[i] - prices[i - 1]) / prices[i - 1] * 100
            returns.append(r)
        if not returns:
            return 0.01
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return var ** 0.5

    @staticmethod
    def _compute_pm_obi(orderbook: OrderBook) -> float:
        """Compute Polymarket order book imbalance: (bid_depth - ask_depth) / total.

        Returns [-1, +1]. Positive = more bids (bullish). Negative = more asks (bearish).
        """
        bid_depth = sum(level.price * level.size for level in orderbook.bids)
        ask_depth = sum(level.price * level.size for level in orderbook.asks)
        total = bid_depth + ask_depth
        if total < 1e-6:
            return 0.0
        return (bid_depth - ask_depth) / total

    @staticmethod
    def _get_window_open_price(
        price_history: list[CEXTick], window_start_ms: int
    ) -> float | None:
        """Find the CEX price nearest to window_start_ms (±5s tolerance)."""
        if not price_history or window_start_ms <= 0:
            return None

        # Find ticks closest to window start
        candidates = [
            t for t in price_history
            if abs(t.timestamp_ms - window_start_ms) <= 5000
        ]

        if candidates:
            # Use the tick closest to window start
            best = min(candidates, key=lambda t: abs(t.timestamp_ms - window_start_ms))
            return best.mid

        # Fallback: if no tick near window start, use oldest available tick
        # (only valid if the oldest tick is after window start)
        oldest = min(price_history, key=lambda t: t.timestamp_ms)
        if oldest.timestamp_ms >= window_start_ms:
            return oldest.mid

        return None

    def get_recent_opportunities(self, n: int = 20) -> list[MomentumSignal]:
        return self._opportunities[-n:]

    @property
    def hit_rate(self) -> float:
        if not self._opportunities:
            return 0.0
        actionable = sum(1 for o in self._opportunities if o.is_actionable)
        return actionable / len(self._opportunities)
