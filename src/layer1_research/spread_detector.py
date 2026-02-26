"""Spread Detector — identifies CEX vs Polymarket spreads >2%."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook

logger = structlog.get_logger(__name__)


@dataclass
class SpreadOpportunity:
    market_id: str
    pm_yes_price: float
    pm_no_price: float
    cex_price: float
    cex_source: CEXFeed
    implied_pm_prob: float
    fair_prob: float
    spread_pct: float
    direction: str  # "BUY_YES" or "BUY_NO"
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def edge(self) -> float:
        return abs(self.fair_prob - self.implied_pm_prob)

    @property
    def is_actionable(self) -> bool:
        return self.spread_pct >= get_settings().min_edge_pct * 100


class SpreadDetector:
    """Compares Polymarket binary market prices against CEX spot to find >2% spreads.

    For BTC 5-min markets:
    - Market question: "Will BTC be above $X at time T?"
    - YES price = implied probability BTC > X
    - Compare against CEX spot to determine fair probability
    """

    # Maximum OTM distance (%) to allow a directional trade.
    # For monthly BTC binary markets, BTC can move significantly,
    # so wider OTM distances are acceptable.
    MAX_OTM_DISTANCE_PCT = 15.0  # 15% for monthly markets

    def __init__(self) -> None:
        self._settings = get_settings()
        self._min_spread_pct = self._settings.min_edge_pct * 100  # 2%
        self._opportunities: list[SpreadOpportunity] = []

    def detect(
        self,
        orderbook: OrderBook,
        cex_tick: CEXTick,
        strike_price: float,
    ) -> SpreadOpportunity | None:
        """Detect spread between Polymarket implied prob and CEX-derived fair prob.

        Args:
            orderbook: Polymarket CLOB orderbook for YES token
            cex_tick: Latest CEX price tick
            strike_price: The strike price in the binary market question
        """
        pm_yes = orderbook.best_ask  # cost to buy YES
        pm_no = 1.0 - orderbook.best_bid  # implied NO price
        implied_prob = orderbook.mid_price

        # Derive fair probability from CEX spot distance to strike
        # Simple model: use distance + recent volatility as proxy
        cex_mid = cex_tick.mid
        distance_pct = (cex_mid - strike_price) / strike_price

        # Sigmoid-based fair probability estimation
        # Positive distance → BTC above strike → higher YES probability
        import math

        # Lower sensitivity for monthly markets (vs 50 for 5-min markets)
        sensitivity = 8.0  # gentler curve for longer-duration binary markets
        fair_prob = 1.0 / (1.0 + math.exp(-sensitivity * distance_pct))
        fair_prob = max(0.01, min(0.99, fair_prob))

        # Calculate spread
        spread_pct = (fair_prob - implied_prob) * 100

        if abs(spread_pct) < self._min_spread_pct:
            return None

        direction = "BUY_YES" if spread_pct > 0 else "BUY_NO"

        # ── Moneyness filter: reject far out-of-the-money directional bets ──
        # BUY_YES needs BTC to finish ABOVE strike → reject if BTC is far below
        # BUY_NO  needs BTC to finish BELOW strike → reject if BTC is far above
        otm_distance_pct = abs(distance_pct) * 100  # as percentage
        is_otm = (
            (direction == "BUY_YES" and cex_mid < strike_price)
            or (direction == "BUY_NO" and cex_mid > strike_price)
        )
        if is_otm and otm_distance_pct > self.MAX_OTM_DISTANCE_PCT:
            logger.debug(
                "spread.otm_rejected",
                market=orderbook.market_id,
                direction=direction,
                otm_pct=round(otm_distance_pct, 3),
                max_otm=self.MAX_OTM_DISTANCE_PCT,
                cex_price=round(cex_mid, 2),
                strike=strike_price,
            )
            return None

        opp = SpreadOpportunity(
            market_id=orderbook.market_id,
            pm_yes_price=pm_yes,
            pm_no_price=pm_no,
            cex_price=cex_mid,
            cex_source=cex_tick.exchange,
            implied_pm_prob=implied_prob,
            fair_prob=fair_prob,
            spread_pct=abs(spread_pct),
            direction=direction,
        )

        self._opportunities.append(opp)
        # Keep rolling window
        if len(self._opportunities) > 1000:
            self._opportunities = self._opportunities[-500:]

        logger.info(
            "spread_detected",
            market=orderbook.market_id,
            spread_pct=round(abs(spread_pct), 3),
            direction=direction,
            pm_mid=round(implied_prob, 4),
            fair=round(fair_prob, 4),
            cex_price=round(cex_mid, 2),
        )

        return opp

    def get_recent_opportunities(self, n: int = 20) -> list[SpreadOpportunity]:
        return self._opportunities[-n:]

    @property
    def hit_rate(self) -> float:
        """Fraction of detected opportunities that were actionable."""
        if not self._opportunities:
            return 0.0
        actionable = sum(1 for o in self._opportunities if o.is_actionable)
        return actionable / len(self._opportunities)
