"""Liquidity Scanner — filters markets with depth >$10K for safe execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from config.settings import get_settings
from src.layer0_ingestion.polymarket_client import OrderBook

logger = structlog.get_logger(__name__)


@dataclass
class LiquidityProfile:
    market_id: str
    total_bid_depth_usd: float
    total_ask_depth_usd: float
    best_bid_size: float
    best_ask_size: float
    spread_bps: float
    depth_imbalance: float  # bid_depth / (bid_depth + ask_depth)
    levels_with_liquidity: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def total_depth_usd(self) -> float:
        return self.total_bid_depth_usd + self.total_ask_depth_usd

    @property
    def is_sufficient(self) -> bool:
        return self.total_depth_usd >= get_settings().liquidity_minimum_usd

    @property
    def order_book_imbalance(self) -> float:
        """Order Book Imbalance: (bid - ask) / (bid + ask), range [-1, +1].

        Positive = more bids → bullish pressure.
        Negative = more asks → bearish pressure.
        """
        total = self.total_bid_depth_usd + self.total_ask_depth_usd
        if total < 1e-6:
            return 0.0
        return (self.total_bid_depth_usd - self.total_ask_depth_usd) / total

    @property
    def max_safe_order_usd(self) -> float:
        """Max order size that won't move price >1%."""
        # 30% of weaker side depth, with $25 floor for small trades
        weaker = min(self.total_bid_depth_usd, self.total_ask_depth_usd)
        return max(weaker * 0.30, 25.0)


class LiquidityScanner:
    """Scans Polymarket orderbooks for sufficient depth before trading."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._min_depth = self._settings.liquidity_minimum_usd
        self._profiles: dict[str, LiquidityProfile] = {}

    def scan(self, orderbook: OrderBook) -> LiquidityProfile:
        """Analyze orderbook depth and return liquidity profile."""
        bid_depth = sum(
            level.price * level.size for level in orderbook.bids
        )
        ask_depth = sum(
            level.price * level.size for level in orderbook.asks
        )
        total = bid_depth + ask_depth

        spread_bps = 0.0
        if orderbook.best_bid > 0 and orderbook.best_ask > 0:
            mid = (orderbook.best_bid + orderbook.best_ask) / 2
            spread_bps = (
                (orderbook.best_ask - orderbook.best_bid) / mid * 10_000
            )

        imbalance = bid_depth / total if total > 0 else 0.5

        profile = LiquidityProfile(
            market_id=orderbook.market_id,
            total_bid_depth_usd=bid_depth,
            total_ask_depth_usd=ask_depth,
            best_bid_size=orderbook.bids[0].size if orderbook.bids else 0,
            best_ask_size=orderbook.asks[0].size if orderbook.asks else 0,
            spread_bps=spread_bps,
            depth_imbalance=imbalance,
            levels_with_liquidity=len(orderbook.bids) + len(orderbook.asks),
        )

        self._profiles[orderbook.market_id] = profile

        if not profile.is_sufficient:
            logger.debug(
                "liquidity_insufficient",
                market=orderbook.market_id,
                depth_usd=round(total, 2),
                min_required=self._min_depth,
            )

        return profile

    def is_tradeable(self, market_id: str) -> bool:
        profile = self._profiles.get(market_id)
        if not profile:
            return False
        return profile.is_sufficient

    def get_profile(self, market_id: str) -> LiquidityProfile | None:
        return self._profiles.get(market_id)

    def get_all_profiles(self) -> dict[str, LiquidityProfile]:
        return dict(self._profiles)

    def get_max_order_size(self, market_id: str) -> float:
        profile = self._profiles.get(market_id)
        if not profile:
            return 0.0
        return profile.max_safe_order_usd
