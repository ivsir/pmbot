"""Live Market Feed — wraps real CEX WebSocket prices and generates
synthetic Polymarket orderbooks for paper trading with real data."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from src.layer0_ingestion.cex_websocket import CEXWebSocketManager, CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel

logger = structlog.get_logger(__name__)


@dataclass
class LiveFeedConfig:
    pm_lag_ticks: int = 5  # PM price lags CEX by N ticks
    pm_spread_bps: int = 200  # Polymarket bid-ask spread
    arb_opportunity_freq: float = 0.08  # 8% of ticks have wider arb
    arb_spread_mult: float = 3.0  # arb spread multiplier
    strike_interval: float = 500.0  # $500 between strikes
    num_strikes: int = 5
    orderbook_depth_levels: int = 8
    base_liquidity_usd: float = 15_000.0


class LiveMarketFeed:
    """Provides real CEX ticks + synthetic PM orderbooks for paper trading.

    CEX prices come from live Binance/Bybit/OKX WebSocket feeds.
    Polymarket orderbooks are synthetically generated based on CEX prices
    with a configurable lag to simulate the ~500ms latency arbitrage window.
    """

    def __init__(
        self,
        cex_manager: CEXWebSocketManager,
        config: Optional[LiveFeedConfig] = None,
    ) -> None:
        self.cfg = config or LiveFeedConfig()
        self._cex_manager = cex_manager

        # Price tracking
        self._btc_price: float = 0.0
        self._pm_price: float = 0.0
        self._price_buffer: list[float] = []  # for lag simulation
        self._tick_count: int = 0

        # Strikes (initialized on first tick when we know BTC price)
        self.strikes: list[float] = []
        self.market_ids: Dict[str, float] = {}
        self._initialized = False

    @property
    def current_btc_price(self) -> float:
        return self._btc_price

    @property
    def current_pm_price(self) -> float:
        return self._pm_price

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def _initialize_strikes(self, btc_price: float) -> None:
        """Set up strike prices around current BTC price."""
        center = round(btc_price / self.cfg.strike_interval) * self.cfg.strike_interval
        self.strikes = [
            center + (i - self.cfg.num_strikes // 2) * self.cfg.strike_interval
            for i in range(self.cfg.num_strikes)
        ]
        self.market_ids = {f"btc_5min_{int(s)}": s for s in self.strikes}
        self._initialized = True
        logger.info(
            "live_feed.strikes_initialized",
            center=center,
            strikes=self.strikes,
        )

    def tick(self) -> tuple:
        """Get current market state from live CEX feeds + synthetic PM orderbooks.

        Returns: (cex_ticks, orderbooks) matching MarketSimulator interface.
        """
        self._tick_count += 1

        # Get real CEX ticks
        all_ticks = self._cex_manager.get_all_ticks()
        cex_ticks: List[CEXTick] = list(all_ticks.values())

        if not cex_ticks:
            # No ticks yet — return empty
            return [], {}

        # Use best bid as the current BTC price
        best = max(cex_ticks, key=lambda t: t.bid)
        self._btc_price = best.mid

        # Initialize strikes on first real tick
        if not self._initialized:
            self._initialize_strikes(self._btc_price)

        # Buffer for PM lag simulation
        self._price_buffer.append(self._btc_price)
        if len(self._price_buffer) > 200:
            self._price_buffer = self._price_buffer[-100:]

        # PM price lags behind CEX
        lag_idx = max(0, len(self._price_buffer) - 1 - self.cfg.pm_lag_ticks)
        lagged_price = self._price_buffer[lag_idx]

        # Occasionally inject wider lag (arb opportunity)
        if random.random() < self.cfg.arb_opportunity_freq:
            lag_idx = max(0, len(self._price_buffer) - 1 - self.cfg.pm_lag_ticks * 3)
            lagged_price = self._price_buffer[lag_idx]

        self._pm_price = lagged_price

        # Generate synthetic PM orderbooks
        orderbooks = self._generate_orderbooks()

        return cex_ticks, orderbooks

    def _generate_orderbooks(self) -> Dict[str, OrderBook]:
        """Generate Polymarket orderbooks based on real CEX price."""
        now_ms = int(time.time() * 1000)
        books: Dict[str, OrderBook] = {}

        for market_id, strike in self.market_ids.items():
            # Fair probability based on PM (lagged) price vs strike
            distance_pct = (self._pm_price - strike) / strike
            sensitivity = 50.0
            fair_prob = 1.0 / (1.0 + math.exp(-sensitivity * distance_pct))
            fair_prob = max(0.02, min(0.98, fair_prob))

            # Add noise
            noise = np.random.normal(0, 0.015)
            mid_prob = max(0.03, min(0.97, fair_prob + noise))

            # Spread
            half_spread = self.cfg.pm_spread_bps / 20_000
            if random.random() < 0.05:
                half_spread *= self.cfg.arb_spread_mult

            # Build orderbook levels
            bids: list[OrderBookLevel] = []
            asks: list[OrderBookLevel] = []
            liq_multiplier = random.uniform(0.5, 1.5)
            base_liq = self.cfg.base_liquidity_usd * liq_multiplier

            for i in range(self.cfg.orderbook_depth_levels):
                bid_price = max(0.01, mid_prob - half_spread - i * 0.01)
                ask_price = min(0.99, mid_prob + half_spread + i * 0.01)
                level_liq = base_liq / (i + 1) * random.uniform(0.7, 1.3)

                bids.append(OrderBookLevel(
                    price=round(bid_price, 4),
                    size=round(level_liq / max(bid_price, 0.01), 2),
                ))
                asks.append(OrderBookLevel(
                    price=round(ask_price, 4),
                    size=round(level_liq / max(ask_price, 0.01), 2),
                ))

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            books[market_id] = OrderBook(
                market_id=market_id,
                timestamp_ms=now_ms,
                bids=bids,
                asks=asks,
            )

        return books
