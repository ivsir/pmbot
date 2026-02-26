"""Synthetic market simulator — generates realistic BTC price action and
Polymarket orderbooks with configurable spread/lag dynamics."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel


@dataclass
class MarketConfig:
    initial_btc_price: float = 67_500.0
    volatility_per_tick: float = 0.0003  # ~0.03% per 100ms tick
    mean_reversion_strength: float = 0.001
    trend_drift: float = 0.0  # slight upward/downward bias
    pm_lag_ms: int = 500  # Polymarket lags CEX by this much
    pm_spread_bps: int = 200  # Polymarket bid-ask spread
    cex_spread_bps: int = 5  # CEX spread (tight)
    arb_opportunity_freq: float = 0.08  # 8% of ticks have arb opportunity
    arb_spread_pct: float = 0.025  # 2.5% spread when arb occurs
    strike_interval: float = 500.0  # $500 between strikes
    num_strikes: int = 5
    orderbook_depth_levels: int = 8
    base_liquidity_usd: float = 15_000.0


class MarketSimulator:
    """Generates synthetic market data streams for paper trading.

    Simulates:
    1. Realistic BTC price via geometric Brownian motion + mean reversion
    2. Multi-exchange CEX ticks (Binance, Bybit, OKX) with small divergence
    3. Polymarket orderbooks with configurable lag behind CEX
    4. Periodic arbitrage opportunities (spread > 2%)
    5. Variable liquidity and occasional liquidity events
    """

    def __init__(self, config: MarketConfig | None = None) -> None:
        self.cfg = config or MarketConfig()
        self._btc_price = self.cfg.initial_btc_price
        self._btc_target = self.cfg.initial_btc_price
        self._pm_price = self.cfg.initial_btc_price  # lagged
        self._tick_count = 0
        self._session_start = time.time()

        # Strike prices around current BTC price
        center = round(self._btc_price / self.cfg.strike_interval) * self.cfg.strike_interval
        self.strikes: list[float] = [
            center + (i - self.cfg.num_strikes // 2) * self.cfg.strike_interval
            for i in range(self.cfg.num_strikes)
        ]
        self.market_ids: dict[str, float] = {
            f"btc_5min_{int(s)}": s for s in self.strikes
        }

        # Price history for the sim
        self._price_history: list[float] = [self._btc_price]
        self._pm_history: list[float] = [self._btc_price]

        # Regime: occasionally inject momentum or mean-reversion phases
        self._regime = "normal"
        self._regime_ticks_left = 0

    def tick(self) -> tuple[list[CEXTick], dict[str, OrderBook]]:
        """Advance simulation by one tick. Returns CEX ticks and PM orderbooks."""
        self._tick_count += 1
        self._update_regime()

        # 1. Evolve BTC price (GBM + mean reversion)
        self._evolve_price()

        # 2. Generate CEX ticks with small exchange-level divergence
        cex_ticks = self._generate_cex_ticks()

        # 3. Update PM price with lag and occasional arb opportunities
        self._update_pm_price()

        # 4. Generate PM orderbooks for each strike
        orderbooks = self._generate_orderbooks()

        return cex_ticks, orderbooks

    @property
    def current_btc_price(self) -> float:
        return self._btc_price

    @property
    def current_pm_price(self) -> float:
        return self._pm_price

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def _evolve_price(self) -> None:
        """Geometric Brownian motion with mean reversion."""
        vol = self.cfg.volatility_per_tick

        # Regime-based adjustments
        if self._regime == "momentum":
            vol *= 2.5
            drift = self.cfg.trend_drift + random.choice([-1, 1]) * 0.001
        elif self._regime == "volatile":
            vol *= 4.0
            drift = 0
        else:
            drift = self.cfg.trend_drift

        # Mean reversion toward target
        mr = self.cfg.mean_reversion_strength * (self._btc_target - self._btc_price)

        # GBM step
        dW = np.random.normal(0, 1)
        dS = self._btc_price * (drift + mr / self._btc_price + vol * dW)
        self._btc_price += dS
        self._btc_price = max(self._btc_price, 10_000)  # floor

        # Slowly drift target
        self._btc_target += np.random.normal(0, 5)

        self._price_history.append(self._btc_price)
        if len(self._price_history) > 10_000:
            self._price_history = self._price_history[-5_000:]

    def _update_pm_price(self) -> None:
        """Polymarket price lags behind CEX with configurable delay."""
        # Smooth lag: PM moves toward CEX price with delay
        lag_factor = 1.0 - math.exp(-1.0 / (self.cfg.pm_lag_ms / 100))

        # Inject arb opportunities
        if random.random() < self.cfg.arb_opportunity_freq:
            # PM stays stale — larger gap
            lag_factor *= 0.2  # slow update
        else:
            lag_factor = min(lag_factor * 1.5, 0.95)  # normal catch-up

        self._pm_price += lag_factor * (self._btc_price - self._pm_price)
        self._pm_history.append(self._pm_price)
        if len(self._pm_history) > 10_000:
            self._pm_history = self._pm_history[-5_000:]

    def _generate_cex_ticks(self) -> list[CEXTick]:
        """Generate ticks for Binance, Bybit, OKX with small divergence."""
        now_ms = int(time.time() * 1000)
        ticks: list[CEXTick] = []
        half_spread = self._btc_price * self.cfg.cex_spread_bps / 20_000

        for exchange in [CEXFeed.BINANCE, CEXFeed.BYBIT, CEXFeed.OKX]:
            # Small exchange-level price divergence (±$2)
            divergence = np.random.normal(0, 2.0)
            mid = self._btc_price + divergence
            tick = CEXTick(
                exchange=exchange,
                symbol="BTCUSDT",
                bid=round(mid - half_spread, 2),
                ask=round(mid + half_spread, 2),
                last=round(mid + np.random.normal(0, 0.5), 2),
                timestamp_ms=now_ms,
                volume_24h=random.uniform(500, 5000),
            )
            ticks.append(tick)

        return ticks

    def _generate_orderbooks(self) -> dict[str, OrderBook]:
        """Generate Polymarket orderbooks for each strike market."""
        now_ms = int(time.time() * 1000)
        books: dict[str, OrderBook] = {}

        for market_id, strike in self.market_ids.items():
            # Fair probability: sigmoid based on distance from strike
            distance_pct = (self._pm_price - strike) / strike
            sensitivity = 50.0
            fair_prob = 1.0 / (1.0 + math.exp(-sensitivity * distance_pct))
            fair_prob = max(0.02, min(0.98, fair_prob))

            # Add noise to make it realistic
            noise = np.random.normal(0, 0.02)
            mid_prob = max(0.03, min(0.97, fair_prob + noise))

            # Spread
            half_spread = self.cfg.pm_spread_bps / 20_000
            # Occasionally widen spread
            if random.random() < 0.05:
                half_spread *= 3

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

    def _update_regime(self) -> None:
        """Occasionally switch market regimes for realistic dynamics."""
        if self._regime_ticks_left > 0:
            self._regime_ticks_left -= 1
            if self._regime_ticks_left == 0:
                self._regime = "normal"
            return

        # Random regime change
        if random.random() < 0.005:  # ~0.5% chance per tick
            self._regime = random.choice(["momentum", "volatile"])
            self._regime_ticks_left = random.randint(20, 100)
