"""Order Book Sniper — optimal order placement for spread capture."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

from config.settings import get_settings
from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel

logger = structlog.get_logger(__name__)


class OrderStrategy(str, Enum):
    MARKET = "market"
    LIMIT_AGGRESSIVE = "limit_aggressive"  # inside spread
    LIMIT_PASSIVE = "limit_passive"  # at best bid/ask
    ICEBERG = "iceberg"  # split into chunks
    MAKER_SETTLEMENT = "maker_settlement"  # near-settlement maker order


@dataclass
class SniperOrder:
    """Optimized order parameters from the sniper."""

    strategy: OrderStrategy
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    urgency: float  # 0-1, higher = more aggressive
    chunks: list[tuple[float, float]] = field(default_factory=list)  # (price, size)
    expected_fill_ms: int = 0
    expected_slippage_bps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "urgency": self.urgency,
            "chunks": self.chunks,
            "expected_fill_ms": self.expected_fill_ms,
            "expected_slippage_bps": self.expected_slippage_bps,
        }


class OrderBookSniper:
    """Analyzes orderbook microstructure to determine optimal order placement.

    Strategies:
    1. Market order — when urgency is high and spread is tight
    2. Limit aggressive — place inside the spread for price improvement
    3. Limit passive — sit at best bid/ask for maker rebate
    4. Iceberg — split large orders to minimize impact
    """

    ICEBERG_THRESHOLD_USD = 5_000  # split orders above this size

    def __init__(self) -> None:
        self._executions: list[SniperOrder] = []
        self._settings = get_settings()

    def compute_optimal_order(
        self,
        orderbook: OrderBook,
        side: str,
        size_usd: float,
        urgency: float = 0.7,
    ) -> SniperOrder:
        """Determine optimal order strategy based on book state.

        Args:
            orderbook: Current CLOB orderbook
            side: "BUY" or "SELL"
            size_usd: Dollar amount to trade
            urgency: 0-1, how aggressively to fill
        """
        spread = orderbook.spread
        mid = orderbook.mid_price
        spread_bps = (spread / mid * 10_000) if mid > 0 else 0

        if side == "BUY":
            reference_levels = orderbook.asks
            best_price = orderbook.best_ask
            available_depth = orderbook.total_ask_depth
        else:
            reference_levels = orderbook.bids
            best_price = orderbook.best_bid
            available_depth = orderbook.total_bid_depth

        # Estimate slippage
        slippage_bps = self._estimate_slippage(
            reference_levels, size_usd, mid
        )

        # Choose strategy
        if urgency >= 0.9 or spread_bps < 50:
            # Very urgent or tight spread → market order
            strategy = OrderStrategy.MARKET
            price = best_price
            expected_fill_ms = 100

        elif size_usd > self.ICEBERG_THRESHOLD_USD:
            # Large order → iceberg
            strategy = OrderStrategy.ICEBERG
            price = best_price
            expected_fill_ms = 2000
            chunks = self._split_iceberg(
                reference_levels, size_usd, side, mid
            )

        elif urgency >= 0.5:
            # Medium urgency → limit aggressive (inside spread)
            strategy = OrderStrategy.LIMIT_AGGRESSIVE
            if side == "BUY":
                # Place slightly above best bid but below best ask
                price = orderbook.best_bid + spread * 0.3
            else:
                price = orderbook.best_ask - spread * 0.3
            expected_fill_ms = 500

        else:
            # Low urgency → limit passive (at best bid/ask for rebate)
            strategy = OrderStrategy.LIMIT_PASSIVE
            if side == "BUY":
                price = orderbook.best_bid
            else:
                price = orderbook.best_ask
            expected_fill_ms = 3000

        # Snap price to tick grid and clamp to valid Polymarket range
        tick = float(self._settings.poly_tick_size)
        decimals = len(self._settings.poly_tick_size.split(".")[-1]) if "." in self._settings.poly_tick_size else 0
        price = round(round(price / tick) * tick, decimals)
        price = max(tick, min(1.0 - tick, price))

        # Compute share count from USD size and enforce Polymarket minimum (5 shares)
        shares = round(size_usd / price if price > 0 else 0, 2)
        if shares < 5:
            shares = 5.0  # Polymarket minimum order size

        order = SniperOrder(
            strategy=strategy,
            price=price,
            size=shares,
            side=side,
            urgency=urgency,
            expected_fill_ms=expected_fill_ms,
            expected_slippage_bps=round(slippage_bps, 2),
        )

        if strategy == OrderStrategy.ICEBERG and chunks:
            order.chunks = chunks

        self._executions.append(order)

        logger.info(
            "sniper.order_computed",
            strategy=strategy.value,
            price=round(price, 4),
            size_usd=round(size_usd, 2),
            slippage_bps=round(slippage_bps, 2),
        )

        return order

    def compute_maker_order(
        self,
        orderbook: OrderBook,
        side: str,
        size_usd: float,
        fair_prob: float,
        secs_until_end: float,
    ) -> SniperOrder:
        """Compute a maker order for near-settlement strategy.

        Places a post-only order at a price derived from our fair probability
        estimate, capped to guarantee minimum profit per contract.

        Args:
            orderbook: Current CLOB orderbook
            side: "BUY" or "SELL"
            size_usd: Dollar amount to trade
            fair_prob: Our model's fair probability (0-1)
            secs_until_end: Seconds until market window ends
        """
        # Time-adaptive price cap: closer to settlement → pay more for fill
        if secs_until_end <= 15:
            price_cap = 0.95
        elif secs_until_end <= 30:
            price_cap = 0.93
        else:
            price_cap = 0.90

        # Base price from fair probability
        price = min(fair_prob, price_cap)
        price = max(price, 0.85)  # floor: minimum $0.05 profit per contract

        # Don't cross the book — stay below best ask (for BUY)
        tick = float(self._settings.poly_tick_size)
        if side == "BUY" and orderbook.best_ask > 0:
            price = min(price, orderbook.best_ask - tick)
        elif side == "SELL" and orderbook.best_bid > 0:
            price = max(price, orderbook.best_bid + tick)

        # Snap to tick grid and clamp
        decimals = len(self._settings.poly_tick_size.split(".")[-1]) if "." in self._settings.poly_tick_size else 0
        price = round(round(price / tick) * tick, decimals)
        price = max(tick, min(1.0 - tick, price))

        # Compute share count, enforce minimum
        shares = round(size_usd / price if price > 0 else 0, 2)
        if shares < 5:
            shares = 5.0

        order = SniperOrder(
            strategy=OrderStrategy.MAKER_SETTLEMENT,
            price=price,
            size=shares,
            side=side,
            urgency=0.0,  # maker = no urgency
            expected_fill_ms=int(secs_until_end * 1000),
            expected_slippage_bps=0.0,  # maker = no slippage
        )

        self._executions.append(order)

        logger.info(
            "sniper.maker_order_computed",
            price=round(price, 4),
            size_usd=round(size_usd, 2),
            shares=shares,
            fair_prob=round(fair_prob, 4),
            price_cap=price_cap,
            secs_until_end=round(secs_until_end, 1),
        )

        return order

    @staticmethod
    def _estimate_slippage(
        levels: list[OrderBookLevel], size_usd: float, mid: float
    ) -> float:
        """Estimate execution slippage by walking the book."""
        if not levels or mid == 0:
            return 100.0  # high penalty for empty book

        remaining = size_usd
        total_cost = 0.0
        total_qty = 0.0

        for level in levels:
            level_usd = level.price * level.size
            take = min(remaining, level_usd)
            total_cost += take
            total_qty += take / level.price
            remaining -= take
            if remaining <= 0:
                break

        if total_qty == 0:
            return 100.0

        avg_fill_price = total_cost / total_qty
        slippage_bps = abs(avg_fill_price - mid) / mid * 10_000
        return slippage_bps

    @staticmethod
    def _split_iceberg(
        levels: list[OrderBookLevel],
        total_usd: float,
        side: str,
        mid: float,
    ) -> list[tuple[float, float]]:
        """Split large order into smaller chunks across price levels."""
        n_chunks = max(3, int(total_usd / 2000))
        chunk_size = total_usd / n_chunks

        chunks: list[tuple[float, float]] = []
        for i, level in enumerate(levels[:n_chunks]):
            price = level.price
            size = chunk_size / price if price > 0 else 0
            chunks.append((round(price, 4), round(size, 4)))

        return chunks
