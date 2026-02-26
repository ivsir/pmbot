"""Hedge Agent — both-sides entry for volatility compression trades."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.layer0_ingestion.polymarket_client import OrderBook, PolymarketClient

logger = structlog.get_logger(__name__)


@dataclass
class HedgePair:
    """Represents a YES + NO hedge pair for the same market."""

    market_id: str
    yes_order_id: str = ""
    no_order_id: str = ""
    yes_price: float = 0.0
    no_price: float = 0.0
    yes_size: float = 0.0
    no_size: float = 0.0
    total_cost: float = 0.0
    max_loss: float = 0.0
    guaranteed_profit: float = 0.0
    is_complete: bool = False
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def cost_basis(self) -> float:
        """Total cost of YES + NO should be < $1 for an arb."""
        return self.yes_price + self.no_price

    @property
    def is_arbitrage(self) -> bool:
        """True if YES + NO < $1 (guaranteed profit on resolution)."""
        return self.cost_basis < 1.0 and self.cost_basis > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "cost_basis": self.cost_basis,
            "is_arbitrage": self.is_arbitrage,
            "guaranteed_profit": self.guaranteed_profit,
            "total_cost": self.total_cost,
            "is_complete": self.is_complete,
        }


class HedgeAgent:
    """Executes both-sides entry on volatility compression opportunities.

    Edge source #3: When YES + NO prices on Polymarket sum to < $1,
    buying both guarantees a profit on market resolution (one side
    always pays $1).

    Also used for volatility hedging: buy YES on one strike, NO on another
    to create bounded-risk positions.
    """

    MIN_HEDGE_PROFIT_PCT = 0.5  # minimum 0.5% guaranteed profit

    def __init__(self, polymarket: PolymarketClient) -> None:
        self._polymarket = polymarket
        self._active_hedges: dict[str, HedgePair] = {}
        self._completed_hedges: list[HedgePair] = []

    def scan_for_hedge(self, orderbook: OrderBook) -> HedgePair | None:
        """Check if YES + NO < $1 (arbitrage opportunity).

        On Polymarket binary markets, YES + NO should sum to ~$1.
        When they sum to < $1 due to market inefficiency, buying both
        locks in a guaranteed profit.
        """
        yes_ask = orderbook.best_ask  # cost to buy YES
        no_bid = orderbook.best_bid  # cost of YES that implies NO price

        # NO price = 1 - YES_bid (what you'd pay to buy the NO token)
        no_ask = 1.0 - no_bid if no_bid > 0 else 1.0

        cost_basis = yes_ask + no_ask

        if cost_basis >= 1.0:
            return None  # no arb

        profit_pct = ((1.0 - cost_basis) / cost_basis) * 100

        if profit_pct < self.MIN_HEDGE_PROFIT_PCT:
            return None

        hedge = HedgePair(
            market_id=orderbook.market_id,
            yes_price=yes_ask,
            no_price=no_ask,
            guaranteed_profit=round(1.0 - cost_basis, 4),
        )

        logger.info(
            "hedge.opportunity",
            market=orderbook.market_id,
            yes=round(yes_ask, 4),
            no=round(no_ask, 4),
            cost_basis=round(cost_basis, 4),
            profit_pct=round(profit_pct, 2),
        )

        return hedge

    async def execute_hedge(
        self, hedge: HedgePair, size_usd: float
    ) -> HedgePair:
        """Execute both sides of the hedge simultaneously."""
        # Split size between YES and NO proportionally
        total_cost = hedge.yes_price + hedge.no_price
        yes_allocation = (hedge.yes_price / total_cost) * size_usd
        no_allocation = (hedge.no_price / total_cost) * size_usd

        yes_size = yes_allocation / hedge.yes_price if hedge.yes_price > 0 else 0
        no_size = no_allocation / hedge.no_price if hedge.no_price > 0 else 0

        # Place both orders
        try:
            yes_result = await self._polymarket.place_order(
                token_id=hedge.market_id,
                side="BUY",
                price=hedge.yes_price,
                size=yes_size,
            )
            hedge.yes_order_id = yes_result.get("orderID", "")
            hedge.yes_size = yes_size

            # For NO token, we sell YES (equivalent to buying NO on CLOB)
            no_result = await self._polymarket.place_order(
                token_id=hedge.market_id,
                side="SELL",
                price=1.0 - hedge.no_price,
                size=no_size,
            )
            hedge.no_order_id = no_result.get("orderID", "")
            hedge.no_size = no_size

            hedge.total_cost = size_usd
            hedge.max_loss = 0  # Both-sides = bounded
            hedge.is_complete = True

            self._active_hedges[hedge.market_id] = hedge

            logger.info(
                "hedge.executed",
                market=hedge.market_id,
                total_cost=round(size_usd, 2),
                guaranteed_profit=round(
                    hedge.guaranteed_profit * min(yes_size, no_size), 2
                ),
            )

        except Exception as exc:
            logger.error(
                "hedge.execution_failed",
                market=hedge.market_id,
                error=str(exc),
            )
            # Attempt to cancel the first leg if second fails
            if hedge.yes_order_id:
                try:
                    await self._polymarket.cancel_order(hedge.yes_order_id)
                except Exception:
                    pass

        return hedge

    def get_active_hedges(self) -> list[HedgePair]:
        return list(self._active_hedges.values())

    def get_completed_hedges(self) -> list[HedgePair]:
        return list(self._completed_hedges)
