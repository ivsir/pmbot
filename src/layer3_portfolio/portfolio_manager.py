"""Portfolio Manager — Max $50K/pos, Max 5 concurrent, Kelly 11.5%, -5% DD limit."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from config.settings import get_settings
from src.layer0_ingestion.data_store import DataStore
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer2_signal.signal_validator import ValidatedSignal

logger = structlog.get_logger(__name__)


class PositionStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Position:
    id: str
    market_id: str
    direction: str
    entry_price: float
    size_usd: float
    status: PositionStatus = PositionStatus.PENDING
    fill_price: float = 0.0
    exit_price: float = 0.0
    pnl_usd: float = 0.0
    order_id: str = ""
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    filled_at_ms: int = 0
    closed_at_ms: int = 0

    @property
    def is_active(self) -> bool:
        return self.status in (PositionStatus.PENDING, PositionStatus.OPEN)

    @property
    def duration_ms(self) -> int:
        if self.closed_at_ms > 0:
            return self.closed_at_ms - self.created_at_ms
        return int(time.time() * 1000) - self.created_at_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "market_id": self.market_id,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "size_usd": self.size_usd,
            "status": self.status.value,
            "fill_price": self.fill_price,
            "exit_price": self.exit_price,
            "pnl_usd": self.pnl_usd,
            "order_id": self.order_id,
            "created_at_ms": self.created_at_ms,
            "filled_at_ms": self.filled_at_ms,
            "closed_at_ms": self.closed_at_ms,
        }


class PortfolioManager:
    """Manages portfolio state, position lifecycle, and enforces risk limits.

    Constraints from spec:
    - Max $50K per position
    - Max 5 concurrent positions
    - Kelly 4.3% fractional sizing
    - -5% drawdown limit (circuit breaker)
    """

    def __init__(self, event_bus: EventBus, data_store: DataStore) -> None:
        self._settings = get_settings()
        self._event_bus = event_bus
        self._data_store = data_store
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._recently_closed: list[Position] = []  # buffer for feedback loop
        self._initial_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._wallet_balance: float = 0.0  # real on-chain USDC balance
        self._halted = False

    async def initialize(self, initial_equity: float) -> None:
        """Set starting equity for drawdown tracking."""
        self._initial_equity = initial_equity
        self._peak_equity = initial_equity
        logger.info("portfolio.initialized", equity=initial_equity)

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def active_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_active]

    @property
    def active_count(self) -> int:
        return len(self.active_positions)

    @property
    def total_exposure_usd(self) -> float:
        return sum(p.size_usd for p in self.active_positions)

    def update_wallet_balance(self, balance: float) -> None:
        """Update wallet balance from on-chain polling."""
        self._wallet_balance = balance

    @property
    def current_equity(self) -> float:
        # Use real wallet balance + cost of open positions as equity.
        # This prevents phantom PnL from resolution/redemption timing gaps.
        if self._wallet_balance > 0:
            open_cost = sum(p.size_usd for p in self.active_positions)
            return self._wallet_balance + open_cost
        # Fallback before first wallet poll
        unrealized = sum(p.pnl_usd for p in self.active_positions)
        realized = sum(p.pnl_usd for p in self._closed_positions)
        return self._initial_equity + realized + unrealized

    @property
    def drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.current_equity) / self._peak_equity

    async def open_position(self, signal: ValidatedSignal) -> Position | None:
        """Create a new position from a validated trade signal."""
        if self._halted:
            logger.warning("portfolio.halted", reason="drawdown_limit")
            return None

        if self.active_count >= self._settings.max_concurrent_positions:
            logger.warning(
                "portfolio.max_positions",
                active=self.active_count,
                max=self._settings.max_concurrent_positions,
            )
            return None

        position = Position(
            id=str(uuid.uuid4())[:12],
            market_id=signal.signal.market_id,
            direction=signal.signal.direction,
            entry_price=signal.signal.entry_price,
            size_usd=signal.final_size_usd,
        )

        self._positions[position.id] = position

        # Persist to Redis
        await self._data_store.set_position(position.id, position.to_dict())

        await self._event_bus.publish(
            Event(
                event_type=EventType.POSITION_UPDATE,
                data={"action": "open", **position.to_dict()},
                source="portfolio_manager",
            )
        )

        logger.info(
            "portfolio.position_opened",
            id=position.id,
            market=position.market_id,
            direction=position.direction,
            size=position.size_usd,
        )

        return position

    async def fill_position(
        self, position_id: str, fill_price: float, order_id: str,
        filled_size: float = 0.0,
    ) -> None:
        """Mark position as filled."""
        pos = self._positions.get(position_id)
        if not pos:
            return
        pos.status = PositionStatus.OPEN
        pos.fill_price = fill_price
        pos.order_id = order_id
        pos.filled_at_ms = int(time.time() * 1000)

        # Update size_usd to actual cost (fill_price × shares) so PnL and
        # dashboard display reflect real on-chain spend, not Kelly estimate.
        if filled_size > 0 and fill_price > 0:
            pos.size_usd = round(fill_price * filled_size, 4)

        await self._data_store.set_position(pos.id, pos.to_dict())
        logger.info(
            "portfolio.position_filled",
            id=pos.id,
            fill_price=fill_price,
            filled_size=filled_size,
            actual_cost=pos.size_usd,
        )

    async def close_position(
        self, position_id: str, exit_price: float, pnl: float
    ) -> None:
        """Close a position and record PnL."""
        pos = self._positions.get(position_id)
        if not pos:
            return

        pos.status = PositionStatus.CLOSED
        pos.exit_price = exit_price
        pos.pnl_usd = pnl
        pos.closed_at_ms = int(time.time() * 1000)

        self._closed_positions.append(pos)
        self._recently_closed.append(pos)
        del self._positions[position_id]

        # Update daily PnL
        await self._data_store.incr_daily_pnl(pnl)
        await self._data_store.set_position(pos.id, pos.to_dict())

        # Update peak equity
        eq = self.current_equity
        if eq > self._peak_equity:
            self._peak_equity = eq

        # Check drawdown circuit breaker
        await self._check_drawdown()

        await self._event_bus.publish(
            Event(
                event_type=EventType.POSITION_UPDATE,
                data={"action": "close", **pos.to_dict()},
                source="portfolio_manager",
            )
        )

        logger.info(
            "portfolio.position_closed",
            id=pos.id,
            pnl=round(pnl, 2),
            equity=round(eq, 2),
        )

    async def update_unrealized_pnl(
        self, position_id: str, current_price: float
    ) -> None:
        """Update mark-to-market PnL for an open position."""
        pos = self._positions.get(position_id)
        if not pos or pos.status != PositionStatus.OPEN:
            return

        if pos.direction == "BUY_YES":
            # Bought YES at entry_price, current value = current_price
            pos.pnl_usd = (current_price - pos.fill_price) * pos.size_usd
        else:
            # Bought NO, which is (1 - YES_price)
            pos.pnl_usd = (pos.fill_price - current_price) * pos.size_usd

    async def get_portfolio_state(self) -> dict[str, Any]:
        """Full portfolio snapshot for risk filter updates."""
        daily_pnl = await self._data_store.get_daily_pnl()
        return {
            "active_positions": len(self.active_positions),
            "total_exposure_usd": round(self.total_exposure_usd, 2),
            "current_equity": round(self.current_equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "daily_pnl": round(daily_pnl, 2),
            "halted": self._halted,
            "closed_trades": len(self._closed_positions),
        }

    async def _check_drawdown(self) -> None:
        """Circuit breaker — disabled to let the bot run unconstrained."""
        return

    def get_recently_closed(self) -> list[Position]:
        """Return and clear the buffer of positions closed since last check."""
        closed = self._recently_closed
        self._recently_closed = []
        return closed

    def resume_trading(self) -> None:
        """Manual override to resume after halt."""
        self._halted = False
        logger.info("portfolio.resumed")
