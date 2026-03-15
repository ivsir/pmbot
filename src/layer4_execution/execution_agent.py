"""Execution Agent — CLOB orders, <500ms latency, limit/market optimization."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

from config.settings import get_settings
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer0_ingestion.polymarket_client import PolymarketClient, OrderBook
from src.layer2_signal.signal_validator import ValidatedSignal
from src.layer3_portfolio.portfolio_manager import PortfolioManager, Position
from src.layer4_execution.order_book_sniper import OrderBookSniper, SniperOrder
from src.layer4_execution.fill_monitor import FillMonitor, FillResult, FillStatus
from src.layer4_execution.hedge_agent import HedgeAgent

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    position_id: str
    order_id: str
    success: bool
    fill_price: float
    fill_size: float
    slippage_bps: float
    execution_time_ms: int
    strategy_used: str
    error: str = ""
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "order_id": self.order_id,
            "success": self.success,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
            "slippage_bps": self.slippage_bps,
            "execution_time_ms": self.execution_time_ms,
            "strategy_used": self.strategy_used,
            "error": self.error,
        }


class ExecutionAgent:
    """Top-level execution agent — coordinates sniper, fill monitor, and hedge agent.

    Responsibilities:
    1. Receive validated trade signals
    2. Determine optimal order strategy (sniper)
    3. Place orders on Polymarket CLOB (<500ms target)
    4. Monitor fills and handle partial fills
    5. Execute hedge trades when applicable
    """

    def __init__(
        self,
        event_bus: EventBus,
        polymarket: PolymarketClient,
        portfolio: PortfolioManager,
    ) -> None:
        self._settings = get_settings()
        self._event_bus = event_bus
        self._polymarket = polymarket
        self._portfolio = portfolio
        self._sniper = OrderBookSniper()
        self._fill_monitor = FillMonitor()
        self._hedge_agent = HedgeAgent(polymarket)
        self._execution_count = 0

    @property
    def fill_monitor(self) -> FillMonitor:
        return self._fill_monitor

    @property
    def hedge_agent(self) -> HedgeAgent:
        return self._hedge_agent

    async def execute(self, signal: ValidatedSignal) -> ExecutionResult:
        """Execute a validated trade signal end-to-end.

        Pipeline: Signal → Position → Sniper → Order → Fill → Result
        """
        # Standby guard: skip execution when wallet has no funds
        if self._polymarket.is_standby:
            return ExecutionResult(
                position_id="",
                order_id="",
                success=False,
                fill_price=0,
                fill_size=0,
                slippage_bps=0,
                execution_time_ms=0,
                strategy_used="none",
                error="STANDBY — no funds in wallet",
            )

        start_ms = int(time.time() * 1000)
        self._execution_count += 1

        # Step 1: Open position in portfolio
        position = await self._portfolio.open_position(signal)
        if not position:
            return ExecutionResult(
                position_id="",
                order_id="",
                success=False,
                fill_price=0,
                fill_size=0,
                slippage_bps=0,
                execution_time_ms=0,
                strategy_used="none",
                error="Portfolio rejected position",
            )

        # Step 2: Resolve CLOB token_id and get correct token's orderbook
        clob_token_id = self._polymarket.get_clob_token_id(
            signal.signal.market_id, signal.signal.direction
        )
        if not clob_token_id:
            return ExecutionResult(
                position_id=position.id,
                order_id="",
                success=False,
                fill_price=0,
                fill_size=0,
                slippage_bps=0,
                execution_time_ms=int(time.time() * 1000) - start_ms,
                strategy_used="none",
                error=f"No CLOB token_id mapping for {signal.signal.market_id[:20]}... direction={signal.signal.direction}",
            )

        # Look up the specific token's orderbook (not the condition_id's)
        ob = self._polymarket.get_cached_orderbook(clob_token_id)
        if not ob:
            # Fallback to condition_id orderbook
            ob = self._polymarket.get_cached_orderbook(signal.signal.market_id)
        if not ob:
            # Try fetching fresh
            try:
                ob = await self._polymarket.get_orderbook(
                    clob_token_id
                )
            except Exception as exc:
                return ExecutionResult(
                    position_id=position.id,
                    order_id="",
                    success=False,
                    fill_price=0,
                    fill_size=0,
                    slippage_bps=0,
                    execution_time_ms=int(time.time() * 1000) - start_ms,
                    strategy_used="none",
                    error=f"Orderbook unavailable: {exc}",
                )

        # Step 3: Compute optimal order via sniper
        side = "BUY"  # Both BUY_YES and BUY_NO are "BUY" on CLOB
        neg_risk = self._polymarket.is_neg_risk(signal.signal.market_id)

        # Cap order size to available wallet balance (leave $0.10 buffer)
        available = self._polymarket.wallet_balance - 0.10
        trade_size = signal.final_size_usd
        if available < 1.0:
            await self._portfolio.close_position(position.id, 0.0, 0.0)
            return ExecutionResult(
                position_id=position.id,
                order_id="",
                success=False,
                fill_price=0,
                fill_size=0,
                slippage_bps=0,
                execution_time_ms=int(time.time() * 1000) - start_ms,
                strategy_used="none",
                error=f"Insufficient balance: ${available + 0.10:.2f}",
            )
        if trade_size > available:
            logger.info(
                "execution.size_capped_to_balance",
                original=round(trade_size, 2),
                capped=round(available, 2),
                balance=round(self._polymarket.wallet_balance, 2),
            )
            trade_size = available

        # Determine if we should use maker strategy (final 60s of window)
        window_end_ms = getattr(signal.signal.research, 'window_end_ms', 0) if signal.signal.research else 0
        now_ms_check = int(time.time() * 1000)
        secs_to_end = (window_end_ms - now_ms_check) / 1000 if window_end_ms > 0 else 999
        use_maker = (
            self._settings.market_mode == "5min_updown"
            and window_end_ms > 0
            and secs_to_end <= 60  # only use maker in final 60 seconds
        )

        if use_maker:
            # Near-settlement maker order: GTD + post_only
            now_ms = int(time.time() * 1000)
            secs_until_end = max(0, (window_end_ms - now_ms) / 1000)
            fair_prob = signal.signal.entry_price  # already computed as maker price

            sniper_order = self._sniper.compute_maker_order(
                orderbook=ob,
                side=side,
                size_usd=trade_size,
                fair_prob=fair_prob,
                secs_until_end=secs_until_end,
            )
        else:
            # For 5min_updown: always cross the spread (taker) for instant fills.
            # Edge decays in milliseconds — sitting on the book loses money.
            if self._settings.market_mode == "5min_updown":
                urgency = 0.95  # forces MARKET strategy in sniper
            else:
                urgency = min(signal.signal.win_probability, 0.95)
            sniper_order = self._sniper.compute_optimal_order(
                orderbook=ob,
                side=side,
                size_usd=trade_size,
                urgency=urgency,
            )

        # Step 4: Place order on CLOB
        try:
            if use_maker:
                # GTD maker order with post_only — zero fees
                expiration_s = int(window_end_ms / 1000) + 30
                order_result = await self._polymarket.place_maker_order(
                    token_id=clob_token_id,
                    side=side,
                    price=sniper_order.price,
                    size=sniper_order.size,
                    expiration_s=expiration_s,
                    neg_risk=neg_risk,
                )
            else:
                order_result = await self._polymarket.place_order(
                    token_id=clob_token_id,
                    side=side,
                    price=sniper_order.price,
                    size=sniper_order.size,
                    order_type="GTC",
                    neg_risk=neg_risk,
                )
            order_id = order_result.get("orderID", "")

            # Step 6: Track fill
            self._fill_monitor.track_order(
                order_id=order_id,
                position_id=position.id,
                requested_price=sniper_order.price,
                requested_size=sniper_order.size,
            )

            # Step 7: Wait for fill
            # 5min_updown orders need longer wait — thin books may take time to match
            if use_maker:
                wait_ms = 30_000  # maker: sit on book until settlement
            elif self._settings.market_mode == "5min_updown":
                wait_ms = 30_000  # taker: give thin books time to match
            else:
                wait_ms = None  # use default MAX_FILL_WAIT_MS
            fill = await self._wait_for_fill(order_id, sniper_order, max_wait_override_ms=wait_ms)

            execution_time = int(time.time() * 1000) - start_ms

            if fill and fill.status == FillStatus.FILLED:
                # Update portfolio (pass filled_size so size_usd reflects actual cost)
                await self._portfolio.fill_position(
                    position.id, fill.fill_price, order_id,
                    filled_size=fill.filled_size,
                )

                result = ExecutionResult(
                    position_id=position.id,
                    order_id=order_id,
                    success=True,
                    fill_price=fill.fill_price,
                    fill_size=fill.filled_size,
                    slippage_bps=fill.slippage_bps,
                    execution_time_ms=execution_time,
                    strategy_used=sniper_order.strategy.value,
                )
            else:
                result = ExecutionResult(
                    position_id=position.id,
                    order_id=order_id,
                    success=False,
                    fill_price=0,
                    fill_size=0,
                    slippage_bps=0,
                    execution_time_ms=execution_time,
                    strategy_used=sniper_order.strategy.value,
                    error="Fill timeout or failure",
                )

        except Exception as exc:
            execution_time = int(time.time() * 1000) - start_ms
            result = ExecutionResult(
                position_id=position.id,
                order_id="",
                success=False,
                fill_price=0,
                fill_size=0,
                slippage_bps=0,
                execution_time_ms=execution_time,
                strategy_used=sniper_order.strategy.value,
                error=str(exc),
            )
            logger.error(
                "execution.order_failed",
                position_id=position.id,
                error=str(exc),
            )

        # Publish execution event
        await self._event_bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED
                if result.success
                else EventType.RISK_ALERT,
                data=result.to_dict(),
                source="execution_agent",
            )
        )

        # Cancel phantom position if order failed (so it doesn't block future trades)
        if not result.success and position:
            await self._portfolio.close_position(position.id, 0.0, 0.0)
            logger.info(
                "execution.phantom_position_cancelled",
                position_id=position.id,
            )

        # Check latency budget
        if result.execution_time_ms > self._settings.latency_order_exec_ms:
            logger.warning(
                "execution.latency_exceeded",
                actual_ms=result.execution_time_ms,
                budget_ms=self._settings.latency_order_exec_ms,
            )

        logger.info(
            "execution.complete",
            position_id=position.id,
            success=result.success,
            time_ms=result.execution_time_ms,
            strategy=result.strategy_used,
        )

        return result

    async def check_and_close_expired(self) -> list[str]:
        """Check for expired orders and handle them."""
        expired = self._fill_monitor.check_expired()
        for order_id in expired:
            logger.warning("execution.order_expired", order_id=order_id)
            # Attempt to cancel
            try:
                await self._polymarket.cancel_order(order_id)
            except Exception:
                pass
        return expired

    async def _wait_for_fill(
        self, order_id: str, sniper: SniperOrder, max_wait_override_ms: int | None = None,
    ) -> FillResult | None:
        """Wait for order fill.

        Paper mode: records simulated fill immediately.
        Live mode: polls CLOB order status every 250ms until filled or timeout.
        """
        if not self._settings.live_trading_enabled:
            # Paper mode: immediate simulated fill
            return self._fill_monitor.record_fill(
                order_id=order_id,
                fill_price=sniper.price,
                filled_size=sniper.size,
                status=FillStatus.FILLED,
            )

        # Live mode: poll order status from CLOB
        poll_interval_ms = 100  # 100ms for faster fill detection
        max_wait_ms = max_wait_override_ms or self._fill_monitor.MAX_FILL_WAIT_MS
        elapsed_ms = 0

        while elapsed_ms < max_wait_ms:
            try:
                order_data = await self._polymarket.get_order_status(order_id)
                if order_data is None:
                    await asyncio.sleep(poll_interval_ms / 1000)
                    elapsed_ms += poll_interval_ms
                    continue

                size_matched = float(order_data.get("size_matched", 0))
                original_size = float(
                    order_data.get("original_size", sniper.size)
                )
                order_status = str(order_data.get("status", "")).upper()

                if size_matched >= original_size * 0.99:
                    # Fully filled
                    avg_price = float(order_data.get("price", sniper.price))
                    fill = self._fill_monitor.record_fill(
                        order_id=order_id,
                        fill_price=avg_price,
                        filled_size=size_matched,
                        status=FillStatus.FILLED,
                    )
                    logger.info(
                        "execution.fill_confirmed",
                        order_id=order_id,
                        fill_price=avg_price,
                        filled_size=size_matched,
                        time_ms=elapsed_ms,
                    )
                    return fill

                elif size_matched > 0 and order_status in (
                    "CANCELLED", "EXPIRED",
                ):
                    # Partial fill
                    avg_price = float(order_data.get("price", sniper.price))
                    fill = self._fill_monitor.record_fill(
                        order_id=order_id,
                        fill_price=avg_price,
                        filled_size=size_matched,
                        status=FillStatus.PARTIAL,
                    )
                    logger.warning(
                        "execution.partial_fill",
                        order_id=order_id,
                        filled=size_matched,
                        requested=original_size,
                    )
                    return fill

                elif order_status in ("CANCELLED", "EXPIRED"):
                    # No fill
                    return self._fill_monitor.record_fill(
                        order_id=order_id,
                        fill_price=0,
                        filled_size=0,
                        status=FillStatus.CANCELLED,
                    )

            except Exception as exc:
                logger.warning(
                    "execution.poll_error",
                    order_id=order_id,
                    error=str(exc),
                )

            await asyncio.sleep(poll_interval_ms / 1000)
            elapsed_ms += poll_interval_ms

        # Timeout: cancel the order
        logger.warning(
            "execution.fill_timeout",
            order_id=order_id,
            waited_ms=max_wait_ms,
        )
        try:
            await self._polymarket.cancel_order(order_id)
        except Exception:
            pass

        return self._fill_monitor.record_fill(
            order_id=order_id,
            fill_price=0,
            filled_size=0,
            status=FillStatus.EXPIRED,
        )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_executions": self._execution_count,
            "fill_metrics": self._fill_monitor.get_metrics(),
            "active_hedges": len(self._hedge_agent.get_active_hedges()),
        }
