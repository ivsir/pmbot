"""Research Orchestrator — coordinates all Layer 1 agents and feeds Layer 2."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine

import structlog

from src.layer0_ingestion.cex_websocket import CEXTick, CEXWebSocketManager
from src.layer0_ingestion.polymarket_client import OrderBook, PolymarketClient
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from config.settings import get_settings
from src.layer1_research.spread_detector import SpreadDetector, SpreadOpportunity
from src.layer1_research.momentum_detector import MomentumDetector, MomentumSignal
from src.layer1_research.latency_arb import LatencyArbDetector, LatencySignal
from src.layer1_research.liquidity_scanner import LiquidityScanner, LiquidityProfile
from src.layer1_research.research_synthesis import ResearchSynthesis, ResearchOutput

logger = structlog.get_logger(__name__)


class ResearchOrchestrator:
    """Orchestrates all research agents, runs them concurrently on each tick,
    and publishes fused ResearchOutput to the event bus."""

    def __init__(
        self,
        event_bus: EventBus,
        polymarket: PolymarketClient,
        cex_manager: CEXWebSocketManager,
    ) -> None:
        self._event_bus = event_bus
        self._polymarket = polymarket
        self._cex = cex_manager

        self._settings = get_settings()
        self._market_mode = self._settings.market_mode

        # Sub-agents
        self._spread_detector = SpreadDetector()
        self._momentum_detector = MomentumDetector()
        self._latency_detector = LatencyArbDetector()
        self._liquidity_scanner = LiquidityScanner()
        self._synthesis = ResearchSynthesis()

        # State — monthly markets
        self._active_markets: dict[str, float] = {}  # market_id → strike_price
        # State — 5-min Up/Down markets
        self._active_updown_markets: dict[str, dict] = {}  # condition_id → {"start_ms", "end_ms"}
        self._signaled_windows: set[str] = set()  # condition_ids already signaled this window
        self._callbacks: list[
            Callable[[ResearchOutput], Coroutine[Any, Any, None]]
        ] = []

    def on_research_output(
        self, cb: Callable[[ResearchOutput], Coroutine[Any, Any, None]]
    ) -> None:
        self._callbacks.append(cb)

    def set_active_markets(self, markets: dict[str, float]) -> None:
        """Set active monthly BTC markets: {market_id: strike_price}."""
        self._active_markets = markets
        logger.info(
            "research_orchestrator.markets_set",
            count=len(markets),
        )

    def set_active_updown_markets(self, markets: dict[str, dict]) -> None:
        """Set active 5-min Up/Down markets: {condition_id: {"start_ms", "end_ms"}}."""
        self._active_updown_markets = markets
        logger.info(
            "research_orchestrator.updown_markets_set",
            count=len(markets),
            market_ids=[k[:20] for k in markets.keys()],
        )

    async def process_tick(self, cex_tick: CEXTick) -> list[ResearchOutput]:
        """Process a new CEX tick — run all research agents in parallel."""
        self._latency_detector.record_cex_tick(cex_tick)

        if self._market_mode == "5min_updown":
            return await self._process_tick_updown(cex_tick)
        else:
            return await self._process_tick_monthly(cex_tick)

    async def _process_tick_monthly(self, cex_tick: CEXTick) -> list[ResearchOutput]:
        """Original monthly market processing logic."""
        results: list[ResearchOutput] = []

        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % 500 == 1:
            cached_keys = list(self._polymarket._orderbooks.keys())
            market_keys = list(self._active_markets.keys())[:3]
            matched = sum(1 for k in self._active_markets if k in self._polymarket._orderbooks)
            logger.info(
                "research.orderbook_debug",
                tick=self._tick_count,
                cached_count=len(cached_keys),
                active_count=len(self._active_markets),
                matched=matched,
                cached_sample=[k[:20] for k in cached_keys[:3]],
                active_sample=[k[:20] for k in market_keys],
                cex_mid=round(cex_tick.mid, 2),
            )

        for market_id, strike_price in self._active_markets.items():
            ob = self._polymarket.get_cached_orderbook(market_id)
            if ob is None:
                continue

            spread_opp, latency_sig, liquidity_prof = await asyncio.gather(
                self._run_spread(ob, cex_tick, strike_price),
                self._run_latency(ob, cex_tick, strike_price),
                self._run_liquidity(ob),
            )

            output = self._synthesis.synthesize(
                spread_opp, latency_sig, liquidity_prof
            )

            if output and output.is_actionable:
                results.append(output)
                await self._event_bus.publish(
                    Event(
                        event_type=EventType.RESEARCH_SIGNAL,
                        data=output.to_dict(),
                        source="research_orchestrator",
                    )
                )
                for cb in self._callbacks:
                    asyncio.create_task(cb(output))

        return results

    async def _process_tick_updown(self, cex_tick: CEXTick) -> list[ResearchOutput]:
        """Process tick for 5-min BTC Up/Down markets using momentum detector."""
        results: list[ResearchOutput] = []

        if not self._active_updown_markets:
            return results

        # Get CEX price history for momentum calculation
        price_history = self._cex.get_price_history(max_age_ms=300_000)

        # Feed candle buffer for ML feature computation
        self._momentum_detector.update_candle_buffer(cex_tick)

        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % 500 == 1:
            logger.info(
                "research.updown_debug",
                tick=self._tick_count,
                active_updown=len(self._active_updown_markets),
                price_history_len=len(price_history),
                cex_mid=round(cex_tick.mid, 2),
            )

        now_ms = int(time.time() * 1000)

        # Prune old entries from signaled windows set
        self._signaled_windows = {
            cid for cid in self._signaled_windows
            if cid in self._active_updown_markets
        }

        for condition_id, window in self._active_updown_markets.items():
            ob = self._polymarket.get_cached_orderbook(condition_id)
            if ob is None:
                continue

            # Skip if we already emitted a signal for this window
            if condition_id in self._signaled_windows:
                continue

            start_ms = window.get("start_ms", 0)
            end_ms = window.get("end_ms", 0)

            secs_until_start = (start_ms - now_ms) / 1000 if start_ms else -999
            secs_until_end = (end_ms - now_ms) / 1000 if end_ms else 0

            if secs_until_start > 0:
                continue  # window hasn't started yet — skip
            if secs_until_end < 5:
                continue  # less than 5s until end — too late for fill

            secs_into_window = 300 - secs_until_end  # how far into the 5-min window

            # T+0: trade from window start
            if secs_into_window < 0:
                continue

            if secs_into_window > 240:
                continue  # only trade in first 4 min of 5-min window

            # Debug: log when we're in the tradeable window
            if self._tick_count % 100 == 1:
                logger.info(
                    "research.window_active",
                    market=condition_id[:20],
                    secs_into=round(secs_into_window),
                    secs_left=round(secs_until_end),
                    has_ob=ob is not None,
                )

            # Run agents concurrently
            momentum_sig, latency_sig, liquidity_prof = await asyncio.gather(
                self._run_momentum(ob, cex_tick, price_history, start_ms, end_ms),
                self._run_latency(ob, cex_tick, 0.0),  # no strike for Up/Down
                self._run_liquidity(ob),
            )

            # Bayesian synthesis — MomentumSignal duck-types SpreadOpportunity
            output = self._synthesis.synthesize(
                momentum_sig, latency_sig, liquidity_prof
            )

            if output and output.is_actionable:
                # One signal per window — mark as signaled
                self._signaled_windows.add(condition_id)
                # Attach window_end_ms for GTD expiration in execution layer
                output.window_end_ms = end_ms
                results.append(output)
                await self._event_bus.publish(
                    Event(
                        event_type=EventType.RESEARCH_SIGNAL,
                        data=output.to_dict(),
                        source="research_orchestrator",
                    )
                )
                for cb in self._callbacks:
                    asyncio.create_task(cb(output))

        return results

    async def process_orderbook(self, orderbook: OrderBook) -> None:
        """Handle incoming orderbook update — update latency tracker."""
        local_recv = orderbook.local_receive_ms or int(time.time() * 1000)
        self._latency_detector.record_pm_update(
            orderbook.market_id,
            local_receive_ms=local_recv,
            server_ts=orderbook.timestamp_ms,
        )

    def provide_feedback(self, output: ResearchOutput, was_profitable: bool) -> None:
        """Feedback loop — update Bayesian likelihoods."""
        self._synthesis.update_likelihoods(was_profitable, output)

    # ── Private runners ──

    async def _run_spread(
        self, ob: OrderBook, tick: CEXTick, strike: float
    ) -> SpreadOpportunity | None:
        try:
            return self._spread_detector.detect(ob, tick, strike)
        except Exception as exc:
            logger.warning("spread_detector.error", error=str(exc))
            return None

    async def _run_latency(
        self, ob: OrderBook, tick: CEXTick, strike: float
    ) -> LatencySignal | None:
        try:
            return self._latency_detector.detect(ob, tick, strike)
        except Exception as exc:
            logger.warning("latency_arb.error", error=str(exc))
            return None

    async def _run_momentum(
        self,
        ob: OrderBook,
        tick: CEXTick,
        price_history: list[CEXTick],
        window_start_ms: int,
        window_end_ms: int,
    ) -> MomentumSignal | None:
        try:
            return self._momentum_detector.detect(
                ob, tick, price_history, window_start_ms, window_end_ms
            )
        except Exception as exc:
            logger.warning("momentum_detector.error", error=str(exc))
            return None

    async def _run_liquidity(self, ob: OrderBook) -> LiquidityProfile | None:
        try:
            return self._liquidity_scanner.scan(ob)
        except Exception as exc:
            logger.warning("liquidity_scanner.error", error=str(exc))
            return None
