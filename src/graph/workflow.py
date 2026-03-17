"""LangGraph workflow — orchestrates the full 5-layer arbitrage pipeline.

Graph topology:
  ingest_data → collect_feedback → research_and_risk → generate_signal
                                                        → [TRADE] → execute
                                                        → [SKIP]  → log_skip
  research and check_risk run in parallel within research_and_risk node.
"""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick, CEXWebSocketManager
from src.layer0_ingestion.data_store import DataStore, PriceTick, SpreadSnapshot
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer0_ingestion.polymarket_client import OrderBook, PolymarketClient
from src.layer1_research.orchestrator import ResearchOrchestrator
from src.layer1_research.research_synthesis import ResearchOutput
from src.layer2_signal.alpha_signal import AlphaSignal, AlphaSignalGenerator
from src.layer2_signal.backtester import Backtester
from src.layer2_signal.risk_filter import RiskFilter, RiskAssessment
from src.layer2_signal.signal_validator import SignalValidator, ValidatedSignal
from src.layer3_portfolio.correlation_monitor import CorrelationMonitor
from src.layer3_portfolio.portfolio_manager import PortfolioManager
from src.layer3_portfolio.tail_risk import TailRiskAgent
from src.layer3_portfolio.platform_risk import PlatformRiskMonitor
from src.layer4_execution.execution_agent import ExecutionAgent, ExecutionResult

logger = structlog.get_logger(__name__)


# ── State schema for the LangGraph workflow ──


class ArbitrageState(TypedDict, total=False):
    """Typed state flowing through the LangGraph arbitrage workflow."""
    cex_tick: CEXTick | None
    orderbook: OrderBook | None
    research_outputs: list[ResearchOutput]
    alpha_signal: AlphaSignal | None
    validated_signal: ValidatedSignal | None
    execution_result: ExecutionResult | None
    risk_alerts: list[dict]
    should_halt: bool
    cycle_start_ms: int
    research_for_trade: ResearchOutput | None


# ── Node functions ──


class ArbitrageNodes:
    """Encapsulates all graph node functions with access to shared components."""

    def __init__(
        self,
        event_bus: EventBus,
        data_store: DataStore,
        polymarket: PolymarketClient,
        cex_manager: CEXWebSocketManager,
        research_orchestrator: ResearchOrchestrator,
        signal_validator: SignalValidator,
        portfolio_manager: PortfolioManager,
        execution_agent: ExecutionAgent,
        tail_risk: TailRiskAgent,
        platform_risk: PlatformRiskMonitor,
        correlation_monitor: CorrelationMonitor,
    ) -> None:
        self._event_bus = event_bus
        self._data_store = data_store
        self._polymarket = polymarket
        self._cex = cex_manager
        self._research = research_orchestrator
        self._validator = signal_validator
        self._portfolio = portfolio_manager
        self._executor = execution_agent
        self._tail_risk = tail_risk
        self._platform_risk = platform_risk
        self._correlation = correlation_monitor
        self._settings = get_settings()
        # Maps position_id → ResearchOutput that generated the trade
        self._trade_research_map: dict[str, ResearchOutput] = {}

    # ── Node: Ingest Data (Layer 0) ──

    async def ingest_data(self, state: dict) -> dict:
        """Capture latest CEX tick and PM orderbook into state."""
        cycle_start = int(time.time() * 1000)

        cex_tick = self._cex.get_best_cex_price()
        self._ingest_count = getattr(self, "_ingest_count", 0) + 1
        if self._ingest_count % 500 == 1:
            logger.info(
                "graph.ingest_debug",
                count=self._ingest_count,
                ticks_dict_size=len(self._cex._ticks),
                has_cex_tick=cex_tick is not None,
                cex_id=id(self._cex),
            )
        if cex_tick:
            # Persist tick
            pt = PriceTick(
                source=cex_tick.exchange.value,
                symbol=cex_tick.symbol,
                bid=cex_tick.bid,
                ask=cex_tick.ask,
                last=cex_tick.last,
                timestamp_ms=cex_tick.timestamp_ms,
                volume=cex_tick.volume_24h,
            )
            await self._data_store.set_latest_tick(pt)

            # Feed tail risk monitor
            self._tail_risk.record_price(cex_tick.mid)

        ingestion_time = int(time.time() * 1000) - cycle_start
        if ingestion_time > self._settings.latency_data_ingestion_ms:
            logger.warning(
                "graph.ingestion_slow",
                actual_ms=ingestion_time,
                budget_ms=self._settings.latency_data_ingestion_ms,
            )

        return {
            "cex_tick": cex_tick,
            "cycle_start_ms": cycle_start,
            "research_outputs": [],
            "alpha_signal": None,
            "validated_signal": None,
            "execution_result": None,
            "risk_alerts": [],
            "should_halt": False,
            "research_for_trade": None,
        }

    # ── Node: Collect Feedback ──

    async def collect_feedback(self, state: dict) -> dict:
        """Process feedback from recently closed positions to update Bayesian priors."""
        recently_closed = self._portfolio.get_recently_closed()
        for pos in recently_closed:
            research = self._trade_research_map.pop(pos.id, None)
            if research:
                was_profitable = pos.pnl_usd > 0
                self._research.provide_feedback(research, was_profitable)
                logger.info(
                    "graph.feedback_recorded",
                    position_id=pos.id,
                    profitable=was_profitable,
                    pnl=round(pos.pnl_usd, 2),
                )
        return {}

    # ── Node: Research + Risk (Layer 1 & 3 — parallel) ──

    async def research_and_risk(self, state: dict) -> dict:
        """Run research agents and risk checks in parallel."""
        cex_tick = state.get("cex_tick")

        self._rar_count = getattr(self, "_rar_count", 0) + 1
        if self._rar_count % 500 == 1:
            logger.info(
                "graph.rar_state_debug",
                count=self._rar_count,
                has_cex_tick=cex_tick is not None,
                state_keys=list(state.keys()),
                cex_in_state="cex_tick" in state,
            )

        # Run research and risk concurrently
        research_coro = self._do_research(cex_tick)
        risk_coro = self._do_risk_check()

        research_result, risk_result = await asyncio.gather(
            research_coro, risk_coro
        )

        return {**research_result, **risk_result}

    async def _do_research(self, cex_tick: CEXTick | None) -> dict:
        """Research sub-task: run all research agents."""
        if not cex_tick:
            self._no_tick_count = getattr(self, "_no_tick_count", 0) + 1
            if self._no_tick_count % 500 == 1:
                logger.warning("graph.no_cex_tick", count=self._no_tick_count)
            return {"research_outputs": []}
        outputs = await self._research.process_tick(cex_tick)
        return {"research_outputs": outputs}

    async def _do_risk_check(self) -> dict:
        """Risk check sub-task: tail risk + platform risk."""
        alerts = []

        # Tail risk
        tail_alerts = await self._tail_risk.check_all()
        for a in tail_alerts:
            alerts.append(
                {
                    "level": a.level.value,
                    "type": a.alert_type,
                    "message": a.message,
                    "action": a.recommended_action,
                }
            )

        # Platform health
        platform = self._platform_risk.status
        if not platform.all_healthy:
            alerts.append(
                {
                    "level": "warning",
                    "type": "platform_degraded",
                    "message": f"CEX up: {platform.cex_count_up}/3, PM: {platform.polymarket_up}",
                    "action": "REDUCE_AGGRESSION",
                }
            )

        # Portfolio halt check
        should_halt = (
            self._portfolio.is_halted
            or self._tail_risk.is_emergency
        )

        # Update risk filter state
        portfolio_state = await self._portfolio.get_portfolio_state()
        self._validator.risk_filter.update_state(
            positions=[p.to_dict() for p in self._portfolio.active_positions],
            daily_pnl=portfolio_state["daily_pnl"],
            equity=portfolio_state["current_equity"],
        )

        return {"risk_alerts": alerts, "should_halt": should_halt}

    # ── Node: Generate Signal (Layer 2) ──

    async def generate_signal(self, state: dict) -> dict:
        """Pick best research output and generate alpha signal."""
        outputs = state.get("research_outputs", [])
        if not outputs:
            return {"validated_signal": None, "research_for_trade": None}

        # Pick highest combined probability
        best = max(outputs, key=lambda o: o.combined_probability)

        # Full validation pipeline (alpha → risk → backtest → TRADE/SKIP)
        validated = await self._validator.validate(best)

        return {"validated_signal": validated, "research_for_trade": best}

    # ── Node: Execute Trade (Layer 4) ──

    async def execute_trade(self, state: dict) -> dict:
        """Execute validated trade signal."""
        validated = state.get("validated_signal")
        if not validated or not validated.is_trade:
            return {"execution_result": None}

        result = await self._executor.execute(validated)

        # Store research→position lineage for feedback loop
        if result.success:
            research = state.get("research_for_trade")
            if research:
                self._trade_research_map[result.position_id] = research

            logger.info(
                "graph.trade_executed",
                position_id=result.position_id,
                fill_price=result.fill_price,
                time_ms=result.execution_time_ms,
            )

        return {"execution_result": result}

    # ── Node: Log Skip ──

    async def log_skip(self, state: dict) -> dict:
        """Log skipped trade for analysis."""
        validated = state.get("validated_signal")
        if validated:
            logger.debug(
                "graph.trade_skipped",
                market=validated.signal.market_id,
                reason=validated.reason,
            )
        return {}

    # ── Node: Report Metrics ──

    async def report_metrics(self, state: dict) -> dict:
        """End-of-cycle metrics and latency check."""
        cycle_start = state.get("cycle_start_ms", 0)
        if cycle_start > 0:
            total_ms = int(time.time() * 1000) - cycle_start
            if total_ms > self._settings.latency_end_to_end_ms:
                logger.warning(
                    "graph.e2e_latency_exceeded",
                    actual_ms=total_ms,
                    budget_ms=self._settings.latency_end_to_end_ms,
                )
            else:
                logger.debug("graph.cycle_complete", latency_ms=total_ms)

        return {}


# ── Routing functions ──


def should_trade_or_skip(state: dict) -> Literal["execute_trade", "log_skip", "report_metrics"]:
    """Route based on validation result."""
    if state.get("should_halt"):
        return "report_metrics"

    validated = state.get("validated_signal")
    if validated and validated.is_trade:
        return "execute_trade"
    return "log_skip"


def has_research_outputs(state: dict) -> Literal["generate_signal", "report_metrics"]:
    """Route based on whether research found opportunities."""
    outputs = state.get("research_outputs", [])
    if outputs and not state.get("should_halt"):
        return "generate_signal"
    return "report_metrics"


# ── Graph builder ──


def build_arbitrage_graph(nodes: ArbitrageNodes) -> StateGraph:
    """Build the LangGraph state machine for the arbitrage pipeline.

    Flow:
    ┌─────────────┐
    │ ingest_data  │
    └──────┬───────┘
           │
    ┌──────▼──────────────┐
    │ collect_feedback     │  (updates Bayesian priors from closed trades)
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │ research_and_risk   │  (research + risk checks run in parallel)
    └──────┬──────────────┘
           │
    ┌──────▼────────┐
    │generate_signal │
    └──────┬────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  ┌────────┐ ┌────────┐
  │execute │ │log_skip│
  └───┬────┘ └───┬────┘
      └─────┬────┘
            ▼
    ┌───────────────┐
    │ report_metrics│
    └───────────────┘
    """

    graph = StateGraph(ArbitrageState)

    # Add nodes
    graph.add_node("ingest_data", nodes.ingest_data)
    graph.add_node("collect_feedback", nodes.collect_feedback)
    graph.add_node("research_and_risk", nodes.research_and_risk)
    graph.add_node("generate_signal", nodes.generate_signal)
    graph.add_node("execute_trade", nodes.execute_trade)
    graph.add_node("log_skip", nodes.log_skip)
    graph.add_node("report_metrics", nodes.report_metrics)

    # Set entry point
    graph.set_entry_point("ingest_data")

    # Edges: ingest → feedback → research+risk (parallel internally)
    graph.add_edge("ingest_data", "collect_feedback")
    graph.add_edge("collect_feedback", "research_and_risk")

    # Conditional: after research+risk, route to signal generation or skip
    graph.add_conditional_edges(
        "research_and_risk",
        has_research_outputs,
        {
            "generate_signal": "generate_signal",
            "report_metrics": "report_metrics",
        },
    )

    # Conditional: after signal generation, trade or skip
    graph.add_conditional_edges(
        "generate_signal",
        should_trade_or_skip,
        {
            "execute_trade": "execute_trade",
            "log_skip": "log_skip",
            "report_metrics": "report_metrics",
        },
    )

    # Terminal edges
    graph.add_edge("execute_trade", "report_metrics")
    graph.add_edge("log_skip", "report_metrics")
    graph.add_edge("report_metrics", END)

    return graph
