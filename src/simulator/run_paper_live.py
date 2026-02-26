"""Launch script for paper trading with REAL CEX data + web dashboard.

Usage:
    python -m src.simulator.run_paper_live

Connects to Binance, Bybit, OKX WebSocket feeds for live BTC prices,
generates synthetic Polymarket orderbooks, and runs the full arbitrage
signal pipeline in paper trading mode. Web dashboard at http://localhost:8080.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXWebSocketManager, CEXFeed
from src.layer0_ingestion.data_store import DataStore, SpreadSnapshot
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer3_portfolio.platform_risk import PlatformRiskMonitor, PlatformStatus
from src.layer3_portfolio.tail_risk import TailRiskAgent, TailRiskAlert, AlertLevel
from src.layer4_execution.fill_monitor import FillMonitor
from src.simulator.live_feed import LiveMarketFeed, LiveFeedConfig
from src.simulator.paper_trader import PaperTrader, PaperPosition
from src.web.server import create_app, start_server


class PaperPortfolioAdapter:
    """Adapts PaperTrader's portfolio state to the interface build_snapshot() expects."""

    def __init__(self, trader: PaperTrader) -> None:
        self._trader = trader
        self._closed_positions: list[Any] = []

    @property
    def active_positions(self) -> list:
        """Return active positions as objects with to_dict()."""
        return [
            _PosProxy(p)
            for p in self._trader._positions.values()
            if p.status == "open"
        ]

    async def get_portfolio_state(self) -> Dict[str, Any]:
        unrealized = sum(p.pnl for p in self._trader._positions.values())
        equity = self._trader._equity + unrealized
        peak = self._trader._stats.peak_equity
        dd = (peak - equity) / peak if peak > 0 else 0.0

        return {
            "active_positions": len(self.active_positions),
            "total_exposure_usd": sum(
                p.size_usd for p in self._trader._positions.values()
                if p.status == "open"
            ),
            "current_equity": round(equity, 2),
            "peak_equity": round(peak, 2),
            "drawdown_pct": round(dd, 4),
            "daily_pnl": round(self._trader._daily_pnl, 2),
            "halted": False,
            "closed_trades": self._trader._stats.total_trades,
            "win_rate": round(self._trader._stats.win_rate, 3),
            "total_pnl": round(self._trader._stats.total_pnl, 2),
        }

    def sync_closed(self) -> None:
        """Sync closed positions from paper trader stats."""
        # Build proxy objects from trade log
        trades = self._trader._stats.pnl_history
        total = self._trader._stats.total_trades
        if total > len(self._closed_positions):
            # Add new closed position proxies
            for i in range(len(self._closed_positions), total):
                pnl = trades[i] if i < len(trades) else 0
                self._closed_positions.append(_ClosedPosProxy(
                    id=f"P{i+1:04d}",
                    pnl=pnl,
                    won=pnl > 0,
                ))


@dataclass
class _PosProxy:
    """Proxy to give PaperPosition a to_dict() matching Position interface."""
    _p: PaperPosition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._p.id,
            "market_id": self._p.market_id,
            "direction": self._p.direction,
            "entry_price": self._p.entry_price,
            "size_usd": self._p.size_usd,
            "status": self._p.status,
            "fill_price": self._p.entry_price,
            "exit_price": self._p.exit_price,
            "pnl_usd": self._p.pnl,
            "order_id": "",
            "created_at_ms": int(self._p.opened_at * 1000),
            "filled_at_ms": int(self._p.opened_at * 1000),
            "closed_at_ms": 0,
        }


@dataclass
class _ClosedPosProxy:
    id: str
    pnl: float
    won: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "market_id": "",
            "direction": "",
            "entry_price": 0,
            "size_usd": 0,
            "status": "closed",
            "fill_price": 0,
            "exit_price": 1.0 if self.won else 0.0,
            "pnl_usd": self.pnl,
            "order_id": "",
            "created_at_ms": 0,
            "filled_at_ms": 0,
            "closed_at_ms": 0,
        }


class _FakeExecutor:
    """Minimal executor stub providing fill_monitor for build_snapshot."""
    def __init__(self, fm: FillMonitor) -> None:
        self.fill_monitor = fm


class _FakePolymarket:
    """Stub so paper trading dashboard doesn't crash on standby/balance checks."""
    is_standby = False
    wallet_balance = 0.0


class PaperTradingSystem:
    """Lightweight system adapter that exposes the interface web dashboard expects.

    Wires together:
    - Real CEX WebSocket feeds
    - LiveMarketFeed (real CEX + synthetic PM orderbooks)
    - PaperTrader (signal pipeline + paper execution)
    - Web dashboard (FastAPI + WebSocket)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._running = False

        # Core infrastructure
        self._event_bus = EventBus(use_kafka=False)
        self._data_store = DataStore()
        self._cex_manager = CEXWebSocketManager()
        self._platform_risk = PlatformRiskMonitor()
        self._tail_risk = TailRiskAgent(event_bus=self._event_bus)

        # Live feed + paper trader
        self._live_feed = LiveMarketFeed(
            cex_manager=self._cex_manager,
            config=LiveFeedConfig(
                pm_lag_ticks=8,  # ~800ms lag at 10 ticks/sec
                pm_spread_bps=200,
                arb_opportunity_freq=0.10,  # 10% chance of wider arb
                arb_spread_mult=3.5,
                num_strikes=5,
                base_liquidity_usd=15_000.0,
            ),
        )

        self._trader = PaperTrader(
            live_feed=self._live_feed,
            show_terminal=True,
        )

        # Share the event bus
        self._trader._event_bus = self._event_bus

        # Tune research synthesis for live data — higher priors so signals fire
        # (real CEX data has less movement than synthetic GBM)
        self._trader._synthesis.BASE_PRIOR = 0.62
        self._trader._synthesis.SPREAD_WEIGHT = 0.55
        self._trader._synthesis.LATENCY_WEIGHT = 0.25
        self._trader._synthesis.LIQUIDITY_WEIGHT = 0.20
        self._trader._synthesis._spread_tp_rate = 0.90
        self._trader._synthesis._latency_tp_rate = 0.85
        self._trader._synthesis._liquidity_tp_rate = 0.92

        # Adapters for web dashboard
        self._portfolio = PaperPortfolioAdapter(self._trader)
        self._signal_validator = self._trader._signal_validator
        self._executor = _FakeExecutor(self._trader._fill_monitor)
        self._polymarket = _FakePolymarket()

    async def start(self) -> None:
        logger.info("paper_system.starting", mode="LIVE CEX DATA")

        # Boot infrastructure
        await self._event_bus.start()
        await self._data_store.start()
        await self._platform_risk.start()

        self._running = True

        # Background tasks
        tasks = [
            asyncio.create_task(self._cex_manager.start()),
            asyncio.create_task(self._platform_risk.check_loop()),
        ]

        # Wait for CEX connections before starting paper trader
        logger.info("paper_system.waiting_for_cex_data")
        await asyncio.sleep(3)

        # Log initial CEX state
        ticks = self._cex_manager.get_all_ticks()
        for exchange, tick in ticks.items():
            logger.info(
                "paper_system.cex_connected",
                exchange=exchange.value,
                price=f"${tick.mid:,.2f}",
            )

        # Start spread tracking in background
        tasks.append(asyncio.create_task(self._spread_tracker_loop()))

        # Start web dashboard
        if self._settings.dashboard_enabled:
            web_app = create_app(self)
            tasks.append(asyncio.create_task(
                start_server(
                    web_app,
                    host=self._settings.dashboard_host,
                    port=self._settings.dashboard_port,
                )
            ))
            logger.info(
                "paper_system.dashboard_started",
                url=f"http://localhost:{self._settings.dashboard_port}",
            )

        # Start the paper trading loop
        tasks.append(asyncio.create_task(self._trader.run()))

        logger.info("paper_system.started", tasks=len(tasks))

        try:
            # Wait for any task to complete (or error)
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in done:
                if t.exception():
                    logger.error("paper_system.task_error", error=str(t.exception()))
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self._trader._running = False
            await self._shutdown()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _spread_tracker_loop(self) -> None:
        """Track spreads between live CEX and simulated PM for the web dashboard."""
        while self._running:
            try:
                if self._live_feed.current_btc_price > 0 and self._live_feed.current_pm_price > 0:
                    cex_price = self._live_feed.current_btc_price
                    pm_price = self._live_feed.current_pm_price
                    spread_pct = (cex_price - pm_price) / cex_price * 100

                    snap = SpreadSnapshot(
                        polymarket_price=pm_price,
                        cex_price=cex_price,
                        spread_pct=round(spread_pct, 4),
                        timestamp_ms=int(time.time() * 1000),
                        cex_source="live_best",
                    )
                    await self._data_store.push_spread(snap)

                    # Sync closed positions
                    self._portfolio.sync_closed()
            except Exception:
                pass
            await asyncio.sleep(1.0)

    async def _shutdown(self) -> None:
        logger.info("paper_system.shutting_down")
        await self._cex_manager.stop()
        await self._platform_risk.stop()
        await self._event_bus.stop()
        await self._data_store.stop()
        logger.info("paper_system.stopped")


def main() -> None:
    """Entry point for live paper trading."""
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    system = PaperTradingSystem()

    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("paper_system.signal_received", signal=sig)
        system._running = False
        system._trader._running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(system.start())
    except KeyboardInterrupt:
        logger.info("paper_system.keyboard_interrupt")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
