"""Main entry point — boots all layers and runs the LangGraph arbitrage loop."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from typing import Any

import structlog
import uvloop

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXWebSocketManager, CEXTick
from src.layer0_ingestion.chainlink_oracle import ChainlinkOracle
from src.layer0_ingestion.data_store import DataStore
from src.layer0_ingestion.event_bus import EventBus
from src.layer0_ingestion.polymarket_client import PolymarketClient
from src.layer1_research.orchestrator import ResearchOrchestrator
from src.layer2_signal.signal_validator import SignalValidator
from src.layer3_portfolio.correlation_monitor import CorrelationMonitor
from src.layer3_portfolio.platform_risk import PlatformRiskMonitor
from src.layer3_portfolio.portfolio_manager import PortfolioManager
from src.layer3_portfolio.tail_risk import TailRiskAgent
from src.layer4_execution.execution_agent import ExecutionAgent
from src.graph.workflow import ArbitrageNodes, build_arbitrage_graph
from src.web.server import create_app, start_server

logger = structlog.get_logger(__name__)

# ── Logging setup ──

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


class ArbitrageSystem:
    """Top-level orchestrator — wires all components and runs the main loop."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._running = False

        if self._settings.live_trading_enabled:
            logger.critical(
                "system.LIVE_TRADING_MODE",
                msg="LIVE TRADING ENABLED — REAL MONEY AT RISK",
            )
        else:
            logger.info("system.paper_trading_mode")

        # Layer 0
        self._event_bus = EventBus(use_kafka=False)
        self._data_store = DataStore()
        self._polymarket = PolymarketClient()
        self._cex_manager = CEXWebSocketManager()
        self._chainlink = ChainlinkOracle(poll_interval_s=2.0)

        # Layer 1
        self._research = ResearchOrchestrator(
            event_bus=self._event_bus,
            polymarket=self._polymarket,
            cex_manager=self._cex_manager,
        )

        # Layer 2
        from src.layer2_signal.alpha_signal import AlphaSignalGenerator
        alpha_gen = AlphaSignalGenerator(
            wallet_balance_fn=lambda: self._polymarket.wallet_balance
        )
        self._signal_validator = SignalValidator(
            event_bus=self._event_bus, alpha_gen=alpha_gen
        )

        # Layer 3
        self._portfolio = PortfolioManager(
            event_bus=self._event_bus, data_store=self._data_store
        )
        self._tail_risk = TailRiskAgent(event_bus=self._event_bus)
        self._platform_risk = PlatformRiskMonitor()
        self._correlation_monitor = CorrelationMonitor()

        # Layer 4
        self._executor = ExecutionAgent(
            event_bus=self._event_bus,
            polymarket=self._polymarket,
            portfolio=self._portfolio,
        )

        # LangGraph
        self._nodes = ArbitrageNodes(
            event_bus=self._event_bus,
            data_store=self._data_store,
            polymarket=self._polymarket,
            cex_manager=self._cex_manager,
            research_orchestrator=self._research,
            signal_validator=self._signal_validator,
            portfolio_manager=self._portfolio,
            execution_agent=self._executor,
            tail_risk=self._tail_risk,
            platform_risk=self._platform_risk,
            correlation_monitor=self._correlation_monitor,
        )

        self._graph = build_arbitrage_graph(self._nodes)
        self._compiled_graph = self._graph.compile()

    async def start(self) -> None:
        """Initialize all components and start background tasks."""
        logger.info("system.starting")

        # Pre-flight validation for live trading
        if self._settings.live_trading_enabled:
            if not self._settings.polygon_private_key:
                raise RuntimeError(
                    "LIVE_TRADING_ENABLED=true but POLYGON_PRIVATE_KEY is not set."
                )
            if not self._settings.polymarket_api_key:
                raise RuntimeError(
                    "LIVE_TRADING_ENABLED=true but POLYMARKET_API_KEY is not set. "
                    "Run: python scripts/generate_poly_creds.py"
                )

        # Boot Layer 0
        await self._event_bus.start()
        await self._data_store.start()
        await self._polymarket.start()
        await self._platform_risk.start()

        # Check wallet balance and initialize portfolio with real equity
        initial_equity = 100_000.0  # default for paper mode
        if self._settings.live_trading_enabled:
            balance = await self._polymarket.check_wallet_balance()
            if balance > 0:
                initial_equity = balance
                logger.info(
                    "system.wallet_balance",
                    balance_usd=round(balance, 2),
                )
            else:
                logger.warning(
                    "system.NO_FUNDS",
                    msg="Wallet has $0 — bot will enter STANDBY mode",
                )
                initial_equity = 0.0

        await self._portfolio.initialize(initial_equity=initial_equity)

        # Discover markets based on mode
        if self._settings.market_mode == "5min_updown":
            await self._discover_updown_markets()
        else:
            await self._discover_monthly_markets()

        self._running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._cex_manager.start()),
            asyncio.create_task(self._platform_risk.check_loop()),
        ]

        # Start balance polling for live mode
        if self._settings.live_trading_enabled:
            tasks.append(asyncio.create_task(self._balance_poll_loop()))

        # Start market rotation, resolution, and redemption loops for 5-min mode
        if self._settings.market_mode == "5min_updown":
            tasks.append(asyncio.create_task(self._market_rotation_loop()))
            tasks.append(asyncio.create_task(self._resolution_monitor_loop()))

        # Start auto-redemption loop for live mode
        if self._settings.live_trading_enabled:
            tasks.append(asyncio.create_task(self._redemption_loop()))

        # Start auto-retrain loop (every 3 hours)
        tasks.append(asyncio.create_task(self._retrain_loop()))

        # Start Chainlink if configured (skip placeholder URLs)
        chainlink_url = self._settings.chainlink_rpc_url
        if chainlink_url and "YOUR_KEY" not in chainlink_url:
            await self._chainlink.start()
            tasks.append(asyncio.create_task(self._chainlink.poll_loop()))

        # Start PM orderbook streaming using CLOB token IDs
        if self._token_ids:
            tasks.append(
                asyncio.create_task(
                    self._polymarket.stream_orderbook(self._token_ids)
                )
            )
            logger.info(
                "system.orderbook_stream_starting",
                token_ids=len(self._token_ids),
            )

        # Register orderbook callback for research orchestrator
        self._polymarket.on_orderbook_update(
            self._research.process_orderbook
        )

        # Start web dashboard
        if self._settings.dashboard_enabled:
            web_app = create_app(self)
            tasks.append(
                asyncio.create_task(
                    start_server(
                        web_app,
                        host=self._settings.dashboard_host,
                        port=self._settings.dashboard_port,
                    )
                )
            )
            logger.info(
                "system.dashboard_started",
                url=f"http://localhost:{self._settings.dashboard_port}",
            )

        logger.info("system.started", background_tasks=len(tasks))

        # Run main arbitrage loop
        try:
            await self._main_loop()
        finally:
            self._running = False
            await self._shutdown()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _discover_monthly_markets(self) -> None:
        """Discover monthly BTC binary markets (original flow)."""
        try:
            btc_markets = await self._polymarket.fetch_btc_markets()
            active_markets = {}
            token_ids: list[str] = []
            for m in btc_markets[:10]:
                market_id = m.get("conditionId", m.get("condition_id", m.get("id", "")))
                question = m.get("question", "")
                strike = self._parse_strike_from_question(question)
                raw_tokens = m.get("clobTokenIds", m.get("tokens", []))
                if isinstance(raw_tokens, str):
                    try:
                        raw_tokens = json.loads(raw_tokens)
                    except (json.JSONDecodeError, TypeError):
                        raw_tokens = []
                if isinstance(raw_tokens, list):
                    for t in raw_tokens:
                        tid = t if isinstance(t, str) else t.get("token_id", "")
                        if tid:
                            token_ids.append(tid)
                if market_id and strike > 0:
                    active_markets[market_id] = strike
                    neg_risk = m.get("negRisk", m.get("neg_risk", False))
                    if isinstance(neg_risk, str):
                        neg_risk = neg_risk.lower() == "true"
                    if isinstance(raw_tokens, list) and raw_tokens:
                        self._polymarket.set_token_mapping(
                            market_id, raw_tokens, neg_risk=bool(neg_risk)
                        )
                    logger.info(
                        "system.market_added",
                        question=question,
                        strike=strike,
                        market_id=str(market_id)[:20] + "...",
                        token_count=len(raw_tokens) if isinstance(raw_tokens, list) else 0,
                    )
            self._research.set_active_markets(active_markets)
            self._token_ids = token_ids
            logger.info(
                "system.markets_loaded",
                count=len(active_markets),
                token_ids=len(token_ids),
            )
        except Exception as exc:
            logger.warning("system.market_discovery_failed", error=str(exc))
            self._token_ids = []
            self._research.set_active_markets({"default_btc_market": 67500.0})

    async def _discover_updown_markets(self) -> None:
        """Discover upcoming 5-min BTC Up/Down markets and subscribe to orderbooks."""
        try:
            lookahead = self._settings.updown_lookahead_minutes
            markets = await self._polymarket.fetch_updown_markets(
                lookahead_minutes=lookahead
            )

            if not markets:
                logger.warning("system.no_updown_markets_found", lookahead=lookahead)
                self._token_ids = getattr(self, "_token_ids", [])
                return

            updown_map: dict[str, dict] = {}
            new_token_ids: list[str] = []

            for m in markets:
                condition_id = m.get("conditionId", m.get("condition_id", ""))
                if not condition_id:
                    continue

                start_ms = m.get("_start_ms", m.get("start_ms", 0))
                end_ms = m.get("_end_ms", m.get("end_ms", 0))
                question = m.get("question", "")

                raw_tokens = m.get("clobTokenIds", m.get("tokens", []))
                if isinstance(raw_tokens, str):
                    try:
                        raw_tokens = json.loads(raw_tokens)
                    except (json.JSONDecodeError, TypeError):
                        raw_tokens = []

                if isinstance(raw_tokens, list):
                    for t in raw_tokens:
                        tid = t if isinstance(t, str) else t.get("token_id", "")
                        if tid:
                            new_token_ids.append(tid)

                neg_risk = m.get("negRisk", m.get("neg_risk", False))
                if isinstance(neg_risk, str):
                    neg_risk = neg_risk.lower() == "true"
                if isinstance(raw_tokens, list) and raw_tokens:
                    self._polymarket.set_token_mapping(
                        condition_id, raw_tokens, neg_risk=bool(neg_risk)
                    )

                updown_map[condition_id] = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }

                logger.info(
                    "system.updown_market_added",
                    question=question[:60],
                    condition_id=condition_id[:20] + "...",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    tokens=len(raw_tokens) if isinstance(raw_tokens, list) else 0,
                )

            self._research.set_active_updown_markets(updown_map)

            # Subscribe to new token orderbooks
            existing = set(getattr(self, "_token_ids", []))
            truly_new = [t for t in new_token_ids if t not in existing]
            if truly_new:
                asyncio.create_task(
                    self._polymarket.stream_orderbook(truly_new)
                )
                logger.info(
                    "system.updown_stream_started",
                    new_tokens=len(truly_new),
                )

            self._token_ids = list(existing | set(new_token_ids))

            logger.info(
                "system.updown_markets_loaded",
                count=len(updown_map),
                total_tokens=len(self._token_ids),
            )
        except Exception as exc:
            logger.warning("system.updown_discovery_failed", error=str(exc))
            self._token_ids = getattr(self, "_token_ids", [])

    async def _market_rotation_loop(self) -> None:
        """Continuously discover new 5-min Up/Down markets every refresh interval."""
        interval = self._settings.updown_market_refresh_interval_s
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._discover_updown_markets()

                # Prune expired markets (window ended >2 min ago)
                now_ms = int(time.time() * 1000)
                current = self._research._active_updown_markets
                active = {
                    k: v for k, v in current.items()
                    if v.get("end_ms", 0) > now_ms - 120_000
                }
                if len(active) < len(current):
                    pruned = len(current) - len(active)
                    self._research.set_active_updown_markets(active)
                    logger.info("system.updown_pruned", pruned=pruned, remaining=len(active))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("system.rotation_error", error=str(exc))
                await asyncio.sleep(10)

    async def _resolution_monitor_loop(self) -> None:
        """Auto-close positions by checking actual on-chain market resolution.

        Polls the CLOB API for each open position's market status.
        When a market is resolved, determines win/loss from tokens[].winner
        and closes the position with real PnL.
        """
        logger.info("system.resolution_monitor_started")
        while self._running:
            try:
                await asyncio.sleep(15)
                now_ms = int(time.time() * 1000)

                active = [
                    (pid, p) for pid, p in self._portfolio._positions.items()
                    if p.is_active
                ]
                if not active:
                    continue

                logger.info(
                    "system.resolution_check",
                    active_positions=len(active),
                    ids=[pid[:12] for pid, _ in active],
                )

                for pos_id, pos in active:
                    age_s = (now_ms - pos.created_at_ms) / 1000

                    # Give 60s for fill before checking
                    if age_s < 60:
                        continue

                    market_id = pos.market_id

                    # Failsafe: force-close any position older than 15 min.
                    # 5-min markets: trade at T+1, window ends T+5, on-chain
                    # resolution takes 5-8 min → total ~13 min max.
                    # Old 600s timer was too short, causing filled positions
                    # to be closed as pnl=0 before resolution arrived.
                    if age_s > 900:
                        logger.info(
                            "system.force_closing_stale",
                            position_id=pos_id[:16],
                            age_s=round(age_s),
                            market_id=market_id[:20] + "...",
                        )
                        # Try to get resolution for PnL accuracy
                        stale_window_start_ms = (pos.created_at_ms // 300_000) * 300_000
                        resolution = await self._polymarket.check_market_resolution(
                            market_id, window_start_ms=stale_window_start_ms
                        )
                        if resolution and resolution.get("resolved"):
                            winning_outcome = resolution.get("winning_outcome", "")
                            won = (
                                (pos.direction == "BUY_YES" and winning_outcome == "Up")
                                or (pos.direction == "BUY_NO" and winning_outcome == "Down")
                            )
                            if won:
                                fp = pos.fill_price or pos.entry_price
                                shares = pos.size_usd / fp if fp > 0 else 0
                                pnl = shares * 1.0 - pos.size_usd
                                exit_price = 1.0
                            else:
                                pnl = -pos.size_usd
                                exit_price = 0.0
                        else:
                            # Can't determine outcome. For filled positions,
                            # USDC was already spent — record as loss so PnL
                            # tracking doesn't hide real losses as "break-even".
                            # If tokens later win, redemption adds USDC back
                            # to wallet (tracked by wallet-based equity).
                            if pos.fill_price > 0:
                                pnl = -pos.size_usd
                                exit_price = 0.0
                                logger.warning(
                                    "system.stale_filled_unresolved",
                                    position_id=pos_id[:16],
                                    cost=pos.size_usd,
                                    age_s=round(age_s),
                                )
                            else:
                                # Unfilled — no money spent
                                pnl = 0.0
                                exit_price = 0.0

                        try:
                            await self._portfolio.close_position(
                                pos_id, exit_price=exit_price, pnl=round(pnl, 2)
                            )
                            logger.info(
                                "system.stale_position_closed",
                                position_id=pos_id[:16],
                                pnl=round(pnl, 2),
                            )
                        except Exception as e:
                            logger.warning(
                                "system.stale_close_failed",
                                position_id=pos_id[:16],
                                error=str(e),
                            )
                        continue

                    # Normal resolution check
                    # Derive window_start_ms for gamma API fallback
                    # (CLOB API drops resolved 5-min markets)
                    window_start_ms = (pos.created_at_ms // 300_000) * 300_000
                    resolution = await self._polymarket.check_market_resolution(
                        market_id, window_start_ms=window_start_ms
                    )

                    logger.info(
                        "system.resolution_result",
                        position_id=pos_id[:12],
                        age_s=round(age_s),
                        resolution=str(resolution)[:100] if resolution else "None",
                    )

                    if resolution is None:
                        continue

                    if not resolution.get("resolved", False):
                        # Not resolved yet — check if stale unfilled
                        if pos.fill_price == 0 and age_s > 600:
                            try:
                                await self._portfolio.close_position(
                                    pos_id, exit_price=0.0, pnl=0.0
                                )
                            except Exception:
                                pass
                        continue

                    # Market resolved — determine win/loss
                    winning_outcome = resolution.get("winning_outcome", "")
                    won = (
                        (pos.direction == "BUY_YES" and winning_outcome == "Up")
                        or (pos.direction == "BUY_NO" and winning_outcome == "Down")
                    )

                    if won:
                        fp = pos.fill_price or pos.entry_price
                        shares = pos.size_usd / fp if fp > 0 else 0
                        payout = shares * 1.0
                        pnl = payout - pos.size_usd
                        exit_price = 1.0
                    else:
                        pnl = -pos.size_usd
                        exit_price = 0.0

                    logger.info(
                        "system.position_resolved",
                        position_id=pos_id[:16],
                        market_id=market_id[:20] + "...",
                        direction=pos.direction,
                        outcome=winning_outcome,
                        won=won,
                        pnl=round(pnl, 2),
                    )

                    try:
                        await self._portfolio.close_position(
                            pos_id, exit_price=exit_price, pnl=round(pnl, 2)
                        )
                    except Exception as close_exc:
                        logger.warning(
                            "system.auto_close_failed",
                            position_id=pos_id[:16],
                            error=str(close_exc),
                        )

                    # Immediately redeem winning positions (don't wait for data-api scan)
                    if won and self._settings.live_trading_enabled:
                        try:
                            tx = await self._polymarket.redeem_winning_position(market_id)
                            if tx:
                                logger.info(
                                    "system.immediate_redeem_success",
                                    condition_id=market_id[:20] + "...",
                                    tx_hash=tx,
                                )
                                await self._polymarket.check_wallet_balance()
                        except Exception as redeem_exc:
                            logger.debug(
                                "system.immediate_redeem_failed",
                                condition_id=market_id[:20] + "...",
                                error=str(redeem_exc),
                            )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("system.resolution_error", error=str(exc))
                await asyncio.sleep(10)

    async def _redemption_loop(self) -> None:
        """Auto-redeem winning positions via CTF contract.

        Polls Gamma API for redeemable positions every 30s.
        For each redeemable position, calls redeemPositions on the CTF
        through the Polymarket proxy wallet.
        """
        # Track redeemed condition_ids to avoid duplicate attempts
        redeemed: set[str] = set()

        while self._running:
            try:
                await asyncio.sleep(30)

                positions = await self._polymarket.fetch_redeemable_positions()
                if not positions:
                    continue

                # Filter to only winning positions ($0 positions waste gas)
                winners = [
                    p for p in positions
                    if float(p.get("curPrice", 0) or 0) > 0
                    and float(p.get("currentValue", 0) or 0) > 0
                ]
                if not winners:
                    continue

                for pos in winners:
                    condition_id = pos.get("conditionId", pos.get("condition_id", ""))
                    if not condition_id or condition_id in redeemed:
                        continue

                    title = pos.get("title", "")[:40]
                    size = pos.get("size", 0)

                    logger.info(
                        "system.redeeming_position",
                        condition_id=condition_id[:20] + "...",
                        title=title,
                        size=size,
                    )

                    tx_hash = await self._polymarket.redeem_winning_position(
                        condition_id
                    )

                    if tx_hash:
                        redeemed.add(condition_id)
                        logger.info(
                            "system.redeem_success",
                            condition_id=condition_id[:20] + "...",
                            tx_hash=tx_hash,
                        )
                        # Refresh wallet balance after redemption
                        await self._polymarket.check_wallet_balance()
                    else:
                        logger.debug(
                            "system.redeem_skipped",
                            condition_id=condition_id[:20] + "...",
                            reason="no_matic_or_proxy_issue",
                        )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("system.redemption_error", error=str(exc))
                await asyncio.sleep(30)

    async def _retrain_loop(self) -> None:
        """Periodically retrain the ML model with live trade data every 3 hours."""
        RETRAIN_INTERVAL_S = 3 * 60 * 60  # 3 hours
        MIN_LIVE_TRADES = 10
        logger.info("system.retrain_loop_started", interval_h=3)

        # Wait 2 min before first retrain check
        await asyncio.sleep(2 * 60)

        while self._running:
            try:
                import pandas as pd
                csv_path = "data/live_trades.csv"

                # Check if we have enough live trades
                if not os.path.exists(csv_path):
                    logger.info("system.retrain_skip", reason="no_csv")
                    await asyncio.sleep(RETRAIN_INTERVAL_S)
                    continue

                df = pd.read_csv(csv_path)
                completed = df[df["outcome"].notna() & (df["outcome"] != "")]
                if len(completed) < MIN_LIVE_TRADES:
                    logger.info(
                        "system.retrain_skip",
                        reason="too_few_trades",
                        trades=len(completed),
                        needed=MIN_LIVE_TRADES,
                    )
                    await asyncio.sleep(RETRAIN_INTERVAL_S)
                    continue

                logger.info(
                    "system.retrain_starting",
                    live_trades=len(completed),
                )

                # Run retrain in a thread to avoid blocking the event loop
                import subprocess
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "scripts/retrain_model.py",
                    "--months", "3",
                    "--min-live-trades", str(MIN_LIVE_TRADES),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    logger.info(
                        "system.retrain_complete",
                        output=stdout.decode()[-200:] if stdout else "",
                    )
                    # Hot-reload the model in the momentum detector
                    try:
                        self._research._momentum_detector._predictor.reload()
                        logger.info("system.model_hot_reloaded")
                    except Exception as e:
                        logger.warning("system.model_reload_failed", error=str(e))
                else:
                    logger.warning(
                        "system.retrain_failed",
                        returncode=proc.returncode,
                        stderr=stderr.decode()[-300:] if stderr else "",
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("system.retrain_error", error=str(exc))

            await asyncio.sleep(RETRAIN_INTERVAL_S)

    async def _balance_poll_loop(self) -> None:
        """Periodically check wallet balance and update standby state."""
        while self._running:
            try:
                balance = await self._polymarket.check_wallet_balance()
                # Sync portfolio equity to real wallet balance
                if balance > 0:
                    self._portfolio.update_wallet_balance(balance)
                    if self._portfolio._initial_equity == 0:
                        await self._portfolio.initialize(initial_equity=balance)
                        logger.info(
                            "system.equity_updated_from_wallet",
                            balance_usd=round(balance, 2),
                        )
            except Exception as exc:
                logger.debug("system.balance_poll_error", error=str(exc))
            await asyncio.sleep(30)  # check every 30 seconds

    async def _main_loop(self) -> None:
        """Main arbitrage loop — invokes LangGraph on each tick cycle."""
        logger.info("system.main_loop_started")
        cycle = 0

        while self._running:
            try:
                cycle += 1

                cycle_start = time.time() * 1000

                # Run one cycle of the LangGraph pipeline
                initial_state: dict[str, Any] = {}
                result = await self._compiled_graph.ainvoke(initial_state)

                # Log periodic stats every 100 cycles
                if cycle % 100 == 0:
                    await self._log_stats(cycle)

                # Adaptive throttle: target ~20 cycles/second (50ms per cycle)
                elapsed_ms = time.time() * 1000 - cycle_start
                target_ms = 50
                sleep_s = max(0, (target_ms - elapsed_ms) / 1000)
                await asyncio.sleep(sleep_s)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                import traceback
                logger.error("system.cycle_error", cycle=cycle, error=str(exc), tb=traceback.format_exc())
                await asyncio.sleep(1)

    async def _log_stats(self, cycle: int) -> None:
        """Periodic stats logging."""
        portfolio = await self._portfolio.get_portfolio_state()
        fill_metrics = self._executor.fill_monitor.get_metrics()
        bt = self._signal_validator.backtester.latest_result

        logger.info(
            "system.stats",
            cycle=cycle,
            positions=portfolio["active_positions"],
            equity=portfolio["current_equity"],
            wallet_balance=round(self._polymarket.wallet_balance, 2),
            standby=self._polymarket.is_standby,
            daily_pnl=portfolio["daily_pnl"],
            drawdown=portfolio["drawdown_pct"],
            halted=portfolio["halted"],
            fill_rate=fill_metrics["fill_rate"],
            avg_slippage=fill_metrics["avg_slippage_bps"],
            bt_win_rate=bt.win_rate if bt else "N/A",
            trade_rate=self._signal_validator.get_trade_rate(),
        )

    async def _shutdown(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("system.shutting_down")
        await self._polymarket.stop()
        await self._cex_manager.stop()
        if self._settings.chainlink_rpc_url and "YOUR_KEY" not in self._settings.chainlink_rpc_url:
            await self._chainlink.stop()
        await self._platform_risk.stop()
        await self._event_bus.stop()
        await self._data_store.stop()
        logger.info("system.stopped")

    @staticmethod
    def _parse_strike_from_question(question: str) -> float:
        """Extract strike price from market question.
        E.g., 'Will BTC be above $67,500 at 12:05 UTC?' → 67500.0
        """
        import re

        match = re.search(r"\$?([\d,]+(?:\.\d+)?)", question)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                return 0.0
        return 0.0


def main() -> None:
    """Entry point."""
    # Use uvloop for better async performance
    uvloop.install()

    # Create event loop first (Python 3.9 requires a running loop for asyncio.Queue)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    system = ArbitrageSystem()

    # Handle graceful shutdown

    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("system.signal_received", signal=sig)
        system._running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(system.start())
    except KeyboardInterrupt:
        logger.info("system.keyboard_interrupt")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
