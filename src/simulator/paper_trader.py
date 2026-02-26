"""Paper Trader — runs the full arbitrage pipeline against simulated or live markets
with terminal dashboard + optional web dashboard.

Supports two modes:
- Synthetic: MarketSimulator generates fake GBM prices (default)
- Live: Real CEX WebSocket feeds + synthetic Polymarket orderbooks
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import structlog

from config.settings import get_settings
from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel
from src.layer0_ingestion.event_bus import EventBus, Event, EventType
from src.layer1_research.spread_detector import SpreadDetector, SpreadOpportunity
from src.layer1_research.latency_arb import LatencyArbDetector
from src.layer1_research.liquidity_scanner import LiquidityScanner
from src.layer1_research.research_synthesis import ResearchSynthesis, ResearchOutput
from src.layer2_signal.alpha_signal import AlphaSignalGenerator, AlphaSignal
from src.layer2_signal.backtester import Backtester, TradeRecord
from src.layer2_signal.risk_filter import RiskFilter
from src.layer2_signal.signal_validator import SignalValidator, ValidatedSignal
from src.layer4_execution.order_book_sniper import OrderBookSniper
from src.layer4_execution.fill_monitor import FillMonitor, FillStatus
from src.simulator.market_simulator import MarketSimulator, MarketConfig

logger = structlog.get_logger(__name__)


@dataclass
class PaperPosition:
    id: str
    market_id: str
    strike: float
    direction: str
    entry_price: float
    size_usd: float
    entry_btc_price: float
    opened_at: float
    expires_at: float  # 5-min market resolution time
    pnl: float = 0.0
    status: str = "open"
    exit_price: float = 0.0
    resolved: bool = False


@dataclass
class PaperTradeStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    pnl_history: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_profit(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0


class PaperTrader:
    """Full paper trading engine with simulated execution and live dashboard.

    Can run in two modes:
    - Synthetic: pass a MarketConfig (default) → uses MarketSimulator
    - Live: pass a LiveMarketFeed → uses real CEX prices
    """

    TICK_INTERVAL_S = 0.1  # 100ms per tick (10 ticks/sec)
    MARKET_DURATION_S = 300.0  # 5-min markets
    INITIAL_EQUITY = 100_000.0

    def __init__(
        self,
        config: Optional[MarketConfig] = None,
        live_feed: Any = None,
        show_terminal: bool = True,
    ) -> None:
        self._settings = get_settings()
        self._live_mode = live_feed is not None
        self._show_terminal = show_terminal

        if live_feed is not None:
            self._feed = live_feed
        else:
            self._feed = MarketSimulator(config)

        self._event_bus = EventBus(use_kafka=False)

        # Research agents
        self._spread_detector = SpreadDetector()
        self._latency_detector = LatencyArbDetector()
        self._liquidity_scanner = LiquidityScanner()
        self._synthesis = ResearchSynthesis()
        # Tune Bayesian prior up for paper trading so signals actually fire
        self._synthesis.BASE_PRIOR = 0.55
        self._synthesis.SPREAD_WEIGHT = 0.50
        self._synthesis.LATENCY_WEIGHT = 0.30
        self._synthesis.LIQUIDITY_WEIGHT = 0.20
        self._synthesis._spread_tp_rate = 0.85
        self._synthesis._latency_tp_rate = 0.80
        self._synthesis._liquidity_tp_rate = 0.90

        # Signal pipeline
        self._alpha_gen = AlphaSignalGenerator()
        self._backtester = Backtester()
        self._risk_filter = RiskFilter()
        self._signal_validator = SignalValidator(
            event_bus=self._event_bus,
            alpha_gen=self._alpha_gen,
            backtester=self._backtester,
            risk_filter=self._risk_filter,
        )

        # Execution
        self._sniper = OrderBookSniper()
        self._fill_monitor = FillMonitor()

        # Portfolio state
        self._equity = self.INITIAL_EQUITY
        self._positions: dict[str, PaperPosition] = {}
        self._stats = PaperTradeStats()
        self._stats.peak_equity = self._equity
        self._daily_pnl = 0.0

        # Dashboard state
        self._last_signals: deque[str] = deque(maxlen=12)
        self._last_trades: deque[str] = deque(maxlen=8)
        self._running = False
        self._start_time = 0.0
        self._market_epoch = 0  # which 5-min market round we're in
        self._next_market_time = 0.0

        # Trade logging
        self._data_dir = Path(__file__).parent.parent.parent / "data"
        self._data_dir.mkdir(exist_ok=True)
        self._trade_log_path = self._data_dir / "trade_history.jsonl"
        self._signal_log_path = self._data_dir / "signal_history.jsonl"
        self._params_path = self._data_dir / "tuned_params.json"
        self._params_mtime: float = 0.0  # for hot-reload detection
        self._last_param_check: float = 0.0

    async def run(self) -> None:
        """Main paper trading loop."""
        await self._event_bus.start()
        self._running = True
        self._start_time = time.time()
        self._next_market_time = self._start_time + self.MARKET_DURATION_S

        # Initialize risk filter
        self._risk_filter.update_state(
            positions=[], daily_pnl=0, equity=self._equity
        )

        mode_label = "LIVE CEX DATA" if self._live_mode else "SYNTHETIC DATA"

        if self._show_terminal:
            print("\033[2J\033[H")  # clear screen
            print("=" * 72)
            print(f"  POLYMARKET 5-MIN BTC ARBITRAGE — PAPER TRADING ({mode_label})")
            print("=" * 72)

            if self._live_mode:
                print("  Waiting for real CEX price feeds...")
                # Wait for first real tick — call tick() to pull from CEX manager
                for _ in range(50):  # up to 5 seconds
                    self._feed.tick()  # pull latest CEX data
                    if self._feed.current_btc_price > 0:
                        break
                    await asyncio.sleep(0.1)
                if self._feed.current_btc_price > 0:
                    print(f"  Live BTC: ${self._feed.current_btc_price:,.2f}")
                    print(f"  Active strikes: {list(self._feed.market_ids.values())}")
                else:
                    print("  (still waiting for CEX data...)")
            else:
                print(f"  Starting BTC: ${self._feed.current_btc_price:,.2f}")

            print(f"  Initial equity: ${self._equity:,.2f}")
            print("=" * 72)
            await asyncio.sleep(2)

        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self.TICK_INTERVAL_S)
        except KeyboardInterrupt:
            pass
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self._event_bus.stop()
            if self._show_terminal:
                self._print_final_report()

    async def _tick(self) -> None:
        """One tick — full pipeline (works with both synthetic and live feed)."""
        now = time.time()

        # Check if 5-min market round is over → resolve positions
        if now >= self._next_market_time:
            await self._resolve_markets()
            self._market_epoch += 1
            self._next_market_time = now + self.MARKET_DURATION_S

        # 1. Get market data (synthetic or live)
        cex_ticks, orderbooks = self._feed.tick()

        # Skip if no data yet (live mode waiting for first tick)
        if not cex_ticks or not orderbooks:
            return

        # Pick best CEX tick
        best_cex = max(cex_ticks, key=lambda t: t.bid)

        # Record for latency detector
        self._latency_detector.record_cex_tick(best_cex)

        # Broadcast CEX tick event for web dashboard
        if self._live_mode:
            await self._event_bus.publish(Event(
                event_type=EventType.PRICE_TICK,
                data={
                    "exchange": best_cex.exchange.value if hasattr(best_cex.exchange, 'value') else str(best_cex.exchange),
                    "bid": best_cex.bid,
                    "ask": best_cex.ask,
                    "mid": best_cex.mid,
                    "last": best_cex.last,
                },
                source="paper_trader",
            ))

        # 2. Run research on each market
        best_research: Optional[ResearchOutput] = None
        best_ob: Optional[OrderBook] = None

        for market_id, strike in self._feed.market_ids.items():
            ob = orderbooks.get(market_id)
            if not ob:
                continue

            # Update latency tracker
            self._latency_detector.record_pm_update(
                market_id, int(time.time() * 1000)
            )

            # Run research agents
            spread_opp = self._spread_detector.detect(ob, best_cex, strike)
            latency_sig = self._latency_detector.detect(ob, best_cex, strike)
            liquidity_prof = self._liquidity_scanner.scan(ob)

            # Synthesize
            research = self._synthesis.synthesize(
                spread_opp, latency_sig, liquidity_prof
            )

            if research and research.is_actionable:
                if best_research is None or research.combined_probability > best_research.combined_probability:
                    best_research = research
                    best_ob = ob

                self._last_signals.append(
                    f"  [{_ts()}] SIGNAL {market_id} | "
                    f"dir={research.direction} edge={research.edge_pct:.1f}% "
                    f"conf={research.combined_probability:.0%}"
                )

        # 3. Validate and execute best signal
        if best_research and best_ob:
            self._log_signal(best_research, validated=True)
            await self._process_signal(best_research, best_ob)

        # 4. Update unrealized PnL
        self._update_positions(self._feed.current_btc_price)

        # 5. Update risk filter state
        positions_data = [
            {"direction": p.direction, "market_id": p.market_id}
            for p in self._positions.values()
        ]
        self._risk_filter.update_state(
            positions=positions_data,
            daily_pnl=self._daily_pnl,
            equity=self._equity,
        )

        # 6. Check for parameter hot-reload from auto-tuner
        self._check_hot_reload()

        # 7. Render terminal dashboard every 5 ticks
        if self._show_terminal and self._feed.tick_count % 5 == 0:
            self._render_dashboard()

    async def _process_signal(self, research: ResearchOutput, ob: OrderBook) -> None:
        """Validate signal and execute paper trade if approved."""
        # Check position limits
        if len(self._positions) >= self._settings.max_concurrent_positions:
            return

        # Don't trade same market twice
        if any(p.market_id == research.market_id for p in self._positions.values()):
            return

        # Validate through full pipeline
        validated = await self._signal_validator.validate(research)

        if not validated.is_trade:
            return

        # Execute paper trade
        strike = self._feed.market_ids.get(research.market_id, 0)
        entry_price = validated.signal.entry_price
        size_usd = validated.final_size_usd

        pos_id = f"P{self._stats.total_trades + 1:04d}"
        position = PaperPosition(
            id=pos_id,
            market_id=research.market_id,
            strike=strike,
            direction=validated.signal.direction,
            entry_price=entry_price,
            size_usd=size_usd,
            entry_btc_price=self._feed.current_btc_price,
            opened_at=time.time(),
            expires_at=self._next_market_time,
        )
        self._positions[pos_id] = position

        self._last_trades.append(
            f"  [{_ts()}] \033[92mOPEN\033[0m {pos_id} | {research.direction} "
            f"${strike:,.0f} @ {entry_price:.4f} | "
            f"size=${size_usd:,.0f} edge={research.edge_pct:.1f}%"
        )

    async def _resolve_markets(self) -> None:
        """Resolve all open positions based on final BTC price vs strike."""
        final_price = self._feed.current_btc_price

        for pos_id, pos in list(self._positions.items()):
            if pos.status != "open":
                continue

            # Determine outcome
            btc_above_strike = final_price > pos.strike

            if pos.direction == "BUY_YES":
                won = btc_above_strike
            else:
                won = not btc_above_strike

            # PnL calculation
            if won:
                # Payout is $1 per contract, cost was entry_price
                pnl = pos.size_usd * ((1.0 / pos.entry_price) - 1.0)
                # Cap realistic profit
                pnl = min(pnl, pos.size_usd * 1.5)
            else:
                pnl = -pos.size_usd  # lose entire stake

            pos.pnl = round(pnl, 2)
            pos.status = "closed"
            pos.resolved = True
            pos.exit_price = 1.0 if won else 0.0

            # Update stats
            self._stats.total_trades += 1
            if won:
                self._stats.wins += 1
            else:
                self._stats.losses += 1
            self._stats.total_pnl += pnl
            self._equity += pnl
            self._daily_pnl += pnl

            if self._equity > self._stats.peak_equity:
                self._stats.peak_equity = self._equity
            dd = (self._stats.peak_equity - self._equity) / self._stats.peak_equity
            if dd > self._stats.max_drawdown_pct:
                self._stats.max_drawdown_pct = dd

            self._stats.pnl_history.append(pnl)

            # Record for backtester
            self._backtester.record_trade(TradeRecord(
                market_id=pos.market_id,
                direction=pos.direction,
                entry_price=pos.entry_price,
                size_usd=pos.size_usd,
                pnl_usd=pnl,
                won=won,
                edge_pct=0,
            ))

            color = "\033[92m" if won else "\033[91m"
            self._last_trades.append(
                f"  [{_ts()}] {color}{'WIN ' if won else 'LOSS'}\033[0m {pos_id} | "
                f"BTC=${final_price:,.0f} vs ${pos.strike:,.0f} | "
                f"PnL={color}${pnl:+,.2f}\033[0m"
            )

            # Log trade to file
            self._log_trade(pos, won, final_price)

            # Feedback to synthesis
            if hasattr(self, '_last_research'):
                self._synthesis.update_likelihoods(won, self._last_research)

        # Clear closed positions
        self._positions = {
            k: v for k, v in self._positions.items() if v.status == "open"
        }

    def _log_trade(self, pos: PaperPosition, won: bool, final_price: float) -> None:
        """Append trade result to JSONL log file."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch_ms": int(time.time() * 1000),
            "pos_id": pos.id,
            "market_id": pos.market_id,
            "strike": pos.strike,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "size_usd": pos.size_usd,
            "entry_btc": pos.entry_btc_price,
            "exit_btc": final_price,
            "pnl": pos.pnl,
            "won": won,
            "market_epoch": self._market_epoch,
            "equity_after": round(self._equity, 2),
            "win_rate": round(self._stats.win_rate, 4),
            "total_trades": self._stats.total_trades,
            "synthesis_params": {
                "base_prior": self._synthesis.BASE_PRIOR,
                "spread_weight": self._synthesis.SPREAD_WEIGHT,
                "latency_weight": self._synthesis.LATENCY_WEIGHT,
                "liquidity_weight": self._synthesis.LIQUIDITY_WEIGHT,
                "spread_tp": self._synthesis._spread_tp_rate,
                "latency_tp": self._synthesis._latency_tp_rate,
                "liquidity_tp": self._synthesis._liquidity_tp_rate,
            },
        }
        try:
            with open(self._trade_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _log_signal(self, research: ResearchOutput, validated: bool) -> None:
        """Append signal to JSONL log file."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch_ms": int(time.time() * 1000),
            "market_id": research.market_id,
            "direction": research.direction,
            "edge_pct": research.edge_pct,
            "combined_prob": research.combined_probability,
            "confidence": research.confidence,
            "btc_price": self._feed.current_btc_price,
            "pm_price": self._feed.current_pm_price,
            "validated": validated,
        }
        try:
            with open(self._signal_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _check_hot_reload(self) -> None:
        """Check if tuned_params.json was updated by the auto-tuner and reload."""
        now = time.time()
        if now - self._last_param_check < 10.0:  # check every 10 seconds
            return
        self._last_param_check = now

        try:
            if not self._params_path.exists():
                return
            mtime = self._params_path.stat().st_mtime
            if mtime <= self._params_mtime:
                return
            self._params_mtime = mtime

            with open(self._params_path) as f:
                params = json.load(f)

            # Apply synthesis parameters
            synth = params.get("synthesis", {})
            if "base_prior" in synth:
                self._synthesis.BASE_PRIOR = synth["base_prior"]
            if "spread_weight" in synth:
                self._synthesis.SPREAD_WEIGHT = synth["spread_weight"]
            if "latency_weight" in synth:
                self._synthesis.LATENCY_WEIGHT = synth["latency_weight"]
            if "liquidity_weight" in synth:
                self._synthesis.LIQUIDITY_WEIGHT = synth["liquidity_weight"]
            if "spread_tp_rate" in synth:
                self._synthesis._spread_tp_rate = synth["spread_tp_rate"]
            if "latency_tp_rate" in synth:
                self._synthesis._latency_tp_rate = synth["latency_tp_rate"]
            if "liquidity_tp_rate" in synth:
                self._synthesis._liquidity_tp_rate = synth["liquidity_tp_rate"]

            logger.info(
                "paper_trader.params_reloaded",
                prior=self._synthesis.BASE_PRIOR,
                spread_w=self._synthesis.SPREAD_WEIGHT,
                spread_tp=self._synthesis._spread_tp_rate,
            )
        except Exception:
            pass

    def _update_positions(self, current_price: float) -> None:
        """Mark-to-market all open positions."""
        for pos in self._positions.values():
            if pos.status != "open":
                continue
            distance = (current_price - pos.strike) / pos.strike
            import math
            current_prob = 1.0 / (1.0 + math.exp(-50 * distance))
            if pos.direction == "BUY_YES":
                pos.pnl = (current_prob - pos.entry_price) * pos.size_usd
            else:
                pos.pnl = (pos.entry_price - current_prob) * pos.size_usd

    def _render_dashboard(self) -> None:
        """Render live terminal dashboard."""
        elapsed = time.time() - self._start_time
        time_to_resolve = max(0, self._next_market_time - time.time())

        btc = self._feed.current_btc_price
        pm = self._feed.current_pm_price
        if btc <= 0:
            return  # no data yet
        lag_pct = abs(btc - pm) / btc * 100

        unrealized = sum(p.pnl for p in self._positions.values())
        total_equity = self._equity + unrealized

        wr = self._stats.win_rate
        wr_color = "\033[92m" if wr >= 0.7 else ("\033[93m" if wr >= 0.5 else "\033[91m")
        pnl_color = "\033[92m" if self._stats.total_pnl >= 0 else "\033[91m"
        unr_color = "\033[92m" if unrealized >= 0 else "\033[91m"

        mode_label = "LIVE" if self._live_mode else "SIM"
        lines: list[str] = []
        lines.append("\033[H")  # move cursor to top
        lines.append("=" * 72)
        lines.append(f"  \033[1mPOLYMARKET 5-MIN BTC ARBITRAGE — PAPER TRADING ({mode_label})\033[0m")
        lines.append("=" * 72)
        lines.append("")

        # Market data
        regime_info = ""
        if not self._live_mode and hasattr(self._feed, '_regime'):
            regime_info = f"   Regime: {self._feed._regime}"
        lines.append(f"  \033[1mMARKET DATA\033[0m      Elapsed: {_fmt_duration(elapsed)}   "
                     f"Market round: #{self._market_epoch + 1}   "
                     f"Resolves in: {_fmt_duration(time_to_resolve)}")
        lines.append(f"  BTC/USD (CEX):   \033[1m${btc:>10,.2f}\033[0m")
        lines.append(f"  BTC/USD (PM):    \033[1m${pm:>10,.2f}\033[0m    "
                     f"Lag: {lag_pct:.3f}%{regime_info}")
        lines.append(f"  Ticks:           {self._feed.tick_count:>10,}")
        lines.append("")

        # Portfolio
        lines.append(f"  \033[1mPORTFOLIO\033[0m")
        lines.append(f"  Equity:          \033[1m${total_equity:>10,.2f}\033[0m   "
                     f"(initial: ${self.INITIAL_EQUITY:,.0f})")
        lines.append(f"  Realized PnL:    {pnl_color}${self._stats.total_pnl:>+10,.2f}\033[0m")
        lines.append(f"  Unrealized PnL:  {unr_color}${unrealized:>+10,.2f}\033[0m")
        lines.append(f"  Max Drawdown:    {self._stats.max_drawdown_pct:>10.2%}")
        lines.append(f"  Daily PnL:       ${self._daily_pnl:>+10,.2f}")
        lines.append("")

        # Performance
        lines.append(f"  \033[1mPERFORMANCE\033[0m")
        lines.append(f"  Trades:          {self._stats.total_trades:>10}   "
                     f"({self._stats.wins}W / {self._stats.losses}L)")
        lines.append(f"  Win Rate:        {wr_color}{wr:>10.1%}\033[0m   "
                     f"(target: >70%)")
        lines.append(f"  Avg Profit:      ${self._stats.avg_profit:>+10,.2f}   "
                     f"(target: >$15)")
        lines.append(f"  Open Positions:  {len(self._positions):>10}   "
                     f"(max: {self._settings.max_concurrent_positions})")
        lines.append("")

        # PnL sparkline
        if self._stats.pnl_history:
            spark = _sparkline(self._stats.pnl_history[-40:])
            lines.append(f"  \033[1mPnL HISTORY\033[0m  {spark}")
            lines.append("")

        # Open positions
        lines.append(f"  \033[1mOPEN POSITIONS\033[0m")
        if self._positions:
            for pos in self._positions.values():
                pc = "\033[92m" if pos.pnl >= 0 else "\033[91m"
                ttl = max(0, pos.expires_at - time.time())
                lines.append(
                    f"    {pos.id} | {pos.direction:8s} ${pos.strike:>8,.0f} "
                    f"@ {pos.entry_price:.4f} | size=${pos.size_usd:>6,.0f} | "
                    f"PnL={pc}${pos.pnl:>+8,.2f}\033[0m | TTL={ttl:.0f}s"
                )
        else:
            lines.append("    (none)")
        lines.append("")

        # Recent signals
        lines.append(f"  \033[1mRECENT SIGNALS\033[0m")
        for s in list(self._last_signals)[-5:]:
            lines.append(s)
        if not self._last_signals:
            lines.append("    (scanning...)")
        lines.append("")

        # Recent trades
        lines.append(f"  \033[1mTRADE LOG\033[0m")
        for t in list(self._last_trades)[-6:]:
            lines.append(t)
        if not self._last_trades:
            lines.append("    (no trades yet)")
        lines.append("")
        lines.append("─" * 72)
        lines.append("  Press Ctrl+C to stop paper trading")
        lines.append("")

        # Pad to fill screen (prevent old content showing)
        while len(lines) < 50:
            lines.append(" " * 72)

        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()

    def _print_final_report(self) -> None:
        """Print final paper trading results."""
        print("\033[2J\033[H")  # clear
        elapsed = time.time() - self._start_time
        print("=" * 72)
        print("  PAPER TRADING SESSION — FINAL REPORT")
        print("=" * 72)
        print(f"  Duration:        {_fmt_duration(elapsed)}")
        print(f"  Ticks:           {self._feed.tick_count:,}")
        print(f"  Market rounds:   {self._market_epoch + 1}")
        print()
        print(f"  Starting equity: ${self.INITIAL_EQUITY:>12,.2f}")
        print(f"  Final equity:    ${self._equity:>12,.2f}")
        print(f"  Total PnL:       ${self._stats.total_pnl:>+12,.2f}")
        print(f"  Max Drawdown:    {self._stats.max_drawdown_pct:>12.2%}")
        print()
        print(f"  Total trades:    {self._stats.total_trades}")
        print(f"  Wins / Losses:   {self._stats.wins} / {self._stats.losses}")
        print(f"  Win rate:        {self._stats.win_rate:.1%}")
        print(f"  Avg profit:      ${self._stats.avg_profit:+,.2f}")
        print()

        if self._stats.pnl_history:
            pnls = np.array(self._stats.pnl_history)
            print(f"  Best trade:      ${np.max(pnls):+,.2f}")
            print(f"  Worst trade:     ${np.min(pnls):+,.2f}")
            if np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(288)  # annualized
                print(f"  Sharpe (est):    {sharpe:.2f}")

        print()
        if self._stats.pnl_history:
            print(f"  PnL curve: {_sparkline(self._stats.pnl_history)}")
        print()
        print("=" * 72)


# ── Helper functions ──

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _sparkline(values: list[float]) -> str:
    """Render a unicode sparkline for PnL history."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(
        blocks[min(len(blocks) - 1, int((v - mn) / rng * (len(blocks) - 1)))]
        for v in values
    )
