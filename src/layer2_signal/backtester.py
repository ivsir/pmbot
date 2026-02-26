"""Backtester — 30-day rolling validation, requires >70% win rate to pass."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from config.settings import get_settings
from src.layer2_signal.alpha_signal import AlphaSignal

logger = structlog.get_logger(__name__)


@dataclass
class TradeRecord:
    """Historical trade result for backtesting."""

    market_id: str
    direction: str
    entry_price: float
    size_usd: float
    pnl_usd: float
    won: bool
    edge_pct: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class BacktestResult:
    """Rolling backtest statistics."""

    window_days: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    is_valid: bool  # meets >70% win rate threshold
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "is_valid": self.is_valid,
        }


class Backtester:
    """Rolling 30-day backtester — validates strategy performance in real-time.

    Maintains a sliding window of recent trades and computes:
    - Win rate (must be >70%)
    - Sharpe ratio (target >2.5)
    - Max drawdown (must be <5%)
    - Expectancy per trade (target >$15)
    """

    WINDOW_DAYS = 30
    MIN_TRADES = 20  # minimum trades before validation kicks in

    def __init__(self) -> None:
        self._settings = get_settings()
        self._trades: deque[TradeRecord] = deque(maxlen=10_000)
        self._latest_result: BacktestResult | None = None

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade."""
        self._trades.append(trade)
        logger.debug(
            "backtester.trade_recorded",
            market=trade.market_id,
            pnl=trade.pnl_usd,
            won=trade.won,
        )

    def record_from_signal(
        self,
        signal: AlphaSignal,
        pnl_usd: float,
        won: bool,
    ) -> None:
        """Convenience: record trade from signal + outcome."""
        self._trades.append(
            TradeRecord(
                market_id=signal.market_id,
                direction=signal.direction,
                entry_price=signal.entry_price,
                size_usd=signal.optimal_size_usd,
                pnl_usd=pnl_usd,
                won=won,
                edge_pct=signal.edge_pct,
            )
        )

    def validate(self) -> BacktestResult:
        """Compute rolling backtest statistics over the last 30 days."""
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (self.WINDOW_DAYS * 86_400_000)

        window_trades = [t for t in self._trades if t.timestamp_ms >= cutoff_ms]

        if not window_trades:
            return self._empty_result()

        wins = [t for t in window_trades if t.won]
        losses = [t for t in window_trades if not t.won]

        total = len(window_trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0

        pnls = [t.pnl_usd for t in window_trades]
        avg_profit = np.mean([t.pnl_usd for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_usd for t in losses]) if losses else 0
        total_pnl = sum(pnls)

        # Sharpe ratio (annualized, assuming ~288 5-min periods per day)
        if len(pnls) > 1 and np.std(pnls) > 0:
            daily_returns = self._aggregate_daily_returns(window_trades)
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(365)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        max_dd = self._compute_max_drawdown(pnls)

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = total_pnl / total if total > 0 else 0

        # Validation: need enough trades and >70% win rate
        is_valid = (
            total >= self.MIN_TRADES
            and win_rate >= self._settings.target_win_rate
        )

        result = BacktestResult(
            window_days=self.WINDOW_DAYS,
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            win_rate=round(win_rate, 4),
            avg_profit=round(float(avg_profit), 2),
            avg_loss=round(float(avg_loss), 2),
            total_pnl=round(total_pnl, 2),
            sharpe_ratio=round(float(sharpe), 3),
            max_drawdown_pct=round(max_dd, 4),
            profit_factor=round(profit_factor, 3),
            expectancy=round(expectancy, 2),
            is_valid=is_valid,
        )

        self._latest_result = result

        logger.info(
            "backtester.validated",
            trades=total,
            win_rate=round(win_rate, 3),
            sharpe=round(float(sharpe), 3),
            pnl=round(total_pnl, 2),
            valid=is_valid,
        )

        return result

    @property
    def latest_result(self) -> BacktestResult | None:
        return self._latest_result

    @property
    def is_strategy_valid(self) -> bool:
        """Quick check: is the strategy currently passing backtest validation."""
        if self._latest_result is None:
            # No backtest yet — allow trading (will validate after MIN_TRADES)
            return len(self._trades) < self.MIN_TRADES
        return self._latest_result.is_valid

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            window_days=self.WINDOW_DAYS,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            avg_profit=0,
            avg_loss=0,
            total_pnl=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            profit_factor=0,
            expectancy=0,
            is_valid=False,
        )

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        # Avoid division by zero
        peak_nonzero = np.where(peak > 0, peak, 1)
        drawdowns = (peak - cumulative) / peak_nonzero
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    @staticmethod
    def _aggregate_daily_returns(trades: list[TradeRecord]) -> list[float]:
        """Group trades by day and sum PnL."""
        from collections import defaultdict
        import datetime

        daily: defaultdict[str, float] = defaultdict(float)
        for t in trades:
            day = datetime.datetime.fromtimestamp(
                t.timestamp_ms / 1000
            ).strftime("%Y-%m-%d")
            daily[day] += t.pnl_usd
        return list(daily.values())
