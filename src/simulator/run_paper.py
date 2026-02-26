"""Launch script for paper trading mode."""

from __future__ import annotations

import asyncio
import logging
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import structlog

# Silence all logging so it doesn't clobber the live dashboard
logging.disable(logging.CRITICAL)

class _NullLogger:
    def msg(self, *a, **kw): pass
    debug = info = warning = error = critical = exception = msg
    def bind(self, **kw): return self
    def new(self, **kw): return self

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer(colors=False)],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=lambda *a, **kw: _NullLogger(),
)

from src.simulator.paper_trader import PaperTrader
from src.simulator.market_simulator import MarketConfig


def main() -> None:
    config = MarketConfig(
        initial_btc_price=67_500.0,
        volatility_per_tick=0.0003,
        pm_lag_ms=500,
        pm_spread_bps=200,
        arb_opportunity_freq=0.08,
        arb_spread_pct=0.025,
        num_strikes=5,
        base_liquidity_usd=15_000.0,
    )

    trader = PaperTrader(config)

    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
