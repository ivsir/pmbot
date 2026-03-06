"""Tests for Layer 3 Portfolio Manager."""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from src.layer0_ingestion.event_bus import EventBus
from src.layer0_ingestion.data_store import DataStore
from src.layer2_signal.signal_validator import ValidatedSignal
from src.layer2_signal.alpha_signal import AlphaSignal
from src.layer2_signal.risk_filter import RiskAssessment
from src.layer3_portfolio.portfolio_manager import PortfolioManager, PositionStatus


def _make_validated_signal(
    market_id: str = "test_market",
    size: float = 1000,
    direction: str = "BUY_YES",
) -> ValidatedSignal:
    signal = AlphaSignal(
        market_id=market_id,
        direction=direction,
        entry=True,
        edge_pct=3.0,
        win_probability=0.75,
        kelly_fraction=0.03,
        optimal_size_usd=size,
        entry_price=0.48,
        expected_profit_usd=15,
        expected_payout=1.08,
    )
    risk = RiskAssessment(signal=signal, passed=True, adjusted_size_usd=size)
    return ValidatedSignal(
        action="TRADE",
        signal=signal,
        risk=risk,
        backtest_valid=True,
        final_size_usd=size,
    )


@pytest.fixture
def mock_data_store():
    store = MagicMock(spec=DataStore)
    store.set_position = AsyncMock()
    store.incr_daily_pnl = AsyncMock(return_value=0.0)
    store.get_daily_pnl = AsyncMock(return_value=0.0)
    return store


@pytest.fixture
def mock_event_bus():
    bus = MagicMock(spec=EventBus)
    bus.publish = AsyncMock()
    return bus


@pytest.mark.asyncio
class TestPortfolioManager:
    async def test_open_position(self, mock_event_bus, mock_data_store):
        pm = PortfolioManager(mock_event_bus, mock_data_store)
        await pm.initialize(100_000)
        signal = _make_validated_signal()
        pos = await pm.open_position(signal)
        assert pos is not None
        assert pos.status == PositionStatus.PENDING
        assert pos.size_usd == 1000
        assert pm.active_count == 1

    async def test_max_concurrent_positions(self, mock_event_bus, mock_data_store):
        pm = PortfolioManager(mock_event_bus, mock_data_store)
        await pm.initialize(100_000)
        for i in range(5):
            await pm.open_position(_make_validated_signal(market_id=f"m{i}"))
        # 6th should be rejected
        pos = await pm.open_position(_make_validated_signal(market_id="m5"))
        assert pos is None
        assert pm.active_count == 5

    async def test_close_position_pnl(self, mock_event_bus, mock_data_store):
        pm = PortfolioManager(mock_event_bus, mock_data_store)
        await pm.initialize(100_000)
        pos = await pm.open_position(_make_validated_signal())
        assert pos is not None
        await pm.close_position(pos.id, exit_price=0.55, pnl=150)
        assert pm.active_count == 0
        assert pm.current_equity == 100_150

    async def test_drawdown_tracking(self, mock_event_bus, mock_data_store):
        """Drawdown is tracked even when circuit breaker is disabled."""
        pm = PortfolioManager(mock_event_bus, mock_data_store)
        await pm.initialize(100_000)
        pos = await pm.open_position(_make_validated_signal())
        assert pos is not None
        await pm.close_position(pos.id, exit_price=0.1, pnl=-5100)
        # Drawdown circuit breaker is intentionally disabled for small bankroll,
        # but drawdown should still be tracked
        assert pm.drawdown_pct > 0

    async def test_resume_after_halt(self, mock_event_bus, mock_data_store):
        """Manual halt/resume works."""
        pm = PortfolioManager(mock_event_bus, mock_data_store)
        await pm.initialize(100_000)
        pm._halted = True
        assert pm.is_halted is True
        pm.resume_trading()
        assert pm.is_halted is False
