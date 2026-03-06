"""Tests for Layer 2 Signal Validator and Alpha Signal Generator."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.layer0_ingestion.event_bus import EventBus
from src.layer1_research.research_synthesis import ResearchOutput
from src.layer1_research.spread_detector import SpreadOpportunity
from src.layer1_research.liquidity_scanner import LiquidityProfile
from src.layer0_ingestion.cex_websocket import CEXFeed
from src.layer2_signal.alpha_signal import AlphaSignalGenerator
from src.layer2_signal.backtester import Backtester, TradeRecord
from src.layer2_signal.risk_filter import RiskFilter
from src.layer2_signal.signal_validator import SignalValidator


def _make_research(
    combined_prob: float = 0.75,
    edge_pct: float = 3.0,
    direction: str = "BUY_YES",
    market_id: str = "test_market",
) -> ResearchOutput:
    spread_opp = SpreadOpportunity(
        market_id=market_id,
        pm_yes_price=0.48,
        pm_no_price=0.52,
        cex_price=67800,
        cex_source=CEXFeed.BINANCE,
        implied_pm_prob=0.50,
        fair_prob=0.75,
        spread_pct=edge_pct,
        direction=direction,
    )
    liquidity = LiquidityProfile(
        market_id=market_id,
        total_bid_depth_usd=15000,
        total_ask_depth_usd=15000,
        best_bid_size=5000,
        best_ask_size=5000,
        spread_bps=200,
        depth_imbalance=0.5,
        levels_with_liquidity=10,
    )
    return ResearchOutput(
        market_id=market_id,
        direction=direction,
        spread_score=0.8,
        latency_score=0.6,
        liquidity_score=0.7,
        combined_probability=combined_prob,
        confidence=0.75,
        spread_opp=spread_opp,
        liquidity_prof=liquidity,
        edge_pct=edge_pct,
        max_safe_size_usd=3000,
    )


class TestAlphaSignalGenerator:
    def test_entry_signal_generated(self):
        """High-confidence research should produce an entry signal."""
        gen = AlphaSignalGenerator()
        research = _make_research(combined_prob=0.75, edge_pct=3.5)
        signal = gen.generate(research)
        assert signal.entry is True
        assert signal.kelly_fraction > 0
        assert signal.optimal_size_usd > 0
        assert signal.direction == "BUY_YES"

    def test_no_entry_low_edge(self):
        """Low edge should produce a SKIP."""
        gen = AlphaSignalGenerator()
        research = _make_research(combined_prob=0.55, edge_pct=0.5)
        signal = gen.generate(research)
        assert signal.entry is False

    def test_kelly_capped(self):
        """Kelly fraction should never exceed the configured cap."""
        gen = AlphaSignalGenerator()
        research = _make_research(combined_prob=0.95, edge_pct=10.0)
        signal = gen.generate(research)
        assert signal.kelly_fraction <= gen._kelly_cap

    def test_expected_profit_positive(self):
        """Entry signals should have positive expected profit."""
        gen = AlphaSignalGenerator()
        research = _make_research(combined_prob=0.80, edge_pct=5.0)
        signal = gen.generate(research)
        if signal.entry:
            assert signal.expected_profit_usd > 0


class TestRiskFilter:
    def test_passes_clean_signal(self):
        """A signal within all limits should pass."""
        rf = RiskFilter()
        rf.update_state(positions=[], daily_pnl=0, equity=100_000)
        gen = AlphaSignalGenerator()
        research = _make_research()
        signal = gen.generate(research)
        assessment = rf.assess(signal)
        assert assessment.passed is True

    def test_rejects_daily_loss_exceeded(self):
        """Should reject when daily loss limit is breached."""
        rf = RiskFilter()
        rf.update_state(positions=[], daily_pnl=-2500, equity=97_500)
        gen = AlphaSignalGenerator()
        research = _make_research()
        signal = gen.generate(research)
        assessment = rf.assess(signal)
        assert assessment.daily_loss_ok is False
        assert assessment.passed is False

    def test_rejects_max_concurrent(self):
        """Should reject when max concurrent positions reached."""
        rf = RiskFilter()
        positions = [
            {"direction": "BUY_YES", "market_id": f"m{i}"}
            for i in range(5)
        ]
        rf.update_state(positions=positions, daily_pnl=0, equity=100_000)
        gen = AlphaSignalGenerator()
        research = _make_research()
        signal = gen.generate(research)
        assessment = rf.assess(signal)
        assert assessment.concentration_ok is False
        assert assessment.passed is False


class TestBacktester:
    def test_valid_with_high_win_rate(self):
        """Strategy with >70% win rate should validate."""
        bt = Backtester()
        for i in range(30):
            bt.record_trade(
                TradeRecord(
                    market_id="m1",
                    direction="BUY_YES",
                    entry_price=0.48,
                    size_usd=1000,
                    pnl_usd=15 if i < 22 else -10,
                    won=i < 22,  # 73% win rate
                    edge_pct=3.0,
                )
            )
        result = bt.validate()
        assert result.total_trades == 30
        assert result.win_rate > 0.70
        assert result.is_valid is True

    def test_invalid_with_low_win_rate(self):
        """Strategy with <70% win rate should fail validation."""
        bt = Backtester()
        for i in range(30):
            bt.record_trade(
                TradeRecord(
                    market_id="m1",
                    direction="BUY_YES",
                    entry_price=0.48,
                    size_usd=1000,
                    pnl_usd=15 if i < 15 else -10,
                    won=i < 15,  # 50% win rate
                    edge_pct=3.0,
                )
            )
        result = bt.validate()
        assert result.win_rate < 0.70
        assert result.is_valid is False

    def test_allows_trading_before_min_trades(self):
        """Before MIN_TRADES reached, strategy should be allowed."""
        bt = Backtester()
        bt.record_trade(
            TradeRecord(
                market_id="m1",
                direction="BUY_YES",
                entry_price=0.48,
                size_usd=1000,
                pnl_usd=15,
                won=True,
                edge_pct=3.0,
            )
        )
        assert bt.is_strategy_valid is True  # under MIN_TRADES


@pytest.mark.asyncio
class TestSignalValidator:
    async def test_trade_on_valid_signal(self):
        """High-quality signal should produce TRADE action."""
        bus = EventBus()
        await bus.start()
        validator = SignalValidator(event_bus=bus)
        validator.risk_filter.update_state(
            positions=[], daily_pnl=0, equity=100_000
        )
        research = _make_research(combined_prob=0.75, edge_pct=3.5)
        result = await validator.validate(research)
        assert result.action == "TRADE"
        assert result.final_size_usd > 0
        await bus.stop()

    async def test_skip_on_low_confidence(self):
        """Low-confidence signal should produce SKIP action."""
        bus = EventBus()
        await bus.start()
        validator = SignalValidator(event_bus=bus)
        research = _make_research(combined_prob=0.40, edge_pct=1.0)
        result = await validator.validate(research)
        assert result.action == "SKIP"
        await bus.stop()
