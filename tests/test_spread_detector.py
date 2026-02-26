"""Tests for Layer 1 Spread Detector."""

from __future__ import annotations

import pytest

from src.layer0_ingestion.cex_websocket import CEXTick, CEXFeed
from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel
from src.layer1_research.spread_detector import SpreadDetector


def _make_orderbook(
    market_id: str = "test_market",
    best_bid: float = 0.48,
    best_ask: float = 0.52,
    bid_size: float = 1000,
    ask_size: float = 1000,
) -> OrderBook:
    return OrderBook(
        market_id=market_id,
        timestamp_ms=1700000000000,
        bids=[OrderBookLevel(price=best_bid, size=bid_size)],
        asks=[OrderBookLevel(price=best_ask, size=ask_size)],
    )


def _make_cex_tick(
    bid: float = 67500.0,
    ask: float = 67510.0,
    exchange: CEXFeed = CEXFeed.BINANCE,
) -> CEXTick:
    return CEXTick(
        exchange=exchange,
        symbol="BTCUSDT",
        bid=bid,
        ask=ask,
        last=bid,
        timestamp_ms=1700000000000,
    )


class TestSpreadDetector:
    def test_no_spread_returns_none(self):
        """When CEX and PM agree, no opportunity should be detected."""
        detector = SpreadDetector()
        # Strike at exactly the CEX price → fair prob ~50%, PM at 50% → no spread
        ob = _make_orderbook(best_bid=0.49, best_ask=0.51)
        tick = _make_cex_tick(bid=67500, ask=67510)
        result = detector.detect(ob, tick, strike_price=67505)
        assert result is None

    def test_large_spread_detected(self):
        """When CEX is well above strike but PM still shows low YES price."""
        detector = SpreadDetector()
        # PM shows YES at ~30% (cheap), but CEX is way above strike → should be ~90%+
        ob = _make_orderbook(best_bid=0.28, best_ask=0.32)
        tick = _make_cex_tick(bid=68000, ask=68010)
        result = detector.detect(ob, tick, strike_price=67000)
        assert result is not None
        assert result.direction == "BUY_YES"
        assert result.spread_pct > 2.0

    def test_buy_no_direction(self):
        """When PM overprices YES relative to CEX."""
        detector = SpreadDetector()
        # PM shows YES at ~80%, but CEX is well below strike → should be ~10%
        ob = _make_orderbook(best_bid=0.78, best_ask=0.82)
        tick = _make_cex_tick(bid=66000, ask=66010)
        result = detector.detect(ob, tick, strike_price=67000)
        assert result is not None
        assert result.direction == "BUY_NO"
        assert result.spread_pct > 2.0

    def test_hit_rate_tracking(self):
        """Spread detector tracks hit rate of actionable opportunities."""
        detector = SpreadDetector()
        ob = _make_orderbook(best_bid=0.28, best_ask=0.32)
        tick = _make_cex_tick(bid=68000, ask=68010)
        detector.detect(ob, tick, strike_price=67000)
        assert detector.hit_rate > 0

    def test_orderbook_properties(self):
        """OrderBook computed properties work correctly."""
        ob = _make_orderbook(best_bid=0.45, best_ask=0.55)
        assert ob.best_bid == 0.45
        assert ob.best_ask == 0.55
        assert ob.mid_price == 0.50
        assert ob.spread == pytest.approx(0.10)
