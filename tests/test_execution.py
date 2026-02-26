"""Tests for Layer 4 Execution components."""

from __future__ import annotations

import pytest

from src.layer0_ingestion.polymarket_client import OrderBook, OrderBookLevel
from src.layer4_execution.order_book_sniper import OrderBookSniper, OrderStrategy
from src.layer4_execution.fill_monitor import FillMonitor, FillStatus


def _make_deep_orderbook() -> OrderBook:
    bids = [
        OrderBookLevel(price=0.48 - i * 0.01, size=500 + i * 100)
        for i in range(10)
    ]
    asks = [
        OrderBookLevel(price=0.52 + i * 0.01, size=500 + i * 100)
        for i in range(10)
    ]
    return OrderBook(
        market_id="test_market",
        timestamp_ms=1700000000000,
        bids=bids,
        asks=asks,
    )


class TestOrderBookSniper:
    def test_market_order_on_high_urgency(self):
        ob = _make_deep_orderbook()
        sniper = OrderBookSniper()
        order = sniper.compute_optimal_order(ob, "BUY", 500, urgency=0.95)
        assert order.strategy == OrderStrategy.MARKET

    def test_iceberg_on_large_order(self):
        ob = _make_deep_orderbook()
        sniper = OrderBookSniper()
        order = sniper.compute_optimal_order(ob, "BUY", 10_000, urgency=0.6)
        assert order.strategy == OrderStrategy.ICEBERG
        assert len(order.chunks) > 0

    def test_limit_passive_on_low_urgency(self):
        ob = _make_deep_orderbook()
        sniper = OrderBookSniper()
        order = sniper.compute_optimal_order(ob, "BUY", 500, urgency=0.2)
        assert order.strategy == OrderStrategy.LIMIT_PASSIVE

    def test_slippage_estimate_reasonable(self):
        ob = _make_deep_orderbook()
        sniper = OrderBookSniper()
        order = sniper.compute_optimal_order(ob, "BUY", 500, urgency=0.7)
        assert order.expected_slippage_bps >= 0
        assert order.expected_slippage_bps < 500  # should be reasonable


class TestFillMonitor:
    def test_track_and_record_fill(self):
        fm = FillMonitor()
        fm.track_order("order1", "pos1", 0.48, 100)
        fill = fm.record_fill("order1", 0.485, 100, FillStatus.FILLED)
        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert fill.slippage_bps > 0

    def test_unknown_order_returns_none(self):
        fm = FillMonitor()
        fill = fm.record_fill("unknown", 0.5, 100)
        assert fill is None

    def test_metrics_computed(self):
        fm = FillMonitor()
        fm.track_order("o1", "p1", 0.48, 100)
        fm.record_fill("o1", 0.482, 100)
        fm.track_order("o2", "p2", 0.50, 200)
        fm.record_fill("o2", 0.503, 200)
        metrics = fm.get_metrics()
        assert metrics["total_fills"] == 2
        assert metrics["fill_rate"] == 1.0
        assert metrics["avg_slippage_bps"] > 0

    def test_fill_rate_calculation(self):
        fm = FillMonitor()
        fm.track_order("o1", "p1", 0.48, 100)
        fm.record_fill("o1", 0.48, 100, FillStatus.FILLED)
        fm.track_order("o2", "p2", 0.50, 200)
        fm.record_fill("o2", 0, 0, FillStatus.CANCELLED)
        assert fm.fill_rate == 0.5
