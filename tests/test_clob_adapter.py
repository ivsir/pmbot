"""Tests for the ClobAdapter async wrapper and live mode integration."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.layer4_execution.clob_adapter import ClobAdapter, ClobOrderResult
from src.layer4_execution.fill_monitor import FillMonitor, FillStatus


# ── ClobAdapter init tests ──


class TestClobAdapterInit:
    def test_not_initialized_raises(self):
        adapter = ClobAdapter()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = adapter.client

    @patch("src.layer4_execution.clob_adapter.get_settings")
    def test_missing_private_key_raises(self, mock_settings):
        mock_settings.return_value.polygon_private_key = ""
        adapter = ClobAdapter()
        adapter._settings = mock_settings.return_value
        with pytest.raises(ValueError, match="POLYGON_PRIVATE_KEY"):
            adapter.initialize()


# ── ClobAdapter order tests ──


class TestClobAdapterOrders:
    @pytest.mark.asyncio
    async def test_place_limit_order_success(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        adapter._client.create_order.return_value = {"mock_signed": True}
        adapter._client.post_order.return_value = {
            "success": True,
            "orderID": "test-order-123",
            "errorMsg": "",
            "status": "LIVE",
        }

        result = await adapter.place_limit_order(
            token_id="token-abc",
            side="BUY",
            price=0.65,
            size=100,
        )

        assert result.success is True
        assert result.order_id == "test-order-123"

    @pytest.mark.asyncio
    async def test_place_limit_order_failure(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        adapter._client.create_order.side_effect = Exception("API error")

        result = await adapter.place_limit_order(
            token_id="token-abc",
            side="BUY",
            price=0.65,
            size=100,
        )

        assert result.success is False
        assert "API error" in result.error_msg

    @pytest.mark.asyncio
    async def test_place_limit_order_rejected(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        adapter._client.create_order.return_value = {"mock_signed": True}
        adapter._client.post_order.return_value = {
            "success": False,
            "orderID": "",
            "errorMsg": "Insufficient balance",
            "status": "REJECTED",
        }

        result = await adapter.place_limit_order(
            token_id="token-abc",
            side="BUY",
            price=0.65,
            size=100,
        )

        assert result.success is False
        assert "Insufficient balance" in result.error_msg


# ── Price tick snapping tests ──


class TestPriceTickSnapping:
    def test_snap_to_001(self):
        tick = 0.01
        price = 0.655
        snapped = round(round(price / tick) * tick, 2)
        assert snapped == 0.66

    def test_snap_preserves_exact(self):
        tick = 0.01
        price = 0.50
        snapped = round(round(price / tick) * tick, 2)
        assert snapped == 0.50

    def test_snap_clamp_high(self):
        tick = 0.01
        price = 1.05
        snapped = round(round(price / tick) * tick, 2)
        clamped = max(tick, min(1.0 - tick, snapped))
        assert clamped == 0.99

    def test_snap_clamp_low(self):
        tick = 0.01
        price = -0.05
        snapped = round(round(price / tick) * tick, 2)
        clamped = max(tick, min(1.0 - tick, snapped))
        assert clamped == 0.01

    def test_snap_to_01(self):
        tick = 0.1
        price = 0.68
        snapped = round(round(price / tick) * tick, 1)
        assert snapped == 0.7

    def test_snap_to_0001(self):
        tick = 0.001
        price = 0.6556
        snapped = round(round(price / tick) * tick, 3)
        assert snapped == 0.656


# ── Fill monitor edge cases for live mode ──


class TestFillMonitorLiveMode:
    def test_partial_fill(self):
        fm = FillMonitor()
        fm.track_order("o1", "p1", 0.50, 100)
        fill = fm.record_fill("o1", 0.505, 60, FillStatus.PARTIAL)
        assert fill is not None
        assert fill.status == FillStatus.PARTIAL
        assert fill.filled_size == 60
        assert fill.fill_ratio == 0.6

    def test_cancelled_fill(self):
        fm = FillMonitor()
        fm.track_order("o1", "p1", 0.50, 100)
        fill = fm.record_fill("o1", 0, 0, FillStatus.CANCELLED)
        assert fill is not None
        assert fill.status == FillStatus.CANCELLED
        assert fill.is_complete is True

    def test_expired_fill(self):
        fm = FillMonitor()
        fm.track_order("o1", "p1", 0.50, 100)
        fill = fm.record_fill("o1", 0, 0, FillStatus.EXPIRED)
        assert fill is not None
        assert fill.status == FillStatus.EXPIRED
        assert fill.is_complete is True

    def test_fill_rate_with_failures(self):
        fm = FillMonitor()
        # 2 filled, 1 cancelled
        fm.track_order("o1", "p1", 0.50, 100)
        fm.record_fill("o1", 0.50, 100, FillStatus.FILLED)
        fm.track_order("o2", "p2", 0.60, 100)
        fm.record_fill("o2", 0.60, 100, FillStatus.FILLED)
        fm.track_order("o3", "p3", 0.70, 100)
        fm.record_fill("o3", 0, 0, FillStatus.CANCELLED)

        assert fm.fill_rate == pytest.approx(2 / 3, rel=0.01)


# ── ClobOrderResult tests ──


class TestClobOrderResult:
    def test_success_result(self):
        r = ClobOrderResult(success=True, order_id="abc-123")
        assert r.success is True
        assert r.order_id == "abc-123"
        assert r.error_msg == ""

    def test_failure_result(self):
        r = ClobOrderResult(success=False, order_id="", error_msg="boom")
        assert r.success is False
        assert r.error_msg == "boom"


# ── Balance check tests ──


class TestClobAdapterBalance:
    @pytest.mark.asyncio
    async def test_get_balance_success(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        # Balance returned as raw USDC (6 decimals)
        adapter._client.get_balance_allowance.return_value = {
            "balance": "50000000",  # 50 USDC
            "allowance": "100000000",
        }

        balance = await adapter.get_collateral_balance()
        assert balance == 50.0

    @pytest.mark.asyncio
    async def test_get_balance_zero(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        adapter._client.get_balance_allowance.return_value = {
            "balance": "0",
            "allowance": "0",
        }

        balance = await adapter.get_collateral_balance()
        assert balance == 0.0

    @pytest.mark.asyncio
    async def test_get_balance_error_returns_zero(self):
        adapter = ClobAdapter()
        adapter._initialized = True
        adapter._client = MagicMock()
        adapter._client.get_balance_allowance.side_effect = Exception("API down")

        balance = await adapter.get_collateral_balance()
        assert balance == 0.0


# ── Standby mode tests ──


class TestStandbyMode:
    @pytest.mark.asyncio
    async def test_standby_on_zero_balance(self):
        from src.layer0_ingestion.polymarket_client import PolymarketClient

        client = PolymarketClient()
        client._live_mode = True
        client._clob_adapter = MagicMock()

        # Mock async get_collateral_balance
        async def mock_balance():
            return 0.0
        client._clob_adapter.get_collateral_balance = mock_balance

        balance = await client.check_wallet_balance()
        assert balance == 0.0
        assert client.is_standby is True

    @pytest.mark.asyncio
    async def test_standby_lifted_with_funds(self):
        from src.layer0_ingestion.polymarket_client import PolymarketClient

        client = PolymarketClient()
        client._live_mode = True
        client._standby = True  # was in standby
        client._clob_adapter = MagicMock()

        async def mock_balance():
            return 100.0
        client._clob_adapter.get_collateral_balance = mock_balance

        balance = await client.check_wallet_balance()
        assert balance == 100.0
        assert client.is_standby is False

    @pytest.mark.asyncio
    async def test_paper_mode_no_standby(self):
        from src.layer0_ingestion.polymarket_client import PolymarketClient

        client = PolymarketClient()
        client._live_mode = False

        balance = await client.check_wallet_balance()
        assert balance == 0.0
        assert client.is_standby is False
