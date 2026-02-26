"""Platform Risk Monitor — gas costs, CLOB uptime, exchange health."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class PlatformStatus:
    polymarket_up: bool = True
    binance_up: bool = True
    bybit_up: bool = True
    okx_up: bool = True
    gas_gwei: float = 0.0
    gas_acceptable: bool = True
    clob_latency_ms: int = 0
    last_check_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def all_healthy(self) -> bool:
        return (
            self.polymarket_up
            and self.gas_acceptable
            and any([self.binance_up, self.bybit_up, self.okx_up])
        )

    @property
    def cex_count_up(self) -> int:
        return sum([self.binance_up, self.bybit_up, self.okx_up])

    def to_dict(self) -> dict[str, Any]:
        return {
            "polymarket_up": self.polymarket_up,
            "binance_up": self.binance_up,
            "bybit_up": self.bybit_up,
            "okx_up": self.okx_up,
            "gas_gwei": self.gas_gwei,
            "gas_acceptable": self.gas_acceptable,
            "clob_latency_ms": self.clob_latency_ms,
            "all_healthy": self.all_healthy,
            "cex_count_up": self.cex_count_up,
        }


class PlatformRiskMonitor:
    """Monitors infrastructure health: exchange connectivity, gas costs, CLOB latency."""

    GAS_THRESHOLD_GWEI = 50.0  # max acceptable gas
    CLOB_LATENCY_THRESHOLD_MS = 2000  # max CLOB response time
    CHECK_INTERVAL_S = 10.0

    def __init__(self) -> None:
        self._settings = get_settings()
        self._status = PlatformStatus()
        self._session: aiohttp.ClientSession | None = None
        self._running = False

    @property
    def status(self) -> PlatformStatus:
        return self._status

    @property
    def is_safe_to_trade(self) -> bool:
        return self._status.all_healthy

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        )
        self._running = True
        logger.info("platform_risk.started")

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        logger.info("platform_risk.stopped")

    async def check_loop(self) -> None:
        """Continuous health check loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as exc:
                logger.warning("platform_risk.check_error", error=str(exc))
            await asyncio.sleep(self.CHECK_INTERVAL_S)

    async def check_all(self) -> PlatformStatus:
        """Run all platform health checks concurrently."""
        results = await asyncio.gather(
            self._check_polymarket(),
            self._check_binance(),
            self._check_bybit(),
            self._check_okx(),
            self._check_gas(),
            return_exceptions=True,
        )

        self._status.polymarket_up = results[0] is True
        self._status.binance_up = results[1] is True
        self._status.bybit_up = results[2] is True
        self._status.okx_up = results[3] is True

        if isinstance(results[4], (int, float)):
            self._status.gas_gwei = float(results[4])
            self._status.gas_acceptable = (
                self._status.gas_gwei <= self.GAS_THRESHOLD_GWEI
            )

        self._status.last_check_ms = int(time.time() * 1000)

        if not self._status.all_healthy:
            logger.warning(
                "platform_risk.degraded",
                status=self._status.to_dict(),
            )

        return self._status

    async def _check_polymarket(self) -> bool:
        if not self._session:
            return False
        try:
            start = time.time()
            async with self._session.get(
                f"{self._settings.polymarket_clob_url}/time"
            ) as resp:
                latency_ms = int((time.time() - start) * 1000)
                self._status.clob_latency_ms = latency_ms
                return (
                    resp.status == 200
                    and latency_ms < self.CLOB_LATENCY_THRESHOLD_MS
                )
        except Exception:
            return False

    async def _check_binance(self) -> bool:
        if not self._session:
            return False
        try:
            async with self._session.get(
                "https://api.binance.com/api/v3/ping"
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _check_bybit(self) -> bool:
        if not self._session:
            return False
        try:
            async with self._session.get(
                "https://api.bybit.com/v5/market/time"
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _check_okx(self) -> bool:
        if not self._session:
            return False
        try:
            async with self._session.get(
                "https://www.okx.com/api/v5/public/time"
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _check_gas(self) -> float:
        """Check current Ethereum gas price via the RPC endpoint."""
        if not self._settings.chainlink_rpc_url:
            return 0.0
        try:
            from web3 import AsyncWeb3, AsyncHTTPProvider

            w3 = AsyncWeb3(AsyncHTTPProvider(self._settings.chainlink_rpc_url))
            gas_wei = await w3.eth.gas_price
            return gas_wei / 1e9  # Convert to gwei
        except Exception:
            return 0.0
