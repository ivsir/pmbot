"""Chainlink price oracle — on-chain BTC/USD feed for reference pricing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import structlog
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.contract import AsyncContract

from config.settings import get_settings

logger = structlog.get_logger(__name__)

AGGREGATOR_V3_ABI: list[dict[str, Any]] = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass
class OraclePrice:
    price: float
    round_id: int
    updated_at: int
    timestamp_ms: int
    source: str = "chainlink"

    @property
    def staleness_ms(self) -> int:
        return self.timestamp_ms - (self.updated_at * 1000)


class ChainlinkOracle:
    """Polls Chainlink BTC/USD aggregator for on-chain reference price."""

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self._settings = get_settings()
        self._poll_interval = poll_interval_s
        self._w3: AsyncWeb3 | None = None
        self._contract: AsyncContract | None = None
        self._decimals: int = 8
        self._latest: OraclePrice | None = None
        self._callbacks: list[
            Callable[[OraclePrice], Coroutine[Any, Any, None]]
        ] = []
        self._running = False

    def on_price(
        self, cb: Callable[[OraclePrice], Coroutine[Any, Any, None]]
    ) -> None:
        self._callbacks.append(cb)

    @property
    def latest(self) -> OraclePrice | None:
        return self._latest

    async def start(self) -> None:
        self._w3 = AsyncWeb3(
            AsyncHTTPProvider(self._settings.chainlink_rpc_url)
        )
        self._contract = self._w3.eth.contract(
            address=self._w3.to_checksum_address(
                self._settings.chainlink_btc_usd_feed
            ),
            abi=AGGREGATOR_V3_ABI,
        )
        self._decimals = await self._contract.functions.decimals().call()
        self._running = True
        logger.info(
            "chainlink_oracle.started",
            feed=self._settings.chainlink_btc_usd_feed,
            decimals=self._decimals,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("chainlink_oracle.stopped")

    async def poll_loop(self) -> None:
        """Continuous polling loop — call as asyncio.create_task."""
        while self._running:
            try:
                price = await self.fetch_price()
                if price:
                    self._latest = price
                    for cb in self._callbacks:
                        asyncio.create_task(cb(price))
            except Exception as exc:
                logger.warning("chainlink_oracle.poll_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    async def fetch_price(self) -> OraclePrice | None:
        if not self._contract:
            return None
        (
            round_id,
            answer,
            _started_at,
            updated_at,
            _answered_in_round,
        ) = await self._contract.functions.latestRoundData().call()

        price = answer / (10**self._decimals)
        now_ms = int(time.time() * 1000)
        return OraclePrice(
            price=price,
            round_id=round_id,
            updated_at=updated_at,
            timestamp_ms=now_ms,
        )
