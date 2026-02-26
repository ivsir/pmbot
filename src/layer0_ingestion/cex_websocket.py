"""CEX WebSocket feeds — Binance, Bybit, OKX real-time BTC price ticks."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine

import orjson
import structlog
import websockets

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class CEXFeed(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"


@dataclass
class CEXTick:
    exchange: CEXFeed
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp_ms: int
    volume_24h: float = 0.0
    local_receive_ms: int = 0  # local clock when message was received

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        if self.mid == 0:
            return 0.0
        return ((self.ask - self.bid) / self.mid) * 10_000


class CEXWebSocketManager:
    """Manages concurrent WebSocket connections to multiple CEXes."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ticks: dict[CEXFeed, CEXTick] = {}
        self._callbacks: list[Callable[[CEXTick], Coroutine[Any, Any, None]]] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Rolling price history for momentum calculation (~5 min at ~2 ticks/sec)
        self._price_history: deque[CEXTick] = deque(maxlen=600)

    def on_tick(self, cb: Callable[[CEXTick], Coroutine[Any, Any, None]]) -> None:
        self._callbacks.append(cb)

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._run_binance()),
            asyncio.create_task(self._run_bybit()),
            asyncio.create_task(self._run_okx()),
        ]
        logger.info("cex_ws.started", feeds=3)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("cex_ws.stopped")

    def get_best_cex_price(self) -> CEXTick | None:
        """Return the most recent tick with best (highest) bid across exchanges."""
        if not self._ticks:
            return None
        return max(self._ticks.values(), key=lambda t: t.bid)

    def get_tick(self, exchange: CEXFeed) -> CEXTick | None:
        return self._ticks.get(exchange)

    def get_all_ticks(self) -> dict[CEXFeed, CEXTick]:
        return dict(self._ticks)

    def get_price_history(self, max_age_ms: int = 300_000) -> list[CEXTick]:
        """Return recent CEX ticks within max_age_ms, for momentum calculation."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - max_age_ms
        return [t for t in self._price_history if t.timestamp_ms >= cutoff]

    async def _emit(self, tick: CEXTick) -> None:
        self._ticks[tick.exchange] = tick
        self._price_history.append(tick)
        for cb in self._callbacks:
            asyncio.create_task(cb(tick))

    # ── Binance ──

    async def _run_binance(self) -> None:
        url = f"{self._settings.binance_ws_url}/btcusdt@bookTicker"
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("binance.ws_connected")
                    backoff = 1  # reset on success
                    async for raw in ws:
                        if not self._running:
                            break
                        local_now = int(time.time() * 1000)
                        data = orjson.loads(raw)
                        tick = CEXTick(
                            exchange=CEXFeed.BINANCE,
                            symbol="BTCUSDT",
                            bid=float(data.get("b", 0)),
                            ask=float(data.get("a", 0)),
                            last=float(data.get("b", 0)),
                            timestamp_ms=local_now,  # Binance bookTicker has no server ts
                            volume_24h=float(data.get("B", 0)),
                            local_receive_ms=local_now,
                        )
                        await self._emit(tick)
            except Exception as exc:
                err = str(exc)
                if "451" in err or "Unavailable" in err:
                    # Geo-blocked — log once and back off heavily
                    if backoff <= 2:
                        logger.warning("binance.geo_blocked", msg="HTTP 451 — Binance unavailable in this region, backing off")
                    backoff = min(backoff * 2, 300)  # max 5 min
                else:
                    logger.warning("binance.ws_error", error=err)
                    backoff = min(backoff * 2, 30)  # max 30s for other errors
                await asyncio.sleep(backoff)

    # ── Bybit ──

    async def _run_bybit(self) -> None:
        url = self._settings.bybit_ws_url
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    sub = {
                        "op": "subscribe",
                        "args": ["tickers.BTCUSDT"],
                    }
                    await ws.send(orjson.dumps(sub).decode())
                    logger.info("bybit.ws_connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        local_now = int(time.time() * 1000)
                        data = orjson.loads(raw)
                        if data.get("topic") != "tickers.BTCUSDT":
                            continue
                        d = data.get("data", {})
                        tick = CEXTick(
                            exchange=CEXFeed.BYBIT,
                            symbol="BTCUSDT",
                            bid=float(d.get("bid1Price", 0)),
                            ask=float(d.get("ask1Price", 0)),
                            last=float(d.get("lastPrice", 0)),
                            timestamp_ms=int(
                                d.get("ts", local_now)
                            ),
                            volume_24h=float(d.get("volume24h", 0)),
                            local_receive_ms=local_now,
                        )
                        await self._emit(tick)
            except Exception as exc:
                logger.warning("bybit.ws_error", error=str(exc))
                await asyncio.sleep(1)

    # ── OKX ──

    async def _run_okx(self) -> None:
        url = self._settings.okx_ws_url
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    sub = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "tickers", "instId": "BTC-USDT"}
                        ],
                    }
                    await ws.send(orjson.dumps(sub).decode())
                    logger.info("okx.ws_connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        local_now = int(time.time() * 1000)
                        data = orjson.loads(raw)
                        if "data" not in data:
                            continue
                        for d in data["data"]:
                            tick = CEXTick(
                                exchange=CEXFeed.OKX,
                                symbol="BTC-USDT",
                                bid=float(d.get("bidPx", 0)),
                                ask=float(d.get("askPx", 0)),
                                last=float(d.get("last", 0)),
                                timestamp_ms=int(
                                    d.get("ts", local_now)
                                ),
                                volume_24h=float(d.get("vol24h", 0)),
                                local_receive_ms=local_now,
                            )
                            await self._emit(tick)
            except Exception as exc:
                logger.warning("okx.ws_error", error=str(exc))
                await asyncio.sleep(1)
