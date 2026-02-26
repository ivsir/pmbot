"""Data store — TimescaleDB for ticks, Redis for hot state, PostgreSQL for positions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import orjson
import structlog
from redis.asyncio import Redis

from config.settings import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class PriceTick:
    source: str
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp_ms: int
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "timestamp_ms": self.timestamp_ms,
            "volume": self.volume,
        }


@dataclass
class SpreadSnapshot:
    polymarket_price: float
    cex_price: float
    spread_pct: float
    timestamp_ms: int
    cex_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pm_price": self.polymarket_price,
            "cex_price": self.cex_price,
            "spread_pct": self.spread_pct,
            "timestamp_ms": self.timestamp_ms,
            "cex_source": self.cex_source,
        }


class DataStore:
    """Unified data access layer — Redis for hot state, with optional TimescaleDB."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis: Redis | None = None
        self._pg_pool: Any = None

    async def start(self) -> None:
        # Redis (optional — falls back to in-memory)
        try:
            self._redis = Redis.from_url(
                self._settings.redis_url, decode_responses=False
            )
            await self._redis.ping()
            logger.info("data_store.redis_connected")
        except Exception as exc:
            logger.warning("data_store.redis_unavailable", error=str(exc))
            self._redis = None

        # PostgreSQL/TimescaleDB connection pool (optional)
        try:
            import asyncpg

            dsn = self._settings.timescale_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
            self._pg_pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
            await self._init_timescale_tables()
            logger.info("data_store.timescale_connected")
        except Exception as exc:
            logger.warning("data_store.timescale_unavailable", error=str(exc))
            self._pg_pool = None

    async def stop(self) -> None:
        if self._redis:
            await self._redis.close()
        if self._pg_pool:
            await self._pg_pool.close()
        logger.info("data_store.stopped")

    # ── In-memory fallbacks when Redis is unavailable ──

    _mem_store: dict[str, Any] = {}
    _mem_lists: dict[str, list[Any]] = {}
    _mem_daily_pnl: float = 0.0

    # ── Redis hot state ──

    async def set_latest_tick(self, tick: PriceTick) -> None:
        key = f"tick:{tick.source}:{tick.symbol}"
        if self._redis:
            await self._redis.set(key, orjson.dumps(tick.to_dict()), ex=30)
        else:
            self._mem_store[key] = tick.to_dict()

    async def get_latest_tick(self, source: str, symbol: str) -> PriceTick | None:
        key = f"tick:{source}:{symbol}"
        if self._redis:
            raw = await self._redis.get(key)
            if not raw:
                return None
            d = orjson.loads(raw)
        else:
            d = self._mem_store.get(key)
            if not d:
                return None
        return PriceTick(
            source=d["source"],
            symbol=d["symbol"],
            bid=d["bid"],
            ask=d["ask"],
            last=d["last"],
            timestamp_ms=d["timestamp_ms"],
            volume=d.get("volume", 0),
        )

    async def push_spread(self, snap: SpreadSnapshot) -> None:
        key = "spreads:history"
        if self._redis:
            await self._redis.lpush(key, orjson.dumps(snap.to_dict()))
            await self._redis.ltrim(key, 0, 9999)
        else:
            lst = self._mem_lists.setdefault(key, [])
            lst.insert(0, snap.to_dict())
            if len(lst) > 10000:
                del lst[10000:]

    async def get_recent_spreads(self, count: int = 100) -> list[dict[str, Any]]:
        key = "spreads:history"
        if self._redis:
            raw_list = await self._redis.lrange(key, 0, count - 1)
            return [orjson.loads(r) for r in raw_list]
        return self._mem_lists.get(key, [])[:count]

    async def set_position(self, position_id: str, data: dict[str, Any]) -> None:
        key = f"position:{position_id}"
        if self._redis:
            await self._redis.set(key, orjson.dumps(data))
        else:
            self._mem_store[key] = data

    async def get_position(self, position_id: str) -> dict[str, Any] | None:
        key = f"position:{position_id}"
        if self._redis:
            raw = await self._redis.get(key)
            return orjson.loads(raw) if raw else None
        return self._mem_store.get(key)

    async def get_all_positions(self) -> list[dict[str, Any]]:
        if self._redis:
            keys = []
            async for key in self._redis.scan_iter(match="position:*"):
                keys.append(key)
            if not keys:
                return []
            values = await self._redis.mget(keys)
            return [orjson.loads(v) for v in values if v]
        return [v for k, v in self._mem_store.items() if k.startswith("position:")]

    async def incr_daily_pnl(self, amount: float) -> float:
        if self._redis:
            key = f"daily_pnl:{_today_key()}"
            new_val = await self._redis.incrbyfloat(key, amount)
            await self._redis.expire(key, 86400)
            return float(new_val)
        self._mem_daily_pnl += amount
        return self._mem_daily_pnl

    async def get_daily_pnl(self) -> float:
        if self._redis:
            key = f"daily_pnl:{_today_key()}"
            val = await self._redis.get(key)
            return float(val) if val else 0.0
        return self._mem_daily_pnl

    async def set_system_state(self, key: str, value: Any) -> None:
        if self._redis:
            await self._redis.set(f"state:{key}", orjson.dumps(value))
        else:
            self._mem_store[f"state:{key}"] = value

    async def get_system_state(self, key: str) -> Any:
        if self._redis:
            raw = await self._redis.get(f"state:{key}")
            return orjson.loads(raw) if raw else None
        return self._mem_store.get(f"state:{key}")

    # ── TimescaleDB persistence ──

    async def insert_tick(self, tick: PriceTick) -> None:
        if not self._pg_pool:
            return
        await self._pg_pool.execute(
            """
            INSERT INTO price_ticks (source, symbol, bid, ask, last, mid, volume, ts)
            VALUES ($1, $2, $3, $4, $5, $6, $7, to_timestamp($8 / 1000.0))
            """,
            tick.source,
            tick.symbol,
            tick.bid,
            tick.ask,
            tick.last,
            tick.mid,
            tick.volume,
            tick.timestamp_ms,
        )

    async def insert_spread(self, snap: SpreadSnapshot) -> None:
        if not self._pg_pool:
            return
        await self._pg_pool.execute(
            """
            INSERT INTO spread_snapshots (pm_price, cex_price, spread_pct, cex_source, ts)
            VALUES ($1, $2, $3, $4, to_timestamp($5 / 1000.0))
            """,
            snap.polymarket_price,
            snap.cex_price,
            snap.spread_pct,
            snap.cex_source,
            snap.timestamp_ms,
        )

    async def query_ticks(
        self, source: str, symbol: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        if not self._pg_pool:
            return []
        rows = await self._pg_pool.fetch(
            """
            SELECT source, symbol, bid, ask, last, mid, volume, ts
            FROM price_ticks
            WHERE source = $1 AND symbol = $2
            ORDER BY ts DESC LIMIT $3
            """,
            source,
            symbol,
            limit,
        )
        return [dict(r) for r in rows]

    # ── Schema init ──

    async def _init_timescale_tables(self) -> None:
        if not self._pg_pool:
            return
        await self._pg_pool.execute(
            """
            CREATE TABLE IF NOT EXISTS price_ticks (
                id BIGSERIAL,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bid DOUBLE PRECISION,
                ask DOUBLE PRECISION,
                last DOUBLE PRECISION,
                mid DOUBLE PRECISION,
                volume DOUBLE PRECISION DEFAULT 0,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await self._pg_pool.execute(
            """
            CREATE TABLE IF NOT EXISTS spread_snapshots (
                id BIGSERIAL,
                pm_price DOUBLE PRECISION,
                cex_price DOUBLE PRECISION,
                spread_pct DOUBLE PRECISION,
                cex_source TEXT,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # Attempt hypertable conversion (idempotent)
        for table in ("price_ticks", "spread_snapshots"):
            try:
                await self._pg_pool.execute(
                    f"SELECT create_hypertable('{table}', 'ts', if_not_exists => TRUE);"
                )
            except Exception:
                pass


def _today_key() -> str:
    import datetime

    return datetime.date.today().isoformat()
