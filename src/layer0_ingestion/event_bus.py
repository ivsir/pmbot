"""Event bus — in-process async pub/sub with optional Kafka producer for persistence."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import orjson
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    PRICE_TICK = "price_tick"
    ORDERBOOK_UPDATE = "orderbook_update"
    ORACLE_PRICE = "oracle_price"
    SPREAD_DETECTED = "spread_detected"
    LATENCY_ARB = "latency_arb"
    LIQUIDITY_UPDATE = "liquidity_update"
    RESEARCH_SIGNAL = "research_signal"
    ALPHA_SIGNAL = "alpha_signal"
    TRADE_SIGNAL = "trade_signal"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    POSITION_UPDATE = "position_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Event:
    event_type: EventType
    data: dict[str, Any]
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    source: str = ""

    def to_bytes(self) -> bytes:
        return orjson.dumps(
            {
                "event_type": self.event_type.value,
                "data": self.data,
                "timestamp_ms": self.timestamp_ms,
                "source": self.source,
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> Event:
        d = orjson.loads(raw)
        return cls(
            event_type=EventType(d["event_type"]),
            data=d["data"],
            timestamp_ms=d["timestamp_ms"],
            source=d.get("source", ""),
        )


SubscriberCallback = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Lightweight async event bus with topic-based routing.

    Optionally forwards events to Kafka for durable storage.
    """

    def __init__(self, use_kafka: bool = False) -> None:
        self._settings = get_settings()
        self._subscribers: dict[EventType, list[SubscriberCallback]] = {}
        self._use_kafka = use_kafka
        self._kafka_producer: Any = None
        self._event_count = 0
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10_000)
        self._running = False

    async def start(self) -> None:
        self._running = True
        if self._use_kafka:
            try:
                from aiokafka import AIOKafkaProducer

                self._kafka_producer = AIOKafkaProducer(
                    bootstrap_servers=self._settings.kafka_bootstrap_servers,
                    value_serializer=lambda v: v,
                )
                await self._kafka_producer.start()
                logger.info("event_bus.kafka_connected")
            except Exception as exc:
                logger.warning(
                    "event_bus.kafka_unavailable", error=str(exc)
                )
                self._kafka_producer = None
        logger.info("event_bus.started", kafka=self._use_kafka)

    async def stop(self) -> None:
        self._running = False
        if self._kafka_producer:
            await self._kafka_producer.stop()
        logger.info(
            "event_bus.stopped", total_events=self._event_count
        )

    def subscribe(
        self, event_type: EventType, callback: SubscriberCallback
    ) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: SubscriberCallback) -> None:
        for et in EventType:
            self.subscribe(et, callback)

    async def publish(self, event: Event) -> None:
        self._event_count += 1

        # Dispatch to in-process subscribers
        callbacks = self._subscribers.get(event.event_type, [])
        if callbacks:
            await asyncio.gather(
                *(cb(event) for cb in callbacks), return_exceptions=True
            )

        # Forward to Kafka
        if self._kafka_producer:
            topic = self._event_type_to_topic(event.event_type)
            try:
                await self._kafka_producer.send(topic, event.to_bytes())
            except Exception as exc:
                logger.warning(
                    "event_bus.kafka_send_error", error=str(exc)
                )

    def _event_type_to_topic(self, et: EventType) -> str:
        mapping = {
            EventType.PRICE_TICK: self._settings.kafka_topic_ticks,
            EventType.ORDERBOOK_UPDATE: self._settings.kafka_topic_ticks,
            EventType.ORACLE_PRICE: self._settings.kafka_topic_ticks,
            EventType.ALPHA_SIGNAL: self._settings.kafka_topic_signals,
            EventType.TRADE_SIGNAL: self._settings.kafka_topic_signals,
            EventType.ORDER_PLACED: self._settings.kafka_topic_executions,
            EventType.ORDER_FILLED: self._settings.kafka_topic_executions,
        }
        return mapping.get(et, self._settings.kafka_topic_ticks)
