"""Centralized configuration — loaded from .env with Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Polymarket ──
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # ── Polymarket Live Trading ──
    polygon_private_key: str = ""
    poly_funder_address: str = ""
    poly_signature_type: int = 1  # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE
    poly_chain_id: int = 137  # Polygon mainnet
    live_trading_enabled: bool = False
    poly_tick_size: str = "0.01"

    # ── Binance ──
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"

    # ── Bybit ──
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_ws_url: str = "wss://stream.bybit.com/v5/public/spot"

    # ── OKX ──
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""
    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"

    # ── Chainlink ──
    chainlink_rpc_url: str = ""
    chainlink_btc_usd_feed: str = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"

    # ── Database ──
    postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/polymarket_arb"
    redis_url: str = "redis://localhost:6379/0"
    timescale_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/timeseries"

    # ── Kafka ──
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_ticks: str = "price-ticks"
    kafka_topic_signals: str = "trade-signals"
    kafka_topic_executions: str = "executions"

    # ── Monitoring ──
    datadog_api_key: str = ""
    datadog_app_key: str = ""

    # ── LLM ──
    openai_api_key: str = ""

    # ── Risk Controls ──
    max_position_usd: float = Field(default=50_000.0)
    max_concurrent_positions: int = Field(default=5)
    daily_loss_limit_usd: float = Field(default=2_000.0)
    correlation_threshold: float = Field(default=0.7)
    liquidity_minimum_usd: float = Field(default=10_000.0)
    kelly_fraction: float = Field(default=0.043)
    max_drawdown_pct: float = Field(default=0.05)
    min_edge_pct: float = Field(default=0.02)
    min_confidence: float = Field(default=0.60)

    # ── Latency Budgets (ms) ──
    latency_data_ingestion_ms: int = 100
    latency_signal_gen_ms: int = 200
    latency_order_exec_ms: int = 500
    latency_end_to_end_ms: int = 1000

    # ── Performance Targets ──
    target_win_rate: float = 0.70
    target_avg_profit: float = 15.0
    target_sharpe: float = 2.5

    # ── 5-Min Up/Down Market Mode ──
    market_mode: str = "5min_updown"  # "5min_updown" or "monthly_reach"
    updown_lookahead_minutes: int = 1500  # ~25h lookahead (markets created 24h ahead)
    updown_market_refresh_interval_s: int = 60
    momentum_sigmoid_sensitivity: float = 50.0

    # ── Web Dashboard ──
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
