# Polymarket 5-Min BTC Arbitrage — Agentic Trading System

A full 5-layer agentic trading system that exploits the ~500ms latency gap between centralized exchanges (Binance, Bybit, OKX) and Polymarket's CLOB for BTC 5-minute binary markets.

## Architecture

```
Layer 0 — DATA INGESTION
  Polymarket CLOB API │ Binance WS │ Bybit WS │ OKX WS │ Chainlink Oracle
  → Event Bus (Kafka) → Redis (hot state) + TimescaleDB (persistence)

Layer 1 — RESEARCH AGENTS
  Orchestrator → Spread Detector (CEX vs PM >2%)
               → Latency Arb (500ms lag exploit)
               → Liquidity Scanner (Depth >$10K)
               → Bayesian Fusion (Research Synthesis)

Layer 2 — SIGNAL GENERATION
  Alpha Signal (Kelly sizing) → Backtester (30-day, >70% win rate)
                               → Risk Filter (Max $50K, correlation)
                               → Signal Validator (Edge >2%, Conf >60%)
                               → TRADE / SKIP

Layer 3 — PORTFOLIO & RISK MANAGEMENT
  Portfolio Manager (Max $50K/pos, 5 concurrent, -5% DD halt)
  Correlation Monitor │ Tail Risk Agent │ Platform Risk Monitor

Layer 4 — EXECUTION
  Execution Agent (CLOB orders, <500ms) → Order Book Sniper
                                        → Fill Monitor
                                        → Hedge Agent (both-sides)
```

**Orchestration:** LangGraph state machine connecting all layers.

## Performance Targets

| Metric | Target |
|---|---|
| Win Rate | >70% |
| Avg Profit/Trade | >$15 |
| Max Drawdown | <5% |
| Sharpe Ratio | >2.5 |
| End-to-End Latency | <1000ms |

## Risk Controls

| Control | Value |
|---|---|
| Max Position | $50,000 |
| Max Concurrent | 5 positions |
| Daily Loss Limit | -$2,000 |
| Correlation Threshold | 0.7 |
| Liquidity Minimum | $10,000 |

## Edge Sources

1. **Polymarket 500ms lag** behind CEX spot prices
2. **CLOB orderbook spread capture** via order book sniper
3. **Both-sides entry** on volatility compression (YES + NO < $1)
4. **Chainlink oracle frontrunning** via on-chain price feed
5. **Market maker rebate optimization** via limit order placement

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 + asyncio |
| Framework | LangGraph (agent orchestration) |
| Database | TimescaleDB + PostgreSQL + Redis |
| Message Bus | Kafka (aiokafka) |
| Monitoring | Datadog + structlog |
| Deployment | Docker Compose (local) / AWS Lambda + Kafka (prod) |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys for Polymarket, Binance, Bybit, OKX (see `.env.example`)

### 2. Infrastructure

```bash
# Start databases, Redis, Kafka
docker-compose up -d
```

### 3. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Start the System

```bash
python -m src.main
```

## Project Structure

```
├── config/
│   └── settings.py              # Pydantic settings (from .env)
├── src/
│   ├── layer0_ingestion/
│   │   ├── polymarket_client.py # Polymarket CLOB API + WS
│   │   ├── cex_websocket.py     # Binance/Bybit/OKX feeds
│   │   ├── chainlink_oracle.py  # On-chain BTC/USD price
│   │   ├── event_bus.py         # Async pub/sub + Kafka
│   │   └── data_store.py        # Redis + TimescaleDB
│   ├── layer1_research/
│   │   ├── orchestrator.py      # Coordinates research agents
│   │   ├── spread_detector.py   # CEX vs PM spread (>2%)
│   │   ├── latency_arb.py       # 500ms lag detection
│   │   ├── liquidity_scanner.py # Depth filtering (>$10K)
│   │   └── research_synthesis.py# Bayesian signal fusion
│   ├── layer2_signal/
│   │   ├── alpha_signal.py      # Entry/exit + Kelly sizing
│   │   ├── backtester.py        # 30-day rolling validation
│   │   ├── risk_filter.py       # Pre-trade risk checks
│   │   └── signal_validator.py  # Final TRADE/SKIP gate
│   ├── layer3_portfolio/
│   │   ├── portfolio_manager.py # Position lifecycle + DD halt
│   │   ├── correlation_monitor.py # Co-movement tracking
│   │   ├── tail_risk.py         # Black swan detection
│   │   └── platform_risk.py     # Exchange health + gas
│   ├── layer4_execution/
│   │   ├── execution_agent.py   # Top-level executor
│   │   ├── order_book_sniper.py # Optimal order placement
│   │   ├── fill_monitor.py      # Slippage tracking
│   │   └── hedge_agent.py       # Both-sides entry
│   ├── graph/
│   │   └── workflow.py          # LangGraph state machine
│   └── main.py                  # Entry point
├── tests/
│   ├── test_spread_detector.py
│   ├── test_signal_validator.py
│   ├── test_portfolio.py
│   └── test_execution.py
├── docker-compose.yml           # TimescaleDB, Postgres, Redis, Kafka
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Sample 5-Min Trade Flow

| Time | Event | Profit |
|---|---|---|
| 00:00.0 | BTC pumps on Binance to $67,520 | — |
| 00:00.2 | PM still shows $67,450 \| Spread: 2.3% | — |
| 00:00.5 | Execute: BUY YES at 48¢ | — |
| 00:05.0 | Market resolves to Binance spot | +$520 |

## Monitoring

The system logs structured events via `structlog`. Key events:

- `spread_detected` — CEX-PM spread exceeds threshold
- `latency_arb_detected` — stale PM price detected
- `signal_validator.TRADE` — trade signal validated
- `execution.complete` — order executed on CLOB
- `portfolio.HALTED` — drawdown circuit breaker triggered
- `tail_risk.EMERGENCY` — black swan detected

## Safety Features

- **Circuit breaker**: Auto-halts at -5% drawdown
- **Daily loss limit**: Stops at -$2,000/day
- **Tail risk agent**: Detects flash crashes, vol spikes, liquidity evaporation
- **Platform monitoring**: Checks exchange uptime before trading
- **Correlation guard**: Prevents concentrated positions
- **Backtest gate**: Requires >70% historical win rate to trade
