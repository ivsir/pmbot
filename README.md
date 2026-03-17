# Polymarket 5-Min BTC Up/Down Trading Bot

Automated trading bot for Polymarket's 5-minute Bitcoin Up/Down binary markets. Detects BTC price displacement on CEX exchanges (Binance, Bybit, OKX) and bets on the direction before Polymarket reprices.

## How It Works

Every 5 minutes, Polymarket creates a new market: "Will BTC be higher or lower than the opening price at the end of this window?" The bot:

1. **Monitors BTC price** via websockets on Binance, Bybit, and OKX
2. **Detects displacement** — when BTC moves away from the 5-min window's opening price
3. **Predicts direction** — sigmoid function (scale=50) converts displacement into P(Up)
4. **Places GTC limit order** on Polymarket CLOB within the first 60 seconds of the window
5. **Auto-redeems** winning positions via gasless relayer or direct web3

The edge: BTC moves on CEX first, Polymarket reprices slower. The bot gets in at ~$0.48-0.52 when the fair price is already $0.70+.

## Architecture

```
CEX Websockets (Binance/Bybit/OKX)
        │
        ▼
Displacement Detection (momentum_detector.py)
   BTC price vs window open → sigmoid(scale=50) → P(Up)
   Filters: velocity confirmation, volatility normalization
        │
        ▼
Research Synthesis (research_synthesis.py)
   Bayesian fusion of momentum + spread + liquidity signals
        │
        ▼
Alpha Signal (alpha_signal.py)
   Kelly sizing → entry price → TRADE or SKIP
        │
        ▼
Execution (execution_agent.py + order_book_sniper.py)
   GTC limit order on Polymarket CLOB → fill monitor → auto-redeem
```

## Key Configuration (.env)

| Setting | Value | Description |
|---------|-------|-------------|
| `KELLY_FRACTION` | 0.15 | Base Kelly fraction for position sizing |
| `KELLY_FRACTION_MAX` | 0.20 | Maximum Kelly fraction cap |
| `DISPLACEMENT_SIGMOID_SCALE` | 50.0 | Sigmoid sensitivity — higher = more decisive |
| `MIN_DISPLACEMENT_PCT` | 0.02 | Minimum BTC displacement to trigger signal |
| `MAX_DRAWDOWN_PCT` | 1.0 | Drawdown limit (1.0 = disabled) |
| `LIVE_TRADING_ENABLED` | true | Enable real money trading |
| `MARKET_MODE` | 5min_updown | Target market type |

## Performance

Proven track record on live Polymarket trading:
- **$30 → $500+** in single-day runs during high BTC volatility
- **75% win rate** in backtest over 3 months (25,000+ windows)
- Best during high-volatility periods (5%+ daily BTC range)
- Fill rate: 76-100% depending on market conditions

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your Polymarket API keys, private key, and CEX websocket URLs
```

### 3. Run

```bash
python -m src.main
```

Dashboard available at `http://localhost:8080`

### 4. Deploy to VPS (recommended)

```bash
# Rsync to VPS
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  ./ root@YOUR_VPS:/home/bot/arbitragemarkets/

# On VPS: install deps and start as systemd service
cd /home/bot/arbitragemarkets
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copy systemd service file, enable, start
```

Mexico/US East VPS recommended for lowest latency to Polymarket CLOB (~1-2ms).

## Project Structure

```
├── config/settings.py              # Pydantic settings from .env
├── models/
│   └── displacement_model.joblib   # ML model (sigmoid fallback if unavailable)
├── src/
│   ├── layer0_ingestion/
│   │   ├── cex_websocket.py        # Binance/Bybit/OKX price feeds
│   │   ├── polymarket_client.py    # Polymarket CLOB API + orderbook WS
│   │   ├── event_bus.py            # Async event pub/sub
│   │   └── data_store.py           # Optional Redis/TimescaleDB
│   ├── layer1_research/
│   │   ├── orchestrator.py         # Coordinates research per 5-min window
│   │   ├── momentum_detector.py    # Displacement detection + sigmoid/ML prediction
│   │   ├── displacement_predictor.py # ML model wrapper + sigmoid fallback
│   │   ├── feature_engine.py       # 24-feature vector for ML model
│   │   ├── spread_detector.py      # CEX vs PM spread measurement
│   │   ├── liquidity_scanner.py    # Orderbook depth analysis
│   │   └── research_synthesis.py   # Bayesian fusion + state persistence
│   ├── layer2_signal/
│   │   ├── alpha_signal.py         # Kelly sizing + entry decision
│   │   ├── risk_filter.py          # Drawdown + position limits
│   │   └── signal_validator.py     # Final trade/skip gate
│   ├── layer3_portfolio/
│   │   ├── portfolio_manager.py    # Position tracking + P&L
│   │   └── platform_risk.py       # Auto-redemption (gasless relayer + web3)
│   ├── layer4_execution/
│   │   ├── execution_agent.py      # Order placement orchestration
│   │   ├── order_book_sniper.py    # Optimal price computation
│   │   └── fill_monitor.py         # Fill tracking + timeout handling
│   ├── graph/workflow.py           # LangGraph state machine
│   ├── web/                        # Live dashboard (port 8080)
│   └── main.py                     # Entry point
├── data/
│   ├── bayesian_state.json         # Persisted Bayesian priors
│   └── live_trades.csv             # Trade log for analysis
└── scripts/
    └── train_xgboost.py            # Model retraining script
```

## Why Sigmoid > ML

The bot uses a simple sigmoid function (`1 / (1 + exp(-50 * displacement))`) instead of the ML model for direction prediction. Testing showed:

- **Identical accuracy** — both achieve 75% on the same test set
- **Higher conviction** — sigmoid at scale=50 gives decisive signals, ML hedges
- **Faster execution** — no feature computation delay, enters earlier, gets cheaper fills
- **No overfitting** — zero trainable parameters vs 300+ trees that can memorize noise

The ML model (GradientBoosting, 24 features) is available as fallback and may outperform during specific market regimes.

## Monitoring

Key log events via `structlog`:
- `displacement_detected` — BTC displacement exceeds threshold
- `alpha_signal.entry` — trade signal generated with Kelly sizing
- `graph.trade_executed` — order filled on CLOB
- `system.stats` — periodic stats (wallet, PnL, win rate, fill rate)
- `system.redeem_success` — winning position auto-redeemed
