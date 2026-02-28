# Polymarket Momentum Arbitrage Bot

**A Bayesian multi-signal fusion system that exploits CEX-to-prediction-market latency in 5-minute BTC binary options, achieving a 64-67% live win rate.**

---

## Live Results

| Metric | Value |
|---|---|
| Win rate | **64-67%** (verified over 14+ live trades) |
| Net PnL | **+$24.40** (from $8.11 starting equity) |
| Avg win | +$4.90 |
| Avg loss | -$5.03 |
| Trade frequency | ~1 trade per 5-15 minutes, 24/7 |
| Market | Polymarket BTC 5-Min Up/Down |

---

## How It Works

### The Core Insight

Every 5 minutes, Polymarket opens a new binary market: *"Will Bitcoin go up or down in the next 5 minutes?"* Both sides start priced near $0.50 (coin-flip odds).

But Bitcoin doesn't move randomly second-to-second. If BTC just jumped up on Binance, it's more likely to **keep going up** for the next few minutes than to suddenly reverse. This is **short-term momentum** — a well-documented phenomenon in crypto microstructure.

Polymarket is slow to react. It takes 200-500ms for betting prices to catch up to what centralized exchanges (CEXs) already show. During that delay window, you can buy the correct side at $0.50 when the fair price should be $0.55-$0.65.

### Why This Creates a Repeatable Edge

1. **CEX price leads PM price** — Binance/Bybit/OKX process millions of BTC trades per second. Polymarket processes dozens.
2. **Binary payout is asymmetric** — Pay ~$0.50, receive $1.00 if correct. Even at 60% accuracy, expected value is positive.
3. **Markets reset every 5 minutes** — Each window is an independent bet. Losses don't compound across windows.
4. **The bot trades faster than humans** — Sub-second signal detection to order placement.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                   │
│                   (~20 cycles/second, <50ms)                 │
│                                                              │
│  ┌──────────────┐                                           │
│  │  Layer 0:     │  Binance ─┐                              │
│  │  Ingestion    │  Bybit  ──┼─► Best CEX Tick              │
│  │              │  OKX    ──┘                              │
│  │              │  Polymarket WS ──► Orderbook Cache        │
│  └──────┬───────┘                                           │
│         │                                                    │
│  ┌──────▼───────┐                                           │
│  │  Layer 1:     │  MomentumDetector ─────┐                 │
│  │  Research     │  LatencyArbDetector ───┼─► Bayesian      │
│  │  (parallel)   │  LiquidityScanner ────┘    Fusion        │
│  └──────┬───────┘                                           │
│         │                                                    │
│  ┌──────▼───────┐                                           │
│  │  Layer 2:     │  AlphaSignal (Kelly sizing)              │
│  │  Signal       │  RiskFilter (5 checks)                   │
│  │  Generation   │  Backtester (70% win rate gate)          │
│  └──────┬───────┘                                           │
│         │                                                    │
│  ┌──────▼───────┐  ┌─────────────┐                         │
│  │  Layer 3:     │  │  Layer 4:    │                         │
│  │  Portfolio    │──│  Execution   │──► Polymarket CLOB      │
│  │  Management   │  │  Agent       │                         │
│  └──────────────┘  └─────────────┘                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Background: Resolution Monitor (polls every 15s)     │   │
│  │  ► Detects market settlement → closes positions       │   │
│  │  ► Feeds PnL back to Bayesian prior updates           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Layer 0: Data Ingestion

- **3 CEX WebSockets** (Binance, Bybit, OKX) — captures BTC/USDT book tickers in real-time
- **Polymarket CLOB WebSocket** — streams orderbook snapshots and incremental price updates
- **Best price selection** — picks the exchange with the highest bid across all CEXs
- **Price history** — rolling deque of 600 ticks (~5 minutes) for momentum calculation
- **Deterministic market discovery** — slug-based lookup: `btc-updown-5m-{floor(unix/300)*300}`

### Layer 1: Research Agents

Three independent agents run in parallel on every tick via `asyncio.gather`:

1. **MomentumDetector** (50% Bayesian weight) — primary signal
2. **LatencyArbDetector** (30% weight) — confirmation signal
3. **LiquidityScanner** (20% weight) — safety filter

Results are fused by `ResearchSynthesis` using weighted Bayesian log-odds.

### Layer 2: Signal Generation

A 5-gate validation pipeline that every signal must pass:

1. **Kelly threshold** — fractional Kelly must exceed 0.5%
2. **Edge threshold** — minimum 2% edge over market price
3. **Confidence threshold** — Bayesian probability must exceed 60%
4. **Risk filter** — position limits, daily loss, correlation, drawdown
5. **Backtester gate** — rolling 30-day win rate must exceed 70% (after 20 trades)

### Layer 3: Portfolio Management

- Position lifecycle: PENDING → OPEN → CLOSED
- Correlation bypass for 5-minute markets (each window is independent)
- Concurrent position limit (default: 5)
- PnL tracking and equity curve

### Layer 4: Execution

- **OrderBook Sniper** — selects order strategy based on urgency and spread
- **CLOB Adapter** — wraps py-clob-client SDK with async ThreadPoolExecutor
- **Order types** — GTC (taker), GTD (maker), FOK (market)
- **Resolution Monitor** — polls CLOB every 15s for market settlement

---

## Signal Chain Deep Dive

### 1. Momentum Detector

Converts CEX price momentum into a directional probability:

```
Step 1: Calculate weighted returns
  ret_1m  × 0.40  (strongest short-term predictor)
  ret_3m  × 0.25  (trend persistence)
  ret_5m  × 0.15  (longer context)
  OBI     × 0.20  (order book imbalance × volatility)

Step 2: Composite momentum
  raw_momentum = W₁ₘ·ret₁ₘ + W₃ₘ·ret₃ₘ + W₅ₘ·ret₅ₘ + W_OBI·obi·vol

Step 3: Z-score normalization
  zscore = raw_momentum / rolling_volatility
  (volatility = std of last 500 one-minute returns, floor at 0.001)

Step 4: Time decay
  0-30s into window  → factor = 0.80  (freshest signal)
  30-60s             → factor = 0.75
  60-120s            → factor = 0.65
  120s+              → factor = 0.50  (PM catching up)

Step 5: Sigmoid mapping
  P(Up) = 1 / (1 + exp(-50 × adjusted_zscore))
```

The sigmoid sensitivity of **50** is the critical calibration parameter. It makes the system effectively binary — even a z-score of +0.02 maps to 62% probability.

### 2. Latency Arbitrage Detector

Detects when CEX has moved but Polymarket hasn't caught up:

```
Trigger conditions (ALL required):
  - CEX price move > 0.05% within 500ms window
  - Polymarket orderbook staleness > 200ms
  - Directional agreement with momentum (if both present)

Confidence = lag_factor × move_factor × 0.85 × measurement_bonus
  lag_factor    = min(pm_lag / 500ms, 1.0)
  move_factor   = min(|cex_move| / 0.2%, 1.0)
  measurement_bonus = 1.0 (real timestamps) or 0.7 (estimated)
```

### 3. Liquidity Scanner

Ensures sufficient market depth for safe execution:

```
Metrics:
  total_bid_depth_usd = Σ(price × size) across all bid levels
  total_ask_depth_usd = Σ(price × size) across all ask levels
  OBI = (bid_depth - ask_depth) / (bid_depth + ask_depth)
  max_safe_order = min(bid_depth, ask_depth) × 0.10

Gate: total depth ≥ $10,000 minimum
```

### 4. Bayesian Fusion (Research Synthesis)

Combines all three signals using weighted log-odds:

```
Prior: P₀ = 0.55 (slight momentum bias)

For each signal i ∈ {momentum, latency, liquidity}:
  score_i     = signal-specific score (0 to 1)
  P(obs|win)  = tp_rate × score + (1-tp_rate) × (1-score)
  P(obs|loss) = fp_rate × score + (1-fp_rate) × (1-score)
  LR_i        = P(obs|win) / P(obs|loss)

Fusion:
  log_posterior = log(P₀/(1-P₀)) + 0.50·log(LR_momentum)
                                   + 0.30·log(LR_latency)
                                   + 0.20·log(LR_liquidity)

  P(profitable) = sigmoid(log_posterior)

TP/FP rates (calibrated):
  Momentum:  tp=0.85, fp=0.30
  Latency:   tp=0.80, fp=0.25
  Liquidity: tp=0.90, fp=0.50

Actionability gate: P(profitable) > 0.60 AND edge > 2%
Direction agreement: momentum and latency must agree (or signal rejected)
```

### 5. Kelly Criterion Sizing

```
For a binary market paying $1 on correct outcome:

  payout_ratio = (1 / entry_price) - 1
  f* = (p × b - q) / b

  where p = win probability, q = 1-p, b = payout_ratio

  Fractional Kelly cap: min(f*, 0.043)
  Position size = bankroll × fractional_kelly

Example at 65% confidence, $0.48 entry:
  b = (1/0.48) - 1 = 1.083
  f* = (0.65 × 1.083 - 0.35) / 1.083 = 0.327
  Capped at 0.043 → size = $175 × 0.043 = $7.53
```

---

## Key Parameters & Calibration

The system's 67% win rate emerges from **5 interdependent parameters calibrated as a unit**. Changing any one without recalibrating the others breaks the system.

| Parameter | Value | What It Controls |
|---|---|---|
| Sigmoid sensitivity | **50.0** | How aggressively momentum maps to probability |
| Time decay curve | **0.80 → 0.50** | Signal strength vs time into window |
| PM price filter | **$0.20 - $0.80** | Only trade when market is uncertain |
| Bayesian TP/FP rates | **0.85/0.80/0.90** | How much to trust each signal source |
| Entry window | **0 - 240 seconds** | When to trade within each 5-min window |

### Why These Are Fragile

The sigmoid at sensitivity=50 saturates extremely fast:

| Z-score | P(Up) |
|---|---|
| 0.00 | 50.0% |
| 0.02 | 62.2% |
| 0.05 | 92.4% |
| 0.10 | 99.3% |

This means the system is **effectively binary** — either momentum is positive (strong BUY_YES) or negative (strong BUY_NO). The Bayesian fusion, direction agreement filter, and PM price filter act as safety nets to block the ~33% of signals where this binary classification is wrong.

**Changing sensitivity from 50 to 5** makes the sigmoid nearly linear: a z-score of +0.5 maps to only ~52.5% instead of ~100%. The bot goes quiet because signals never reach the 60% confidence threshold.

**Changing the entry window** to final 60 seconds means PM prices are already $0.80-$0.90 (not $0.50). The Kelly sizing, PM price filter, and payout math all assume entry near $0.50. The whole calibration breaks.

**Changing OBI weights** shifts the z-score distribution because OBI contributes `0.20 × obi` to the z-score (volatility cancels out), while price returns contribute `returns/vol`. Different distributions invalidate the TP/FP rates.

---

## The Math: Why 64-67% Is Profitable

### Expected Value Per Trade

```
At 65% win rate with $0.48 average entry:
  Win:  +$0.52 per share (pays $1.00, cost $0.48)
  Loss: -$0.48 per share

  EV = 0.65 × $0.52 - 0.35 × $0.48
     = $0.338 - $0.168
     = +$0.17 per share

  Minus ~1.5% taker fee on $0.48: -$0.007
  Net EV ≈ +$0.16 per share per trade
```

### Annualized Returns

```
At 1 trade per 10 minutes, 24/7:
  144 trades/day × $0.16 EV × $7.50 size ≈ $172/day

  (Actual trade frequency varies — signal must pass 5 gates)
  Conservative estimate: 20-40 qualifying trades/day
  → $24-$48/day at $7.50/trade
```

### Kelly Criterion Justification

```
Full Kelly at 65% win, 1.08 payout ratio:
  f* = (0.65 × 1.08 - 0.35) / 1.08 = 0.325 (32.5%)

We use 4.3% fractional Kelly (1/7.5 of full Kelly).
This sacrifices ~60% of theoretical growth rate
but reduces drawdown risk by ~87%.

At 4.3% Kelly with $175 bankroll:
  Max position = $7.53
  Max single-trade loss = $7.53 (7.5% of equity at $100)
  Expected max drawdown (20-trade window): ~15-20%
```

---

## Known Limitations

### 1. Taker Fees Erode Edge
Polymarket charges up to **1.56% taker fee** at 50-cent prices (fee formula: `C × 0.25 × (p × (1-p))²`). On a $0.48 entry, this is ~$0.007/share — roughly 4% of the $0.17 per-share EV. Maker orders (zero fees + rebates) would capture 100% of edge.

### 2. Stale Orderbook Risk
The WebSocket stores orderbooks under both `token_id` and `condition_id`. Incremental `price_changes` update the token-level book but may not propagate to the condition-level book. The research pipeline reads condition-level orderbooks — if these go stale, the OBI signal becomes biased, systematically pushing trades in one direction.

### 3. Paper vs Live Fill Divergence
Paper trading records **instant fills** at the computed price. Live trading polls for 5 seconds (taker) or 30 seconds (maker) with 250ms intervals. Paper results consistently overstate live performance because they assume no slippage and 100% fill rate.

### 4. Sensitivity = 50 Is a Razor's Edge
The sigmoid saturates so quickly that the system is effectively a binary classifier. This works well when BTC has clear short-term momentum but fails during:
- Low-volatility chop (z-scores near zero → signals flip rapidly)
- Flash crashes/pumps (extreme z-scores → 99.9% confidence on potentially wrong direction)
- Regime changes (volatility shifts invalidate the rolling std normalization)

### 5. Static TP/FP Rates
The Bayesian fusion uses hardcoded true-positive/false-positive rates. The `update_likelihoods` method exists but converges slowly (alpha=0.05, ~20 trades to move halfway). If market microstructure changes (e.g., PM reduces latency), the rates become miscalibrated before the bot can adapt.

### 6. Single-Asset Concentration
All trades are on BTC Up/Down. A sustained regime of low BTC volatility would eliminate momentum signals entirely, producing zero qualifying trades.

---

## Future Research Directions

### 1. Near-Settlement Maker Strategy
Trade in the **final 60 seconds** of each window using `GTD + post_only` maker orders at $0.90-$0.95. By this point, BTC direction is ~85% determined. Zero taker fees, plus daily maker rebates. Requires recalibrating the entire parameter set for end-of-window dynamics.

### 2. Adaptive Sigmoid Sensitivity
Replace the fixed sensitivity=50 with a volatility-regime-dependent value:
- High vol (>1% hourly): sensitivity=30 (wider z-score range, less saturation)
- Normal vol: sensitivity=50 (current calibration)
- Low vol (<0.2% hourly): sensitivity=80 (amplify weak signals)

### 3. Online Bayesian Learning
Replace static TP/FP rates with a faster online update (alpha=0.15-0.20) using the resolution feedback loop that already exists. The `update_likelihoods` method in `ResearchSynthesis` supports this — only the learning rate needs adjustment, with guardrails to prevent divergence.

### 4. Multi-Asset Expansion
Polymarket has introduced ETH and SOL Up/Down 5-minute markets. The same momentum-latency framework should apply, but each asset needs its own sensitivity calibration (different volatility profiles, different CEX liquidity).

### 5. Cross-Window Momentum Persistence
Currently each window is treated as independent. In reality, BTC momentum persists across multiple 5-minute windows (autocorrelation). A meta-signal tracking the last N window outcomes could improve the Bayesian prior from 0.55 to a dynamic value.

### 6. Reinforcement Learning for Parameter Tuning
Use the resolution feedback loop as a reward signal to train an RL agent that adjusts the 5 key parameters (sensitivity, time decay, TP/FP rates, entry window, price filter) as a coordinated system rather than individually.

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Polymarket CLOB API credentials (API key, secret, passphrase)
- Polygon wallet with USDC (for live trading)
- At least one CEX API key (Binance, Bybit, or OKX) for price feeds
- Optional: Redis, TimescaleDB (for persistence — bot works without them)

### Configuration

Create a `.env` file in the project root:

```env
# Polymarket CLOB
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_API_PASSPHRASE=your_passphrase

# Polygon Wallet (for live trading)
POLYGON_PRIVATE_KEY=your_private_key
POLY_FUNDER_ADDRESS=your_proxy_address
LIVE_TRADING_ENABLED=true

# CEX Price Feeds (at least one required)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Sizing
MAX_POSITION_USD=175        # Bankroll for Kelly sizing
KELLY_FRACTION=0.043        # 4.3% fractional Kelly cap

# Strategy
MARKET_MODE=5min_updown
MOMENTUM_SIGMOID_SENSITIVITY=50.0
```

### Running

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Clear bytecode cache (important after any code changes)
find . -type d -name __pycache__ -exec rm -rf {} +

# Start the bot
python -m src.main

# Dashboard available at http://localhost:8080
```

### Monitoring

- **Dashboard**: `http://localhost:8080` — live equity curve, trade history, signal visualization
- **Logs**: `bot.log` — structured logging via `structlog`
- **Key log events**:
  - `portfolio.position_opened` — new trade placed
  - `graph.trade_executed` — order filled on CLOB
  - `portfolio.position_closed` — position resolved with PnL
  - `system.stats` — periodic summary (every 100 cycles)

---

## Project Structure

```
src/
├── layer0_ingestion/
│   ├── cex_websocket.py      # Binance/Bybit/OKX WebSocket feeds
│   ├── polymarket_client.py   # CLOB WebSocket + REST + market discovery
│   ├── event_bus.py           # Internal pub/sub for cross-layer communication
│   └── data_store.py          # Redis/TimescaleDB persistence (optional)
│
├── layer1_research/
│   ├── momentum_detector.py   # CEX momentum → P(Up) via sigmoid
│   ├── latency_arb.py         # CEX-PM lag detection
│   ├── liquidity_scanner.py   # Orderbook depth analysis
│   ├── spread_detector.py     # Monthly market spread detection (unused in 5min mode)
│   ├── research_synthesis.py  # Bayesian fusion of all signals
│   └── orchestrator.py        # Runs all agents in parallel per tick
│
├── layer2_signal/
│   ├── alpha_signal.py        # Kelly sizing + entry price calculation
│   ├── risk_filter.py         # Position limits, daily loss, correlation
│   ├── signal_validator.py    # 5-gate validation pipeline
│   └── backtester.py          # Rolling win rate gate (70% threshold)
│
├── layer3_portfolio/
│   ├── portfolio_manager.py   # Position lifecycle + PnL tracking
│   ├── correlation_monitor.py # Cross-position correlation
│   ├── tail_risk.py           # Extreme move detection
│   └── platform_risk.py       # CEX/PM health monitoring
│
├── layer4_execution/
│   ├── execution_agent.py     # Trade execution orchestration
│   ├── clob_adapter.py        # py-clob-client SDK wrapper (async)
│   └── order_book_sniper.py   # Order strategy selection
│
├── graph/
│   └── workflow.py            # LangGraph state machine (main loop)
│
├── web/
│   ├── server.py              # Dashboard HTTP server
│   └── static/app.js          # Dashboard frontend
│
└── main.py                    # Entry point + market discovery loops
```

---

## Citation

If you use this system or its methodology in your research, please cite:

```
Polymarket Momentum Arbitrage Bot
https://github.com/[your-repo]
A Bayesian multi-signal fusion system for prediction market arbitrage.
```

---

## License

MIT — see LICENSE file for details.
