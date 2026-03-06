# Architecture: Polymarket 5-Min BTC Displacement Bot

**Status**: Live-profitable, March 2026
**Market**: Polymarket 5-minute BTC Up/Down binary options
**Model**: Gradient Boosted Classifier with isotonic calibration (24 features)
**Backtest**: 76.4% WR, 34,811 trades, +$11,688 over 6 months

---

## The Edge

Polymarket lists rolling 5-minute binary markets: *"Will BTC be up or down at the end of this window?"*

- YES token pays $1 if BTC closes above the window open price
- NO token pays $1 if BTC closes below
- Tokens open near 50c (equal odds)

**The inefficiency**: Within the first 60 seconds of a window, BTC has already moved on Binance/Bybit/OKX, but Polymarket tokens still trade near 50c. If BTC is up +0.05% from window open at T+30s, there's a ~73% chance it finishes the window up. But the YES token is still priced at 50-52c.

The bot buys the directionally correct token at a discount, holds to resolution, and collects $1 on wins.

### Why displacement persists

BTC price displacement within a 5-minute window is mean-*continuing*, not mean-reverting:

| Displacement at entry | P(same direction at close) | Backtest trades |
|----------------------|---------------------------|-----------------|
| 0.02% | 68% | ~25,000 |
| 0.05% | 73% | ~12,000 |
| 0.10% | 84% | ~4,000 |
| 0.20% | 88% | ~1,500 |
| 0.50% | 94% | ~300 |

This is the core alpha. Momentum at the 5-minute horizon is persistent because:
1. Institutional order flow arrives in waves (TWAP/VWAP), sustaining directional pressure
2. 5 minutes is too short for mean reversion to dominate
3. Retail reaction to price moves amplifies the initial displacement

### Why PM misprices

Polymarket's 5-min markets are thinly traded prediction markets, not efficient derivatives:
- Low liquidity (often <$10K depth per side)
- No market makers running Black-Scholes — tokens trade on retail sentiment
- PM orderbook updates within ~21ms of CEX, but *prices* don't reprice until human traders react
- The bot doesn't exploit latency (21ms is too fast) — it exploits the *behavioral* lag of PM participants adjusting their limit orders

---

## System Architecture

```
Binance/Bybit/OKX WebSocket ticks (~6/sec)
    |
    v
+================================+
|  Layer 0: INGESTION            |
|  CEXWebSocketManager           |  3 exchange feeds, rolling 5-min history
|  PolymarketClient              |  CLOB orderbook streaming + REST trading
|  EventBus                      |  In-memory pub/sub (optional Kafka)
+================================+
    |
    v
+================================+
|  Layer 1: RESEARCH             |
|  MomentumDetector              |  Displacement + ML model -> P(Up)
|    FeatureEngine (24 features) |  Returns, vol, volume, structure, trend
|    DisplacementPredictor       |  GBC model or sigmoid fallback
|  LatencyArbDetector            |  PM staleness confirmation
|  LiquidityScanner              |  Orderbook depth safety
|  ResearchSynthesis             |  Bayesian fusion (bypassed for P(Up))
+================================+
    |
    v
+================================+
|  Layer 2: SIGNAL               |
|  AlphaSignalGenerator          |  Kelly sizing: f* = (p*b - q) / b
|  SignalValidator               |  Edge, Kelly, risk gates
|  RiskFilter                    |  Position limits, correlation, drawdown
+================================+
    |
    v
+================================+
|  Layer 3: PORTFOLIO            |
|  PortfolioManager              |  Max 5 concurrent, equity tracking
|  ResolutionMonitor             |  Polls CLOB every 15s for settlement
|  AutoRedemption                |  On-chain CTF token redemption
+================================+
    |
    v
+================================+
|  Layer 4: EXECUTION            |
|  ExecutionAgent                |  Taker (0-240s) / Maker (240-300s)
|  OrderBookSniper               |  5-share minimum, tick grid snapping
|  CLOBAdapter                   |  EIP-712 signing, proxy wallet routing
+================================+
```

### LangGraph Pipeline

The system runs as a LangGraph state machine at ~20 cycles/sec:

```
ingest_data -> collect_feedback -> research_and_risk (parallel)
            -> generate_signal -> [TRADE] -> execute
                                  [SKIP]  -> log_skip
            -> report_metrics -> END
```

One signal per 5-min window per market (duplicate signals blocked by correlation filter).

---

## ML Model: Displacement Predictor

### The problem with sigmoid

The original model used a simple sigmoid: `P(Up) = 1 / (1 + exp(-10 * displacement))`. This treats all 0.05% displacements identically regardless of context. But a 0.05% move in a quiet overnight session is far more meaningful than 0.05% during a volatile US open.

### GradientBoostingClassifier + Isotonic Calibration

**Model**: `sklearn.ensemble.GradientBoostingClassifier` wrapped in `CalibratedClassifierCV(method='isotonic')`

**Hyperparameters**:
- 300 trees, max depth 4, learning rate 0.05
- Subsample 0.8 (stochastic gradient boosting)
- Min samples per leaf: 50 (prevents overfitting to rare regimes)

**Training**: Walk-forward on 6 months of Binance 1-min klines (~52K 5-min windows). Train on months 1-4, validate on months 5-6.

**Metrics**:
- Accuracy: 70.1%
- AUC: 0.768
- Brier score: 0.196 (vs sigmoid 0.202 — better calibrated)
- Precision: 70.8%, Recall: 67.6%

### 24-Feature Vector

Features are computed identically for backtest (`compute_from_candles`) and live trading (`compute_from_ticks` via `RollingCandleBuffer`):

| Group | Features | Why it matters |
|-------|----------|---------------|
| **Core** | `displacement_pct` | Primary signal — how far BTC moved from window open |
| **Multi-timeframe returns** | `return_1m, return_2m, return_3m, return_5m` | Momentum persistence across horizons |
| **Velocity** | `velocity_15s, acceleration, displacement_vs_1m` | Is the move accelerating or stalling? |
| **Volatility regime** | `rolling_stdev_5m, rolling_stdev_15m, z_displacement` | 0.05% in low-vol = strong signal; in high-vol = noise |
| **Volume** | `volume_ratio_1m, volume_ratio_5m, volume_trend` | High-volume moves have higher follow-through |
| **Candle structure** | `body_ratio, upper_wick_ratio, lower_wick_ratio` | Strong bodies = conviction; long wicks = rejection |
| **Trend alignment** | `trend_15m, trend_30m, trend_60m` | With-trend displacements persist +2-5% more |
| **Time of day** | `hour_sin, hour_cos, is_us_session` | BTC behavior differs by session (Asia/Europe/US) |
| **Window timing** | `secs_into_window` | Later entries have more data but worse prices |

### Fallback

If the model file (`models/displacement_model.joblib`) is missing or prediction fails, the system falls back to the sigmoid automatically. The `DisplacementPredictor` class handles this transparently.

### Why GBC over alternatives

- **Random Forest** compresses probabilities toward 0.50 — bad for Kelly sizing which needs calibrated P(Up)
- **Logistic Regression** can't capture feature interactions (low-vol + high-volume + with-trend = strong signal)
- **Neural nets** overfit on 52K samples and don't provide probability calibration out of the box
- **GBC + isotonic calibration** gives the best Brier score and calibrated probabilities for Kelly

---

## Kelly Criterion Sizing

The bot sizes every trade using the Kelly criterion — the mathematically optimal bet size for maximizing long-run bankroll growth:

```
f* = (p * b - q) / b

where:
  p = P(win)           # from ML model (fair_up_prob)
  q = 1 - p
  b = payout ratio     # (1 / entry_price) - 1
```

**Critical**: The alpha signal uses `fair_up_prob` from the ML model directly, NOT the Bayesian `combined_probability`. The Bayesian synthesis dampens displacement probability (~0.63 -> 0.50), which kills Kelly sizing and blocks +EV trades.

### Example

BTC displaced +0.08%, ML model says P(Up) = 0.65, YES token at 48c:

```
b = (1 / 0.48) - 1 = 1.083
f* = (0.65 * 1.083 - 0.35) / 1.083 = 0.327
kelly = min(0.327, 0.115) = 0.115  (capped at 11.5%)
size = wallet_balance * 0.115
```

### Entry gates

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Kelly > 0.005 | Skip near-zero edge | Not worth the execution cost |
| Edge >= 1% | Minimum mispricing | `abs(fair_prob - pm_mid) * 100` |
| Size >= $0.10 | Platform minimum | Below this, CLOB rejects |
| Entry price 5c-80c | Price quality band | Below 5c = token nearly worthless; above 80c = no edge |

---

## Execution

### Taker strategy (0-240s into window)

For the first 4 minutes, speed matters — the bot crosses the spread:
- Places a limit order at or above the best ask (urgency = 0.95)
- Fills within ~1.6 seconds average
- CLOB minimum: 5 shares per order (~$2.50 at 50c tokens)

### Maker strategy (240-300s into window)

In the final minute, the bot posts passive limit orders:
- GTD (Good-Till-Date) expiry at window end
- Post-only = zero taker fees
- Price capped at 85-95c (ensures profitable payout even on expensive tokens)

### Proxy wallet architecture

Polymarket uses an EIP-1167 minimal proxy pattern:
- Factory (`0xaB45c...052`) deploys proxy wallets via CREATE2
- Proxy is derived deterministically from EOA address
- CTF tokens are held by the proxy, not the EOA
- To redeem: call `factory.proxy()` which routes to the proxy wallet
- The bot auto-redeems winning positions after resolution

---

## Position Lifecycle

```
SIGNAL FIRED
    |
    v
ORDER PLACED (CLOB)  -->  Fill monitor polls every 500ms
    |
    v
FILLED (position OPEN)
    |  fill_price * filled_size = actual cost (not estimated)
    |
    v
RESOLUTION MONITOR polls every 15s
    |
    +---> Market resolves YES --> P&L = +($1 * shares - cost)
    |                             Auto-redeem winning tokens
    |
    +---> Market resolves NO  --> P&L = -(cost)
    |                             Skip redemption (tokens worthless)
    |
    +---> 900s stale timer    --> Force close, P&L = -(cost)
                                  Safety net for missed resolutions
```

Resolution typically takes 5-8 minutes after window ends (on-chain settlement).

---

## Configuration

All parameters live in `config/settings.py` with `.env` overrides (`.env` takes precedence):

```env
# Displacement model
MIN_DISPLACEMENT_PCT=0.02    # Minimum BTC move to trigger signal
ML_MODEL_ENABLED=true        # Use ML model vs sigmoid fallback

# Signal filters (all optional)
REQUIRE_VELOCITY_CONFIRM=false   # Price velocity must agree with displacement
REQUIRE_VOL_NORMALIZATION=false  # Z-score gating
REQUIRE_OBI_CONFIRM=false        # PM orderbook imbalance must agree

# Kelly sizing
KELLY_FRACTION=0.12          # 12% fractional Kelly cap
MAX_BID_USD=2.50             # Per-trade cap
MAX_CONCURRENT_POSITIONS=5   # Portfolio limit

# Execution
LIVE_TRADING_ENABLED=true
MARKET_MODE=5min_updown
```

---

## File Map

| File | Role |
|------|------|
| `src/main.py` | System boot, market rotation, resolution loop |
| `src/layer0_ingestion/cex_websocket.py` | Binance/Bybit/OKX WebSocket feeds |
| `src/layer0_ingestion/polymarket_client.py` | CLOB orderbook + REST trading |
| `src/layer1_research/momentum_detector.py` | Core: displacement -> ML model -> P(Up) |
| `src/layer1_research/feature_engine.py` | 24-feature extraction (candle + tick modes) |
| `src/layer1_research/displacement_predictor.py` | Model loader with sigmoid fallback |
| `src/layer1_research/orchestrator.py` | Coordinates research agents per tick |
| `src/layer2_signal/alpha_signal.py` | Kelly sizing, uses fair_up_prob directly |
| `src/layer2_signal/risk_filter.py` | Position limits, correlation, drawdown |
| `src/layer3_portfolio/portfolio_manager.py` | Equity tracking, resolution monitor |
| `src/layer4_execution/execution_agent.py` | Taker/maker routing |
| `src/layer4_execution/clob_adapter.py` | CLOB SDK wrapper, proxy redemption |
| `scripts/train_displacement_model.py` | ML training pipeline |
| `scripts/watchdog.sh` | 24h auto-restart daemon |
