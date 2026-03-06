#!/usr/bin/env python3
"""
6-Month Backtest: Original Momentum Model vs Displacement Model

Uses real Binance 1-minute BTC/USDT klines to simulate both models
on Polymarket 5-min Up/Down windows.

Data: ~6 months of 1-min candles (~260,000 data points)
Windows: Every 5-minute interval (00:00, 00:05, 00:10, ...)
Entry: T+1min (60s into the 5-min window)
"""

import math
import time
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import requests

# ─── Config ───────────────────────────────────────────────────────────────────

# Original model (commit cb5421b)
OLD_SENSITIVITY = 50.0
OLD_W_1M = 0.4
OLD_W_3M = 0.25
OLD_W_5M = 0.15
OLD_W_OBI = 0.20
OLD_MIN_CONFIDENCE = 0.60      # original .env default
OLD_KELLY_CAP = 0.043          # original settings.py
OLD_MIN_EDGE_PCT = 2.0         # min_edge_pct=0.02 * 100
OLD_PRICE_QUALITY_MIN = 0.20
OLD_PRICE_QUALITY_MAX = 0.80
OLD_USE_BAYESIAN = True        # original used combined_probability

# New model (displacement)
NEW_SCALE = 10.0
NEW_MIN_DISPLACEMENT = 0.02    # percent
NEW_MIN_CONFIDENCE = 0.55      # current .env
NEW_KELLY_CAP = 0.115          # current settings
NEW_MIN_EDGE_PCT = 2.0
NEW_PRICE_QUALITY_MIN = 0.05   # lowered for 5-min tokens
NEW_PRICE_QUALITY_MAX = 0.80
NEW_USE_BAYESIAN = False        # uses fair_up_prob directly

BANKROLL = 100.0               # starting bankroll for sizing

# PM mid-price simulation parameters
# At T+1min, PM partially reflects displacement. We model this as:
#   pm_mid = 0.50 + efficiency * (fair_up - 0.50)
# efficiency=0 means PM is stuck at 50/50, efficiency=1 means PM fully reflects our model
PM_EFFICIENCY = 0.40  # PM absorbs ~40% of true displacement signal by T+1min


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines from Binance between start_ms and end_ms."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_klines.extend(batch)
        # Next batch starts after last candle
        current_start = batch[-1][0] + 60_000  # 1-min interval
        if len(batch) < 1000:
            break
        time.sleep(0.15)  # rate limit

    return all_klines


@dataclass
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_klines(raw: list) -> list[Candle]:
    """Parse raw Binance klines into Candle objects."""
    candles = []
    for k in raw:
        candles.append(Candle(
            open_time_ms=k[0],
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
        ))
    return candles


# ─── Original Momentum Model (cb5421b) ───────────────────────────────────────

def old_model_signal(
    candles_before_entry: list[Candle],
    return_history: deque,
) -> tuple[float | None, str | None, float]:
    """
    Simulate the original momentum model.

    Returns (fair_up_prob, direction, adjusted_zscore) or (None, None, 0)
    """
    if len(candles_before_entry) < 6:
        return None, None, 0.0

    # Calculate returns: 1m, 3m, 5m
    latest_price = candles_before_entry[-1].close

    def calc_return(lookback_candles: int) -> float | None:
        idx = max(0, len(candles_before_entry) - 1 - lookback_candles)
        old_price = candles_before_entry[idx].close
        if old_price == 0:
            return None
        return (latest_price - old_price) / old_price

    ret_1m = calc_return(1)
    ret_3m = calc_return(3)
    ret_5m = calc_return(5)

    if ret_1m is None:
        return None, None, 0.0

    return_history.append(ret_1m)

    # Rolling volatility
    if len(return_history) < 5:
        vol = 0.001
    else:
        vol = float(np.std(list(return_history)))
    if vol < 1e-8:
        vol = 0.001

    # OBI simulation: use price vs mid of high/low as proxy
    # In live, this comes from PM orderbook. In backtest, approximate as 0.
    obi = 0.0

    # Raw momentum
    raw_momentum = (
        OLD_W_1M * (ret_1m or 0.0)
        + OLD_W_3M * (ret_3m or 0.0)
        + OLD_W_5M * (ret_5m or 0.0)
        + OLD_W_OBI * obi * vol
    )
    zscore = raw_momentum / vol

    # Time factor: at T+1min (60s into window), secs_into_window=60
    time_factor = 0.75  # 30-60s bracket

    adjusted_zscore = zscore * time_factor

    # Sigmoid (clamp to avoid overflow — exactly the problem in live)
    x = -OLD_SENSITIVITY * adjusted_zscore
    x = max(-500, min(500, x))
    fair_up_prob = 1.0 / (1.0 + math.exp(x))
    fair_up_prob = max(0.01, min(0.99, fair_up_prob))

    direction = "BUY_YES" if fair_up_prob > 0.50 else "BUY_NO"

    return fair_up_prob, direction, adjusted_zscore


def old_model_bayesian(fair_up_prob: float) -> float:
    """Simulate Bayesian dampening from ResearchSynthesis."""
    # The old model passed through ResearchSynthesis which applied Bayesian fusion
    # with BASE_PRIOR=0.55, and spread_score derived from fair_up_prob

    # Spread score: 1 - exp(-spread_pct/5) where spread_pct = |fair - 0.5| * 100
    spread_pct = abs(fair_up_prob - 0.5) * 100
    spread_score = float(np.clip(1 - np.exp(-spread_pct / 5), 0, 1))

    # Latency and liquidity scores at neutral (no real data in backtest)
    latency_score = 0.3
    liquidity_score = 0.2

    # Bayesian fusion (simplified from research_synthesis.py)
    base_prior = 0.55

    # TP/FP rates from original
    spread_tp, spread_fp = 0.85, 0.30
    latency_tp, latency_fp = 0.80, 0.25
    liquidity_tp, liquidity_fp = 0.90, 0.50

    def lr(score, tp, fp):
        p_true = tp * score + (1 - tp) * (1 - score)
        p_false = fp * score + (1 - fp) * (1 - score)
        if p_false < 1e-10:
            return 10.0
        return p_true / p_false

    lr_s = lr(spread_score, spread_tp, spread_fp)
    lr_l = lr(latency_score, latency_tp, latency_fp)
    lr_q = lr(liquidity_score, liquidity_tp, liquidity_fp)

    log_prior = np.log(base_prior / (1 - base_prior))
    log_post = (
        log_prior
        + 0.50 * np.log(lr_s + 1e-10)
        + 0.30 * np.log(lr_l + 1e-10)
        + 0.20 * np.log(lr_q + 1e-10)
    )

    combined = 1.0 / (1.0 + np.exp(-log_post))
    return float(np.clip(combined, 0.01, 0.99))


# ─── New Displacement Model ──────────────────────────────────────────────────

def new_model_signal(
    window_open_price: float,
    entry_price: float,
) -> tuple[float | None, str | None, float]:
    """
    Simulate the displacement model.

    Returns (fair_up_prob, direction, displacement_pct) or (None, None, 0)
    """
    if window_open_price <= 0:
        return None, None, 0.0

    displacement_pct = (entry_price - window_open_price) / window_open_price * 100

    # Filter: skip noise
    if abs(displacement_pct) < NEW_MIN_DISPLACEMENT:
        return None, None, displacement_pct

    # Sigmoid
    fair_up_prob = 1.0 / (1.0 + math.exp(-NEW_SCALE * displacement_pct))
    fair_up_prob = max(0.01, min(0.99, fair_up_prob))

    direction = "BUY_YES" if fair_up_prob > 0.50 else "BUY_NO"

    return fair_up_prob, direction, displacement_pct


# ─── Kelly & Entry Logic ─────────────────────────────────────────────────────

def kelly_entry(
    win_prob: float,
    pm_mid: float,
    direction: str,
    model: str,  # "old" or "new"
    bankroll: float,
) -> tuple[bool, float, float]:
    """
    Simulate Kelly sizing and entry decision.

    Returns (should_enter, size_usd, kelly_frac)
    """
    if model == "old":
        min_conf = OLD_MIN_CONFIDENCE
        kelly_cap = OLD_KELLY_CAP
        min_edge = OLD_MIN_EDGE_PCT
        pq_min, pq_max = OLD_PRICE_QUALITY_MIN, OLD_PRICE_QUALITY_MAX
    else:
        min_conf = NEW_MIN_CONFIDENCE
        kelly_cap = NEW_KELLY_CAP
        min_edge = NEW_MIN_EDGE_PCT
        pq_min, pq_max = NEW_PRICE_QUALITY_MIN, NEW_PRICE_QUALITY_MAX

    # Entry price
    if direction == "BUY_YES":
        entry_price = pm_mid  # approximate: buy Yes at mid
    else:
        entry_price = 1.0 - pm_mid  # buy No at 1-mid

    # Edge
    if direction == "BUY_YES":
        spread_pct = (win_prob - pm_mid) * 100
    else:
        spread_pct = ((1.0 - win_prob) - (1.0 - pm_mid)) * 100
        # Simplifies to: (pm_mid - win_prob) * 100
        # But edge check uses |fair - implied|
        spread_pct = abs(win_prob - pm_mid) * 100  # actually just the absolute difference

    edge_pct = abs(win_prob - pm_mid) * 100

    # Payout ratio
    if 0 < entry_price < 1:
        payout_ratio = (1.0 / entry_price) - 1.0
    else:
        return False, 0.0, 0.0

    # Kelly
    q = 1.0 - win_prob
    kelly_raw = (win_prob * payout_ratio - q) / payout_ratio
    kelly_raw = max(0.0, kelly_raw)
    kelly_frac = min(kelly_raw, kelly_cap)

    # Size: flat $1 bet for backtest comparison (avoids compounding overflow)
    size = 1.0

    # Price quality check
    price_quality_ok = pq_min <= entry_price <= pq_max

    # Entry decision
    should_enter = (
        kelly_frac > 0.005
        and edge_pct >= min_edge
        and win_prob >= min_conf
        and size >= 0.10  # min trade
        and price_quality_ok
    )

    return should_enter, size, kelly_frac


# ─── PM Mid-Price Simulation ─────────────────────────────────────────────────

def simulate_pm_mid(fair_up_prob: float, efficiency: float = PM_EFFICIENCY) -> float:
    """Simulate what PM mid-price would be at T+1min.

    PM market makers partially absorb the displacement by T+1min.
    efficiency=0.0 → PM stays at 0.50
    efficiency=1.0 → PM fully reflects our fair_up_prob
    """
    # Add some noise to PM price
    noise = np.random.normal(0, 0.02)  # 2% noise
    pm_mid = 0.50 + efficiency * (fair_up_prob - 0.50) + noise
    return float(np.clip(pm_mid, 0.05, 0.95))


# ─── Main Backtest ────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    window_start_ms: int
    direction: str
    fair_up_prob: float
    pm_mid: float
    entry_price_token: float
    size_usd: float
    kelly: float
    won: bool
    pnl: float
    displacement_pct: float = 0.0
    zscore: float = 0.0
    model: str = ""


def run_backtest():
    """Run the full 6-month backtest."""
    print("=" * 80)
    print("6-MONTH BACKTEST: Original Momentum vs Displacement Model")
    print("=" * 80)

    # Date range: 6 months back from today
    now_ms = int(time.time() * 1000)
    six_months_ago_ms = now_ms - (180 * 24 * 60 * 60 * 1000)  # ~180 days

    print(f"\nFetching Binance 1-min BTC/USDT klines...")
    print(f"Period: {time.strftime('%Y-%m-%d', time.gmtime(six_months_ago_ms/1000))} → "
          f"{time.strftime('%Y-%m-%d', time.gmtime(now_ms/1000))}")

    raw_klines = fetch_binance_klines("BTCUSDT", "1m", six_months_ago_ms, now_ms)
    candles = parse_klines(raw_klines)
    print(f"Fetched {len(candles):,} candles")

    if len(candles) < 1000:
        print("ERROR: Not enough data")
        return

    # Build time-indexed lookup: open_time_ms → candle
    candle_map: dict[int, Candle] = {}
    for c in candles:
        candle_map[c.open_time_ms] = c

    # Generate 5-min windows
    # Windows start on round 5-min marks
    first_ms = candles[0].open_time_ms
    last_ms = candles[-1].open_time_ms

    # Align to 5-min boundary
    window_interval = 5 * 60 * 1000
    first_window = (first_ms // window_interval + 1) * window_interval

    windows = []
    t = first_window
    while t + window_interval <= last_ms:
        windows.append(t)
        t += window_interval

    print(f"Total 5-min windows: {len(windows):,}")

    # Run both models on each window
    old_trades: list[TradeResult] = []
    new_trades: list[TradeResult] = []
    old_return_history: deque[float] = deque(maxlen=500)

    old_bankroll = BANKROLL
    new_bankroll = BANKROLL

    old_skipped_reasons = {"no_signal": 0, "kelly_reject": 0, "no_data": 0}
    new_skipped_reasons = {"no_signal": 0, "kelly_reject": 0, "no_data": 0}

    np.random.seed(42)  # reproducible PM noise

    for i, ws_ms in enumerate(windows):
        # Window: ws_ms to ws_ms + 5min
        # Entry at T+1min: ws_ms + 60000
        # Resolve at T+5min: ws_ms + 300000

        entry_time_ms = ws_ms + 60_000
        resolve_time_ms = ws_ms + 300_000

        # Get candles
        window_open_candle = candle_map.get(ws_ms)
        entry_candle = candle_map.get(entry_time_ms)
        resolve_candle = candle_map.get(resolve_time_ms)

        if not window_open_candle or not entry_candle or not resolve_candle:
            old_skipped_reasons["no_data"] += 1
            new_skipped_reasons["no_data"] += 1
            continue

        window_open_price = window_open_candle.open
        entry_btc_price = entry_candle.close  # BTC price at T+1min
        resolve_price = resolve_candle.close   # BTC price at T+5min

        # Outcome: Up if resolve > open, Down otherwise
        outcome_up = resolve_price > window_open_price

        # ── OLD MODEL ──
        # Get candles leading up to entry (for momentum calculation)
        lookback_times = [entry_time_ms - i * 60_000 for i in range(10, -1, -1)]
        candles_before = [candle_map[t] for t in lookback_times if t in candle_map]

        old_fair, old_dir, old_z = old_model_signal(candles_before, old_return_history)

        if old_fair is not None and old_dir is not None:
            # Bayesian dampening
            if OLD_USE_BAYESIAN:
                old_win_prob = old_model_bayesian(old_fair)
            else:
                old_win_prob = old_fair if old_dir == "BUY_YES" else 1.0 - old_fair

            # Simulate PM mid
            old_pm_mid = simulate_pm_mid(old_fair)

            # Spread check
            old_spread = abs(old_fair - old_pm_mid) * 100
            if old_spread < OLD_MIN_EDGE_PCT:
                old_skipped_reasons["no_signal"] += 1
            else:
                # For old model: win_prob is combined_probability (Bayesian)
                old_enter, old_size, old_kelly = kelly_entry(
                    old_win_prob, old_pm_mid, old_dir, "old", old_bankroll
                )

                if old_enter:
                    # Determine win/loss
                    if old_dir == "BUY_YES":
                        won = outcome_up
                        token_price = old_pm_mid  # buy Yes at mid
                    else:
                        won = not outcome_up
                        token_price = 1.0 - old_pm_mid  # buy No

                    payout_ratio = (1.0 / token_price) - 1.0 if 0 < token_price < 1 else 1.0
                    pnl = old_size * payout_ratio if won else -old_size
                    old_bankroll += pnl

                    old_trades.append(TradeResult(
                        window_start_ms=ws_ms,
                        direction=old_dir,
                        fair_up_prob=old_fair,
                        pm_mid=old_pm_mid,
                        entry_price_token=token_price,
                        size_usd=old_size,
                        kelly=old_kelly,
                        won=won,
                        pnl=pnl,
                        zscore=old_z,
                        model="old",
                    ))
                else:
                    old_skipped_reasons["kelly_reject"] += 1
        else:
            old_skipped_reasons["no_signal"] += 1

        # ── NEW MODEL ──
        new_fair, new_dir, new_disp = new_model_signal(window_open_price, entry_btc_price)

        if new_fair is not None and new_dir is not None:
            # Simulate PM mid
            new_pm_mid = simulate_pm_mid(new_fair)

            # Spread check
            new_spread = abs(new_fair - new_pm_mid) * 100
            if new_spread < NEW_MIN_EDGE_PCT:
                new_skipped_reasons["no_signal"] += 1
            else:
                # New model: uses fair_up_prob directly (bypass Bayesian)
                new_win_prob = new_fair if new_dir == "BUY_YES" else 1.0 - new_fair

                new_enter, new_size, new_kelly = kelly_entry(
                    new_win_prob, new_pm_mid, new_dir, "new", new_bankroll
                )

                if new_enter:
                    if new_dir == "BUY_YES":
                        won = outcome_up
                        token_price = new_pm_mid
                    else:
                        won = not outcome_up
                        token_price = 1.0 - new_pm_mid

                    payout_ratio = (1.0 / token_price) - 1.0 if 0 < token_price < 1 else 1.0
                    pnl = new_size * payout_ratio if won else -new_size
                    new_bankroll += pnl

                    new_trades.append(TradeResult(
                        window_start_ms=ws_ms,
                        direction=new_dir,
                        fair_up_prob=new_fair,
                        pm_mid=new_pm_mid,
                        entry_price_token=token_price,
                        size_usd=new_size,
                        kelly=new_kelly,
                        won=won,
                        pnl=pnl,
                        displacement_pct=new_disp,
                        model="new",
                    ))
                else:
                    new_skipped_reasons["kelly_reject"] += 1
        else:
            new_skipped_reasons["no_signal"] += 1

        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(windows):,} windows "
                  f"(old: {len(old_trades)} trades, new: {len(new_trades)} trades)")

    # ─── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for label, trades, reasons, final_bank in [
        ("ORIGINAL MOMENTUM (cb5421b)", old_trades, old_skipped_reasons, old_bankroll),
        ("DISPLACEMENT MODEL (current)", new_trades, new_skipped_reasons, new_bankroll),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

        if not trades:
            print(f"  Total trades:     0")
            print(f"  Skipped reasons:  {reasons}")
            print(f"  Final bankroll:   ${final_bank:.2f}")
            continue

        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        total_pnl = sum(t.pnl for t in trades)

        buy_yes = [t for t in trades if t.direction == "BUY_YES"]
        buy_no = [t for t in trades if t.direction == "BUY_NO"]
        buy_yes_wins = sum(1 for t in buy_yes if t.won)
        buy_no_wins = sum(1 for t in buy_no if t.won)

        # Monthly breakdown
        months: dict[str, list[TradeResult]] = {}
        for t in trades:
            mo = time.strftime("%Y-%m", time.gmtime(t.window_start_ms / 1000))
            months.setdefault(mo, []).append(t)

        # Drawdown
        equity_curve = [BANKROLL]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t.pnl)
        peak = BANKROLL
        max_dd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Avg trade stats
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        print(f"  Total trades:     {len(trades):,}")
        print(f"  Win rate:         {len(wins)/len(trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"  Direction split:  {len(buy_yes)} BUY_YES ({buy_yes_wins}W) | "
              f"{len(buy_no)} BUY_NO ({buy_no_wins}W)")
        print(f"  BUY_YES WR:       {buy_yes_wins/len(buy_yes)*100:.1f}%" if buy_yes else "  BUY_YES WR:       N/A")
        print(f"  BUY_NO WR:        {buy_no_wins/len(buy_no)*100:.1f}%" if buy_no else "  BUY_NO WR:        N/A")
        print(f"  Total P&L:        ${total_pnl:+,.2f}")
        print(f"  Avg win:          ${avg_win:+.2f}")
        print(f"  Avg loss:         ${avg_loss:+.2f}")
        print(f"  Profit factor:    {abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)):.2f}" if losses else "  Profit factor:    inf")
        print(f"  Max drawdown:     {max_dd*100:.1f}%")
        print(f"  Final bankroll:   ${final_bank:,.2f} (started ${BANKROLL})")
        print(f"  ROI:              {(final_bank - BANKROLL) / BANKROLL * 100:+.1f}%")
        print(f"  Skipped:          {reasons}")

        # Monthly breakdown
        print(f"\n  Monthly Breakdown:")
        print(f"  {'Month':<10} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'BUY_YES':>10} {'BUY_NO':>10}")
        for mo in sorted(months.keys()):
            mt = months[mo]
            mw = sum(1 for t in mt if t.won)
            mpnl = sum(t.pnl for t in mt)
            m_yes = sum(1 for t in mt if t.direction == "BUY_YES")
            m_no = sum(1 for t in mt if t.direction == "BUY_NO")
            print(f"  {mo:<10} {len(mt):>7} {mw/len(mt)*100:>6.1f}% ${mpnl:>+9.2f} {m_yes:>10} {m_no:>10}")

        # Signal distribution for new model
        if label.startswith("DISPLACEMENT"):
            disps = [t.displacement_pct for t in trades]
            print(f"\n  Displacement Stats:")
            print(f"    Mean:   {np.mean(disps):+.4f}%")
            print(f"    Std:    {np.std(disps):.4f}%")
            print(f"    Min:    {np.min(disps):+.4f}%")
            print(f"    Max:    {np.max(disps):+.4f}%")

            # Win rate by displacement bucket
            print(f"\n  Win Rate by |Displacement|:")
            buckets = [(0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 999)]
            for lo, hi in buckets:
                bt = [t for t in trades if lo <= abs(t.displacement_pct) < hi]
                if bt:
                    bw = sum(1 for t in bt if t.won)
                    bpnl = sum(t.pnl for t in bt)
                    print(f"    {lo:.2f}-{hi:.2f}%: {len(bt):>5} trades, "
                          f"{bw/len(bt)*100:.1f}% WR, ${bpnl:+,.2f}")

        # Signal distribution for old model
        if label.startswith("ORIGINAL"):
            zscores = [t.zscore for t in trades]
            fairs = [t.fair_up_prob for t in trades]
            print(f"\n  Z-Score Stats:")
            print(f"    Mean:   {np.mean(zscores):+.4f}")
            print(f"    Std:    {np.std(zscores):.4f}")
            print(f"    Mean fair_up: {np.mean(fairs):.4f}")
            print(f"    fair_up < 0.10 or > 0.90: {sum(1 for f in fairs if f < 0.10 or f > 0.90)} / {len(fairs)}")

    # ─── Head-to-Head on Same Windows ─────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("HEAD-TO-HEAD: Windows where BOTH models traded")
    print(f"{'=' * 80}")

    old_ws = {t.window_start_ms: t for t in old_trades}
    new_ws = {t.window_start_ms: t for t in new_trades}
    common = set(old_ws.keys()) & set(new_ws.keys())

    if common:
        both_same_dir = 0
        both_diff_dir = 0
        old_wins_h2h = 0
        new_wins_h2h = 0
        for ws in common:
            ot = old_ws[ws]
            nt = new_ws[ws]
            if ot.direction == nt.direction:
                both_same_dir += 1
            else:
                both_diff_dir += 1
            if ot.won:
                old_wins_h2h += 1
            if nt.won:
                new_wins_h2h += 1

        print(f"  Windows both traded: {len(common)}")
        print(f"  Same direction:      {both_same_dir} ({both_same_dir/len(common)*100:.1f}%)")
        print(f"  Opposite direction:  {both_diff_dir} ({both_diff_dir/len(common)*100:.1f}%)")
        print(f"  Old model WR:        {old_wins_h2h/len(common)*100:.1f}%")
        print(f"  New model WR:        {new_wins_h2h/len(common)*100:.1f}%")
    else:
        print("  No common windows found")

    print(f"\n{'=' * 80}")
    print("PM_EFFICIENCY sensitivity (displacement model only)")
    print(f"{'=' * 80}")

    # Quick sensitivity analysis on PM efficiency
    for eff in [0.0, 0.2, 0.4, 0.6, 0.8]:
        np.random.seed(42)
        bank = BANKROLL
        w, l = 0, 0
        for i, ws_ms in enumerate(windows):
            entry_time_ms = ws_ms + 60_000
            resolve_time_ms = ws_ms + 300_000
            woc = candle_map.get(ws_ms)
            ec = candle_map.get(entry_time_ms)
            rc = candle_map.get(resolve_time_ms)
            if not woc or not ec or not rc:
                continue

            outcome_up = rc.close > woc.open
            fair, dirn, disp = new_model_signal(woc.open, ec.close)
            if fair is None or dirn is None:
                continue

            pm_mid = simulate_pm_mid(fair, efficiency=eff)
            spread = abs(fair - pm_mid) * 100
            if spread < NEW_MIN_EDGE_PCT:
                continue

            win_prob = fair if dirn == "BUY_YES" else 1.0 - fair
            enter, size, _ = kelly_entry(win_prob, pm_mid, dirn, "new", bank)
            if not enter:
                continue

            if dirn == "BUY_YES":
                won = outcome_up
                tp = pm_mid
            else:
                won = not outcome_up
                tp = 1.0 - pm_mid

            pr = (1.0 / tp) - 1.0 if 0 < tp < 1 else 1.0
            pnl = size * pr if won else -size
            bank += pnl
            if won:
                w += 1
            else:
                l += 1

        total = w + l
        wr = w / total * 100 if total else 0
        print(f"  efficiency={eff:.1f}: {total:>5} trades, {wr:.1f}% WR, "
              f"P&L=${bank - BANKROLL:+,.2f}, final=${bank:,.2f}")


if __name__ == "__main__":
    run_backtest()
