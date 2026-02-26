"""Backtest: Momentum strategy win rate on 5-min BTC windows.

Uses Binance kline API for historical 1-minute BTC/USDT candles.
Simulates the exact momentum model the bot uses:
  - 1m/3m/5m weighted returns → z-score → sigmoid → P(Up)
  - If P(Up) > 0.5 → BUY_YES, else BUY_NO
  - Outcome: did BTC go up or down in that 5-min window?

No Polymarket data needed — we're testing the CEX signal accuracy.
"""

import math
import time
import requests
import numpy as np
from collections import deque
from dataclasses import dataclass

# ── Strategy parameters (matching bot config) ──
SENSITIVITY = 5.0       # sigmoid sensitivity (.env value)
W_1M = 0.40
W_3M = 0.25
W_5M = 0.15
W_OBI = 0.20            # OBI not available in backtest, set to 0
MIN_EDGE_PCT = 2.0       # minimum spread to trigger trade

# ── Backtest parameters ──
LOOKBACK_DAYS = 7        # how many days of history
WINDOW_MINUTES = 5       # 5-min windows


@dataclass
class WindowResult:
    window_start_ts: int
    predicted_direction: str  # "UP" or "DOWN"
    actual_direction: str
    fair_up_prob: float
    zscore: float
    ret_1m: float
    ret_3m: float
    ret_5m: float
    btc_open: float
    btc_close: float
    won: bool
    edge_pct: float


def fetch_binance_klines(days: int = 7) -> list[dict]:
    """Fetch 1-minute BTC/USDT candles from Binance."""
    all_candles = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 86400 * 1000)

    print(f"Fetching {days} days of 1-min BTC candles from Binance...")

    current = start_ms
    while current < end_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": current,
            "limit": 1000,
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"  Error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        if not data:
            break

        for k in data:
            all_candles.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })

        current = data[-1][0] + 60_000  # next minute
        time.sleep(0.1)  # rate limit

    print(f"  Fetched {len(all_candles)} candles")
    return all_candles


def calculate_return(candles: list[dict], idx: int, lookback_ms: int) -> float | None:
    """Calculate return over lookback_ms ending at candle idx."""
    current_price = candles[idx]["close"]
    target_time = candles[idx]["open_time"] - lookback_ms

    # Find candle closest to target_time
    for j in range(idx - 1, max(idx - 500, -1), -1):
        if j < 0:
            return None
        if candles[j]["open_time"] <= target_time:
            past_price = candles[j]["close"]
            if past_price > 0:
                return (current_price - past_price) / past_price
            return None
    return None


def run_backtest(candles: list[dict]) -> list[WindowResult]:
    """Run the momentum strategy on historical 5-min windows."""
    results = []
    return_history = deque(maxlen=500)

    # Group candles into 5-minute windows
    # Windows align to 5-min boundaries (like Polymarket)
    window_size = WINDOW_MINUTES * 60 * 1000  # 5 min in ms

    if not candles:
        return results

    first_ts = candles[0]["open_time"]
    last_ts = candles[-1]["open_time"]

    # Align to 5-min boundaries
    window_start = (first_ts // window_size + 1) * window_size

    # Build index for fast candle lookup
    candle_by_time = {}
    for i, c in enumerate(candles):
        # Round to minute
        minute_ts = (c["open_time"] // 60000) * 60000
        candle_by_time[minute_ts] = i

    windows_processed = 0

    while window_start + window_size <= last_ts:
        window_end = window_start + window_size

        # Find the candle at window start and end
        start_minute = (window_start // 60000) * 60000
        end_minute = ((window_end - 60000) // 60000) * 60000  # last candle in window

        if start_minute not in candle_by_time or end_minute not in candle_by_time:
            window_start += window_size
            continue

        start_idx = candle_by_time[start_minute]
        end_idx = candle_by_time[end_minute]

        btc_open = candles[start_idx]["open"]
        btc_close = candles[end_idx]["close"]
        actual_up = btc_close > btc_open
        actual_direction = "UP" if actual_up else "DOWN"

        # ── Simulate momentum signal at decision time ──
        # We make the decision based on data BEFORE the window starts
        # (or at the start of the window, using trailing returns)
        decision_idx = start_idx

        ret_1m = calculate_return(candles, decision_idx, 60_000)
        ret_3m = calculate_return(candles, decision_idx, 180_000)
        ret_5m = calculate_return(candles, decision_idx, 300_000)

        if ret_1m is None:
            window_start += window_size
            continue

        return_history.append(ret_1m)

        # Rolling volatility
        if len(return_history) >= 10:
            vol = float(np.std(list(return_history)))
        else:
            vol = 0.001
        if vol < 1e-8:
            vol = 0.001

        # Composite momentum (OBI = 0 in backtest)
        raw_momentum = (
            W_1M * (ret_1m or 0.0)
            + W_3M * (ret_3m or 0.0)
            + W_5M * (ret_5m or 0.0)
        )
        zscore = raw_momentum / vol

        # Sigmoid → P(Up)
        # Using time_factor = 0.5 (start of window, conservative)
        time_factor = 0.5
        adjusted_zscore = zscore * time_factor
        fair_up_prob = 1.0 / (1.0 + math.exp(-SENSITIVITY * adjusted_zscore))
        fair_up_prob = max(0.01, min(0.99, fair_up_prob))

        # Edge check — only trade if edge > MIN_EDGE_PCT
        # Assume Polymarket implied = 0.50 (fair coin)
        implied_up = 0.50
        edge_pct = abs(fair_up_prob - implied_up) * 100

        if edge_pct < MIN_EDGE_PCT:
            window_start += window_size
            continue

        # Predicted direction
        predicted = "UP" if fair_up_prob > 0.5 else "DOWN"
        won = (predicted == actual_direction)

        results.append(WindowResult(
            window_start_ts=window_start,
            predicted_direction=predicted,
            actual_direction=actual_direction,
            fair_up_prob=fair_up_prob,
            zscore=adjusted_zscore,
            ret_1m=ret_1m,
            ret_3m=ret_3m or 0,
            ret_5m=ret_5m or 0,
            btc_open=btc_open,
            btc_close=btc_close,
            won=won,
            edge_pct=edge_pct,
        ))

        windows_processed += 1
        window_start += window_size

    return results


def analyze_results(results: list[WindowResult]):
    """Print detailed backtest analysis."""
    if not results:
        print("No results to analyze!")
        return

    total = len(results)
    wins = sum(1 for r in results if r.won)
    losses = total - wins
    win_rate = wins / total * 100

    # Break down by direction
    up_trades = [r for r in results if r.predicted_direction == "UP"]
    down_trades = [r for r in results if r.predicted_direction == "DOWN"]
    up_wins = sum(1 for r in up_trades if r.won)
    down_wins = sum(1 for r in down_trades if r.won)

    # Break down by edge size
    low_edge = [r for r in results if r.edge_pct < 5]
    med_edge = [r for r in results if 5 <= r.edge_pct < 15]
    high_edge = [r for r in results if r.edge_pct >= 15]

    # Break down by confidence (distance from 50%)
    high_conf = [r for r in results if abs(r.fair_up_prob - 0.5) > 0.15]
    med_conf = [r for r in results if 0.05 < abs(r.fair_up_prob - 0.5) <= 0.15]
    low_conf = [r for r in results if abs(r.fair_up_prob - 0.5) <= 0.05]

    # Simulate P&L (buy at ~$0.50 implied, win pays $1, lose pays $0)
    # More realistic: buy at fair_up_prob price
    total_pnl = 0
    trade_size = 4.30  # matching bot's Kelly size
    for r in results:
        if r.predicted_direction == "UP":
            entry_price = r.fair_up_prob  # buy Up token
        else:
            entry_price = 1.0 - r.fair_up_prob  # buy Down token

        shares = trade_size / entry_price if entry_price > 0 else 0
        if r.won:
            profit = shares * 1.0 - trade_size  # win: shares × $1 - cost
        else:
            profit = -trade_size  # lose: lose cost basis
        total_pnl += profit

    # Time-of-day analysis
    from datetime import datetime, timezone
    hour_wins = {}
    hour_total = {}
    for r in results:
        dt = datetime.fromtimestamp(r.window_start_ts / 1000, tz=timezone.utc)
        h = dt.hour
        hour_total[h] = hour_total.get(h, 0) + 1
        if r.won:
            hour_wins[h] = hour_wins.get(h, 0) + 1

    # Streak analysis
    max_win_streak = 0
    max_lose_streak = 0
    current_streak = 0
    current_type = None
    for r in results:
        if r.won:
            if current_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_type == "lose":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "lose"
            max_lose_streak = max(max_lose_streak, current_streak)

    print("\n" + "=" * 70)
    print("  MOMENTUM STRATEGY BACKTEST RESULTS")
    print(f"  Period: {LOOKBACK_DAYS} days | Window: {WINDOW_MINUTES}min | Sensitivity: {SENSITIVITY}")
    print("=" * 70)

    print(f"\n  OVERALL:")
    print(f"    Total windows traded:  {total}")
    print(f"    Wins:                  {wins}")
    print(f"    Losses:                {losses}")
    print(f"    Win Rate:              {win_rate:.1f}%")
    print(f"    Simulated P&L:         ${total_pnl:.2f} (at ${trade_size}/trade)")

    print(f"\n  BY DIRECTION:")
    if up_trades:
        print(f"    BUY_YES (Up):   {up_wins}/{len(up_trades)} = {up_wins/len(up_trades)*100:.1f}%")
    if down_trades:
        print(f"    BUY_NO (Down):  {down_wins}/{len(down_trades)} = {down_wins/len(down_trades)*100:.1f}%")

    print(f"\n  BY EDGE SIZE:")
    if low_edge:
        lw = sum(1 for r in low_edge if r.won)
        print(f"    Edge <5%:       {lw}/{len(low_edge)} = {lw/len(low_edge)*100:.1f}%")
    if med_edge:
        mw = sum(1 for r in med_edge if r.won)
        print(f"    Edge 5-15%:     {mw}/{len(med_edge)} = {mw/len(med_edge)*100:.1f}%")
    if high_edge:
        hw = sum(1 for r in high_edge if r.won)
        print(f"    Edge >15%:      {hw}/{len(high_edge)} = {hw/len(high_edge)*100:.1f}%")

    print(f"\n  BY CONFIDENCE:")
    if high_conf:
        hcw = sum(1 for r in high_conf if r.won)
        print(f"    High (>65%):    {hcw}/{len(high_conf)} = {hcw/len(high_conf)*100:.1f}%")
    if med_conf:
        mcw = sum(1 for r in med_conf if r.won)
        print(f"    Medium (55-65%): {mcw}/{len(med_conf)} = {mcw/len(med_conf)*100:.1f}%")
    if low_conf:
        lcw = sum(1 for r in low_conf if r.won)
        print(f"    Low (50-55%):   {lcw}/{len(low_conf)} = {lcw/len(low_conf)*100:.1f}%")

    print(f"\n  STREAKS:")
    print(f"    Max win streak:   {max_win_streak}")
    print(f"    Max lose streak:  {max_lose_streak}")

    print(f"\n  HOURLY WIN RATE (UTC):")
    for h in sorted(hour_total.keys()):
        w = hour_wins.get(h, 0)
        t = hour_total[h]
        bar = "█" * int(w / t * 20) if t > 0 else ""
        print(f"    {h:02d}:00  {w:3d}/{t:3d} = {w/t*100:5.1f}%  {bar}")

    # Sensitivity sweep
    print(f"\n  SENSITIVITY SWEEP (testing different values):")
    for sens in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        sw = 0
        st = 0
        ret_hist = deque(maxlen=500)
        for r in results:
            ret_hist.append(r.ret_1m)
            v = float(np.std(list(ret_hist))) if len(ret_hist) >= 10 else 0.001
            if v < 1e-8:
                v = 0.001
            raw = W_1M * r.ret_1m + W_3M * r.ret_3m + W_5M * r.ret_5m
            z = raw / v * 0.5
            p = 1.0 / (1.0 + math.exp(-sens * z))
            p = max(0.01, min(0.99, p))
            edge = abs(p - 0.5) * 100
            if edge >= MIN_EDGE_PCT:
                pred = "UP" if p > 0.5 else "DOWN"
                if pred == r.actual_direction:
                    sw += 1
                st += 1
        if st > 0:
            marker = " ◄ current" if sens == SENSITIVITY else ""
            print(f"    sens={sens:5.1f}: {sw}/{st} = {sw/st*100:5.1f}% ({st} trades){marker}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    candles = fetch_binance_klines(days=LOOKBACK_DAYS)
    if candles:
        results = run_backtest(candles)
        analyze_results(results)
