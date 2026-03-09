#!/usr/bin/env python3
"""Backtest: ML model vs Sigmoid on 3 months of Binance data.

Simulates realistic Polymarket 5-min Up/Down trading:
  - Entry at T+60s into each 5-min window
  - Uses displacement from window open price
  - Applies velocity, vol-normalization, and min-edge filters
  - Simulates PM entry prices based on displacement (mid ~50c +/- edge)
  - Tracks P&L with realistic binary payouts

Usage:
    python scripts/backtest_ml_vs_sigmoid.py [--months 3]
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.layer1_research.feature_engine import FeatureEngine, FEATURE_NAMES, N_FEATURES
from scripts.train_displacement_model import fetch_binance_klines, parse_klines, Candle


def sigmoid(x, scale=10.0):
    return 1.0 / (1.0 + math.exp(-scale * x))


def run_backtest(candles, candle_map, model=None, label="Sigmoid", scale=10.0,
                 min_displacement=0.02, min_edge_pct=0.02, bet_size=5.0):
    """Run backtest over all 5-min windows.

    Returns dict of results.
    """
    first_ms = candles[0].open_time_ms
    last_ms = candles[-1].open_time_ms
    window_interval = 5 * 60 * 1000
    first_window = (first_ms // window_interval + 1) * window_interval

    # Need 60 candles of warmup for features
    first_window = max(first_window, first_ms + 60 * 60 * 1000)

    trades = []
    t = first_window

    while t + window_interval <= last_ms:
        entry_time_ms = t + 60 * 1000  # T+60s
        resolve_time_ms = t + 300_000

        window_open_candle = candle_map.get(t)
        entry_candle = candle_map.get(entry_time_ms)
        resolve_candle = candle_map.get(resolve_time_ms)

        if not window_open_candle or not entry_candle or not resolve_candle:
            t += window_interval
            continue

        window_open_price = window_open_candle.open
        entry_price_btc = entry_candle.close

        # Displacement at entry
        displacement_pct = (entry_price_btc - window_open_price) / window_open_price * 100

        # Skip noise
        if abs(displacement_pct) < min_displacement:
            t += window_interval
            continue

        # Velocity check: compare entry candle close vs 15s-ago candle
        vel_candle_ms = entry_time_ms - 15 * 1000
        # Round to nearest minute
        vel_candle_ms = (vel_candle_ms // 60000) * 60000
        vel_candle = candle_map.get(vel_candle_ms)
        if vel_candle:
            velocity = (entry_candle.close - vel_candle.close) / vel_candle.close * 100
            # Velocity must agree with displacement
            if displacement_pct > 0 and velocity < 0:
                t += window_interval
                continue
            if displacement_pct < 0 and velocity > 0:
                t += window_interval
                continue
        else:
            velocity = 0.0

        # Compute P(Up) — ML or sigmoid
        if model is not None:
            features = FeatureEngine.compute_from_candles(
                candle_map=candle_map,
                window_start_ms=t,
                entry_time_ms=entry_time_ms,
            )
            if features is None:
                t += window_interval
                continue
            X = np.nan_to_num(features.reshape(1, -1), nan=0.0, posinf=10.0, neginf=-10.0)
            fair_up_prob = float(model.predict_proba(X)[0, 1])
            fair_up_prob = max(0.01, min(0.99, fair_up_prob))
        else:
            fair_up_prob = sigmoid(displacement_pct, scale)
            fair_up_prob = max(0.01, min(0.99, fair_up_prob))

        # Direction follows displacement
        direction = "BUY_YES" if displacement_pct > 0 else "BUY_NO"

        # Simulate PM pricing: mid starts ~50c, moves slightly with displacement
        # Real PM markets have mid ~48-52c for most of the window
        pm_mid = 0.50 + displacement_pct * 0.5  # slight PM reaction
        pm_mid = max(0.20, min(0.80, pm_mid))

        # Edge calculation
        if direction == "BUY_YES":
            edge = fair_up_prob - pm_mid
            entry_price = pm_mid + 0.01  # buy at ask (mid + 1c spread)
        else:
            edge = (1.0 - fair_up_prob) - (1.0 - pm_mid)
            entry_price = (1.0 - pm_mid) + 0.01

        entry_price = max(0.10, min(0.90, entry_price))

        # Min edge filter
        if edge < min_edge_pct:
            t += window_interval
            continue

        # Kelly sizing
        prob = fair_up_prob if direction == "BUY_YES" else (1.0 - fair_up_prob)
        payout = (1.0 / entry_price) - 1.0
        q = 1.0 - prob
        kelly = (prob * payout - q) / payout
        if kelly < 0.005:
            t += window_interval
            continue

        # Outcome
        btc_went_up = resolve_candle.close > window_open_candle.open
        won = (direction == "BUY_YES" and btc_went_up) or \
              (direction == "BUY_NO" and not btc_went_up)

        if won:
            pnl = bet_size * payout
        else:
            pnl = -bet_size

        trades.append({
            "timestamp_ms": t,
            "direction": direction,
            "displacement_pct": displacement_pct,
            "fair_prob": fair_up_prob,
            "pm_mid": pm_mid,
            "entry_price": entry_price,
            "edge": edge,
            "kelly": kelly,
            "won": won,
            "pnl": pnl,
        })

        t += window_interval

    # Summarize
    if not trades:
        return {"label": label, "trades": 0}

    wins = sum(1 for t in trades if t["won"])
    losses = len(trades) - wins
    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in trades if t["won"]]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in trades if not t["won"]]) if losses else 0

    # Running equity curve
    equity = [0.0]
    for trade in trades:
        equity.append(equity[-1] + trade["pnl"])
    equity = np.array(equity)
    max_dd = 0.0
    peak = 0.0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    # Monthly breakdown
    monthly = {}
    for trade in trades:
        month = time.strftime("%Y-%m", time.gmtime(trade["timestamp_ms"] / 1000))
        if month not in monthly:
            monthly[month] = {"wins": 0, "losses": 0, "pnl": 0.0}
        if trade["won"]:
            monthly[month]["wins"] += 1
        else:
            monthly[month]["losses"] += 1
        monthly[month]["pnl"] += trade["pnl"]

    # Win rate by direction
    yes_trades = [t for t in trades if t["direction"] == "BUY_YES"]
    no_trades = [t for t in trades if t["direction"] == "BUY_NO"]
    yes_wr = sum(1 for t in yes_trades if t["won"]) / len(yes_trades) * 100 if yes_trades else 0
    no_wr = sum(1 for t in no_trades if t["won"]) / len(no_trades) * 100 if no_trades else 0

    return {
        "label": label,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "final_equity": equity[-1],
        "monthly": monthly,
        "yes_trades": len(yes_trades),
        "yes_wr": yes_wr,
        "no_trades": len(no_trades),
        "no_wr": no_wr,
        "avg_edge": np.mean([t["edge"] for t in trades]),
        "avg_kelly": np.mean([t["kelly"] for t in trades]),
        "avg_displacement": np.mean([abs(t["displacement_pct"]) for t in trades]),
        "equity_curve": equity,
    }


def print_results(r):
    if r["trades"] == 0:
        print(f"\n  {r['label']}: NO TRADES")
        return

    print(f"\n  {'='*65}")
    print(f"  {r['label']} BACKTEST RESULTS")
    print(f"  {'='*65}")
    print(f"  Total trades:    {r['trades']:,}")
    print(f"  Win rate:        {r['win_rate']:.1f}% ({r['wins']}W / {r['losses']}L)")
    print(f"  Total P&L:       ${r['total_pnl']:+,.2f}")
    print(f"  Avg win:         ${r['avg_win']:+.2f}")
    print(f"  Avg loss:        ${r['avg_loss']:.2f}")
    print(f"  Max drawdown:    ${r['max_drawdown']:.2f}")
    print(f"  Final equity:    ${r['final_equity']:+,.2f}")
    print(f"  Avg edge:        {r['avg_edge']*100:.2f}%")
    print(f"  Avg Kelly:       {r['avg_kelly']*100:.2f}%")
    print(f"  Avg |displace|:  {r['avg_displacement']:.4f}%")
    print(f"\n  Direction breakdown:")
    print(f"    BUY_YES: {r['yes_trades']:,} trades, {r['yes_wr']:.1f}% WR")
    print(f"    BUY_NO:  {r['no_trades']:,} trades, {r['no_wr']:.1f}% WR")
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':>10} {'Trades':>8} {'WR':>7} {'P&L':>10}")
    for month in sorted(r["monthly"]):
        m = r["monthly"][month]
        total = m["wins"] + m["losses"]
        wr = m["wins"] / total * 100 if total else 0
        print(f"  {month:>10} {total:>8} {wr:>6.1f}% ${m['pnl']:>+9.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--bet-size", type=float, default=5.0)
    args = parser.parse_args()

    print("=" * 70)
    print("BACKTEST: ML MODEL vs SIGMOID (3-month Binance data)")
    print("=" * 70)

    # Fetch data
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (args.months * 30 * 24 * 60 * 60 * 1000)
    print(f"\nFetching {args.months} months of Binance 1-min klines...")
    raw = fetch_binance_klines("BTCUSDT", "1m", start_ms, now_ms)
    candles = parse_klines(raw)
    candle_map = {c.open_time_ms: c for c in candles}
    print(f"Fetched {len(candles):,} candles")
    print(f"Period: {time.strftime('%Y-%m-%d', time.gmtime(candles[0].open_time_ms/1000))} -> "
          f"{time.strftime('%Y-%m-%d', time.gmtime(candles[-1].open_time_ms/1000))}")

    # Load ML model
    print("\nLoading ML model...")
    import joblib
    model_path = "models/displacement_model.joblib"
    if os.path.exists(model_path):
        artifact = joblib.load(model_path)
        ml_model = artifact["model"]
        print(f"  Model loaded: AUC={artifact['metrics'].get('auc', 'N/A'):.4f}")
    else:
        print("  NO ML MODEL FOUND — skipping ML backtest")
        ml_model = None

    # Run backtests
    print(f"\nRunning Sigmoid (scale=10) backtest (bet=${args.bet_size})...")
    sig_results = run_backtest(candles, candle_map, model=None, label="SIGMOID (scale=10)",
                               scale=10.0, bet_size=args.bet_size)
    print_results(sig_results)

    if ml_model:
        print(f"\nRunning ML model backtest (bet=${args.bet_size})...")
        ml_results = run_backtest(candles, candle_map, model=ml_model, label="ML MODEL",
                                   bet_size=args.bet_size)
        print_results(ml_results)

        # Head-to-head
        print(f"\n  {'='*65}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"  {'='*65}")
        print(f"  {'Metric':<25} {'Sigmoid':>15} {'ML':>15} {'Diff':>10}")
        print(f"  {'-'*65}")
        if sig_results["trades"] and ml_results["trades"]:
            metrics = [
                ("Trades", sig_results["trades"], ml_results["trades"]),
                ("Win Rate %", sig_results["win_rate"], ml_results["win_rate"]),
                ("Total P&L $", sig_results["total_pnl"], ml_results["total_pnl"]),
                ("Max Drawdown $", sig_results["max_drawdown"], ml_results["max_drawdown"]),
                ("Avg Edge %", sig_results["avg_edge"]*100, ml_results["avg_edge"]*100),
                ("BUY_YES WR %", sig_results["yes_wr"], ml_results["yes_wr"]),
                ("BUY_NO WR %", sig_results["no_wr"], ml_results["no_wr"]),
            ]
            for name, sig_val, ml_val in metrics:
                diff = ml_val - sig_val
                print(f"  {name:<25} {sig_val:>15.2f} {ml_val:>15.2f} {diff:>+10.2f}")

    # Also test sigmoid with higher scale (original sensitivity=50)
    print(f"\n\nRunning Sigmoid (scale=50) backtest for comparison...")
    sig50_results = run_backtest(candles, candle_map, model=None, label="SIGMOID (scale=50)",
                                  scale=50.0, bet_size=args.bet_size)
    print_results(sig50_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_results = [sig_results, sig50_results]
    if ml_model:
        all_results.insert(1, ml_results)
    print(f"  {'Model':<25} {'Trades':>8} {'WR':>7} {'P&L':>12} {'MaxDD':>10}")
    print(f"  {'-'*62}")
    for r in all_results:
        if r["trades"]:
            print(f"  {r['label']:<25} {r['trades']:>8} {r['win_rate']:>6.1f}% ${r['total_pnl']:>+10.2f} ${r['max_drawdown']:>8.2f}")


if __name__ == "__main__":
    main()
