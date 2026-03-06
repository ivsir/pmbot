#!/usr/bin/env python3
"""Backtest comparison: ML model vs sigmoid on held-out data.

Runs both models on the same 5-min windows with realistic PM simulation
and Kelly sizing. Compares WR, calibration, P&L, and trade selection.

Usage:
    python scripts/backtest_ml_vs_sigmoid.py [--months 2] [--model models/displacement_model.joblib]
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import requests
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.layer1_research.feature_engine import FeatureEngine, FEATURE_NAMES, N_FEATURES


# ─── Config ───────────────────────────────────────────────────────────────────

SIGMOID_SCALE = 10.0
MIN_DISPLACEMENT_PCT = 0.03
KELLY_CAP = 0.115
MIN_EDGE_PCT = 2.0
PM_EFFICIENCY = 0.40
BANKROLL = 100.0


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_binance_klines(symbol, interval, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current_start, "endTime": end_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_klines.extend(batch)
        current_start = batch[-1][0] + 60_000
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return all_klines


class Candle:
    __slots__ = ("open_time_ms", "open", "high", "low", "close", "volume")
    def __init__(self, k):
        self.open_time_ms = k[0]
        self.open = float(k[1])
        self.high = float(k[2])
        self.low = float(k[3])
        self.close = float(k[4])
        self.volume = float(k[5])


# ─── Simulation ───────────────────────────────────────────────────────────────

def simulate_pm_mid(fair_up_prob, efficiency=PM_EFFICIENCY):
    noise = np.random.normal(0, 0.02)
    pm_mid = 0.50 + efficiency * (fair_up_prob - 0.50) + noise
    return float(np.clip(pm_mid, 0.05, 0.95))


def kelly_sizing(win_prob, entry_price):
    if entry_price <= 0 or entry_price >= 1:
        return 0.0, 0.0
    payout = (1.0 / entry_price) - 1.0
    q = 1.0 - win_prob
    kelly = (win_prob * payout - q) / payout
    kelly = max(0.0, min(kelly, KELLY_CAP))
    return kelly, payout


def run_backtest(candle_map, windows, model, entry_offset_s=60):
    """Run both ML and sigmoid on windows, return results dicts."""
    results = {"ml": [], "sigmoid": []}

    for ws_ms in windows:
        entry_ms = ws_ms + entry_offset_s * 1000
        resolve_ms = ws_ms + 300_000

        woc = candle_map.get(ws_ms)
        rc = candle_map.get(resolve_ms)
        ec = candle_map.get(entry_ms)
        if not woc or not rc or not ec:
            continue

        window_open = woc.open
        entry_price_btc = ec.close
        if window_open <= 0:
            continue

        displacement_pct = (entry_price_btc - window_open) / window_open * 100
        outcome_up = rc.close > window_open

        # Skip noise
        if abs(displacement_pct) < MIN_DISPLACEMENT_PCT:
            continue

        # Compute features for ML
        features = FeatureEngine.compute_from_candles(
            candle_map=candle_map,
            window_start_ms=ws_ms,
            entry_time_ms=entry_ms,
        )

        # ── Sigmoid model ──
        sig_prob = 1.0 / (1.0 + math.exp(-SIGMOID_SCALE * displacement_pct))
        sig_prob = max(0.01, min(0.99, sig_prob))
        sig_dir = "BUY_YES" if displacement_pct > 0 else "BUY_NO"
        sig_win_prob = sig_prob if sig_dir == "BUY_YES" else 1.0 - sig_prob

        # ── ML model ──
        if features is not None:
            X = np.nan_to_num(features.reshape(1, -1), nan=0.0, posinf=10.0, neginf=-10.0)
            ml_prob = float(model.predict_proba(X)[0, 1])
            ml_prob = max(0.01, min(0.99, ml_prob))
        else:
            ml_prob = sig_prob  # fallback
        ml_dir = "BUY_YES" if displacement_pct > 0 else "BUY_NO"
        ml_win_prob = ml_prob if ml_dir == "BUY_YES" else 1.0 - ml_prob

        # Simulate PM mid
        pm_mid = simulate_pm_mid(sig_prob)  # same for both

        # Process each model
        for label, win_prob, fair_prob in [
            ("ml", ml_win_prob, ml_prob),
            ("sigmoid", sig_win_prob, sig_prob),
        ]:
            direction = "BUY_YES" if displacement_pct > 0 else "BUY_NO"

            if direction == "BUY_YES":
                token_price = pm_mid
                spread = (fair_prob - pm_mid) * 100
            else:
                token_price = 1.0 - pm_mid
                spread = ((1.0 - fair_prob) - (1.0 - pm_mid)) * 100

            if spread < MIN_EDGE_PCT:
                continue

            kelly, payout = kelly_sizing(win_prob, token_price)
            if kelly < 0.005:
                continue
            if not (0.05 <= token_price <= 0.80):
                continue

            won = (direction == "BUY_YES" and outcome_up) or \
                  (direction == "BUY_NO" and not outcome_up)

            pnl = 1.0 * payout if won else -1.0

            results[label].append({
                "ws_ms": ws_ms,
                "direction": direction,
                "displacement": displacement_pct,
                "fair_prob": fair_prob,
                "win_prob": win_prob,
                "pm_mid": pm_mid,
                "kelly": kelly,
                "token_price": token_price,
                "won": won,
                "pnl": pnl,
            })

    return results


def print_results(results):
    print(f"\n{'='*70}")
    print(f"{'ML MODEL':^35} vs {'SIGMOID':^35}")
    print(f"{'='*70}")

    for label in ["ml", "sigmoid"]:
        trades = results[label]
        if not trades:
            print(f"\n  {label.upper()}: No trades")
            continue

        wins = [t for t in trades if t["won"]]
        total_pnl = sum(t["pnl"] for t in trades)
        buy_yes = [t for t in trades if t["direction"] == "BUY_YES"]
        buy_no = [t for t in trades if t["direction"] == "BUY_NO"]

        print(f"\n  {label.upper()}:")
        print(f"    Trades:     {len(trades):,}")
        print(f"    Win rate:   {len(wins)/len(trades)*100:.1f}%")
        print(f"    Total P&L:  ${total_pnl:+,.0f}")
        print(f"    BUY_YES:    {len(buy_yes)} ({sum(1 for t in buy_yes if t['won'])}W)")
        print(f"    BUY_NO:     {len(buy_no)} ({sum(1 for t in buy_no if t['won'])}W)")

        # By displacement bucket
        print(f"    WR by |Displacement|:")
        disps = np.array([abs(t["displacement"]) for t in trades])
        wons = np.array([t["won"] for t in trades])
        for lo, hi in [(0.03, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 999)]:
            mask = (disps >= lo) & (disps < hi)
            if mask.sum() > 0:
                wr = wons[mask].mean() * 100
                print(f"      {lo:.2f}-{hi:.2f}%: {mask.sum():>5} trades, {wr:.1f}% WR")

    # Head-to-head
    ml_ws = {t["ws_ms"]: t for t in results["ml"]}
    sig_ws = {t["ws_ms"]: t for t in results["sigmoid"]}
    common = set(ml_ws.keys()) & set(sig_ws.keys())

    if common:
        ml_wins_h2h = sum(1 for ws in common if ml_ws[ws]["won"])
        sig_wins_h2h = sum(1 for ws in common if sig_ws[ws]["won"])
        both_right = sum(1 for ws in common if ml_ws[ws]["won"] and sig_ws[ws]["won"])
        ml_only = sum(1 for ws in common if ml_ws[ws]["won"] and not sig_ws[ws]["won"])
        sig_only = sum(1 for ws in common if not ml_ws[ws]["won"] and sig_ws[ws]["won"])

        print(f"\n  HEAD-TO-HEAD ({len(common)} common windows):")
        print(f"    ML WR:        {ml_wins_h2h/len(common)*100:.1f}%")
        print(f"    Sigmoid WR:   {sig_wins_h2h/len(common)*100:.1f}%")
        print(f"    Both right:   {both_right}")
        print(f"    ML only right: {ml_only}")
        print(f"    Sig only right: {sig_only}")

    # Trade selection differences
    ml_only_trades = set(ml_ws.keys()) - set(sig_ws.keys())
    sig_only_trades = set(sig_ws.keys()) - set(ml_ws.keys())
    if ml_only_trades:
        ml_unique = [ml_ws[ws] for ws in ml_only_trades]
        ml_unique_wr = sum(1 for t in ml_unique if t["won"]) / len(ml_unique) * 100
        print(f"\n    ML-only trades: {len(ml_unique)} ({ml_unique_wr:.1f}% WR)")
    if sig_only_trades:
        sig_unique = [sig_ws[ws] for ws in sig_only_trades]
        sig_unique_wr = sum(1 for t in sig_unique if t["won"]) / len(sig_unique) * 100
        print(f"    Sig-only trades: {len(sig_unique)} ({sig_unique_wr:.1f}% WR)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=2, help="Months of test data")
    parser.add_argument("--model", default="models/displacement_model.joblib")
    args = parser.parse_args()

    print("=" * 70)
    print("BACKTEST: ML Model vs Sigmoid")
    print("=" * 70)

    # Load model
    artifact = joblib.load(args.model)
    model = artifact["model"]
    print(f"Model loaded: {args.model}")
    print(f"Training metrics: {artifact.get('metrics', {})}")

    # Fetch recent test data (last N months)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (args.months * 30 * 24 * 60 * 60 * 1000)

    print(f"\nFetching {args.months} months of Binance data...")
    raw = fetch_binance_klines("BTCUSDT", "1m", start_ms, now_ms)
    candles = [Candle(k) for k in raw]
    candle_map = {c.open_time_ms: c for c in candles}
    print(f"Fetched {len(candles):,} candles")

    # Generate windows
    first_ms = candles[0].open_time_ms
    last_ms = candles[-1].open_time_ms
    interval = 5 * 60 * 1000
    first_window = (first_ms // interval + 1) * interval
    windows = []
    t = first_window
    while t + interval <= last_ms:
        windows.append(t)
        t += interval
    print(f"Windows: {len(windows):,}")

    np.random.seed(42)
    results = run_backtest(candle_map, windows, model)
    print_results(results)


if __name__ == "__main__":
    main()
