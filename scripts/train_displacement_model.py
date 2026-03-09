#!/usr/bin/env python3
"""Train ML model to predict 5-min BTC Up/Down window outcome.

Usage:
    python scripts/train_displacement_model.py [--months 6] [--output models/displacement_model.joblib]

Fetches Binance 1-min klines, engineers features, trains GradientBoostingClassifier
with CalibratedClassifierCV, and saves the model.
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.layer1_research.feature_engine import (
    FeatureEngine, FEATURE_NAMES, N_FEATURES, SimpleCandle,
)


# ─── Data Fetching (from backtest_6mo.py) ─────────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines from Binance between start_ms and end_ms."""
    url = "https://api.binance.us/api/v3/klines"
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
        current_start = batch[-1][0] + 60_000
        if len(batch) < 1000:
            break
        time.sleep(0.15)

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


# ─── Dataset Building ─────────────────────────────────────────────────────────

def build_dataset(
    candles: list[Candle],
    candle_map: dict[int, Candle],
    entry_offset_s: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix and labels from candle data.

    For each 5-min window, compute features at entry_offset_s into the window.
    Label = 1 if BTC closes above window open, 0 otherwise.

    Returns:
        X: shape (n_samples, 24)
        y: shape (n_samples,)
        timestamps: shape (n_samples,) — window_start_ms for ordering
    """
    first_ms = candles[0].open_time_ms
    last_ms = candles[-1].open_time_ms

    # Align to 5-min boundaries
    window_interval = 5 * 60 * 1000
    first_window = (first_ms // window_interval + 1) * window_interval

    X_list = []
    y_list = []
    ts_list = []

    t = first_window
    while t + window_interval <= last_ms:
        entry_time_ms = t + entry_offset_s * 1000
        resolve_time_ms = t + 300_000  # T+5min

        window_open_candle = candle_map.get(t)
        resolve_candle = candle_map.get(resolve_time_ms)

        if not window_open_candle or not resolve_candle:
            t += window_interval
            continue

        # Compute features
        features = FeatureEngine.compute_from_candles(
            candle_map=candle_map,
            window_start_ms=t,
            entry_time_ms=entry_time_ms,
        )

        if features is not None:
            # Label: 1 if close > open (UP), 0 otherwise
            outcome_up = 1 if resolve_candle.close > window_open_candle.open else 0

            X_list.append(features)
            y_list.append(outcome_up)
            ts_list.append(t)

        t += window_interval

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    timestamps = np.array(ts_list, dtype=np.int64)

    return X, y, timestamps


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(X, y, timestamps, train_ratio=0.67):
    """Walk-forward train: first train_ratio for training, rest for testing.

    Returns (model, metrics, X_test, y_test, y_prob).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        roc_auc_score, brier_score_loss, log_loss,
        classification_report,
    )

    # Time-ordered split
    n = len(X)
    split_idx = int(n * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_train, ts_test = timestamps[:split_idx], timestamps[split_idx:]

    print(f"\n  Train/Test split: {len(X_train):,} train / {len(X_test):,} test")
    print(f"  Train period: {time.strftime('%Y-%m-%d', time.gmtime(ts_train[0]/1000))} → "
          f"{time.strftime('%Y-%m-%d', time.gmtime(ts_train[-1]/1000))}")
    print(f"  Test period:  {time.strftime('%Y-%m-%d', time.gmtime(ts_test[0]/1000))} → "
          f"{time.strftime('%Y-%m-%d', time.gmtime(ts_test[-1]/1000))}")
    print(f"  Train class balance: {y_train.mean():.3f} UP")
    print(f"  Test class balance:  {y_test.mean():.3f} UP")

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

    # Base model
    base_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=50,
        max_features="sqrt",
        random_state=42,
    )

    # Calibrated wrapper
    print("\n  Training GradientBoostingClassifier with isotonic calibration...")
    model = CalibratedClassifierCV(
        estimator=base_model,
        cv=TimeSeriesSplit(n_splits=3),
        method="isotonic",
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
    }

    print(f"\n  {'='*60}")
    print(f"  TEST SET METRICS")
    print(f"  {'='*60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  AUC:          {metrics['auc']:.4f}")
    print(f"  Brier Score:  {metrics['brier_score']:.4f}  (lower is better, 0.25=random)")
    print(f"  Log Loss:     {metrics['log_loss']:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], digits=4))

    # Calibration analysis
    print(f"\n  Calibration Curve (predicted vs actual):")
    print(f"  {'Predicted':>12} {'Actual':>10} {'Count':>8}")
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_test, y_prob, n_bins=10, strategy="quantile",
        )
        for mp, fp in zip(mean_pred, fraction_pos):
            print(f"  {mp:>12.3f} {fp:>10.3f}")
    except Exception as e:
        print(f"  Could not compute calibration curve: {e}")

    # Feature importance
    print(f"\n  Feature Importances:")
    importances = np.zeros(N_FEATURES)
    n_estimators = 0
    for cal_est in model.calibrated_classifiers_:
        importances += cal_est.estimator.feature_importances_
        n_estimators += 1
    importances /= max(n_estimators, 1)

    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        bar = "#" * int(importances[i] * 200)
        print(f"    {FEATURE_NAMES[i]:<25} {importances[i]:.4f} {bar}")

    return model, metrics, X_test, y_test, y_prob


# ─── Sigmoid Comparison ───────────────────────────────────────────────────────

def compare_with_sigmoid(X_test, y_test, y_prob_ml, scale=10.0):
    """Compare ML predictions with sigmoid baseline on the test set."""
    # Sigmoid uses displacement_pct (feature index 0)
    displacement = X_test[:, 0]
    y_prob_sigmoid = 1.0 / (1.0 + np.exp(-scale * displacement))

    # Convert both to directional predictions
    # ML: predict UP if prob > 0.5
    ml_pred = (y_prob_ml > 0.5).astype(int)
    sig_pred = (y_prob_sigmoid > 0.5).astype(int)

    ml_correct = (ml_pred == y_test).sum()
    sig_correct = (sig_pred == y_test).sum()

    print(f"\n  {'='*60}")
    print(f"  ML vs SIGMOID COMPARISON (on test set)")
    print(f"  {'='*60}")
    print(f"  ML accuracy:      {ml_correct/len(y_test)*100:.2f}% ({ml_correct}/{len(y_test)})")
    print(f"  Sigmoid accuracy: {sig_correct/len(y_test)*100:.2f}% ({sig_correct}/{len(y_test)})")
    print(f"  Improvement:      {(ml_correct - sig_correct)/len(y_test)*100:+.2f}%")

    # Brier scores
    from sklearn.metrics import brier_score_loss
    brier_ml = brier_score_loss(y_test, y_prob_ml)
    brier_sig = brier_score_loss(y_test, y_prob_sigmoid)
    print(f"  ML Brier score:      {brier_ml:.4f}")
    print(f"  Sigmoid Brier score: {brier_sig:.4f}")
    print(f"  Brier improvement:   {brier_sig - brier_ml:+.4f} (positive = ML better)")

    # Breakdown by displacement bucket
    print(f"\n  Accuracy by |Displacement| Bucket:")
    print(f"  {'Bucket':>15} {'ML WR':>8} {'Sig WR':>8} {'Count':>7} {'ML Edge':>8}")
    abs_disp = np.abs(displacement)
    buckets = [(0.00, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10),
               (0.10, 0.20), (0.20, 0.50), (0.50, 999)]
    for lo, hi in buckets:
        mask = (abs_disp >= lo) & (abs_disp < hi)
        if mask.sum() == 0:
            continue
        ml_wr = (ml_pred[mask] == y_test[mask]).mean() * 100
        sig_wr = (sig_pred[mask] == y_test[mask]).mean() * 100
        label = f"{lo:.2f}-{hi:.2f}%"
        print(f"  {label:>15} {ml_wr:>7.1f}% {sig_wr:>7.1f}% {mask.sum():>7} {ml_wr - sig_wr:>+7.1f}%")

    # Trade simulation: ML model with Kelly
    print(f"\n  Simulated P&L (flat $1 bets, trades where |disp| >= 0.02%):")
    for label, probs, name in [
        ("ML", y_prob_ml, "ml"),
        ("Sigmoid", y_prob_sigmoid, "sigmoid"),
    ]:
        mask = abs_disp >= 0.02
        wins = 0
        losses = 0
        total_pnl = 0.0
        for i in np.where(mask)[0]:
            direction_up = probs[i] > 0.5
            prob = probs[i] if direction_up else 1.0 - probs[i]

            # Kelly check
            entry_price = 0.50  # simplified
            payout = (1.0 / entry_price) - 1.0
            q = 1.0 - prob
            kelly = (prob * payout - q) / payout
            if kelly < 0.005:
                continue

            won = (direction_up and y_test[i] == 1) or (not direction_up and y_test[i] == 0)
            pnl = 1.0 if won else -1.0
            total_pnl += pnl
            if won:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        wr = wins / total * 100 if total else 0
        print(f"  {label:<10} {total:>6} trades, {wr:.1f}% WR, P&L ${total_pnl:+,.0f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train displacement ML model")
    parser.add_argument("--months", type=int, default=6, help="Months of data to fetch")
    parser.add_argument("--output", default="models/displacement_model.joblib",
                        help="Output model path")
    parser.add_argument("--entry-offset", type=int, default=60,
                        help="Entry time offset in seconds (default: 60 = T+1min)")
    args = parser.parse_args()

    print("=" * 70)
    print("DISPLACEMENT ML MODEL TRAINING")
    print("=" * 70)

    # 1. Fetch data
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (args.months * 30 * 24 * 60 * 60 * 1000)

    print(f"\nFetching Binance 1-min BTC/USDT klines...")
    print(f"Period: {time.strftime('%Y-%m-%d', time.gmtime(start_ms/1000))} → "
          f"{time.strftime('%Y-%m-%d', time.gmtime(now_ms/1000))}")

    raw_klines = fetch_binance_klines("BTCUSDT", "1m", start_ms, now_ms)
    candles = parse_klines(raw_klines)
    print(f"Fetched {len(candles):,} candles")

    if len(candles) < 1000:
        print("ERROR: Not enough data")
        sys.exit(1)

    # Build candle map
    candle_map: dict[int, Candle] = {c.open_time_ms: c for c in candles}

    # 2. Build dataset
    print(f"\nBuilding feature dataset (entry at T+{args.entry_offset}s)...")
    X, y, timestamps = build_dataset(candles, candle_map, entry_offset_s=args.entry_offset)
    print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.3f} UP ({y.sum():,} UP / {(1-y).sum():,.0f} DOWN)")

    if len(X) < 500:
        print("ERROR: Not enough samples")
        sys.exit(1)

    # 3. Train
    model, metrics, X_test, y_test, y_prob = train_model(X, y, timestamps)

    # 4. Compare with sigmoid
    compare_with_sigmoid(X_test, y_test, y_prob)

    # 5. Save model
    import joblib
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "metrics": metrics,
        "trained_at": time.time(),
        "train_months": args.months,
        "n_samples": len(X),
        "entry_offset_s": args.entry_offset,
    }
    joblib.dump(artifact, args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Artifact keys: {list(artifact.keys())}")


if __name__ == "__main__":
    main()
