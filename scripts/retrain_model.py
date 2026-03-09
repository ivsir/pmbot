#!/usr/bin/env python3
"""Retrain ML model using live trade data + Binance history.

Combines:
  1. Live trade features + outcomes from data/live_trades.csv
  2. Fresh Binance 1-min klines (last N months)

Trains a new GradientBoostingClassifier with isotonic calibration,
validates on held-out data, and saves if it outperforms the current model.

Usage:
    python scripts/retrain_model.py [--months 3] [--min-live-trades 20]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.layer1_research.feature_engine import FEATURE_NAMES, N_FEATURES
from scripts.train_displacement_model import (
    fetch_binance_klines, parse_klines, build_dataset, Candle,
)

LIVE_CSV = "data/live_trades.csv"
MODEL_PATH = "models/displacement_model.joblib"


def load_live_data() -> tuple[np.ndarray, np.ndarray] | None:
    """Load feature vectors and outcomes from live trade CSV."""
    if not os.path.exists(LIVE_CSV):
        print(f"  No live trade data at {LIVE_CSV}")
        return None

    df = pd.read_csv(LIVE_CSV)

    # Filter to rows with outcomes
    df = df[df["outcome"].notna() & (df["outcome"] != "")]
    df["outcome"] = df["outcome"].astype(int)

    if len(df) < 5:
        print(f"  Only {len(df)} live trades with outcomes — too few")
        return None

    # Extract feature columns
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    missing = set(FEATURE_NAMES) - set(feature_cols)
    if missing:
        print(f"  Warning: missing feature columns: {missing}")

    X = df[feature_cols].values.astype(np.float64)
    y = df["outcome"].values.astype(np.int32)

    # Pad missing features with zeros if needed
    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]))
        X = np.hstack([X, pad])

    print(f"  Live trades loaded: {len(X)} samples, {y.mean():.1%} win rate")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Retrain ML model with live data")
    parser.add_argument("--months", type=int, default=3, help="Months of Binance data")
    parser.add_argument("--min-live-trades", type=int, default=20,
                        help="Minimum live trades to include live data")
    parser.add_argument("--live-weight", type=float, default=3.0,
                        help="Weight multiplier for live trade samples")
    parser.add_argument("--output", default=MODEL_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and evaluate but don't save")
    args = parser.parse_args()

    print("=" * 70)
    print("MODEL RETRAINING WITH LIVE DATA")
    print("=" * 70)

    # 1. Load live trade data
    print(f"\n1. Loading live trade data from {LIVE_CSV}...")
    live_data = load_live_data()

    # 2. Fetch Binance data
    print(f"\n2. Fetching {args.months} months of Binance data...")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (args.months * 30 * 24 * 60 * 60 * 1000)

    raw = fetch_binance_klines("BTCUSDT", "1m", start_ms, now_ms)
    candles = parse_klines(raw)
    candle_map = {c.open_time_ms: c for c in candles}
    print(f"  Fetched {len(candles):,} candles")

    # 3. Build Binance dataset
    print(f"\n3. Building Binance feature dataset...")
    X_binance, y_binance, ts_binance = build_dataset(candles, candle_map)
    print(f"  Binance samples: {len(X_binance):,}")

    # 4. Combine datasets
    print(f"\n4. Combining datasets...")
    X_all = X_binance
    y_all = y_binance

    if live_data and len(live_data[0]) >= args.min_live_trades:
        X_live, y_live = live_data

        # Oversample live data (it's more valuable — real market conditions)
        weight = int(args.live_weight)
        X_live_weighted = np.tile(X_live, (weight, 1))
        y_live_weighted = np.tile(y_live, weight)

        X_all = np.vstack([X_binance, X_live_weighted])
        y_all = np.concatenate([y_binance, y_live_weighted])
        print(f"  Combined: {len(X_all):,} samples "
              f"({len(X_binance):,} Binance + {len(X_live_weighted):,} live [{weight}x weighted])")
    else:
        print(f"  Using Binance data only (live trades < {args.min_live_trades} threshold)")

    # 5. Train
    print(f"\n5. Training model...")
    from scripts.train_displacement_model import train_model, compare_with_sigmoid
    model, metrics, X_test, y_test, y_prob = train_model(X_all, y_all, np.arange(len(X_all)))

    # 6. Compare with sigmoid
    compare_with_sigmoid(X_test, y_test, y_prob)

    # 7. Compare with current model
    print(f"\n6. Comparing with current model...")
    import joblib
    current_metrics = {}
    if os.path.exists(args.output):
        current = joblib.load(args.output)
        current_metrics = current.get("metrics", {})
        print(f"  Current model: accuracy={current_metrics.get('accuracy', 'N/A'):.4f}, "
              f"AUC={current_metrics.get('auc', 'N/A'):.4f}")
    print(f"  New model:     accuracy={metrics['accuracy']:.4f}, "
          f"AUC={metrics['auc']:.4f}")

    improved = metrics["auc"] >= current_metrics.get("auc", 0)

    if args.dry_run:
        print(f"\n  DRY RUN — model not saved")
        return

    if improved or not current_metrics:
        artifact = {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "n_features": N_FEATURES,
            "metrics": metrics,
            "trained_at": time.time(),
            "train_months": args.months,
            "n_samples": len(X_all),
            "n_live_samples": len(live_data[0]) if live_data else 0,
            "entry_offset_s": 60,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        joblib.dump(artifact, args.output)
        print(f"\n  Model saved to {args.output}")
    else:
        print(f"\n  New model did NOT improve AUC — keeping current model")
        print(f"  Current AUC: {current_metrics.get('auc', 0):.4f} vs New: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
