"""Auto-Tuner — analyzes paper trading history and refines strategy parameters.

Run via cron every 30 minutes:
    python -m src.simulator.auto_tuner

Reads:  data/trade_history.jsonl, data/signal_history.jsonl
Writes: data/tuned_params.json  (hot-reloaded by paper trader)
Logs:   data/tuner_log.jsonl    (audit trail of all adjustments)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRADE_LOG = DATA_DIR / "trade_history.jsonl"
SIGNAL_LOG = DATA_DIR / "signal_history.jsonl"
PARAMS_FILE = DATA_DIR / "tuned_params.json"
TUNER_LOG = DATA_DIR / "tuner_log.jsonl"

# ── Guardrails: parameter bounds to prevent degenerate strategies ──

BOUNDS = {
    "base_prior":     (0.50, 0.72),
    "spread_weight":  (0.30, 0.70),
    "latency_weight": (0.10, 0.40),
    "liquidity_weight": (0.10, 0.35),
    "spread_tp_rate": (0.60, 0.95),
    "latency_tp_rate": (0.55, 0.92),
    "liquidity_tp_rate": (0.65, 0.95),
}


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_current_params() -> Dict[str, Any]:
    """Load existing tuned params or return defaults."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return json.load(f)
    return {
        "synthesis": {
            "base_prior": 0.62,
            "spread_weight": 0.55,
            "latency_weight": 0.25,
            "liquidity_weight": 0.20,
            "spread_tp_rate": 0.90,
            "latency_tp_rate": 0.85,
            "liquidity_tp_rate": 0.92,
        },
        "version": 0,
        "last_tuned": "",
    }


@dataclass
class TunerAnalysis:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_edge_winners: float = 0.0
    avg_edge_losers: float = 0.0
    total_signals: int = 0
    signal_conversion_rate: float = 0.0
    avg_confidence_winners: float = 0.0
    avg_confidence_losers: float = 0.0
    direction_bias: str = ""  # "BUY_YES" or "BUY_NO" if biased
    best_strike_delta: float = 0.0  # which strike distance performs best
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


def analyze(trades: List[Dict], signals: List[Dict]) -> TunerAnalysis:
    """Analyze trading history and compute metrics for parameter tuning."""
    a = TunerAnalysis()

    if not trades:
        a.recommendations.append("No trades yet — keeping current parameters")
        return a

    a.total_trades = len(trades)
    a.wins = sum(1 for t in trades if t.get("won"))
    a.losses = a.total_trades - a.wins
    a.win_rate = a.wins / a.total_trades if a.total_trades > 0 else 0
    a.total_pnl = sum(t.get("pnl", 0) for t in trades)
    a.avg_pnl = a.total_pnl / a.total_trades if a.total_trades > 0 else 0

    pnls = [t.get("pnl", 0) for t in trades]
    if len(pnls) >= 2 and np.std(pnls) > 0:
        a.sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(288))

    # Drawdown
    equity_curve = []
    eq = 100_000.0
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.get("pnl", 0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(eq)
    a.max_drawdown = max_dd

    # Direction analysis
    yes_wins = sum(1 for t in trades if t.get("won") and t.get("direction") == "BUY_YES")
    yes_total = sum(1 for t in trades if t.get("direction") == "BUY_YES")
    no_wins = sum(1 for t in trades if t.get("won") and t.get("direction") == "BUY_NO")
    no_total = sum(1 for t in trades if t.get("direction") == "BUY_NO")

    yes_wr = yes_wins / yes_total if yes_total > 0 else 0
    no_wr = no_wins / no_total if no_total > 0 else 0
    if abs(yes_wr - no_wr) > 0.15:
        a.direction_bias = "BUY_YES" if yes_wr > no_wr else "BUY_NO"

    # Strike distance analysis (how far from ATM performs best)
    for t in trades:
        btc = t.get("entry_btc", 0)
        strike = t.get("strike", 0)
        if btc > 0 and strike > 0:
            delta = abs(btc - strike) / btc * 100
            if t.get("won"):
                a.best_strike_delta += delta
    if a.wins > 0:
        a.best_strike_delta /= a.wins

    # Signal analysis
    a.total_signals = len(signals)
    validated_signals = sum(1 for s in signals if s.get("validated"))
    a.signal_conversion_rate = validated_signals / a.total_signals if a.total_signals > 0 else 0

    return a


def compute_adjustments(
    analysis: TunerAnalysis, current: Dict[str, Any], trades: List[Dict]
) -> Dict[str, Any]:
    """Compute new parameters based on analysis. Uses conservative adjustments."""
    synth = dict(current.get("synthesis", {}))
    version = current.get("version", 0) + 1
    changes: List[str] = []

    if analysis.total_trades < 3:
        # Not enough data — keep params, maybe loosen priors to get more trades
        if analysis.total_trades == 0:
            synth["base_prior"] = clamp(synth.get("base_prior", 0.62) + 0.02, *BOUNDS["base_prior"])
            changes.append(f"No trades: raised base_prior to {synth['base_prior']:.3f}")
        return {
            "synthesis": synth,
            "version": version,
            "last_tuned": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "analysis": {"total_trades": analysis.total_trades, "changes": changes},
        }

    # ── Adjustment logic ──

    # 1. Win rate tuning
    target_wr = 0.70
    wr_gap = analysis.win_rate - target_wr

    if wr_gap < -0.15:
        # Win rate way too low — tighten priors (fewer but better trades)
        synth["base_prior"] = clamp(synth.get("base_prior", 0.62) - 0.02, *BOUNDS["base_prior"])
        changes.append(f"WR={analysis.win_rate:.1%} << 70%: lowered base_prior to {synth['base_prior']:.3f}")
    elif wr_gap < -0.05:
        # Win rate below target — slightly tighten
        synth["base_prior"] = clamp(synth.get("base_prior", 0.62) - 0.01, *BOUNDS["base_prior"])
        changes.append(f"WR={analysis.win_rate:.1%} < 70%: lowered base_prior to {synth['base_prior']:.3f}")
    elif wr_gap > 0.10:
        # Win rate very high — can afford to loosen for more volume
        synth["base_prior"] = clamp(synth.get("base_prior", 0.62) + 0.015, *BOUNDS["base_prior"])
        changes.append(f"WR={analysis.win_rate:.1%} >> 70%: raised base_prior to {synth['base_prior']:.3f}")

    # 2. Sharpe ratio — if negative, reduce exposure via tighter spread filter
    if analysis.sharpe < 0 and analysis.total_trades >= 5:
        synth["spread_tp_rate"] = clamp(
            synth.get("spread_tp_rate", 0.90) - 0.03, *BOUNDS["spread_tp_rate"]
        )
        changes.append(f"Sharpe={analysis.sharpe:.2f}: tightened spread_tp to {synth['spread_tp_rate']:.3f}")
    elif analysis.sharpe > 2.0:
        synth["spread_tp_rate"] = clamp(
            synth.get("spread_tp_rate", 0.90) + 0.02, *BOUNDS["spread_tp_rate"]
        )
        changes.append(f"Sharpe={analysis.sharpe:.2f}: loosened spread_tp to {synth['spread_tp_rate']:.3f}")

    # 3. Drawdown protection
    if analysis.max_drawdown > 0.03:
        synth["base_prior"] = clamp(synth.get("base_prior", 0.62) - 0.01, *BOUNDS["base_prior"])
        changes.append(f"DD={analysis.max_drawdown:.1%}: lowered base_prior for safety")

    # 4. Direction bias correction — shift weights
    if analysis.direction_bias and analysis.total_trades >= 8:
        changes.append(f"Direction bias detected: {analysis.direction_bias} performing better")

    # 5. Signal conversion rate — if too few signals convert to trades, loosen
    if analysis.signal_conversion_rate < 0.05 and analysis.total_signals > 50:
        synth["spread_weight"] = clamp(
            synth.get("spread_weight", 0.55) + 0.03, *BOUNDS["spread_weight"]
        )
        changes.append(f"Low signal conversion ({analysis.signal_conversion_rate:.1%}): "
                       f"raised spread_weight to {synth['spread_weight']:.3f}")

    # 6. TP rate calibration from actual outcomes
    # Compute actual true-positive rates from trade data
    recent = trades[-20:] if len(trades) > 20 else trades
    actual_tp = sum(1 for t in recent if t.get("won")) / len(recent) if recent else 0.5

    if actual_tp > 0:
        # Blend actual TP with current setting (exponential moving average)
        alpha = 0.3  # learning rate
        for key, bounds_key in [
            ("spread_tp_rate", "spread_tp_rate"),
            ("latency_tp_rate", "latency_tp_rate"),
            ("liquidity_tp_rate", "liquidity_tp_rate"),
        ]:
            old = synth.get(key, 0.85)
            new_val = old * (1 - alpha) + actual_tp * alpha
            synth[key] = clamp(round(new_val, 4), *BOUNDS[bounds_key])

        changes.append(f"Calibrated TP rates toward actual={actual_tp:.1%}")

    # 7. Weight normalization — ensure weights sum to ~1.0
    w_sum = synth.get("spread_weight", 0.55) + synth.get("latency_weight", 0.25) + synth.get("liquidity_weight", 0.20)
    if abs(w_sum - 1.0) > 0.05:
        for k in ["spread_weight", "latency_weight", "liquidity_weight"]:
            synth[k] = round(synth.get(k, 0.33) / w_sum, 4)
        changes.append(f"Normalized weights (sum was {w_sum:.3f})")

    if not changes:
        changes.append("No adjustments needed — parameters performing well")

    return {
        "synthesis": synth,
        "version": version,
        "last_tuned": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis": {
            "total_trades": analysis.total_trades,
            "win_rate": round(analysis.win_rate, 4),
            "total_pnl": round(analysis.total_pnl, 2),
            "sharpe": round(analysis.sharpe, 3),
            "max_drawdown": round(analysis.max_drawdown, 4),
            "avg_pnl": round(analysis.avg_pnl, 2),
            "signal_conversion": round(analysis.signal_conversion_rate, 4),
            "direction_bias": analysis.direction_bias,
            "changes": changes,
        },
    }


def log_tuning_run(result: Dict[str, Any]) -> None:
    """Append tuning result to audit log."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch_ms": int(time.time() * 1000),
        **result,
    }
    try:
        with open(TUNER_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def main() -> None:
    """Run one tuning cycle."""
    print(f"{'='*60}")
    print(f"  AUTO-TUNER — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Load data
    trades = load_jsonl(TRADE_LOG)
    signals = load_jsonl(SIGNAL_LOG)
    current = load_current_params()

    print(f"  Trade history:  {len(trades)} trades")
    print(f"  Signal history: {len(signals)} signals")
    print(f"  Current params: v{current.get('version', 0)}")
    print()

    # Analyze
    analysis = analyze(trades, signals)
    print(f"  Win rate:       {analysis.win_rate:.1%} ({analysis.wins}W / {analysis.losses}L)")
    print(f"  Total PnL:      ${analysis.total_pnl:+,.2f}")
    print(f"  Avg PnL/trade:  ${analysis.avg_pnl:+,.2f}")
    print(f"  Sharpe (est):   {analysis.sharpe:.2f}")
    print(f"  Max Drawdown:   {analysis.max_drawdown:.2%}")
    print(f"  Signal conv:    {analysis.signal_conversion_rate:.1%}")
    if analysis.direction_bias:
        print(f"  Direction bias: {analysis.direction_bias}")
    print()

    # Compute new params
    result = compute_adjustments(analysis, current, trades)

    # Show changes
    print("  ADJUSTMENTS:")
    for change in result.get("analysis", {}).get("changes", []):
        print(f"    -> {change}")
    print()

    # Show new params
    synth = result["synthesis"]
    print("  NEW PARAMETERS:")
    print(f"    base_prior:     {synth.get('base_prior', 0):.4f}")
    print(f"    spread_weight:  {synth.get('spread_weight', 0):.4f}")
    print(f"    latency_weight: {synth.get('latency_weight', 0):.4f}")
    print(f"    liquidity_w:    {synth.get('liquidity_weight', 0):.4f}")
    print(f"    spread_tp:      {synth.get('spread_tp_rate', 0):.4f}")
    print(f"    latency_tp:     {synth.get('latency_tp_rate', 0):.4f}")
    print(f"    liquidity_tp:   {synth.get('liquidity_tp_rate', 0):.4f}")
    print()

    # Write new params (paper trader will hot-reload)
    DATA_DIR.mkdir(exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Written to: {PARAMS_FILE}")

    # Audit log
    log_tuning_run(result)
    print(f"  Logged to:  {TUNER_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
