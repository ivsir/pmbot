"""Combined-filter backtest: Grid search for optimal 70%+ win rate config.

Stacks multiple filters (sensitivity, entry timing, edge threshold,
PM efficiency model, volatility regime, return persistence) and tests
all 1,920 combinations to find the best tradeoff between win rate and P&L.

Key innovation: PM price estimation model that accounts for how quickly
Polymarket absorbs CEX price moves, giving realistic entry price estimates.
"""

import math
import time
import requests
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from itertools import product

# ── Grid search ranges ──
SENSITIVITY_GRID = [1.0, 2.0, 3.0, 5.0]
ENTRY_OFFSET_GRID = [0, 1, 2, 3]           # minutes into 5-min window
MIN_EDGE_GRID = [2.0, 5.0, 10.0, 15.0, 20.0]
PM_EFFICIENCY_GRID = [0.3, 0.5, 0.7]
VOL_REGIME_GRID = [None, "low", "medium", "high"]
RETURN_PERSISTENCE_GRID = [False, True]

# ── Signal model weights (matching production) ──
W_1M = 0.40
W_3M = 0.25
W_5M = 0.15

# ── Trade sizing ──
TRADE_SIZE = 4.30

# ── Time factors by entry offset ──
TIME_FACTORS = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.85}

# ── Window parameters ──
WINDOW_MINUTES = 5
LOOKBACK_DAYS = 7


@dataclass
class FilterConfig:
    sensitivity: float
    entry_offset_min: int
    min_edge_pct: float
    pm_efficiency: float
    vol_regime: str | None
    require_persistence: bool

    @property
    def label(self) -> str:
        parts = [
            f"s={self.sensitivity}",
            f"T+{self.entry_offset_min}",
            f"e>{self.min_edge_pct}%",
            f"eff={self.pm_efficiency}",
        ]
        if self.vol_regime:
            parts.append(f"vol={self.vol_regime}")
        if self.require_persistence:
            parts.append("persist")
        return " | ".join(parts)


@dataclass
class WindowData:
    window_start_ms: int
    btc_open: float
    btc_close: float
    actual_direction: str
    returns_at_offset: dict       # {offset: {r1, r3, r5, zscore, vol, raw}}
    btc_at_offset: dict           # {offset: btc_price}
    btc_move_pct_at_offset: dict  # {offset: pct_move_from_open}
    rolling_vol: float


@dataclass
class ComboResult:
    config: FilterConfig
    wins: int = 0
    losses: int = 0
    no_signal: int = 0
    total_pnl_at_50c: float = 0.0
    total_pnl_at_pm_est: float = 0.0
    trade_pnls: list = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades * 100 if self.total_trades > 0 else 0

    @property
    def ev_per_trade_50c(self) -> float:
        return self.total_pnl_at_50c / self.total_trades if self.total_trades > 0 else 0

    @property
    def ev_per_trade_pm(self) -> float:
        return self.total_pnl_at_pm_est / self.total_trades if self.total_trades > 0 else 0


def fetch_binance_klines(days: int = 7) -> list[dict]:
    """Fetch 1-minute BTC/USDT candles from Binance."""
    all_candles = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 86400 * 1000)
    print(f"Fetching {days} days of 1-min BTC candles from Binance...")
    current = start_ms
    while current < end_ms:
        resp = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": current, "limit": 1000,
        })
        if resp.status_code != 200:
            print(f"  Error {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        if not data:
            break
        for k in data:
            all_candles.append({
                "time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        current = data[-1][0] + 60_000
        time.sleep(0.08)
    print(f"  Got {len(all_candles)} candles")
    return all_candles


def get_return(candles, idx, lookback_minutes):
    """Return over last N minutes from candle at idx."""
    if idx < lookback_minutes:
        return None
    past = candles[idx - lookback_minutes]["close"]
    now = candles[idx]["close"]
    return (now - past) / past if past > 0 else None


def precompute_window_data(candles, idx_by_time) -> list[WindowData]:
    """Pre-compute signals at all offsets for every 5-min window."""
    windows = []
    return_history = deque(maxlen=500)

    window_ms = WINDOW_MINUTES * 60 * 1000
    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]
    ws = ((first_ts // window_ms) + 1) * window_ms

    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000

        if start_t not in idx_by_time or end_t not in idx_by_time:
            ws += window_ms
            continue

        start_idx = idx_by_time[start_t]
        end_idx = idx_by_time[end_t]
        btc_open = candles[start_idx]["open"]
        btc_close = candles[end_idx]["close"]
        actual = "UP" if btc_close > btc_open else "DOWN"

        returns_at_offset = {}
        btc_at_offset = {0: btc_open}
        btc_move_at_offset = {0: 0.0}

        for offset in range(0, 4):
            entry_t = start_t + offset * 60000
            if entry_t not in idx_by_time:
                continue

            entry_idx = idx_by_time[entry_t]
            btc_at_entry = candles[entry_idx]["close"]
            btc_at_offset[offset] = btc_at_entry
            btc_move_at_offset[offset] = (
                (btc_at_entry - btc_open) / btc_open if btc_open > 0 else 0
            )

            r1 = get_return(candles, entry_idx, 1)
            r3 = get_return(candles, entry_idx, 3)
            r5 = get_return(candles, entry_idx, 5)

            if r1 is None:
                continue

            vol = (
                float(np.std(list(return_history)))
                if len(return_history) >= 10
                else 0.001
            )
            if vol < 1e-8:
                vol = 0.001

            raw = W_1M * (r1 or 0) + W_3M * (r3 or 0) + W_5M * (r5 or 0)
            zscore = raw / vol

            returns_at_offset[offset] = {
                "r1": r1,
                "r3": r3,
                "r5": r5,
                "zscore": zscore,
                "vol": vol,
                "raw": raw,
            }

        # Update return_history only at offset=0 (sequential, one per window)
        if 0 in returns_at_offset and returns_at_offset[0]["r1"] is not None:
            return_history.append(returns_at_offset[0]["r1"])

        rolling_vol = (
            float(np.std(list(return_history)))
            if len(return_history) >= 10
            else 0.001
        )

        windows.append(WindowData(
            window_start_ms=ws,
            btc_open=btc_open,
            btc_close=btc_close,
            actual_direction=actual,
            returns_at_offset=returns_at_offset,
            btc_at_offset=btc_at_offset,
            btc_move_pct_at_offset=btc_move_at_offset,
            rolling_vol=rolling_vol,
        ))

        ws += window_ms

    return windows


def estimate_pm_entry_price(btc_move_pct: float, pm_efficiency: float) -> float:
    """Estimate PM token price based on BTC move and PM absorption speed.

    At T+0: PM ~ 50c (market just opened)
    At T+N: PM adjusts toward BTC direction, scaled by efficiency.
    """
    adjustment = btc_move_pct * pm_efficiency * 100
    pm_up_price = 0.50 + adjustment
    return max(0.01, min(0.99, pm_up_price))


def apply_filters(
    window: WindowData,
    config: FilterConfig,
    vol_p33: float,
    vol_p66: float,
) -> dict | None:
    """Apply all stacked filters. Returns trade info or None."""

    offset = config.entry_offset_min
    if offset not in window.returns_at_offset:
        return None

    data = window.returns_at_offset[offset]
    zscore = data["zscore"]
    r1 = data["r1"]
    r3 = data["r3"]

    # ── Filter 1: Volatility Regime ──
    if config.vol_regime is not None:
        v = window.rolling_vol
        if config.vol_regime == "low" and v > vol_p33:
            return None
        elif config.vol_regime == "medium" and (v < vol_p33 or v > vol_p66):
            return None
        elif config.vol_regime == "high" and v < vol_p66:
            return None

    # ── Filter 2: Return Persistence (1m and 3m agree in direction) ──
    if config.require_persistence:
        if r1 is not None and r3 is not None and r1 != 0 and r3 != 0:
            if (r1 > 0) != (r3 > 0):
                return None
        else:
            return None

    # ── Filter 3: Compute direction and edge ──
    tf = TIME_FACTORS.get(offset, 0.5)
    adj_zscore = zscore * tf
    prob = 1.0 / (1.0 + math.exp(-config.sensitivity * adj_zscore))
    prob = max(0.01, min(0.99, prob))
    edge = abs(prob - 0.5) * 100

    # ── Filter 4: Minimum Edge ──
    if edge < config.min_edge_pct:
        return None

    direction = "UP" if prob > 0.5 else "DOWN"

    # ── Filter 5: Entry Price Cap (PM estimation) ──
    btc_move = window.btc_move_pct_at_offset.get(offset, 0.0)
    pm_up_price = estimate_pm_entry_price(btc_move, config.pm_efficiency)

    if direction == "UP":
        entry_price = pm_up_price
    else:
        entry_price = 1.0 - pm_up_price

    if entry_price > 0.80:
        return None

    # ── All filters passed ──
    won = direction == window.actual_direction

    pnl_at_50c = TRADE_SIZE if won else -TRADE_SIZE

    if won:
        shares = TRADE_SIZE / entry_price if entry_price > 0 else 0
        pnl_at_pm = shares * 1.0 - TRADE_SIZE
    else:
        pnl_at_pm = -TRADE_SIZE

    return {
        "won": won,
        "direction": direction,
        "edge": edge,
        "prob": prob,
        "entry_price": entry_price,
        "pnl_at_50c": pnl_at_50c,
        "pnl_at_pm": pnl_at_pm,
    }


def run_grid_search(windows: list[WindowData]) -> list[ComboResult]:
    """Run all filter combinations across pre-computed windows."""
    # Compute volatility percentiles for regime filtering
    all_vols = [w.rolling_vol for w in windows]
    vol_p33 = float(np.percentile(all_vols, 33))
    vol_p66 = float(np.percentile(all_vols, 66))
    print(f"  Vol percentiles: P33={vol_p33:.6f}, P66={vol_p66:.6f}")

    configs = [
        FilterConfig(sens, offset, edge, eff, vol_regime, persist)
        for sens, offset, edge, eff, vol_regime, persist
        in product(
            SENSITIVITY_GRID,
            ENTRY_OFFSET_GRID,
            MIN_EDGE_GRID,
            PM_EFFICIENCY_GRID,
            VOL_REGIME_GRID,
            RETURN_PERSISTENCE_GRID,
        )
    ]

    total_combos = len(configs)
    print(f"  Testing {total_combos} filter combinations across {len(windows)} windows...")

    results = []
    for i, config in enumerate(configs):
        r = ComboResult(config=config)
        for window in windows:
            trade = apply_filters(window, config, vol_p33, vol_p66)
            if trade is None:
                r.no_signal += 1
                continue
            if trade["won"]:
                r.wins += 1
            else:
                r.losses += 1
            r.total_pnl_at_50c += trade["pnl_at_50c"]
            r.total_pnl_at_pm_est += trade["pnl_at_pm"]
            r.trade_pnls.append(trade["pnl_at_pm"])

        results.append(r)
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{total_combos}")

    return results


def print_results(results: list[ComboResult]):
    """Print ranked results."""
    valid = [r for r in results if r.total_trades >= 20]
    by_pnl = sorted(valid, key=lambda r: r.total_pnl_at_pm_est, reverse=True)
    by_wr = sorted(valid, key=lambda r: r.win_rate, reverse=True)

    print("\n" + "=" * 110)
    print("  COMBINED-FILTER BACKTEST RESULTS")
    print("=" * 110)

    # ── Top 20 by P&L ──
    print(f"\n  TOP 20 BY REALISTIC P&L (estimated PM entry price):")
    print("  " + "-" * 105)
    print(f"  {'Config':<55} {'WR%':>6} {'Trades':>7} {'PnL@PM':>10} {'PnL@50c':>10} {'EV/trade':>9}")
    print("  " + "-" * 105)
    for r in by_pnl[:20]:
        print(
            f"  {r.config.label:<55} {r.win_rate:>5.1f}% {r.total_trades:>7} "
            f"${r.total_pnl_at_pm_est:>+9.2f} ${r.total_pnl_at_50c:>+9.2f} "
            f"${r.ev_per_trade_pm:>+8.2f}"
        )

    # ── Top 20 by Win Rate (min 30 trades) ──
    by_wr_30 = [r for r in by_wr if r.total_trades >= 30]
    print(f"\n  TOP 20 BY WIN RATE (min 30 trades):")
    print("  " + "-" * 105)
    print(f"  {'Config':<55} {'WR%':>6} {'Trades':>7} {'PnL@PM':>10} {'PnL@50c':>10} {'EV/trade':>9}")
    print("  " + "-" * 105)
    for r in by_wr_30[:20]:
        print(
            f"  {r.config.label:<55} {r.win_rate:>5.1f}% {r.total_trades:>7} "
            f"${r.total_pnl_at_pm_est:>+9.2f} ${r.total_pnl_at_50c:>+9.2f} "
            f"${r.ev_per_trade_pm:>+8.2f}"
        )

    # ── Sweet Spot: High WR + Positive P&L ──
    sweet = [
        r for r in valid
        if r.win_rate >= 65 and r.total_pnl_at_pm_est > 0 and r.total_trades >= 30
    ]
    sweet.sort(key=lambda r: r.total_pnl_at_pm_est, reverse=True)
    print(f"\n  SWEET SPOT: WR >= 65% AND PM P&L > $0 AND 30+ trades:")
    print("  " + "-" * 105)
    if sweet:
        for r in sweet[:15]:
            print(
                f"  {r.config.label:<55} {r.win_rate:>5.1f}% {r.total_trades:>7} "
                f"${r.total_pnl_at_pm_est:>+9.2f} EV=${r.ev_per_trade_pm:>+.2f}"
            )
    else:
        print("  (none found)")

    # ── Summary ──
    above_70 = [r for r in valid if r.win_rate >= 70 and r.total_trades >= 20]
    above_70_30 = [r for r in valid if r.win_rate >= 70 and r.total_trades >= 30]
    above_65 = [r for r in valid if r.win_rate >= 65 and r.total_trades >= 30]

    print(f"\n  SUMMARY:")
    print(f"    Total valid combos (20+ trades): {len(valid)}")
    print(f"    Combos with WR >= 65% (30+ trades): {len(above_65)}")
    print(f"    Combos with WR >= 70% (20+ trades): {len(above_70)}")
    print(f"    Combos with WR >= 70% (30+ trades): {len(above_70_30)}")

    if above_70:
        best = max(above_70, key=lambda r: r.total_pnl_at_pm_est)
        print(f"\n    Best 70%+ config (by PM P&L):")
        print(f"      {best.config.label}")
        print(f"      WR={best.win_rate:.1f}% trades={best.total_trades} PM_PnL=${best.total_pnl_at_pm_est:+.2f}")

    if above_70_30:
        best30 = max(above_70_30, key=lambda r: r.total_pnl_at_pm_est)
        print(f"\n    Best 70%+ config (30+ trades, by PM P&L):")
        print(f"      {best30.config.label}")
        print(f"      WR={best30.win_rate:.1f}% trades={best30.total_trades} PM_PnL=${best30.total_pnl_at_pm_est:+.2f}")

    # ── Filter importance analysis ──
    print(f"\n  FILTER IMPORTANCE (avg win rate by filter value, 30+ trades):")
    valid_30 = [r for r in valid if r.total_trades >= 30]

    # By sensitivity
    print(f"\n    Sensitivity:")
    for s in SENSITIVITY_GRID:
        subset = [r for r in valid_30 if r.config.sensitivity == s]
        if subset:
            avg_wr = sum(r.win_rate for r in subset) / len(subset)
            avg_trades = sum(r.total_trades for r in subset) / len(subset)
            print(f"      s={s:<5} avg_WR={avg_wr:5.1f}%  avg_trades={avg_trades:5.0f}  (n={len(subset)} configs)")

    # By entry offset
    print(f"\n    Entry Offset:")
    for o in ENTRY_OFFSET_GRID:
        subset = [r for r in valid_30 if r.config.entry_offset_min == o]
        if subset:
            avg_wr = sum(r.win_rate for r in subset) / len(subset)
            avg_trades = sum(r.total_trades for r in subset) / len(subset)
            print(f"      T+{o}min  avg_WR={avg_wr:5.1f}%  avg_trades={avg_trades:5.0f}  (n={len(subset)} configs)")

    # By min edge
    print(f"\n    Min Edge:")
    for e in MIN_EDGE_GRID:
        subset = [r for r in valid_30 if r.config.min_edge_pct == e]
        if subset:
            avg_wr = sum(r.win_rate for r in subset) / len(subset)
            avg_trades = sum(r.total_trades for r in subset) / len(subset)
            print(f"      e>{e:<5}% avg_WR={avg_wr:5.1f}%  avg_trades={avg_trades:5.0f}  (n={len(subset)} configs)")

    # By vol regime
    print(f"\n    Vol Regime:")
    for v in VOL_REGIME_GRID:
        subset = [r for r in valid_30 if r.config.vol_regime == v]
        if subset:
            avg_wr = sum(r.win_rate for r in subset) / len(subset)
            label = v or "none"
            print(f"      {label:<8} avg_WR={avg_wr:5.1f}%  (n={len(subset)} configs)")

    # By persistence
    print(f"\n    Return Persistence:")
    for p in RETURN_PERSISTENCE_GRID:
        subset = [r for r in valid_30 if r.config.require_persistence == p]
        if subset:
            avg_wr = sum(r.win_rate for r in subset) / len(subset)
            label = "required" if p else "off"
            print(f"      {label:<10} avg_WR={avg_wr:5.1f}%  (n={len(subset)} configs)")

    # ── Overfitting warning ──
    total_combos = len(results)
    print(f"\n  WARNING: {total_combos} combinations tested on ~{len(results)} configs.")
    print(f"  High risk of overfitting. Validate best configs on out-of-sample data (14+ days).")
    print(f"  Configs with <50 trades are especially unreliable.")
    print("=" * 110)


def main():
    candles = fetch_binance_klines(days=LOOKBACK_DAYS)
    if not candles:
        print("No candles fetched!")
        return

    # Build time-indexed lookup
    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    print("Pre-computing window data...")
    windows = precompute_window_data(candles, idx_by_time)
    print(f"  {len(windows)} windows pre-computed")

    results = run_grid_search(windows)
    print_results(results)


if __name__ == "__main__":
    main()
