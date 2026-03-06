"""Backtest: Displacement model with three confirmation filters.

Uses the SAME formulas as the live trading system:
  1. Displacement → sigmoid(scale) → fair_up_prob
  2. Compare fair_up_prob vs simulated PM mid → spread/edge
  3. Kelly criterion gates entry (f* > 0.5%)
  4. Binary market PnL: win → (1 - entry_price), loss → -entry_price
  5. PM price modeled as partially efficient (adjusts toward displacement)

NEW FILTERS:
  A. Velocity confirmation — price velocity must agree with displacement direction
  B. Volatility normalization — displacement z-score must exceed threshold
  C. PM OBI gating — simulated PM order book must not strongly disagree
"""

import math
import time
import numpy as np
import requests


# ── Live system constants (from settings.py / .env) ──
KELLY_CAP = 0.12
MIN_EDGE_PCT = 0.02  # 2% minimum edge
PRICE_FLOOR = 0.05
PRICE_CEIL = 0.80
MIN_KELLY = 0.005
SCALE = 10.0  # sigmoid scale (confirmed irrelevant for WR, fix at 10)


def fetch_binance_klines(days: int = 7) -> list[dict]:
    all_candles = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 86400 * 1000)
    print(f"Fetching {days} days of 1-min BTC candles...")
    current = start_ms
    while current < end_ms:
        resp = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": current, "limit": 1000,
        })
        if resp.status_code != 200:
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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def simulate_pm_price(displacement_pct: float, pm_efficiency: float = 0.3) -> float:
    """Model PM mid-price for UP token. PM adjusts at 30% of model rate."""
    return sigmoid(pm_efficiency * displacement_pct)


def compute_velocity(candles: list[dict], entry_idx: int) -> float:
    """Compute velocity: price change over the previous candle (1 min).

    With 1-min candle data, the smallest lookback is 1 candle.
    Returns pct change: (entry_close - prev_close) / prev_close * 100
    """
    if entry_idx < 1:
        return 0.0
    prev_close = candles[entry_idx - 1]["close"]
    entry_close = candles[entry_idx]["close"]
    if prev_close <= 0:
        return 0.0
    return (entry_close - prev_close) / prev_close * 100


def compute_rolling_stdev(candles: list[dict], entry_idx: int, lookback: int = 5) -> float:
    """Compute rolling stdev of 1-candle returns over `lookback` candles.

    Returns stdev in percentage terms (same units as displacement_pct).
    """
    start = max(0, entry_idx - lookback)
    if entry_idx - start < 3:
        return 0.01  # insufficient data, return small default
    returns = []
    for i in range(start + 1, entry_idx + 1):
        prev = candles[i - 1]["close"]
        curr = candles[i]["close"]
        if prev > 0:
            returns.append((curr - prev) / prev * 100)
    if len(returns) < 3:
        return 0.01
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return max(var ** 0.5, 1e-6)


def simulate_pm_obi(displacement_pct: float, noise_std: float = 0.15) -> float:
    """Simulate PM order book imbalance.

    In reality, PM OBI weakly tracks displacement — when BTC is up,
    bid side tends to be slightly heavier. We model this as:
      obi = clip(0.3 * displacement_pct + noise, -1, 1)

    The noise represents disagreement among PM traders.
    For reproducibility we use a deterministic hash-based "noise".
    """
    # Use displacement as seed for deterministic but varied "noise"
    # This simulates that OBI weakly agrees with displacement but isn't perfect
    pseudo_noise = math.sin(displacement_pct * 1000) * noise_std
    raw_obi = 0.3 * displacement_pct + pseudo_noise
    return max(-1.0, min(1.0, raw_obi))


def run_single_config(
    candles: list[dict],
    idx_by_time: dict[int, int],
    first_ts: int,
    last_ts: int,
    days: int,
    # Parameters
    min_disp: float,
    offset: int,
    use_velocity: bool,
    use_vol_norm: bool,
    min_z: float,
    use_obi: bool,
    obi_limit: float,
) -> dict | None:
    """Run a single backtest configuration. Returns stats dict or None."""
    window_ms = 5 * 60 * 1000
    wins, losses, skipped = 0, 0, 0
    total_pnl = 0.0
    entry_sum = 0.0
    edge_sum = 0.0
    kelly_pass = 0
    velocity_filtered = 0
    vol_filtered = 0
    obi_filtered = 0

    ws = ((first_ts // window_ms) + 1) * window_ms
    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000
        entry_t = start_t + offset * 60000

        if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
            ws += window_ms
            continue

        start_idx = idx_by_time[start_t]
        end_idx = idx_by_time[end_t]
        entry_idx = idx_by_time[entry_t]

        btc_start = candles[start_idx]["open"]
        btc_end = candles[end_idx]["close"]
        btc_entry = candles[entry_idx]["close"]
        actual_up = btc_end > btc_start

        # ── Step 1: Displacement ──
        displacement_pct = (btc_entry - btc_start) / btc_start * 100

        if abs(displacement_pct) < min_disp:
            skipped += 1
            ws += window_ms
            continue

        # ── Filter A: Velocity confirmation ──
        if use_velocity:
            velocity = compute_velocity(candles, entry_idx)
            if displacement_pct > 0 and velocity < 0:
                velocity_filtered += 1
                ws += window_ms
                continue
            if displacement_pct < 0 and velocity > 0:
                velocity_filtered += 1
                ws += window_ms
                continue

        # ── Filter B: Volatility-normalized threshold ──
        if use_vol_norm:
            stdev = compute_rolling_stdev(candles, entry_idx, lookback=5)
            z_disp = displacement_pct / stdev if stdev > 1e-8 else 0.0
            if abs(z_disp) < min_z:
                vol_filtered += 1
                ws += window_ms
                continue

        # ── Filter C: PM OBI gating ──
        if use_obi:
            obi = simulate_pm_obi(displacement_pct)
            if displacement_pct > 0 and obi < -obi_limit:
                obi_filtered += 1
                ws += window_ms
                continue
            if displacement_pct < 0 and obi > obi_limit:
                obi_filtered += 1
                ws += window_ms
                continue

        # ── Step 2: fair_up_prob via sigmoid ──
        fair_up_prob = sigmoid(SCALE * displacement_pct)
        fair_up_prob = max(0.01, min(0.99, fair_up_prob))

        # ── Step 3: Simulate PM mid-price ──
        pm_up_mid = simulate_pm_price(displacement_pct, pm_efficiency=0.3)

        # ── Step 4: Spread / direction ──
        spread_pct = (fair_up_prob - pm_up_mid) * 100
        if abs(spread_pct) < MIN_EDGE_PCT:
            skipped += 1
            ws += window_ms
            continue

        direction_yes = spread_pct > 0

        # ── Step 5: Entry price ──
        entry_price = pm_up_mid if direction_yes else 1.0 - pm_up_mid

        if entry_price < PRICE_FLOOR or entry_price > PRICE_CEIL:
            skipped += 1
            ws += window_ms
            continue

        # ── Step 6: Kelly criterion ──
        win_prob = fair_up_prob if direction_yes else 1.0 - fair_up_prob
        payout_ratio = (1.0 / entry_price) - 1.0
        q = 1.0 - win_prob
        kelly_raw = (win_prob * payout_ratio - q) / payout_ratio
        kelly_raw = max(0.0, kelly_raw)
        kelly_fraction = min(kelly_raw, KELLY_CAP)

        if kelly_fraction < MIN_KELLY:
            skipped += 1
            ws += window_ms
            continue

        kelly_pass += 1

        # ── Step 7: Binary market PnL ──
        we_predicted_up = direction_yes
        we_were_right = (we_predicted_up == actual_up)

        edge_sum += abs(spread_pct)
        entry_sum += entry_price

        if we_were_right:
            pnl = 1.0 - entry_price
            wins += 1
        else:
            pnl = -entry_price
            losses += 1
        total_pnl += pnl

        ws += window_ms

    total = wins + losses
    if total < 5:
        return None

    return {
        "min_disp": min_disp,
        "offset": offset,
        "use_velocity": use_velocity,
        "use_vol_norm": use_vol_norm,
        "min_z": min_z,
        "use_obi": use_obi,
        "obi_limit": obi_limit,
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wins / total * 100,
        "ev": total_pnl / total,
        "avg_entry": entry_sum / total,
        "avg_edge": edge_sum / total,
        "tpd": total / days,
        "dpnl": total_pnl / days,
        "velocity_filtered": velocity_filtered,
        "vol_filtered": vol_filtered,
        "obi_filtered": obi_filtered,
    }


def filters_label(cfg: dict) -> str:
    """Short label for which filters are active."""
    parts = []
    if cfg["use_velocity"]:
        parts.append("V")
    if cfg["use_vol_norm"]:
        parts.append(f"Z>{cfg['min_z']:.1f}")
    if cfg["use_obi"]:
        parts.append(f"OBI<{cfg['obi_limit']:.1f}")
    return "+".join(parts) if parts else "none"


def run():
    days = 7
    candles = fetch_binance_klines(days=days)
    if not candles:
        return

    window_ms = 5 * 60 * 1000
    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]

    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    print("\n" + "=" * 110)
    print("  BACKTEST: DISPLACEMENT MODEL + CONFIRMATION FILTERS — 7 DAYS")
    print("=" * 110)

    all_configs = []

    # ═════════════════════════════════════════════════════════════════
    # SECTION 1: BASELINE (no filters) for comparison
    # ═════════════════════════════════════════════════════════════════
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  SECTION 1: BASELINE — No filters (current live system)                                           │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    header = f"  {'cfg':>22} │ {'Trades':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} {'Trd/day':>7}"
    print(header)
    print(f"  {'─'*22} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*7}")

    baselines = {}
    for min_disp in [0.01, 0.02, 0.03, 0.05]:
        for offset in [0, 1, 2, 3]:
            result = run_single_config(
                candles, idx_by_time, first_ts, last_ts, days,
                min_disp=min_disp, offset=offset,
                use_velocity=False, use_vol_norm=False, min_z=0, use_obi=False, obi_limit=0,
            )
            if result:
                label = f"d{min_disp:.3f}% T+{offset} (base)"
                is_live = (min_disp == 0.02 and offset == 1)
                marker = " ◄ LIVE" if is_live else ""
                print(
                    f"  {label:>22} │ {result['total']:>6} {result['wr']:>5.1f}% "
                    f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} {result['tpd']:>6.0f}{marker}"
                )
                baselines[(min_disp, offset)] = result
                result["section"] = "baseline"
                all_configs.append(result)

    # ═════════════════════════════════════════════════════════════════
    # SECTION 2: VELOCITY ONLY — does velocity confirmation help?
    # ═════════════════════════════════════════════════════════════════
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  SECTION 2: VELOCITY CONFIRMATION ONLY — filter out direction-reversing signals                    │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    header = f"  {'cfg':>22} │ {'Trades':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} {'VelFilt':>7} │ {'ΔWR':>5} {'ΔEV':>8}"
    print(header)
    print(f"  {'─'*22} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*7} ┼ {'─'*5} {'─'*8}")

    for min_disp in [0.01, 0.02, 0.03, 0.05]:
        for offset in [0, 1, 2, 3]:
            result = run_single_config(
                candles, idx_by_time, first_ts, last_ts, days,
                min_disp=min_disp, offset=offset,
                use_velocity=True, use_vol_norm=False, min_z=0, use_obi=False, obi_limit=0,
            )
            if result:
                label = f"d{min_disp:.3f}% T+{offset} +Vel"
                base = baselines.get((min_disp, offset))
                delta_wr = (result["wr"] - base["wr"]) if base else 0
                delta_ev = (result["ev"] - base["ev"]) if base else 0
                print(
                    f"  {label:>22} │ {result['total']:>6} {result['wr']:>5.1f}% "
                    f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} {result['velocity_filtered']:>7} │ "
                    f"{delta_wr:>+4.1f}% ${delta_ev:>+6.3f}"
                )
                result["section"] = "velocity"
                all_configs.append(result)

    # ═════════════════════════════════════════════════════════════════
    # SECTION 3: VOLATILITY NORMALIZATION ONLY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  SECTION 3: VOLATILITY NORMALIZATION ONLY — adaptive threshold via z-score                        │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    header = f"  {'cfg':>28} │ {'Trades':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} {'VolFilt':>7} │ {'ΔWR':>5} {'ΔEV':>8}"
    print(header)
    print(f"  {'─'*28} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*7} ┼ {'─'*5} {'─'*8}")

    for min_disp in [0.01, 0.02, 0.03]:
        for offset in [1, 2]:
            for min_z in [0.5, 1.0, 1.5, 2.0]:
                result = run_single_config(
                    candles, idx_by_time, first_ts, last_ts, days,
                    min_disp=min_disp, offset=offset,
                    use_velocity=False, use_vol_norm=True, min_z=min_z, use_obi=False, obi_limit=0,
                )
                if result:
                    label = f"d{min_disp:.3f}% T+{offset} Z>{min_z:.1f}"
                    base = baselines.get((min_disp, offset))
                    delta_wr = (result["wr"] - base["wr"]) if base else 0
                    delta_ev = (result["ev"] - base["ev"]) if base else 0
                    print(
                        f"  {label:>28} │ {result['total']:>6} {result['wr']:>5.1f}% "
                        f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} {result['vol_filtered']:>7} │ "
                        f"{delta_wr:>+4.1f}% ${delta_ev:>+6.3f}"
                    )
                    result["section"] = "vol_norm"
                    all_configs.append(result)

    # ═════════════════════════════════════════════════════════════════
    # SECTION 4: OBI GATING ONLY
    # ═════════════════════════════════════════════════════════════════
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  SECTION 4: PM OBI GATING ONLY — skip when PM order book strongly disagrees                       │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    header = f"  {'cfg':>28} │ {'Trades':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} {'ObiFilt':>7} │ {'ΔWR':>5} {'ΔEV':>8}"
    print(header)
    print(f"  {'─'*28} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*7} ┼ {'─'*5} {'─'*8}")

    for min_disp in [0.01, 0.02, 0.03]:
        for offset in [1, 2]:
            for obi_limit in [0.1, 0.2, 0.3, 0.5]:
                result = run_single_config(
                    candles, idx_by_time, first_ts, last_ts, days,
                    min_disp=min_disp, offset=offset,
                    use_velocity=False, use_vol_norm=False, min_z=0, use_obi=True, obi_limit=obi_limit,
                )
                if result:
                    label = f"d{min_disp:.3f}% T+{offset} OBI<{obi_limit:.1f}"
                    base = baselines.get((min_disp, offset))
                    delta_wr = (result["wr"] - base["wr"]) if base else 0
                    delta_ev = (result["ev"] - base["ev"]) if base else 0
                    print(
                        f"  {label:>28} │ {result['total']:>6} {result['wr']:>5.1f}% "
                        f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} {result['obi_filtered']:>7} │ "
                        f"{delta_wr:>+4.1f}% ${delta_ev:>+6.3f}"
                    )
                    result["section"] = "obi"
                    all_configs.append(result)

    # ═════════════════════════════════════════════════════════════════
    # SECTION 5: ALL FILTERS COMBINED — best combos
    # ═════════════════════════════════════════════════════════════════
    print(f"\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  SECTION 5: ALL FILTERS COMBINED — Velocity + Vol-Norm + OBI                                      │")
    print(f"└──────────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    header = f"  {'cfg':>38} │ {'Trades':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ {'VFilt':>5} {'ZFilt':>5} {'OFilt':>5} │ {'ΔWR':>5} {'ΔEV':>8}"
    print(header)
    print(f"  {'─'*38} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*5} {'─'*5} {'─'*5} ┼ {'─'*5} {'─'*8}")

    combo_configs = []
    for min_disp in [0.01, 0.02, 0.03]:
        for offset in [1, 2]:
            for min_z in [0.5, 1.0, 1.5]:
                for obi_limit in [0.1, 0.2, 0.3]:
                    # Velocity + Vol-Norm + OBI
                    result = run_single_config(
                        candles, idx_by_time, first_ts, last_ts, days,
                        min_disp=min_disp, offset=offset,
                        use_velocity=True, use_vol_norm=True, min_z=min_z,
                        use_obi=True, obi_limit=obi_limit,
                    )
                    if result:
                        label = f"d{min_disp:.3f}% T+{offset} V+Z>{min_z:.1f}+OBI<{obi_limit:.1f}"
                        base = baselines.get((min_disp, offset))
                        delta_wr = (result["wr"] - base["wr"]) if base else 0
                        delta_ev = (result["ev"] - base["ev"]) if base else 0
                        print(
                            f"  {label:>38} │ {result['total']:>6} {result['wr']:>5.1f}% "
                            f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} │ "
                            f"{result['velocity_filtered']:>5} {result['vol_filtered']:>5} {result['obi_filtered']:>5} │ "
                            f"{delta_wr:>+4.1f}% ${delta_ev:>+6.3f}"
                        )
                        result["section"] = "combined"
                        combo_configs.append(result)
                        all_configs.append(result)

    # Also test Velocity + Vol-Norm (no OBI)
    print(f"\n  --- Velocity + Vol-Norm (no OBI) ---")
    print(header)
    print(f"  {'─'*38} ┼ {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*5} {'─'*5} {'─'*5} ┼ {'─'*5} {'─'*8}")

    for min_disp in [0.01, 0.02, 0.03]:
        for offset in [1, 2]:
            for min_z in [0.5, 1.0, 1.5]:
                result = run_single_config(
                    candles, idx_by_time, first_ts, last_ts, days,
                    min_disp=min_disp, offset=offset,
                    use_velocity=True, use_vol_norm=True, min_z=min_z,
                    use_obi=False, obi_limit=0,
                )
                if result:
                    label = f"d{min_disp:.3f}% T+{offset} V+Z>{min_z:.1f}"
                    base = baselines.get((min_disp, offset))
                    delta_wr = (result["wr"] - base["wr"]) if base else 0
                    delta_ev = (result["ev"] - base["ev"]) if base else 0
                    print(
                        f"  {label:>38} │ {result['total']:>6} {result['wr']:>5.1f}% "
                        f"${result['ev']:>+7.3f} ${result['dpnl']:>+6.1f} │ "
                        f"{result['velocity_filtered']:>5} {result['vol_filtered']:>5} {'':>5} │ "
                        f"{delta_wr:>+4.1f}% ${delta_ev:>+6.3f}"
                    )
                    result["section"] = "vel+vol"
                    all_configs.append(result)

    # ═════════════════════════════════════════════════════════════════
    # SUMMARY: TOP CONFIGS ACROSS ALL SECTIONS
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  SUMMARY: TOP 20 CONFIGS BY WIN RATE (min 10 trades/week)")
    print(f"{'='*110}\n")

    viable = [c for c in all_configs if c["total"] >= 10]
    by_wr = sorted(viable, key=lambda x: (-x["wr"], -x["ev"]))

    print(
        f"  {'#':>2} {'WR%':>6} {'Trades':>6} {'EV/trd':>8} {'$/day':>7} {'Trd/d':>5} │ "
        f"{'min_d':>5} {'T+':>2} {'Filters':>18} │ {'VFilt':>5} {'ZFilt':>5} {'OFilt':>5}"
    )
    print(
        f"  {'─'*2} {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*5} ┼ "
        f"{'─'*5} {'─'*2} {'─'*18} ┼ {'─'*5} {'─'*5} {'─'*5}"
    )

    for i, c in enumerate(by_wr[:20]):
        fl = filters_label(c)
        is_live = (c["min_disp"] == 0.02 and c["offset"] == 1
                   and not c["use_velocity"] and not c["use_vol_norm"] and not c["use_obi"])
        marker = " ◄ LIVE" if is_live else ""
        print(
            f"  {i+1:>2} {c['wr']:>5.1f}% {c['total']:>6} "
            f"${c['ev']:>+7.3f} ${c['dpnl']:>+6.1f} {c['tpd']:>5.0f} │ "
            f"{c['min_disp']:>4.3f} {c['offset']:>2} {fl:>18} │ "
            f"{c['velocity_filtered']:>5} {c['vol_filtered']:>5} {c['obi_filtered']:>5}{marker}"
        )

    print(f"\n{'='*110}")
    print(f"  SUMMARY: TOP 20 CONFIGS BY DAILY PnL (min 10 trades/week)")
    print(f"{'='*110}\n")

    by_dpnl = sorted(viable, key=lambda x: -x["dpnl"])

    print(
        f"  {'#':>2} {'$/day':>7} {'WR%':>6} {'Trades':>6} {'EV/trd':>8} {'Trd/d':>5} │ "
        f"{'min_d':>5} {'T+':>2} {'Filters':>18} │ {'VFilt':>5} {'ZFilt':>5} {'OFilt':>5}"
    )
    print(
        f"  {'─'*2} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*5} ┼ "
        f"{'─'*5} {'─'*2} {'─'*18} ┼ {'─'*5} {'─'*5} {'─'*5}"
    )

    for i, c in enumerate(by_dpnl[:20]):
        fl = filters_label(c)
        is_live = (c["min_disp"] == 0.02 and c["offset"] == 1
                   and not c["use_velocity"] and not c["use_vol_norm"] and not c["use_obi"])
        marker = " ◄ LIVE" if is_live else ""
        print(
            f"  {i+1:>2} ${c['dpnl']:>+6.1f} {c['wr']:>5.1f}% {c['total']:>6} "
            f"${c['ev']:>+7.3f} {c['tpd']:>5.0f} │ "
            f"{c['min_disp']:>4.3f} {c['offset']:>2} {fl:>18} │ "
            f"{c['velocity_filtered']:>5} {c['vol_filtered']:>5} {c['obi_filtered']:>5}{marker}"
        )

    # ═════════════════════════════════════════════════════════════════
    # FILTER IMPACT: How much does each filter improve WR over baseline?
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  FILTER IMPACT ANALYSIS: Average WR improvement over baseline")
    print(f"{'='*110}\n")

    sections = {
        "velocity": "Velocity only",
        "vol_norm": "Vol-norm only",
        "obi": "OBI only",
        "vel+vol": "Velocity + Vol-norm",
        "combined": "All three",
    }

    for section, label in sections.items():
        section_cfgs = [c for c in all_configs if c.get("section") == section and c["total"] >= 10]
        if not section_cfgs:
            continue
        deltas = []
        for c in section_cfgs:
            base = baselines.get((c["min_disp"], c["offset"]))
            if base:
                deltas.append(c["wr"] - base["wr"])
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            max_delta = max(deltas)
            min_delta = min(deltas)
            print(f"  {label:>25}: avg ΔWR = {avg_delta:>+5.1f}%  (range: {min_delta:>+5.1f}% to {max_delta:>+5.1f}%)  [{len(section_cfgs)} configs]")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    run()
