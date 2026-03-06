"""Backtest: High-frequency configs — maximize trades/day while staying above 60% WR.

Sweeps very low displacement thresholds with and without filters
to find the sweet spot for "trade almost every window."
"""

import math
import time
import requests


KELLY_CAP = 0.12
MIN_EDGE_PCT = 0.02
PRICE_FLOOR = 0.05
PRICE_CEIL = 0.80
MIN_KELLY = 0.005
SCALE = 10.0


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
    return sigmoid(pm_efficiency * displacement_pct)


def compute_velocity(candles, entry_idx):
    if entry_idx < 1:
        return 0.0
    prev = candles[entry_idx - 1]["close"]
    curr = candles[entry_idx]["close"]
    if prev <= 0:
        return 0.0
    return (curr - prev) / prev * 100


def compute_rolling_stdev(candles, entry_idx, lookback=5):
    start = max(0, entry_idx - lookback)
    if entry_idx - start < 3:
        return 0.01
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


def run_config(candles, idx_by_time, first_ts, last_ts, days,
               min_disp, offset, use_vel, use_vol, min_z):
    window_ms = 5 * 60 * 1000
    wins = losses = skipped = vel_filt = vol_filt = 0
    total_pnl = entry_sum = edge_sum = 0.0
    windows_checked = 0

    ws = ((first_ts // window_ms) + 1) * window_ms
    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000
        entry_t = start_t + offset * 60000

        if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
            ws += window_ms
            continue

        windows_checked += 1
        si = idx_by_time[start_t]
        ei = idx_by_time[end_t]
        ni = idx_by_time[entry_t]

        btc_start = candles[si]["open"]
        btc_end = candles[ei]["close"]
        btc_entry = candles[ni]["close"]
        actual_up = btc_end > btc_start

        disp = (btc_entry - btc_start) / btc_start * 100

        if abs(disp) < min_disp:
            skipped += 1
            ws += window_ms
            continue

        if use_vel and offset > 0:
            vel = compute_velocity(candles, ni)
            if (disp > 0 and vel < 0) or (disp < 0 and vel > 0):
                vel_filt += 1
                ws += window_ms
                continue

        if use_vol:
            stdev = compute_rolling_stdev(candles, ni)
            z = disp / stdev if stdev > 1e-8 else 0.0
            if abs(z) < min_z:
                vol_filt += 1
                ws += window_ms
                continue

        fair_up = sigmoid(SCALE * disp)
        fair_up = max(0.01, min(0.99, fair_up))
        pm_mid = simulate_pm_price(disp, 0.3)
        spread = (fair_up - pm_mid) * 100

        if abs(spread) < MIN_EDGE_PCT:
            skipped += 1
            ws += window_ms
            continue

        dir_yes = spread > 0
        entry_price = pm_mid if dir_yes else 1.0 - pm_mid

        if entry_price < PRICE_FLOOR or entry_price > PRICE_CEIL:
            skipped += 1
            ws += window_ms
            continue

        win_prob = fair_up if dir_yes else 1.0 - fair_up
        payout = (1.0 / entry_price) - 1.0
        q = 1.0 - win_prob
        kelly = min(max((win_prob * payout - q) / payout, 0), KELLY_CAP)

        if kelly < MIN_KELLY:
            skipped += 1
            ws += window_ms
            continue

        right = (dir_yes == actual_up)
        edge_sum += abs(spread)
        entry_sum += entry_price

        if right:
            total_pnl += 1.0 - entry_price
            wins += 1
        else:
            total_pnl -= entry_price
            losses += 1

        ws += window_ms

    total = wins + losses
    if total < 5:
        return None
    return {
        "total": total, "wins": wins, "losses": losses,
        "wr": wins / total * 100, "ev": total_pnl / total,
        "tpd": total / days, "dpnl": total_pnl / days,
        "avg_entry": entry_sum / total, "avg_edge": edge_sum / total,
        "windows": windows_checked, "coverage": total / windows_checked * 100,
        "skipped": skipped, "vel_filt": vel_filt, "vol_filt": vol_filt,
        "min_disp": min_disp, "offset": offset,
        "use_vel": use_vel, "use_vol": use_vol, "min_z": min_z,
    }


def run():
    days = 7
    candles = fetch_binance_klines(days=days)
    if not candles:
        return

    window_ms = 5 * 60 * 1000
    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]
    total_windows = 0
    ws = ((first_ts // window_ms) + 1) * window_ms
    while ws + window_ms <= last_ts:
        ws += window_ms
        total_windows += 1

    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    print(f"\n{'='*110}")
    print(f"  HIGH-FREQUENCY BACKTEST — Target: Trade almost every window (max {total_windows} = {total_windows/days:.0f}/day)")
    print(f"{'='*110}")

    all_results = []

    # ═══════════════════════════════════════════
    # SECTION 1: No filters — raw displacement
    # ═══════════════════════════════════════════
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  NO FILTERS — How many windows does each displacement threshold catch?                        │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    print(f"  {'Config':>20} │ {'Trades':>6} {'Trd/d':>5} {'Cover%':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ {'Skip':>5} {'Windows':>7}")
    print(f"  {'─'*20} ┼ {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*5} {'─'*7}")

    for min_disp in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
        for offset in [0, 1, 2]:
            r = run_config(candles, idx_by_time, first_ts, last_ts, days,
                           min_disp, offset, False, False, 0)
            if r:
                label = f"d{min_disp:.3f}% T+{offset}"
                live = (min_disp == 0.02 and offset == 1)
                m = " ◄" if live else ""
                print(f"  {label:>20} │ {r['total']:>6} {r['tpd']:>5.0f} {r['coverage']:>5.1f}% "
                      f"{r['wr']:>5.1f}% ${r['ev']:>+7.3f} ${r['dpnl']:>+6.1f} │ "
                      f"{r['skipped']:>5} {r['windows']:>7}{m}")
                all_results.append(r)

    # ═══════════════════════════════════════════
    # SECTION 2: Velocity only (lightweight filter)
    # ═══════════════════════════════════════════
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  VELOCITY ONLY — light filter, minimal trade reduction                                        │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    print(f"  {'Config':>20} │ {'Trades':>6} {'Trd/d':>5} {'Cover%':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ {'VFilt':>5}")
    print(f"  {'─'*20} ┼ {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*5}")

    for min_disp in [0.0, 0.001, 0.005, 0.01, 0.02]:
        for offset in [0, 1, 2]:
            r = run_config(candles, idx_by_time, first_ts, last_ts, days,
                           min_disp, offset, True, False, 0)
            if r:
                label = f"d{min_disp:.3f}% T+{offset} +V"
                print(f"  {label:>20} │ {r['total']:>6} {r['tpd']:>5.0f} {r['coverage']:>5.1f}% "
                      f"{r['wr']:>5.1f}% ${r['ev']:>+7.3f} ${r['dpnl']:>+6.1f} │ "
                      f"{r['vel_filt']:>5}")
                all_results.append(r)

    # ═══════════════════════════════════════════
    # SECTION 3: Vol-norm with low z threshold
    # ═══════════════════════════════════════════
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  LOW Z-THRESHOLD — adaptive filter that still allows most trades                               │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    print(f"  {'Config':>24} │ {'Trades':>6} {'Trd/d':>5} {'Cover%':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ {'ZFilt':>5}")
    print(f"  {'─'*24} ┼ {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*5}")

    for min_disp in [0.0, 0.005, 0.01]:
        for offset in [0, 1]:
            for min_z in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
                r = run_config(candles, idx_by_time, first_ts, last_ts, days,
                               min_disp, offset, False, True, min_z)
                if r:
                    label = f"d{min_disp:.3f}% T+{offset} Z>{min_z:.2f}"
                    print(f"  {label:>24} │ {r['total']:>6} {r['tpd']:>5.0f} {r['coverage']:>5.1f}% "
                          f"{r['wr']:>5.1f}% ${r['ev']:>+7.3f} ${r['dpnl']:>+6.1f} │ "
                          f"{r['vol_filt']:>5}")
                    all_results.append(r)

    # ═══════════════════════════════════════════
    # SECTION 4: Velocity + low z combined
    # ═══════════════════════════════════════════
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  VELOCITY + LOW Z — best of both, minimal trade loss                                          │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────────┘\n")

    print(f"  {'Config':>28} │ {'Trades':>6} {'Trd/d':>5} {'Cover%':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ {'VF':>4} {'ZF':>4}")
    print(f"  {'─'*28} ┼ {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ {'─'*4} {'─'*4}")

    for min_disp in [0.0, 0.005, 0.01]:
        for offset in [0, 1]:
            for min_z in [0.1, 0.2, 0.3, 0.5]:
                r = run_config(candles, idx_by_time, first_ts, last_ts, days,
                               min_disp, offset, True, True, min_z)
                if r:
                    label = f"d{min_disp:.3f}% T+{offset} V+Z>{min_z:.1f}"
                    print(f"  {label:>28} │ {r['total']:>6} {r['tpd']:>5.0f} {r['coverage']:>5.1f}% "
                          f"{r['wr']:>5.1f}% ${r['ev']:>+7.3f} ${r['dpnl']:>+6.1f} │ "
                          f"{r['vel_filt']:>4} {r['vol_filt']:>4}")
                    all_results.append(r)

    # ═══════════════════════════════════════════
    # SUMMARY: Best configs above 60% WR, sorted by trades/day
    # ═══════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  BEST HIGH-FREQUENCY CONFIGS: WR >= 60%, sorted by trades/day")
    print(f"{'='*110}\n")

    viable = [r for r in all_results if r["wr"] >= 60 and r["total"] >= 20]
    by_tpd = sorted(viable, key=lambda x: -x["tpd"])

    print(f"  {'#':>2} {'Trd/d':>5} {'Cover%':>6} {'Win%':>6} {'EV/trd':>8} {'$/day':>7} │ "
          f"{'min_d':>6} {'T+':>2} {'Vel':>3} {'Z':>4} │ {'VF':>4} {'ZF':>4}")
    print(f"  {'─'*2} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*7} ┼ "
          f"{'─'*6} {'─'*2} {'─'*3} {'─'*4} ┼ {'─'*4} {'─'*4}")

    seen = set()
    count = 0
    for r in by_tpd:
        key = (r["min_disp"], r["offset"], r["use_vel"], r["use_vol"], r["min_z"])
        if key in seen:
            continue
        seen.add(key)
        count += 1
        if count > 25:
            break
        z_str = f"{r['min_z']:.1f}" if r["use_vol"] else "  -"
        v_str = "Y" if r["use_vel"] else "-"
        print(f"  {count:>2} {r['tpd']:>5.0f} {r['coverage']:>5.1f}% "
              f"{r['wr']:>5.1f}% ${r['ev']:>+7.3f} ${r['dpnl']:>+6.1f} │ "
              f"{r['min_disp']:>5.3f} {r['offset']:>2} {v_str:>3} {z_str:>4} │ "
              f"{r['vel_filt']:>4} {r['vol_filt']:>4}")

    # Highlight the sweet spot
    print(f"\n  Max possible windows/day: {total_windows/days:.0f}")
    print(f"  Total configs tested: {len(all_results)}")
    print(f"  Configs with WR >= 60%: {len(viable)}")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    run()
