"""Backtest: Exhaustive parameter sweep to maximize WR within 5-min windows.

Tests every tunable knob:
  1. Entry timing: T+0.5, T+1, T+1.5, T+2, T+2.5, T+3 minutes
  2. Min displacement threshold: 0.001% to 0.10%
  3. Sigmoid scale: 5, 8, 10, 12, 15, 20
  4. Displacement acceleration filter: require displacement to be increasing
  5. Volume filter: only trade high-volume candles
  6. Candle body filter: require strong candle (close near high/low)
  7. Displacement + reversal rejection: skip if price already reversing
  8. Two-candle confirmation: require 2 consecutive candles in same direction
"""

import math
import time
import requests
from collections import defaultdict


def fetch_binance_klines(days: int = 14) -> list[dict]:
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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def run():
    days = 14
    candles = fetch_binance_klines(days=days)
    if not candles:
        return

    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]
    window_ms = 5 * 60 * 1000

    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    # Precompute median volume for volume filter
    volumes = [c["volume"] for c in candles]
    volumes.sort()
    median_vol = volumes[len(volumes) // 2]

    def evaluate(offset_min, min_disp, scale, filters=None):
        """Run backtest with given params. offset_min can be float (0.5 = 30s)."""
        filters = filters or {}
        wins = losses = skipped = 0

        ws = ((first_ts // window_ms) + 1) * window_ms
        while ws + window_ms <= last_ts:
            we = ws + window_ms
            start_t = (ws // 60000) * 60000
            end_t = ((we - 60000) // 60000) * 60000

            # Entry time — support fractional minutes
            if offset_min == int(offset_min):
                entry_t = start_t + int(offset_min) * 60000
                if entry_t not in idx_by_time:
                    ws += window_ms
                    continue
                ni = idx_by_time[entry_t]
            else:
                # Interpolate between two candles
                lo_t = start_t + int(offset_min) * 60000
                hi_t = lo_t + 60000
                if lo_t not in idx_by_time or hi_t not in idx_by_time:
                    ws += window_ms
                    continue
                frac = offset_min - int(offset_min)
                lo_i = idx_by_time[lo_t]
                hi_i = idx_by_time[hi_t]
                # Use weighted average of close prices
                ni = lo_i  # use lo candle index for filters
                btc_entry_interp = candles[lo_i]["close"] * (1 - frac) + candles[hi_i]["close"] * frac

            if start_t not in idx_by_time or end_t not in idx_by_time:
                ws += window_ms
                continue

            si = idx_by_time[start_t]
            ei = idx_by_time[end_t]

            btc_start = candles[si]["open"]
            btc_end = candles[ei]["close"]

            if offset_min == int(offset_min):
                btc_entry = candles[ni]["close"]
            else:
                btc_entry = btc_entry_interp

            actual_up = btc_end > btc_start
            disp = (btc_entry - btc_start) / btc_start * 100

            if abs(disp) < min_disp:
                skipped += 1
                ws += window_ms
                continue

            # ── Optional filters ──

            # Volume filter: entry candle volume > X percentile of median
            if 'min_vol_mult' in filters:
                entry_vol = candles[ni]["volume"]
                if entry_vol < median_vol * filters['min_vol_mult']:
                    skipped += 1
                    ws += window_ms
                    continue

            # Candle body filter: entry candle should have strong body
            # (close near high for up, close near low for down)
            if 'min_body_ratio' in filters:
                c = candles[ni]
                body = abs(c["close"] - c["open"])
                wick = c["high"] - c["low"]
                if wick > 0:
                    body_ratio = body / wick
                    if body_ratio < filters['min_body_ratio']:
                        skipped += 1
                        ws += window_ms
                        continue

            # Acceleration filter: displacement at entry > displacement at entry-1min
            if filters.get('require_acceleration'):
                prev_t = start_t + max(0, int(offset_min) - 1) * 60000
                if prev_t in idx_by_time and prev_t != start_t:
                    prev_i = idx_by_time[prev_t]
                    prev_disp = (candles[prev_i]["close"] - btc_start) / btc_start * 100
                    # Displacement should be growing in same direction
                    if disp > 0 and prev_disp > 0 and abs(disp) < abs(prev_disp):
                        skipped += 1
                        ws += window_ms
                        continue
                    if disp < 0 and prev_disp < 0 and abs(disp) < abs(prev_disp):
                        skipped += 1
                        ws += window_ms
                        continue

            # Reversal rejection: skip if the last candle reversed direction
            if filters.get('reject_reversal'):
                entry_c = candles[ni]
                candle_dir_up = entry_c["close"] > entry_c["open"]
                if (disp > 0) != candle_dir_up:
                    skipped += 1
                    ws += window_ms
                    continue

            # Two-candle confirmation: both entry candle and previous must agree
            if filters.get('two_candle_confirm'):
                if ni > 0:
                    prev_c = candles[ni - 1]
                    entry_c = candles[ni]
                    prev_up = prev_c["close"] > prev_c["open"]
                    curr_up = entry_c["close"] > entry_c["open"]
                    if prev_up != curr_up:
                        skipped += 1
                        ws += window_ms
                        continue

            # Max displacement filter: skip if displacement too large (overextended)
            if 'max_disp' in filters:
                if abs(disp) > filters['max_disp']:
                    skipped += 1
                    ws += window_ms
                    continue

            # Direction
            dir_yes = disp > 0
            right = (dir_yes == actual_up)
            if right:
                wins += 1
            else:
                losses += 1

            ws += window_ms

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        return {"total": total, "wins": wins, "losses": losses, "wr": wr, "skipped": skipped}

    # ═══════════════════════════════════════════════════════════
    # 1. Entry timing sweep
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  1. ENTRY TIMING — When to read displacement within the 5-min window")
    print(f"{'='*90}\n")

    print(f"  {'Entry Time':>12} │ {'d≥0.001%':>14} │ {'d≥0.005%':>14} │ {'d≥0.010%':>14} │ {'d≥0.020%':>14} │ {'d≥0.040%':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14}")

    for offset in [0.5, 1, 1.5, 2, 2.5, 3]:
        row = f"  {'T+'+str(offset)+'min':>12} │"
        for md in [0.001, 0.005, 0.010, 0.020, 0.040]:
            r = evaluate(offset, md, 10.0)
            row += f" {r['wr']:>5.1f}% ({r['total']:>4}) │"
        print(row)

    # ═══════════════════════════════════════════════════════════
    # 2. Min displacement threshold sweep
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  2. MIN DISPLACEMENT THRESHOLD — How much BTC must move before we trade")
    print(f"{'='*90}\n")

    print(f"  {'Threshold':>12} │ {'T+1min':>14} │ {'T+2min':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14}")

    for md in [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]:
        r1 = evaluate(1, md, 10.0)
        r2 = evaluate(2, md, 10.0)
        print(f"  {md:>11.3f}% │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 3. Sigmoid scale sweep
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  3. SIGMOID SCALE — Confidence mapping aggressiveness (current=10)")
    print(f"     (Note: scale only affects Kelly sizing, not direction. WR is identical.)")
    print(f"{'='*90}\n")

    print(f"  Sigmoid scale only changes fair_up_prob magnitude → affects Kelly/sizing.")
    print(f"  Direction is always: displacement>0 → BUY_YES, <0 → BUY_NO")
    print(f"  So WR is identical across all scales for same displacement filter.\n")

    # ═══════════════════════════════════════════════════════════
    # 4. Volume filter
    # ═══════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print(f"  4. VOLUME FILTER — Only trade when entry candle volume > X × median")
    print(f"{'='*90}\n")

    print(f"  {'Vol Filter':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14}")

    for vol_mult in [0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        label = "none" if vol_mult == 0 else f">{vol_mult:.1f}x"
        filters = {'min_vol_mult': vol_mult} if vol_mult > 0 else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 5. Candle body filter
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  5. CANDLE BODY FILTER — Require strong candle body (body/wick ratio)")
    print(f"{'='*90}\n")

    print(f"  {'Body Ratio':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14}")

    for br in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        label = "none" if br == 0 else f"≥{br:.1f}"
        filters = {'min_body_ratio': br} if br > 0 else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 6. Acceleration filter
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  6. ACCELERATION FILTER — Only trade when displacement is growing")
    print(f"{'='*90}\n")

    print(f"  {'Filter':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14} │ {'d≥0.01% T+2':>14} │ {'d≥0.02% T+2':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14}")

    for accel in [False, True]:
        label = "accel" if accel else "none"
        filters = {'require_acceleration': True} if accel else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        r3 = evaluate(2, 0.010, 10.0, filters)
        r4 = evaluate(2, 0.020, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4}) │ {r3['wr']:>5.1f}% ({r3['total']:>4}) │ {r4['wr']:>5.1f}% ({r4['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 7. Reversal rejection
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  7. REVERSAL REJECTION — Skip if entry candle opposes displacement direction")
    print(f"{'='*90}\n")

    print(f"  {'Filter':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14} │ {'d≥0.01% T+2':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14}")

    for rev in [False, True]:
        label = "reject_rev" if rev else "none"
        filters = {'reject_reversal': True} if rev else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        r3 = evaluate(2, 0.010, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4}) │ {r3['wr']:>5.1f}% ({r3['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 8. Two-candle confirmation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  8. TWO-CANDLE CONFIRMATION — Both entry + previous candle must agree on direction")
    print(f"{'='*90}\n")

    print(f"  {'Filter':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14} │ {'d≥0.01% T+2':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14}")

    for tc in [False, True]:
        label = "2-candle" if tc else "none"
        filters = {'two_candle_confirm': True} if tc else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        r3 = evaluate(2, 0.010, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4}) │ {r3['wr']:>5.1f}% ({r3['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 9. Max displacement (overextension) filter
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  9. MAX DISPLACEMENT — Skip overextended moves that might revert")
    print(f"{'='*90}\n")

    print(f"  {'Max Disp':>12} │ {'d≥0.01% T+1':>14} │ {'d≥0.02% T+1':>14}")
    print(f"  {'─'*12} ┼ {'─'*14} ┼ {'─'*14}")

    for mx in [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
        label = "none" if mx == 0 else f"≤{mx:.2f}%"
        filters = {'max_disp': mx} if mx > 0 else {}
        r1 = evaluate(1, 0.010, 10.0, filters)
        r2 = evaluate(1, 0.020, 10.0, filters)
        print(f"  {label:>12} │ {r1['wr']:>5.1f}% ({r1['total']:>4}) │ {r2['wr']:>5.1f}% ({r2['total']:>4})")

    # ═══════════════════════════════════════════════════════════
    # 10. BEST COMBOS — Stack the best filters together
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  10. BEST COMBINATIONS — Stacking top filters")
    print(f"{'='*90}\n")

    combos = [
        ("baseline",                    1, 0.020, {}),
        ("+ reject_reversal",           1, 0.020, {'reject_reversal': True}),
        ("+ acceleration",              1, 0.020, {'require_acceleration': True}),
        ("+ rev + accel",               1, 0.020, {'reject_reversal': True, 'require_acceleration': True}),
        ("+ body≥0.3",                  1, 0.020, {'min_body_ratio': 0.3}),
        ("+ rev + body≥0.3",            1, 0.020, {'reject_reversal': True, 'min_body_ratio': 0.3}),
        ("+ 2candle",                   1, 0.020, {'two_candle_confirm': True}),
        ("+ rev + 2candle",             1, 0.020, {'reject_reversal': True, 'two_candle_confirm': True}),
        ("+ max_disp≤0.15",             1, 0.020, {'max_disp': 0.15}),
        ("+ rev + max≤0.15",            1, 0.020, {'reject_reversal': True, 'max_disp': 0.15}),
        ("d0.03% baseline",             1, 0.030, {}),
        ("d0.03% + rev",                1, 0.030, {'reject_reversal': True}),
        ("d0.04% baseline",             1, 0.040, {}),
        ("d0.04% + rev",                1, 0.040, {'reject_reversal': True}),
        ("d0.05% baseline",             1, 0.050, {}),
        ("d0.05% + rev",                1, 0.050, {'reject_reversal': True}),
        ("d0.06% baseline",             1, 0.060, {}),
        ("d0.08% baseline",             1, 0.080, {}),
        ("d0.10% baseline",             1, 0.100, {}),
        ("T+2 d0.02%",                  2, 0.020, {}),
        ("T+2 d0.02% + rev",            2, 0.020, {'reject_reversal': True}),
        ("T+2 d0.03%",                  2, 0.030, {}),
        ("T+2 d0.03% + rev",            2, 0.030, {'reject_reversal': True}),
        ("T+2 d0.04%",                  2, 0.040, {}),
        ("T+2 d0.05%",                  2, 0.050, {}),
    ]

    print(f"  {'Config':>28} │ {'Trades':>6} {'WR%':>6} │ {'Δ WR':>6} │ {'Trades/day':>10}")
    print(f"  {'─'*28} ┼ {'─'*6} {'─'*6} ┼ {'─'*6} ┼ {'─'*10}")

    baseline_wr = None
    for label, offset, md, filters in combos:
        r = evaluate(offset, md, 10.0, filters)
        if baseline_wr is None:
            baseline_wr = r['wr']
        delta = r['wr'] - baseline_wr
        tpd = r['total'] / days
        marker = " ★" if r['wr'] >= baseline_wr + 2 and r['total'] >= 100 else ""
        print(f"  {label:>28} │ {r['total']:>6} {r['wr']:>5.1f}% │ {delta:>+5.1f}% │ {tpd:>9.0f}/d{marker}")

    print(f"\n{'='*90}")
    print(f"  ★ = 2%+ WR improvement with ≥100 trades (statistically meaningful)")
    print(f"{'='*90}")


if __name__ == "__main__":
    run()
