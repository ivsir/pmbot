"""Backtest: Maximize WR at T+1min entry (cheap fills under 50¢).

Goal: Stay at T+1 for cheap PM fills, stack filters to push WR as high as possible.
"""

import math
import time
import requests


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


def evaluate(candles, idx_by_time, first_ts, last_ts,
             min_disp, filters=None):
    """T+1 entry only. Returns WR and trade details."""
    filters = filters or {}
    window_ms = 5 * 60 * 1000
    wins = losses = skipped = 0

    # Precompute median volume
    volumes = sorted([c["volume"] for c in candles])
    median_vol = volumes[len(volumes) // 2]

    ws = ((first_ts // window_ms) + 1) * window_ms
    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000
        entry_t = start_t + 60000  # T+1min always

        if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
            ws += window_ms
            continue

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

        # ── Filters ──

        # Reversal rejection: entry candle must agree with displacement
        if filters.get('reject_reversal'):
            entry_c = candles[ni]
            candle_up = entry_c["close"] > entry_c["open"]
            if (disp > 0) != candle_up:
                skipped += 1
                ws += window_ms
                continue

        # Two-candle confirmation
        if filters.get('two_candle'):
            if ni > 0:
                prev_c = candles[ni - 1]
                entry_c = candles[ni]
                if (prev_c["close"] > prev_c["open"]) != (entry_c["close"] > entry_c["open"]):
                    skipped += 1
                    ws += window_ms
                    continue

        # Volume filter
        if 'min_vol_mult' in filters:
            if candles[ni]["volume"] < median_vol * filters['min_vol_mult']:
                skipped += 1
                ws += window_ms
                continue

        # Candle body ratio
        if 'min_body_ratio' in filters:
            c = candles[ni]
            body = abs(c["close"] - c["open"])
            wick = c["high"] - c["low"]
            if wick > 0 and (body / wick) < filters['min_body_ratio']:
                skipped += 1
                ws += window_ms
                continue

        # Entry candle range filter: the first candle (window open → T+1)
        # must have moved significantly relative to its range
        if 'min_candle_move_pct' in filters:
            c = candles[ni]
            candle_range = c["high"] - c["low"]
            candle_move = abs(c["close"] - c["open"])
            if candle_range > 0:
                move_pct = candle_move / c["open"] * 100
                if move_pct < filters['min_candle_move_pct']:
                    skipped += 1
                    ws += window_ms
                    continue

        # Wick rejection: if displacing UP, top wick should be small
        # (price closed near the high = strong conviction)
        if filters.get('wick_rejection'):
            c = candles[ni]
            full_range = c["high"] - c["low"]
            if full_range > 0:
                if disp > 0:
                    # Bullish: reject if big upper wick (selling pressure)
                    upper_wick = c["high"] - max(c["close"], c["open"])
                    if upper_wick / full_range > 0.4:
                        skipped += 1
                        ws += window_ms
                        continue
                else:
                    # Bearish: reject if big lower wick (buying pressure)
                    lower_wick = min(c["close"], c["open"]) - c["low"]
                    if lower_wick / full_range > 0.4:
                        skipped += 1
                        ws += window_ms
                        continue

        # Pre-window momentum: check the candle BEFORE the window
        # If BTC was already moving in this direction before the window, more confident
        if filters.get('pre_window_momentum'):
            pre_t = start_t - 60000
            if pre_t in idx_by_time:
                pre_c = candles[idx_by_time[pre_t]]
                pre_up = pre_c["close"] > pre_c["open"]
                if (disp > 0) != pre_up:
                    skipped += 1
                    ws += window_ms
                    continue

        # Displacement velocity: how fast did we get to this displacement?
        # Compare displacement at T+30s (T+0 candle close) vs T+1min
        if filters.get('require_velocity'):
            t0_candle_t = start_t
            if t0_candle_t in idx_by_time:
                t0_i = idx_by_time[t0_candle_t]
                t0_disp = (candles[t0_i]["close"] - btc_start) / btc_start * 100
                # Require displacement grew from T+0 to T+1
                if abs(disp) <= abs(t0_disp) * 1.0:
                    skipped += 1
                    ws += window_ms
                    continue

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


def run():
    days = 14
    candles = fetch_binance_klines(days=days)
    if not candles:
        return

    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]

    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    print(f"\n{'='*95}")
    print(f"  T+1min ENTRY OPTIMIZATION — Maximize WR while keeping cheap fills")
    print(f"  Goal: enter at T+1 (PM still near 50¢), filter for highest-conviction setups")
    print(f"{'='*95}")

    # ═══════════════════════════════════════════════
    # Individual filter impact at each displacement
    # ═══════════════════════════════════════════════
    print(f"\n  INDIVIDUAL FILTERS (each tested alone at T+1)")
    print(f"  {'─'*91}")

    filter_configs = [
        ("baseline",            {}),
        ("reject_reversal",     {'reject_reversal': True}),
        ("2-candle confirm",    {'two_candle': True}),
        ("wick_rejection",      {'wick_rejection': True}),
        ("pre_window_momentum", {'pre_window_momentum': True}),
        ("vol>1.5x median",    {'min_vol_mult': 1.5}),
        ("body_ratio≥0.4",     {'min_body_ratio': 0.4}),
        ("velocity (growing)", {'require_velocity': True}),
    ]

    for md in [0.020, 0.030, 0.040, 0.050]:
        print(f"\n  Min disp: {md:.3f}%")
        print(f"  {'Filter':>24} │ {'Trades':>6} {'WR%':>6} │ {'Δ WR':>6} │ {'Trades/day':>10}")
        print(f"  {'─'*24} ┼ {'─'*6} {'─'*6} ┼ {'─'*6} ┼ {'─'*10}")

        baseline_wr = None
        for label, filters in filter_configs:
            r = evaluate(candles, idx_by_time, first_ts, last_ts, md, filters)
            if baseline_wr is None:
                baseline_wr = r['wr']
            delta = r['wr'] - baseline_wr
            tpd = r['total'] / days
            print(f"  {label:>24} │ {r['total']:>6} {r['wr']:>5.1f}% │ {delta:>+5.1f}% │ {tpd:>9.0f}/d")

    # ═══════════════════════════════════════════════
    # Best stacked combos
    # ═══════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  STACKED FILTER COMBINATIONS — Best of each stacked together")
    print(f"{'='*95}")

    combos = [
        # (label, min_disp, filters)
        ("d0.02% baseline",                     0.020, {}),
        ("d0.02% + rev",                        0.020, {'reject_reversal': True}),
        ("d0.02% + 2candle",                    0.020, {'two_candle': True}),
        ("d0.02% + wick",                       0.020, {'wick_rejection': True}),
        ("d0.02% + rev + wick",                 0.020, {'reject_reversal': True, 'wick_rejection': True}),
        ("d0.02% + 2candle + wick",             0.020, {'two_candle': True, 'wick_rejection': True}),
        ("d0.02% + rev + 2candle",              0.020, {'reject_reversal': True, 'two_candle': True}),
        ("d0.02% + rev + 2candle + wick",       0.020, {'reject_reversal': True, 'two_candle': True, 'wick_rejection': True}),
        ("d0.02% + pre_mom",                    0.020, {'pre_window_momentum': True}),
        ("d0.02% + pre_mom + rev",              0.020, {'pre_window_momentum': True, 'reject_reversal': True}),
        ("d0.02% + pre_mom + 2candle",          0.020, {'pre_window_momentum': True, 'two_candle': True}),
        ("",                                    0, {}),  # separator
        ("d0.03% baseline",                     0.030, {}),
        ("d0.03% + rev",                        0.030, {'reject_reversal': True}),
        ("d0.03% + 2candle",                    0.030, {'two_candle': True}),
        ("d0.03% + rev + wick",                 0.030, {'reject_reversal': True, 'wick_rejection': True}),
        ("d0.03% + rev + 2candle",              0.030, {'reject_reversal': True, 'two_candle': True}),
        ("d0.03% + 2candle + wick",             0.030, {'two_candle': True, 'wick_rejection': True}),
        ("d0.03% + rev + 2candle + wick",       0.030, {'reject_reversal': True, 'two_candle': True, 'wick_rejection': True}),
        ("d0.03% + pre_mom + rev",              0.030, {'pre_window_momentum': True, 'reject_reversal': True}),
        ("d0.03% + pre_mom + 2candle",          0.030, {'pre_window_momentum': True, 'two_candle': True}),
        ("",                                    0, {}),  # separator
        ("d0.04% baseline",                     0.040, {}),
        ("d0.04% + rev",                        0.040, {'reject_reversal': True}),
        ("d0.04% + 2candle",                    0.040, {'two_candle': True}),
        ("d0.04% + rev + wick",                 0.040, {'reject_reversal': True, 'wick_rejection': True}),
        ("d0.04% + rev + 2candle",              0.040, {'reject_reversal': True, 'two_candle': True}),
        ("d0.04% + 2candle + wick",             0.040, {'two_candle': True, 'wick_rejection': True}),
        ("d0.04% + pre_mom",                    0.040, {'pre_window_momentum': True}),
        ("d0.04% + pre_mom + rev",              0.040, {'pre_window_momentum': True, 'reject_reversal': True}),
        ("d0.04% + pre_mom + 2candle",          0.040, {'pre_window_momentum': True, 'two_candle': True}),
        ("",                                    0, {}),  # separator
        ("d0.05% baseline",                     0.050, {}),
        ("d0.05% + rev",                        0.050, {'reject_reversal': True}),
        ("d0.05% + 2candle",                    0.050, {'two_candle': True}),
        ("d0.05% + rev + 2candle",              0.050, {'reject_reversal': True, 'two_candle': True}),
        ("d0.05% + pre_mom + 2candle",          0.050, {'pre_window_momentum': True, 'two_candle': True}),
        ("d0.05% + rev + 2candle + wick",       0.050, {'reject_reversal': True, 'two_candle': True, 'wick_rejection': True}),
        ("",                                    0, {}),  # separator
        ("d0.06% baseline",                     0.060, {}),
        ("d0.06% + rev",                        0.060, {'reject_reversal': True}),
        ("d0.06% + 2candle",                    0.060, {'two_candle': True}),
        ("d0.06% + pre_mom",                    0.060, {'pre_window_momentum': True}),
        ("d0.08% baseline",                     0.080, {}),
        ("d0.08% + 2candle",                    0.080, {'two_candle': True}),
        ("d0.10% baseline",                     0.100, {}),
    ]

    baseline_r = evaluate(candles, idx_by_time, first_ts, last_ts, 0.020, {})
    base_wr = baseline_r['wr']

    print(f"\n  {'Config':>36} │ {'Trades':>6} {'WR%':>6} │ {'Δ vs 0.02%':>10} │ {'Trades/day':>10}")
    print(f"  {'─'*36} ┼ {'─'*6} {'─'*6} ┼ {'─'*10} ┼ {'─'*10}")

    for label, md, filters in combos:
        if not label:
            print(f"  {'─'*36} ┼ {'─'*6} {'─'*6} ┼ {'─'*10} ┼ {'─'*10}")
            continue
        r = evaluate(candles, idx_by_time, first_ts, last_ts, md, filters)
        delta = r['wr'] - base_wr
        tpd = r['total'] / days
        star = " ★" if r['wr'] >= 80 and r['total'] >= 50 * days else ""
        print(f"  {label:>36} │ {r['total']:>6} {r['wr']:>5.1f}% │ {delta:>+9.1f}% │ {tpd:>9.0f}/d{star}")

    # ═══════════════════════════════════════════════
    # Optimal frontier: WR vs trades/day
    # ═══════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  OPTIMAL FRONTIER — Best WR for each trade frequency band")
    print(f"  ★ = recommended (best WR with ≥50 trades/day)")
    print(f"{'='*95}")

    # Collect all results
    all_results = []
    for label, md, filters in combos:
        if not label:
            continue
        r = evaluate(candles, idx_by_time, first_ts, last_ts, md, filters)
        r['label'] = label
        r['tpd'] = r['total'] / days
        all_results.append(r)

    # Sort by WR descending
    all_results.sort(key=lambda x: (-x['wr'], -x['total']))

    print(f"\n  Top 15 by WR (with ≥{30*days} trades = ≥30/day):\n")
    print(f"  {'#':>3} {'Config':>36} │ {'WR%':>6} {'Trades/day':>10}")
    print(f"  {'─'*3} {'─'*36} ┼ {'─'*6} {'─'*10}")
    shown = 0
    for r in all_results:
        if r['total'] < 30 * days:
            continue
        shown += 1
        if shown > 15:
            break
        star = " ◀ BEST" if shown == 1 else ""
        print(f"  {shown:>3} {r['label']:>36} │ {r['wr']:>5.1f}% {r['tpd']:>9.0f}/d{star}")

    print(f"\n{'='*95}")


if __name__ == "__main__":
    run()
