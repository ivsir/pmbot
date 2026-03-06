"""Backtest: Does a 15-min trend filter improve displacement model WR?

Compares:
  1. Baseline: 5-min displacement only (current model)
  2. With-trend: only trade when 15-min trend agrees with 5-min displacement
  3. Counter-trend: only trade when 15-min trend DISAGREES
  4. Trend-weighted: use 15-min trend to boost/dampen confidence

Also tests 30-min and 60-min trend windows.
"""

import math
import time
import requests


SCALE = 10.0
MIN_EDGE_PCT = 0.02
PRICE_FLOOR = 0.05
PRICE_CEIL = 0.80
MIN_KELLY = 0.005
KELLY_FRAC = 0.12


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


def simulate_pm_price(disp, eff=0.3):
    return sigmoid(eff * disp)


def run_trend_backtest(candles, idx_by_time, first_ts, last_ts,
                       min_disp, offset, trend_minutes, trend_mode):
    """
    trend_mode:
      'none'     — baseline, no trend filter
      'with'     — only trade when trend agrees with displacement
      'counter'  — only trade when trend disagrees
      'boost'    — scale sigmoid by trend strength (multiply scale by 1+trend_factor)
    """
    window_ms = 5 * 60 * 1000
    wins = losses = skipped = 0
    with_trend_wins = with_trend_losses = 0
    counter_trend_wins = counter_trend_losses = 0

    ws = ((first_ts // window_ms) + 1) * window_ms

    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000
        entry_t = start_t + offset * 60000

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

        # ── 15/30/60-min trend ──
        trend_t = start_t - trend_minutes * 60000
        if trend_t not in idx_by_time:
            # Find nearest candle within 2 minutes
            found = False
            for delta in range(0, 120001, 60000):
                if trend_t - delta in idx_by_time:
                    trend_t = trend_t - delta
                    found = True
                    break
                if trend_t + delta in idx_by_time:
                    trend_t = trend_t + delta
                    found = True
                    break
            if not found:
                ws += window_ms
                continue

        ti = idx_by_time[trend_t]
        btc_trend_start = candles[ti]["close"]
        trend_return = (btc_entry - btc_trend_start) / btc_trend_start * 100
        trend_up = trend_return > 0

        # Displacement direction
        disp_up = disp > 0
        trend_agrees = (disp_up == trend_up)

        # Track with/counter-trend WR regardless of mode
        if trend_agrees:
            if (disp_up == actual_up):
                with_trend_wins += 1
            else:
                with_trend_losses += 1
        else:
            if (disp_up == actual_up):
                counter_trend_wins += 1
            else:
                counter_trend_losses += 1

        # Apply trend filter
        if trend_mode == 'with' and not trend_agrees:
            skipped += 1
            ws += window_ms
            continue
        elif trend_mode == 'counter' and trend_agrees:
            skipped += 1
            ws += window_ms
            continue

        # Signal generation
        if trend_mode == 'boost':
            # Scale displacement confidence by trend agreement
            trend_factor = min(abs(trend_return) / 0.1, 1.0)  # normalize trend strength
            if trend_agrees:
                effective_scale = SCALE * (1.0 + 0.5 * trend_factor)  # boost up to 1.5x
            else:
                effective_scale = SCALE * (1.0 - 0.3 * trend_factor)  # dampen up to 0.7x
            fair_up = sigmoid(effective_scale * disp)
        else:
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
        kelly = min(max((win_prob * payout - q) / payout, 0), KELLY_FRAC)

        if kelly < MIN_KELLY:
            skipped += 1
            ws += window_ms
            continue

        # Evaluate outcome
        right = (dir_yes == actual_up)
        if right:
            wins += 1
        else:
            losses += 1

        ws += window_ms

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0

    wt_total = with_trend_wins + with_trend_losses
    ct_total = counter_trend_wins + counter_trend_losses

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "skipped": skipped,
        "with_trend_wins": with_trend_wins,
        "with_trend_losses": with_trend_losses,
        "with_trend_wr": with_trend_wins / wt_total * 100 if wt_total > 0 else 0,
        "with_trend_total": wt_total,
        "counter_trend_wins": counter_trend_wins,
        "counter_trend_losses": counter_trend_losses,
        "counter_trend_wr": counter_trend_wins / ct_total * 100 if ct_total > 0 else 0,
        "counter_trend_total": ct_total,
    }


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

    # ═══════════════════════════════════════════════════════════
    # Section 1: With-trend vs counter-trend WR breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  TREND FILTER ANALYSIS — {days} DAYS — Does higher-TF trend improve 5-min displacement WR?")
    print(f"{'='*100}")

    for min_disp in [0.001, 0.005, 0.010, 0.020]:
        print(f"\n  ┌─────────────────────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Min displacement: {min_disp:.3f}%   Entry: T+1min                                       │")
        print(f"  └─────────────────────────────────────────────────────────────────────────────────────┘\n")

        print(f"  {'Trend Window':>14} │ {'ALL':>14} │ {'WITH trend':>14} │ {'COUNTER trend':>14} │ {'Delta':>6}")
        print(f"  {'─'*14} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*14} ┼ {'─'*6}")

        for trend_min in [5, 10, 15, 30, 60]:
            r = run_trend_backtest(candles, idx_by_time, first_ts, last_ts,
                                   min_disp, 1, trend_min, 'none')

            all_str = f"{r['wr']:.1f}% ({r['total']:>4})"
            wt_str = f"{r['with_trend_wr']:.1f}% ({r['with_trend_total']:>4})"
            ct_str = f"{r['counter_trend_wr']:.1f}% ({r['counter_trend_total']:>4})"
            delta = r['with_trend_wr'] - r['counter_trend_wr']
            delta_str = f"{delta:+.1f}%"

            print(f"  {trend_min:>11} min │ {all_str:>14} │ {wt_str:>14} │ {ct_str:>14} │ {delta_str:>6}")

    # ═══════════════════════════════════════════════════════════
    # Section 2: Filter impact on trade count and WR
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  FILTER IMPACT — What happens when we ONLY trade with-trend?")
    print(f"{'='*100}")

    for min_disp in [0.001, 0.010, 0.020]:
        print(f"\n  Min disp: {min_disp:.3f}%")
        print(f"  {'Mode':>20} │ {'Trend':>5} │ {'Trades':>6} {'WR%':>6} │ {'vs baseline':>11}")
        print(f"  {'─'*20} ┼ {'─'*5} ┼ {'─'*6} {'─'*6} ┼ {'─'*11}")

        baseline = run_trend_backtest(candles, idx_by_time, first_ts, last_ts,
                                      min_disp, 1, 15, 'none')

        for trend_min in [15, 30, 60]:
            for mode, label in [('with', 'with-trend'), ('counter', 'counter-trend'), ('boost', 'trend-boost')]:
                r = run_trend_backtest(candles, idx_by_time, first_ts, last_ts,
                                       min_disp, 1, trend_min, mode)
                delta = r['wr'] - baseline['wr']
                print(f"  {label:>20} │ {trend_min:>4}m │ {r['total']:>6} {r['wr']:>5.1f}% │ {delta:>+10.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Section 3: Compounding impact
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  COMPOUNDING IMPACT — $25 start, Kelly=12%, 14 days")
    print(f"{'='*100}\n")

    def compound_sim(candles, idx_by_time, first_ts, last_ts,
                     min_disp, trend_min, trend_mode):
        """Quick compound sim to compare end bankroll."""
        window_ms = 5 * 60 * 1000
        bankroll = 25.0
        peak = 25.0
        max_dd = 0.0
        wins = losses = 0

        ws = ((first_ts // window_ms) + 1) * window_ms
        while ws + window_ms <= last_ts:
            we = ws + window_ms
            start_t = (ws // 60000) * 60000
            end_t = ((we - 60000) // 60000) * 60000
            entry_t = start_t + 1 * 60000

            if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
                ws += window_ms
                continue

            si, ei, ni = idx_by_time[start_t], idx_by_time[end_t], idx_by_time[entry_t]
            btc_start = candles[si]["open"]
            btc_end = candles[ei]["close"]
            btc_entry = candles[ni]["close"]
            actual_up = btc_end > btc_start
            disp = (btc_entry - btc_start) / btc_start * 100

            if abs(disp) < min_disp:
                ws += window_ms
                continue

            # Trend filter
            trend_t = start_t - trend_min * 60000
            if trend_t not in idx_by_time:
                found = False
                for delta in range(0, 120001, 60000):
                    if trend_t - delta in idx_by_time:
                        trend_t = trend_t - delta
                        found = True
                        break
                    if trend_t + delta in idx_by_time:
                        trend_t = trend_t + delta
                        found = True
                        break
                if not found:
                    ws += window_ms
                    continue

            ti = idx_by_time[trend_t]
            trend_return = (btc_entry - candles[ti]["close"]) / candles[ti]["close"] * 100
            trend_agrees = (disp > 0) == (trend_return > 0)

            if trend_mode == 'with' and not trend_agrees:
                ws += window_ms
                continue
            elif trend_mode == 'counter' and trend_agrees:
                ws += window_ms
                continue

            fair_up = sigmoid(SCALE * disp)
            fair_up = max(0.01, min(0.99, fair_up))
            pm_mid = simulate_pm_price(disp, 0.3)
            spread = (fair_up - pm_mid) * 100

            if abs(spread) < MIN_EDGE_PCT:
                ws += window_ms
                continue

            dir_yes = spread > 0
            entry_price = pm_mid if dir_yes else 1.0 - pm_mid

            if entry_price < PRICE_FLOOR or entry_price > PRICE_CEIL:
                ws += window_ms
                continue

            win_prob = fair_up if dir_yes else 1.0 - fair_up
            payout = (1.0 / entry_price) - 1.0
            q = 1.0 - win_prob
            kelly = min(max((win_prob * payout - q) / payout, 0), KELLY_FRAC)

            if kelly < MIN_KELLY:
                ws += window_ms
                continue

            size = bankroll * KELLY_FRAC
            if size < 0.10:
                ws += window_ms
                continue

            right = (dir_yes == actual_up)
            if right:
                bankroll += size * payout
                wins += 1
            else:
                bankroll -= size
                losses += 1

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            if bankroll < 0.05:
                break

            ws += window_ms

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        return {"total": total, "wr": wr, "end": bankroll, "max_dd": max_dd * 100}

    def fmt(n):
        a = abs(n)
        s = "-" if n < 0 else ""
        if a >= 1e12: return f"{s}{a/1e12:.1f}T"
        if a >= 1e9: return f"{s}{a/1e9:.1f}B"
        if a >= 1e6: return f"{s}{a/1e6:.1f}M"
        if a >= 1e3: return f"{s}{a/1e3:.1f}K"
        return f"{s}{a:.2f}"

    print(f"  {'Config':>28} │ {'Trades':>6} {'WR%':>6} │ {'End Balance':>12} {'MaxDD':>6}")
    print(f"  {'─'*28} ┼ {'─'*6} {'─'*6} ┼ {'─'*12} {'─'*6}")

    for min_disp in [0.001, 0.010, 0.020]:
        # Baseline
        r = compound_sim(candles, idx_by_time, first_ts, last_ts, min_disp, 15, 'none')
        print(f"  {'d'+str(min_disp)+'% baseline':>28} │ {r['total']:>6} {r['wr']:>5.1f}% │ ${fmt(r['end']):>11} {r['max_dd']:>5.1f}%")

        # With 15-min trend
        r = compound_sim(candles, idx_by_time, first_ts, last_ts, min_disp, 15, 'with')
        print(f"  {'d'+str(min_disp)+'% +15m trend':>28} │ {r['total']:>6} {r['wr']:>5.1f}% │ ${fmt(r['end']):>11} {r['max_dd']:>5.1f}%")

        # With 30-min trend
        r = compound_sim(candles, idx_by_time, first_ts, last_ts, min_disp, 30, 'with')
        print(f"  {'d'+str(min_disp)+'% +30m trend':>28} │ {r['total']:>6} {r['wr']:>5.1f}% │ ${fmt(r['end']):>11} {r['max_dd']:>5.1f}%")

        # With 60-min trend
        r = compound_sim(candles, idx_by_time, first_ts, last_ts, min_disp, 60, 'with')
        print(f"  {'d'+str(min_disp)+'% +60m trend':>28} │ {r['total']:>6} {r['wr']:>5.1f}% │ ${fmt(r['end']):>11} {r['max_dd']:>5.1f}%")

        print()

    print(f"{'='*100}")


if __name__ == "__main__":
    run()
