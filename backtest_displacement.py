"""Backtest: Displacement vs Momentum model comparison.

Core insight: The 5-min Up/Down market settles based on whether BTC is
ABOVE or BELOW the window opening price. The current momentum model
uses recent returns (1m/3m/5m), which can disagree with displacement.

Example failure: BTC is $50 above window open (PM prices Up at 94¢),
but last 1min return is flat → momentum model says "no momentum → BUY_NO"
→ loses because BTC stays above open.

This backtest compares:
1. MOMENTUM (current model): recent returns → zscore → sigmoid → direction
2. DISPLACEMENT: (price_now - price_window_open) / price_window_open → direction
3. COMBINED: displacement as primary, momentum as confirmation filter
"""

import math
import time
import requests
import numpy as np
from collections import deque

# ── Parameters ──
W_1M = 0.40
W_3M = 0.25
W_5M = 0.15
MIN_EDGE_PCT = 2.0


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
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        current = data[-1][0] + 60_000
        time.sleep(0.08)
    print(f"  Got {len(all_candles)} candles")
    return all_candles


def get_return(candles, idx, lookback_minutes):
    if idx < lookback_minutes:
        return None
    past = candles[idx - lookback_minutes]["close"]
    now = candles[idx]["close"]
    return (now - past) / past if past > 0 else None


def compute_momentum_signal(candles, idx, return_history, sensitivity, time_factor):
    """Current model: momentum returns → zscore → sigmoid."""
    r1 = get_return(candles, idx, 1)
    r3 = get_return(candles, idx, 3)
    r5 = get_return(candles, idx, 5)
    if r1 is None:
        return None, None

    return_history.append(r1)
    vol = float(np.std(list(return_history))) if len(return_history) >= 10 else 0.001
    if vol < 1e-8:
        vol = 0.001

    raw = W_1M * r1 + W_3M * (r3 or 0) + W_5M * (r5 or 0)
    zscore = raw / vol
    adj = zscore * time_factor
    prob = 1.0 / (1.0 + math.exp(-sensitivity * adj))
    prob = max(0.01, min(0.99, prob))
    edge = abs(prob - 0.5) * 100
    if edge < MIN_EDGE_PCT:
        return None, prob
    direction = "UP" if prob > 0.5 else "DOWN"
    return direction, prob


def compute_displacement_signal(candles, entry_idx, window_open_price, min_disp_pct=0.01):
    """Proposed model: how far is BTC from window open price?"""
    current_price = candles[entry_idx]["close"]
    displacement_pct = (current_price - window_open_price) / window_open_price * 100

    if abs(displacement_pct) < min_disp_pct:
        return None, displacement_pct, current_price

    direction = "UP" if displacement_pct > 0 else "DOWN"
    return direction, displacement_pct, current_price


def run():
    candles = fetch_binance_klines(days=7)
    if not candles:
        return

    window_ms = 5 * 60 * 1000
    first_ts = candles[0]["time"]
    last_ts = candles[-1]["time"]

    idx_by_time = {}
    for i, c in enumerate(candles):
        t = (c["time"] // 60000) * 60000
        idx_by_time[t] = i

    entry_offsets = [0, 1, 2, 3, 4]  # minutes into window

    print("\n" + "=" * 80)
    print("  DISPLACEMENT vs MOMENTUM BACKTEST — 7 DAYS")
    print("=" * 80)

    # ── Strategy 1: MOMENTUM (current model) at various sensitivities ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  STRATEGY 1: MOMENTUM (current model)                  │")
    print("└─────────────────────────────────────────────────────────┘")

    for sensitivity in [5.0, 50.0]:
        print(f"\n  Sensitivity = {sensitivity}")
        print(f"  {'Entry':>8} │ {'Trades':>6} │ {'Win%':>6} │ {'Wins':>5} │ {'Losses':>6} │ {'NoSig':>5}")
        print(f"  {'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}")

        for offset in entry_offsets:
            # Time factor from original model
            if offset == 0:
                tf = 0.8
            elif offset == 1:
                tf = 0.75
            elif offset == 2:
                tf = 0.65
            else:
                tf = 0.5

            wins, losses, no_sig = 0, 0, 0
            return_history = deque(maxlen=500)

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
                actual = "UP" if btc_end > btc_start else "DOWN"

                direction, prob = compute_momentum_signal(
                    candles, entry_idx, return_history, sensitivity, tf
                )

                if direction is None:
                    no_sig += 1
                elif direction == actual:
                    wins += 1
                else:
                    losses += 1

                ws += window_ms

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            print(f"  T+{offset}min  │ {total:>6} │ {wr:>5.1f}% │ {wins:>5} │ {losses:>6} │ {no_sig:>5}")

    # ── Strategy 2: PURE DISPLACEMENT ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  STRATEGY 2: PURE DISPLACEMENT (price vs window open)  │")
    print("└─────────────────────────────────────────────────────────┘")

    for min_disp in [0.0, 0.01, 0.02, 0.05, 0.10]:
        print(f"\n  Min displacement = {min_disp}%")
        print(f"  {'Entry':>8} │ {'Trades':>6} │ {'Win%':>6} │ {'Wins':>5} │ {'Losses':>6} │ {'NoSig':>5}")
        print(f"  {'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}")

        for offset in entry_offsets:
            wins, losses, no_sig = 0, 0, 0

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
                actual = "UP" if btc_end > btc_start else "DOWN"

                direction, disp_pct, _ = compute_displacement_signal(
                    candles, entry_idx, btc_start, min_disp_pct=min_disp
                )

                if direction is None:
                    no_sig += 1
                elif direction == actual:
                    wins += 1
                else:
                    losses += 1

                ws += window_ms

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            print(f"  T+{offset}min  │ {total:>6} │ {wr:>5.1f}% │ {wins:>5} │ {losses:>6} │ {no_sig:>5}")

    # ── Strategy 3: DISPLACEMENT + MOMENTUM AGREEMENT ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  STRATEGY 3: COMBINED (displacement + momentum agree)  │")
    print("└─────────────────────────────────────────────────────────┘")

    for sensitivity in [5.0, 50.0]:
        for min_disp in [0.0, 0.02, 0.05]:
            print(f"\n  Sensitivity={sensitivity}, Min disp={min_disp}%")
            print(f"  {'Entry':>8} │ {'Trades':>6} │ {'Win%':>6} │ {'Wins':>5} │ {'Losses':>6} │ {'NoSig':>5} │ {'Disagree':>8}")
            print(f"  {'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*8}")

            for offset in entry_offsets:
                if offset == 0:
                    tf = 0.8
                elif offset == 1:
                    tf = 0.75
                elif offset == 2:
                    tf = 0.65
                else:
                    tf = 0.5

                wins, losses, no_sig, disagree = 0, 0, 0, 0
                return_history = deque(maxlen=500)

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
                    actual = "UP" if btc_end > btc_start else "DOWN"

                    # Displacement signal
                    disp_dir, disp_pct, _ = compute_displacement_signal(
                        candles, entry_idx, btc_start, min_disp_pct=min_disp
                    )

                    # Momentum signal
                    mom_dir, prob = compute_momentum_signal(
                        candles, entry_idx, return_history, sensitivity, tf
                    )

                    if disp_dir is None:
                        no_sig += 1
                    elif mom_dir is not None and mom_dir != disp_dir:
                        disagree += 1
                        no_sig += 1  # skip disagreements
                    elif disp_dir is not None:
                        # Use displacement direction (momentum either agrees or has no signal)
                        if disp_dir == actual:
                            wins += 1
                        else:
                            losses += 1
                    else:
                        no_sig += 1

                    ws += window_ms

                total = wins + losses
                wr = wins / total * 100 if total > 0 else 0
                print(f"  T+{offset}min  │ {total:>6} │ {wr:>5.1f}% │ {wins:>5} │ {losses:>6} │ {no_sig:>5} │ {disagree:>8}")

    # ── Strategy 4: DISPLACEMENT-ONLY with P&L estimate ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  STRATEGY 4: DISPLACEMENT with realistic P&L           │")
    print("│  (Estimates PM price from displacement magnitude)       │")
    print("└─────────────────────────────────────────────────────────┘")

    print(f"\n  Assumes: PM prices token at ~50¢ + (displacement × efficiency)")
    print(f"  We BUY the token our model favors at estimated PM price")
    print(f"  Win → payout $1/share. Lose → payout $0/share.")

    for pm_efficiency in [0.3, 0.5, 0.7]:
        for min_disp in [0.0, 0.02, 0.05, 0.10]:
            wins, losses, no_sig = 0, 0, 0
            total_pnl = 0.0
            offset = 1  # T+1min — balance of early entry + some displacement info

            return_history = deque(maxlen=500)

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
                actual = "UP" if btc_end > btc_start else "DOWN"

                direction, disp_pct, _ = compute_displacement_signal(
                    candles, entry_idx, btc_start, min_disp_pct=min_disp
                )

                if direction is None:
                    no_sig += 1
                    ws += window_ms
                    continue

                # Estimate PM token price based on displacement
                # If BTC is up 0.1%, PM Up token ~ 50¢ + (0.1% × efficiency × 100)
                pm_token_price = 0.50 + abs(disp_pct) * pm_efficiency
                pm_token_price = max(0.05, min(0.95, pm_token_price))

                # We always buy the token matching our direction
                cost = pm_token_price
                if direction == actual:
                    pnl = 1.0 - cost  # won: paid cost, received $1
                    wins += 1
                else:
                    pnl = -cost  # lost: paid cost, received $0
                    losses += 1
                total_pnl += pnl

                ws += window_ms

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            ev = total_pnl / total if total > 0 else 0
            if total > 0:
                print(f"  eff={pm_efficiency} disp>={min_disp:>4.2f}% │ "
                      f"Trades={total:>5} WR={wr:>5.1f}% "
                      f"P&L=${total_pnl:>+8.1f} EV/trade=${ev:>+.3f}")

    # ── Head-to-head: cases where momentum and displacement DISAGREE ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  HEAD-TO-HEAD: When momentum and displacement disagree │")
    print("│  Who is right more often?                              │")
    print("└─────────────────────────────────────────────────────────┘")

    for sensitivity in [50.0]:
        for offset in [0, 1, 2]:
            if offset == 0:
                tf = 0.8
            elif offset == 1:
                tf = 0.75
            else:
                tf = 0.65

            mom_right, disp_right, both_wrong = 0, 0, 0
            agree_right, agree_wrong = 0, 0
            return_history = deque(maxlen=500)

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
                actual = "UP" if btc_end > btc_start else "DOWN"

                disp_dir, disp_pct, _ = compute_displacement_signal(
                    candles, entry_idx, btc_start, min_disp_pct=0.0
                )
                mom_dir, prob = compute_momentum_signal(
                    candles, entry_idx, return_history, sensitivity, tf
                )

                if mom_dir and disp_dir:
                    if mom_dir != disp_dir:
                        # DISAGREEMENT — who's right?
                        if disp_dir == actual and mom_dir != actual:
                            disp_right += 1
                        elif mom_dir == actual and disp_dir != actual:
                            mom_right += 1
                        else:
                            both_wrong += 1
                    else:
                        # AGREEMENT
                        if mom_dir == actual:
                            agree_right += 1
                        else:
                            agree_wrong += 1

                ws += window_ms

            disagree_total = mom_right + disp_right + both_wrong
            agree_total = agree_right + agree_wrong
            print(f"\n  T+{offset}min (sensitivity={sensitivity}):")
            print(f"    Agreement: {agree_total} trades, "
                  f"{agree_right}/{agree_total} correct ({agree_right/agree_total*100:.1f}%)" if agree_total > 0 else "    Agreement: 0 trades")
            if disagree_total > 0:
                print(f"    Disagreement: {disagree_total} trades")
                print(f"      Displacement right: {disp_right} ({disp_right/disagree_total*100:.1f}%)")
                print(f"      Momentum right:     {mom_right} ({mom_right/disagree_total*100:.1f}%)")
                print(f"      Both wrong:         {both_wrong} ({both_wrong/disagree_total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    run()
