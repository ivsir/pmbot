"""Backtest: OLD Momentum Model vs NEW Displacement Model — side-by-side.

OLD (momentum):  weighted 1m/3m/5m returns → z-score → sigmoid(50) → fair_up_prob
                 Bayesian dampening, MIN_CONFIDENCE=0.60, variable Kelly

NEW (displacement): (price - window_open) / window_open → sigmoid(10) → fair_up_prob
                    Direct fair_up_prob, MIN_CONFIDENCE=0.55, fixed 11.5% Kelly
"""

import math
import time
import requests
import numpy as np
from collections import deque

# ── Shared ──
KELLY_CAP = 0.115
MIN_EDGE_PCT = 2.0
BANKROLL = 80.0

# ── OLD config (momentum) ──
OLD_SENSITIVITY = 50.0
OLD_MIN_CONFIDENCE = 0.60
OLD_W_1M = 0.40
OLD_W_3M = 0.25
OLD_W_5M = 0.15
OLD_W_OBI = 0.20  # simulated as 0 (no PM orderbook in backtest)

# ── NEW config (displacement) ──
NEW_SCALE = 10.0
NEW_MIN_CONFIDENCE = 0.55
NEW_MIN_DISPLACEMENT = 0.02  # percent


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


# ═══════════════════════════════════════════
#  OLD MODEL: Momentum (weighted returns → z-score → sigmoid(50))
# ═══════════════════════════════════════════

def momentum_signal(candles, entry_idx, return_history):
    """Simulate the old momentum model exactly as it was in production.

    Returns: (direction, fair_up_prob, win_prob_dampened, displacement_pct) or Nones
    """
    if entry_idx < 5:
        return None, None, None, None

    # Calculate returns exactly like the old code
    ret_1m = _calc_return(candles, entry_idx, 1)
    ret_3m = _calc_return(candles, entry_idx, 3)
    ret_5m = _calc_return(candles, entry_idx, 5)

    if ret_1m is None:
        return None, None, None, None

    return_history.append(ret_1m)

    # Rolling volatility (std of recent 1m returns)
    if len(return_history) < 5:
        vol = 0.001
    else:
        vol = float(np.std(list(return_history)))
    if vol < 1e-8:
        vol = 0.001

    # Weighted raw momentum (OBI=0 since we don't have PM orderbook)
    raw_momentum = (
        OLD_W_1M * (ret_1m or 0.0)
        + OLD_W_3M * (ret_3m or 0.0)
        + OLD_W_5M * (ret_5m or 0.0)
    )
    zscore = raw_momentum / vol

    # Time factor (assume ~60s into window = 0.75)
    time_factor = 0.75
    adjusted_zscore = zscore * time_factor

    # Sigmoid with sensitivity=50
    fair_up_prob = 1.0 / (1.0 + math.exp(-OLD_SENSITIVITY * adjusted_zscore))
    fair_up_prob = max(0.01, min(0.99, fair_up_prob))

    # Direction
    direction = "UP" if fair_up_prob > 0.50 else "DOWN"

    # Bayesian dampening: combined_prob ≈ 0.50 + (raw - 0.50) * 0.10
    dampened = 0.50 + (fair_up_prob - 0.50) * 0.10
    win_prob = dampened if direction == "UP" else 1.0 - dampened

    return direction, fair_up_prob, win_prob, adjusted_zscore


def _calc_return(candles, idx, lookback_min):
    if idx < lookback_min:
        return None
    old_price = candles[idx - lookback_min]["close"]
    new_price = candles[idx]["close"]
    if old_price == 0:
        return None
    return (new_price - old_price) / old_price


# ═══════════════════════════════════════════
#  NEW MODEL: Displacement (price vs window open → sigmoid(10))
# ═══════════════════════════════════════════

def displacement_signal(candles, entry_idx, window_open_price):
    """Simulate the new displacement model exactly as deployed.

    Returns: (direction, fair_up_prob, win_prob, displacement_pct) or Nones
    """
    current_price = candles[entry_idx]["close"]
    displacement_pct = (current_price - window_open_price) / window_open_price * 100

    if abs(displacement_pct) < NEW_MIN_DISPLACEMENT:
        return None, None, None, None

    # Sigmoid with scale=10
    fair_up_prob = 1.0 / (1.0 + math.exp(-NEW_SCALE * displacement_pct))
    fair_up_prob = max(0.01, min(0.99, fair_up_prob))

    direction = "UP" if displacement_pct > 0 else "DOWN"

    # NEW config: win_prob = fair_up_prob directly (no dampening)
    win_prob = fair_up_prob if direction == "UP" else 1.0 - fair_up_prob

    return direction, fair_up_prob, win_prob, displacement_pct


# ═══════════════════════════════════════════
#  KELLY + SIZING
# ═══════════════════════════════════════════

def kelly_gate(win_prob, entry_price, min_confidence):
    if entry_price <= 0 or entry_price >= 1:
        return False, 0.0
    payout_ratio = (1.0 / entry_price) - 1.0
    q = 1.0 - win_prob
    kelly_raw = (win_prob * payout_ratio - q) / payout_ratio
    kelly_raw = max(0.0, kelly_raw)
    kelly_fraction = min(kelly_raw, KELLY_CAP)
    passes = kelly_fraction > 0.005 and win_prob >= min_confidence
    return passes, kelly_fraction


def sim_entry_price(displacement_pct, pm_efficiency):
    """Simulate PM token price based on displacement magnitude."""
    pm_token_price = 0.50 + abs(displacement_pct) * pm_efficiency
    return max(0.05, min(0.95, pm_token_price))


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

    entry_offset = 1  # T+1min

    print("\n" + "=" * 80)
    print("  OLD MOMENTUM MODEL vs NEW DISPLACEMENT MODEL — 7 DAY BACKTEST")
    print("=" * 80)

    pm_eff = 0.5  # default PM efficiency

    # ════════════════════════════════════════
    # Run BOTH models across all PM efficiency levels
    # ════════════════════════════════════════

    for label, model_fn_name in [("OLD (momentum s=50)", "momentum"), ("NEW (displacement s=10)", "displacement")]:
        print(f"\n┌{'─'*66}┐")
        print(f"│  {label:^64}│")
        print(f"└{'─'*66}┘")

        for pm_eff in [0.3, 0.5, 0.7]:
            wins, losses, no_sig, kelly_reject = 0, 0, 0, 0
            total_pnl = 0.0
            sizes = []
            up_trades, down_trades = 0, 0
            return_history = deque(maxlen=500)

            ws = ((first_ts // window_ms) + 1) * window_ms
            while ws + window_ms <= last_ts:
                we = ws + window_ms
                start_t = (ws // 60000) * 60000
                end_t = ((we - 60000) // 60000) * 60000
                entry_t = start_t + entry_offset * 60000

                if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
                    ws += window_ms
                    continue

                start_idx = idx_by_time[start_t]
                end_idx = idx_by_time[end_t]
                entry_idx = idx_by_time[entry_t]

                btc_start = candles[start_idx]["open"]
                btc_end = candles[end_idx]["close"]
                actual = "UP" if btc_end > btc_start else "DOWN"

                # Run the appropriate model
                if model_fn_name == "momentum":
                    direction, fair_up, win_prob, extra = momentum_signal(
                        candles, entry_idx, return_history
                    )
                    min_conf = OLD_MIN_CONFIDENCE
                    fixed_kelly = False
                else:
                    direction, fair_up, win_prob, extra = displacement_signal(
                        candles, entry_idx, btc_start
                    )
                    min_conf = NEW_MIN_CONFIDENCE
                    fixed_kelly = True

                if direction is None:
                    no_sig += 1
                    ws += window_ms
                    continue

                # Estimate PM entry price
                disp_pct = abs(extra) if model_fn_name == "displacement" else 0.0
                if model_fn_name == "momentum":
                    # For momentum, estimate displacement for PM pricing
                    disp_pct = abs((candles[entry_idx]["close"] - btc_start) / btc_start * 100)
                entry_price = sim_entry_price(disp_pct, pm_eff)

                # Price quality filter (20-80%)
                if entry_price < 0.20 or entry_price > 0.80:
                    kelly_reject += 1
                    ws += window_ms
                    continue

                ok, kelly = kelly_gate(win_prob, entry_price, min_conf)
                if not ok:
                    kelly_reject += 1
                    ws += window_ms
                    continue

                # Sizing
                if fixed_kelly:
                    size = max(BANKROLL * KELLY_CAP, 1.0)
                else:
                    size = max(BANKROLL * kelly, 1.0)
                sizes.append(size)

                if direction == "UP":
                    up_trades += 1
                else:
                    down_trades += 1

                won = direction == actual
                if won:
                    wins += 1
                    total_pnl += size * ((1.0 / entry_price) - 1.0)
                else:
                    losses += 1
                    total_pnl -= size

                ws += window_ms

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            avg_sz = sum(sizes) / len(sizes) if sizes else 0
            print(f"  pm_eff={pm_eff} │ Trades={total:>5} WR={wr:>5.1f}% "
                  f"P&L=${total_pnl:>+9.1f} avg_sz=${avg_sz:>.2f} "
                  f"│ UP={up_trades} DOWN={down_trades} "
                  f"kelly_rej={kelly_reject} no_sig={no_sig}")

    # ════════════════════════════════════════
    # SIDE-BY-SIDE SUMMARY (pm_eff=0.5)
    # ════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  SUMMARY: OLD MOMENTUM vs NEW DISPLACEMENT (pm_eff=0.5, T+1min)")
    print("=" * 80)

    pm_eff = 0.5
    results = {}

    for label, model_fn_name, min_conf, fixed_k in [
        ("OLD", "momentum", OLD_MIN_CONFIDENCE, False),
        ("NEW", "displacement", NEW_MIN_CONFIDENCE, True),
    ]:
        w, l, pnl, kr, ns = 0, 0, 0.0, 0, 0
        sizes = []
        up_t, down_t = 0, 0
        fair_ups = []
        return_history = deque(maxlen=500)

        ws = ((first_ts // window_ms) + 1) * window_ms
        while ws + window_ms <= last_ts:
            we = ws + window_ms
            start_t = (ws // 60000) * 60000
            end_t = ((we - 60000) // 60000) * 60000
            entry_t = start_t + entry_offset * 60000

            if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
                ws += window_ms
                continue

            start_idx = idx_by_time[start_t]
            end_idx = idx_by_time[end_t]
            entry_idx = idx_by_time[entry_t]

            btc_start = candles[start_idx]["open"]
            btc_end = candles[end_idx]["close"]
            actual = "UP" if btc_end > btc_start else "DOWN"

            if model_fn_name == "momentum":
                direction, fair_up, win_prob, extra = momentum_signal(
                    candles, entry_idx, return_history
                )
            else:
                direction, fair_up, win_prob, extra = displacement_signal(
                    candles, entry_idx, btc_start
                )

            if direction is None:
                ns += 1
                ws += window_ms
                continue

            fair_ups.append(fair_up)

            disp_pct = abs(extra) if model_fn_name == "displacement" else 0.0
            if model_fn_name == "momentum":
                disp_pct = abs((candles[entry_idx]["close"] - btc_start) / btc_start * 100)
            entry_price = sim_entry_price(disp_pct, pm_eff)

            if entry_price < 0.20 or entry_price > 0.80:
                kr += 1
                ws += window_ms
                continue

            ok, kelly = kelly_gate(win_prob, entry_price, min_conf)
            if not ok:
                kr += 1
                ws += window_ms
                continue

            size = max(BANKROLL * KELLY_CAP, 1.0) if fixed_k else max(BANKROLL * kelly, 1.0)
            sizes.append(size)

            if direction == "UP":
                up_t += 1
            else:
                down_t += 1

            if direction == actual:
                w += 1
                pnl += size * ((1.0 / entry_price) - 1.0)
            else:
                l += 1
                pnl -= size

            ws += window_ms

        total = w + l
        wr = w / total * 100 if total else 0
        ev = pnl / total if total else 0
        avg_sz = sum(sizes) / len(sizes) if sizes else 0
        avg_fair = sum(fair_ups) / len(fair_ups) if fair_ups else 0.5
        results[label] = {
            "w": w, "l": l, "total": total, "wr": wr, "pnl": pnl,
            "ev": ev, "kr": kr, "ns": ns, "avg_sz": avg_sz,
            "up": up_t, "down": down_t, "avg_fair_up": avg_fair,
        }

    o = results["OLD"]
    n = results["NEW"]

    print(f"\n  {'':>24} {'OLD (momentum)':>18} {'NEW (displace)':>18}")
    print(f"  {'─'*24} {'─'*18} {'─'*18}")
    print(f"  {'Model':>24} {'ret→zscore→sig50':>18} {'disp→sig10':>18}")
    print(f"  {'Win Prob Source':>24} {'Bayesian dampen':>18} {'fair_up direct':>18}")
    print(f"  {'MIN_CONFIDENCE':>24} {OLD_MIN_CONFIDENCE:>18.2f} {NEW_MIN_CONFIDENCE:>18.2f}")
    print(f"  {'Kelly Mode':>24} {'variable':>18} {'fixed 11.5%':>18}")
    print(f"  {'Avg fair_up_prob':>24} {o['avg_fair_up']:>18.4f} {n['avg_fair_up']:>18.4f}")
    print(f"  {'─'*24} {'─'*18} {'─'*18}")
    print(f"  {'Trades':>24} {o['total']:>18} {n['total']:>18}")
    print(f"  {'Win Rate':>24} {o['wr']:>17.1f}% {n['wr']:>17.1f}%")
    print(f"  {'Wins / Losses':>24} {str(o['w'])+'/'+str(o['l']):>18} {str(n['w'])+'/'+str(n['l']):>18}")
    print(f"  {'UP / DOWN trades':>24} {str(o['up'])+'/'+str(o['down']):>18} {str(n['up'])+'/'+str(n['down']):>18}")
    o_pnl, n_pnl = o['pnl'], n['pnl']
    o_ev, n_ev = o['ev'], n['ev']
    o_sz, n_sz = o['avg_sz'], n['avg_sz']
    print(f"  {'P&L':>24} {'$' + f'{o_pnl:+.1f}':>17} {'$' + f'{n_pnl:+.1f}':>17}")
    print(f"  {'EV per trade':>24} {'$' + f'{o_ev:+.3f}':>17} {'$' + f'{n_ev:+.3f}':>17}")
    print(f"  {'Avg Trade Size':>24} {'$' + f'{o_sz:.2f}':>17} {'$' + f'{n_sz:.2f}':>17}")
    print(f"  {'Kelly Rejected':>24} {o['kr']:>18} {n['kr']:>18}")
    print(f"  {'No Signal':>24} {o['ns']:>18} {n['ns']:>18}")
    print(f"  {'─'*24} {'─'*18} {'─'*18}")

    if o['total'] > 0 and n['total'] > 0:
        delta_wr = n['wr'] - o['wr']
        delta_trades = n['total'] - o['total']
        delta_pnl = n_pnl - o_pnl
        print(f"  {'WR Delta':>24} {'':>18} {'+' if delta_wr > 0 else ''}{delta_wr:.1f}%")
        print(f"  {'Trade Count Delta':>24} {'':>18} {'+' if delta_trades > 0 else ''}{delta_trades}")
        print(f"  {'P&L Delta':>24} {'':>18} {'$' + f'{delta_pnl:+.1f}'}")
    elif o['total'] == 0:
        print(f"  OLD produced 0 trades (Bayesian dampening + high MIN_CONF killed all signals)")
        print(f"  NEW: {n['total']} trades, {n['wr']:.1f}% WR, ${n_pnl:+.1f} P&L")

    print()


if __name__ == "__main__":
    run()
