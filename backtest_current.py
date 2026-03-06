"""Backtest: Current live configuration — exact match to .env + settings.py.

Config:
  Model: Displacement (scale=10)
  Entry: T+0 (from window start)
  Filters: velocity confirm ON, vol normalization ON (z>=1.0), OBI OFF
  Min displacement: 0.02%
  Kelly: 12% cap, uses fair_up_prob directly (no Bayesian dampening)
  Min confidence: 0.55 (but removed from entry gate — Kelly gatekeeps)
  Price quality: 0.05 - 0.80
  PM efficiency sweep: 0.0 to 1.0
"""

import math
import time
import requests
import sys


# ── Current live config ──
SCALE = 10.0
MIN_DISPLACEMENT = 0.02  # percent
KELLY_CAP = 0.12
BANKROLL = 80.0
MIN_EDGE_PCT = 2.0
PRICE_FLOOR = 0.05
PRICE_CEIL = 0.80
VELOCITY_LOOKBACK_S = 15  # candles (using 1-min data, so 15 candles)
REQUIRE_VELOCITY = True
REQUIRE_VOL_NORM = True
MIN_Z_DISP = 1.0


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
        time.sleep(0.05)
    print(f"  Got {len(all_candles)} candles ({days}d)")
    return all_candles


def compute_velocity(candles, idx, lookback=15):
    """Price change over last `lookback` candles, in percent."""
    if idx < lookback:
        ref = candles[0]["close"]
    else:
        ref = candles[idx - lookback]["close"]
    if ref <= 0:
        return 0.0
    return (candles[idx]["close"] - ref) / ref * 100


def compute_rolling_stdev(candles, idx, window=60):
    """Rolling stdev of 1-candle returns over last `window` candles."""
    start = max(0, idx - window)
    prices = [candles[i]["close"] for i in range(start, idx + 1) if candles[i]["close"] > 0]
    if len(prices) < 10:
        return 0.01
    returns = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i-1]) / prices[i-1] * 100
        returns.append(r)
    if not returns:
        return 0.01
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return max(var ** 0.5, 1e-8)


def sim_pm_price(displacement_pct, pm_efficiency):
    """Simulate PM token price given displacement and PM efficiency.

    pm_efficiency=0.0: PM stays at 50¢ (maximum edge)
    pm_efficiency=1.0: PM fully reflects displacement (zero edge)
    """
    fair = 1.0 / (1.0 + math.exp(-SCALE * displacement_pct))
    pm_mid = 0.50 + pm_efficiency * (fair - 0.50)
    return max(0.01, min(0.99, pm_mid))


def run():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
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

    entry_offset = 0  # T+0

    print(f"\n{'='*90}")
    print(f"  BACKTEST: Current Live Config — Displacement Model T+0")
    print(f"  Scale={SCALE}, min_disp={MIN_DISPLACEMENT}%, velocity={'ON' if REQUIRE_VELOCITY else 'OFF'}, "
          f"vol_norm={'ON' if REQUIRE_VOL_NORM else 'OFF'} (z>={MIN_Z_DISP})")
    print(f"  Kelly={KELLY_CAP*100:.0f}%, bankroll=${BANKROLL}, entry=T+{entry_offset}min")
    print(f"{'='*90}")

    # Sweep PM efficiency levels
    print(f"\n  {'PM_eff':>6} │ {'Trades':>6} {'WR':>6} {'W/L':>10} {'PnL':>10} {'EV/trade':>9} │ {'UP':>4}/{'DN':>4} │ {'NoSig':>5} {'Filt':>5} {'Kelly':>5}")
    print(f"  {'─'*6} │ {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*9} │ {'─'*4}/{'─'*4} │ {'─'*5} {'─'*5} {'─'*5}")

    for pm_eff in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        wins, losses = 0, 0
        total_pnl = 0.0
        up_trades, down_trades = 0, 0
        no_sig, filtered, kelly_rej = 0, 0, 0

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

            # Displacement at entry
            entry_price_btc = candles[entry_idx]["close"]
            displacement_pct = (entry_price_btc - btc_start) / btc_start * 100

            # Filter 1: min displacement
            if abs(displacement_pct) < MIN_DISPLACEMENT:
                no_sig += 1
                ws += window_ms
                continue

            # Filter 2: velocity confirmation
            if REQUIRE_VELOCITY:
                vel = compute_velocity(candles, entry_idx, lookback=15)
                if displacement_pct > 0 and vel < 0:
                    filtered += 1
                    ws += window_ms
                    continue
                if displacement_pct < 0 and vel > 0:
                    filtered += 1
                    ws += window_ms
                    continue

            # Filter 3: vol normalization
            if REQUIRE_VOL_NORM:
                stdev = compute_rolling_stdev(candles, entry_idx)
                z = displacement_pct / stdev if stdev > 1e-8 else 0.0
                if abs(z) < MIN_Z_DISP:
                    filtered += 1
                    ws += window_ms
                    continue

            # Model: sigmoid
            fair_up = 1.0 / (1.0 + math.exp(-SCALE * displacement_pct))
            fair_up = max(0.01, min(0.99, fair_up))

            direction = "UP" if displacement_pct > 0 else "DOWN"
            win_prob = fair_up if direction == "UP" else 1.0 - fair_up

            # Simulate PM price
            pm_mid = sim_pm_price(displacement_pct, pm_eff)
            if direction == "UP":
                entry_price = pm_mid  # buying YES token
            else:
                entry_price = 1.0 - pm_mid  # buying NO token

            # Price quality filter
            if entry_price < PRICE_FLOOR or entry_price > PRICE_CEIL:
                kelly_rej += 1
                ws += window_ms
                continue

            # Kelly
            if entry_price > 0 and entry_price < 1:
                payout = (1.0 / entry_price) - 1.0
            else:
                payout = 1.0
            q = 1.0 - win_prob
            kelly_raw = (win_prob * payout - q) / payout
            kelly_raw = max(0.0, kelly_raw)
            kelly_frac = min(kelly_raw, KELLY_CAP)

            if kelly_frac < 0.005:
                kelly_rej += 1
                ws += window_ms
                continue

            # Size: fixed 12% of bankroll
            size = max(BANKROLL * KELLY_CAP, 0.10)

            if direction == "UP":
                up_trades += 1
            else:
                down_trades += 1

            won = direction == actual
            if won:
                wins += 1
                total_pnl += size * payout
            else:
                losses += 1
                total_pnl -= size

            ws += window_ms

        total = wins + losses
        wr = wins / total * 100 if total else 0
        ev = total_pnl / total if total else 0
        print(f"  {pm_eff:>6.2f} │ {total:>6} {wr:>5.1f}% {f'{wins}/{losses}':>10} "
              f"${total_pnl:>+9.1f} ${ev:>+8.3f} │ {up_trades:>4}/{down_trades:>4} │ "
              f"{no_sig:>5} {filtered:>5} {kelly_rej:>5}")

    # Also test T+1 for comparison
    print(f"\n{'='*90}")
    print(f"  COMPARISON: T+0 vs T+1 vs T+2 at pm_eff=0.5 and pm_eff=0.9")
    print(f"{'='*90}")

    for pm_eff in [0.5, 0.9]:
        print(f"\n  PM efficiency = {pm_eff}")
        print(f"  {'Entry':>6} │ {'Trades':>6} {'WR':>6} {'PnL':>10} {'EV/trade':>9} │ {'UP/DN':>10}")
        print(f"  {'─'*6} │ {'─'*6} {'─'*6} {'─'*10} {'─'*9} │ {'─'*10}")

        for t_offset in [0, 1, 2, 3]:
            wins, losses = 0, 0
            total_pnl = 0.0
            up_t, down_t = 0, 0

            ws = ((first_ts // window_ms) + 1) * window_ms
            while ws + window_ms <= last_ts:
                we = ws + window_ms
                start_t = (ws // 60000) * 60000
                end_t = ((we - 60000) // 60000) * 60000
                entry_t = start_t + t_offset * 60000

                if start_t not in idx_by_time or end_t not in idx_by_time or entry_t not in idx_by_time:
                    ws += window_ms
                    continue

                start_idx = idx_by_time[start_t]
                end_idx = idx_by_time[end_t]
                entry_idx = idx_by_time[entry_t]

                btc_start = candles[start_idx]["open"]
                btc_end = candles[end_idx]["close"]
                actual = "UP" if btc_end > btc_start else "DOWN"

                entry_price_btc = candles[entry_idx]["close"]
                displacement_pct = (entry_price_btc - btc_start) / btc_start * 100

                if abs(displacement_pct) < MIN_DISPLACEMENT:
                    ws += window_ms
                    continue

                # Velocity filter
                if REQUIRE_VELOCITY:
                    vel = compute_velocity(candles, entry_idx, lookback=15)
                    if (displacement_pct > 0 and vel < 0) or (displacement_pct < 0 and vel > 0):
                        ws += window_ms
                        continue

                # Vol norm filter
                if REQUIRE_VOL_NORM:
                    stdev = compute_rolling_stdev(candles, entry_idx)
                    z = displacement_pct / stdev if stdev > 1e-8 else 0.0
                    if abs(z) < MIN_Z_DISP:
                        ws += window_ms
                        continue

                fair_up = 1.0 / (1.0 + math.exp(-SCALE * displacement_pct))
                fair_up = max(0.01, min(0.99, fair_up))
                direction = "UP" if displacement_pct > 0 else "DOWN"
                win_prob = fair_up if direction == "UP" else 1.0 - fair_up

                pm_mid = sim_pm_price(displacement_pct, pm_eff)
                entry_price = pm_mid if direction == "UP" else 1.0 - pm_mid

                if entry_price < PRICE_FLOOR or entry_price > PRICE_CEIL:
                    ws += window_ms
                    continue

                payout = (1.0 / entry_price) - 1.0 if 0 < entry_price < 1 else 1.0
                q = 1.0 - win_prob
                kelly_raw = max(0.0, (win_prob * payout - q) / payout)
                if min(kelly_raw, KELLY_CAP) < 0.005:
                    ws += window_ms
                    continue

                size = max(BANKROLL * KELLY_CAP, 0.10)

                if direction == "UP":
                    up_t += 1
                else:
                    down_t += 1

                if direction == actual:
                    wins += 1
                    total_pnl += size * payout
                else:
                    losses += 1
                    total_pnl -= size

                ws += window_ms

            total = wins + losses
            wr = wins / total * 100 if total else 0
            ev = total_pnl / total if total else 0
            print(f"  T+{t_offset:>3} │ {total:>6} {wr:>5.1f}% ${total_pnl:>+9.1f} ${ev:>+8.3f} │ {up_t:>4}/{down_t:>4}")

    print(f"\n{'='*90}")
    print(f"  KEY: pm_eff=0.0 means PM stays at 50¢ (best case)")
    print(f"       pm_eff=1.0 means PM fully reflects displacement (worst case)")
    print(f"       Live PM is likely 0.85-0.95 efficient (21ms lag)")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    run()
