"""Backtest: Compounding simulation with Kelly sizing on real bankroll.

Simulates actual trading with:
  - Starting bankroll (e.g. $10, $25, $50, $100)
  - Kelly fraction of LIVE wallet balance each trade (11.5% or 12%)
  - Binary PnL: win → size * (1/entry_price - 1), loss → -size
  - Tracks bankroll over time, drawdowns, and final balance
"""

import math
import time
import requests


KELLY_FRAC = 0.12  # from .env
MIN_TRADE_SIZE = 0.10  # minimum $0.10
MAX_BID_FRAC = 0.12  # max bid as fraction of bankroll (= kelly_frac)
SCALE = 10.0
MIN_EDGE_PCT = 0.02
PRICE_FLOOR = 0.05
PRICE_CEIL = 0.80
MIN_KELLY = 0.005


def fmt(n: float, sign=False) -> str:
    """Compact dollar formatting for large numbers."""
    prefix = "+" if sign and n > 0 else ""
    a = abs(n)
    s = "-" if n < 0 else prefix
    if a >= 1e12:
        return f"{s}{a/1e12:.1f}T"
    if a >= 1e9:
        return f"{s}{a/1e9:.1f}B"
    if a >= 1e6:
        return f"{s}{a/1e6:.1f}M"
    if a >= 1e3:
        return f"{s}{a/1e3:.1f}K"
    return f"{s}{a:.2f}"


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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def simulate_pm_price(disp, eff=0.3):
    return sigmoid(eff * disp)


def run_compound_sim(candles, idx_by_time, first_ts, last_ts, days,
                     start_bankroll, min_disp, offset, kelly_frac):
    """Run a single compounding simulation. Returns detailed results."""
    window_ms = 5 * 60 * 1000
    bankroll = start_bankroll
    peak = start_bankroll
    max_dd = 0.0
    wins = losses = skipped = 0
    trade_log = []  # (window_num, bankroll_before, size, pnl, bankroll_after)
    daily_bankroll = {}  # day_num → end-of-day bankroll

    ws = ((first_ts // window_ms) + 1) * window_ms
    window_num = 0

    while ws + window_ms <= last_ts:
        we = ws + window_ms
        start_t = (ws // 60000) * 60000
        end_t = ((we - 60000) // 60000) * 60000
        entry_t = start_t + offset * 60000
        window_num += 1

        # Track daily bankroll (every 288 windows ≈ 1 day)
        day_num = window_num // 288
        daily_bankroll[day_num] = bankroll

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
        kelly = min(max((win_prob * payout - q) / payout, 0), kelly_frac)

        if kelly < MIN_KELLY:
            skipped += 1
            ws += window_ms
            continue

        # Position size: kelly_frac of current bankroll (scales with balance)
        size = bankroll * kelly_frac
        if size < MIN_TRADE_SIZE:
            skipped += 1
            ws += window_ms
            continue

        # Binary PnL
        right = (dir_yes == actual_up)
        bankroll_before = bankroll

        if right:
            pnl = size * payout  # win: size * (1/entry - 1)
            bankroll += pnl
            wins += 1
        else:
            pnl = -size  # lose entire position
            bankroll -= size
            losses += 1

        # Track peak and drawdown
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        trade_log.append((window_num, bankroll_before, size, pnl, bankroll))

        # Bust check
        if bankroll < 0.05:
            break

        ws += window_ms

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0

    return {
        "start": start_bankroll,
        "end": round(bankroll, 2),
        "total": total,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "max_dd": max_dd * 100,
        "peak": round(peak, 2),
        "growth": round((bankroll / start_bankroll - 1) * 100, 1),
        "daily_bankroll": daily_bankroll,
        "trade_log": trade_log,
        "min_disp": min_disp,
        "offset": offset,
    }


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

    print(f"\n{'='*110}")
    print(f"  COMPOUNDING SIMULATION — Kelly={KELLY_FRAC*100:.0f}% of bankroll (fractional, no fixed cap) — 7 DAYS")
    print(f"{'='*110}")

    # ═══════════════════════════════════════════
    # Test different starting bankrolls
    # ═══════════════════════════════════════════
    configs = [
        # (min_disp, offset, label)
        (0.001, 1, "d0.001% T+1 (max freq)"),
        (0.005, 1, "d0.005% T+1"),
        (0.010, 1, "d0.010% T+1"),
        (0.020, 1, "d0.020% T+1 (old live)"),
        (0.001, 2, "d0.001% T+2"),
        (0.010, 2, "d0.010% T+2"),
    ]

    for start_bank in [10, 25, 50, 100]:
        print(f"\n┌──────────────────────────────────────────────────────────────────────────────┐")
        print(f"│  Starting: ${start_bank:<6}  Kelly: {KELLY_FRAC*100:.0f}% of balance per trade (fractional)      │")
        print(f"└──────────────────────────────────────────────────────────────────────────────┘\n")

        print(f"  {'Config':>28} │ {'Trades':>6} {'WR%':>5} │ {'Start':>7} {'End':>10} {'Peak':>10} {'MaxDD':>6} {'$/day':>10}")
        print(f"  {'─'*28} ┼ {'─'*6} {'─'*5} ┼ {'─'*7} {'─'*10} {'─'*10} {'─'*6} {'─'*10}")

        for min_disp, offset, label in configs:
            r = run_compound_sim(candles, idx_by_time, first_ts, last_ts, days,
                                 start_bank, min_disp, offset, KELLY_FRAC)
            profit_per_day = (r['end'] - r['start']) / days
            print(
                f"  {label:>28} │ {r['total']:>6} {r['wr']:>4.1f}% │ "
                f"${r['start']:>6} ${fmt(r['end']):>9} ${fmt(r['peak']):>9} "
                f"{r['max_dd']:>5.1f}% ${fmt(profit_per_day):>9}"
            )

    # ═══════════════════════════════════════════
    # Detailed day-by-day for best configs
    # ═══════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  DAY-BY-DAY BANKROLL PROGRESSION (Starting $25)")
    print(f"{'='*110}\n")

    for min_disp, offset, label in configs:
        r = run_compound_sim(candles, idx_by_time, first_ts, last_ts, days,
                             25.0, min_disp, offset, KELLY_FRAC)

        print(f"  {label}:")
        daily = r["daily_bankroll"]
        for d in sorted(daily.keys()):
            if d <= days:
                print(f"    Day {d}: ${fmt(daily[d]):>12}")
        print(f"    Final: ${fmt(r['end']):>12}  "
              f"WR={r['wr']:.1f}%  Trades={r['total']}  MaxDD={r['max_dd']:.1f}%")
        print()

    # ═══════════════════════════════════════════
    # Streak analysis — worst losing streaks
    # ═══════════════════════════════════════════
    print(f"{'='*110}")
    print(f"  WORST LOSING STREAKS (d0.001% T+1, $25 start)")
    print(f"{'='*110}\n")

    r = run_compound_sim(candles, idx_by_time, first_ts, last_ts, days,
                         25.0, 0.001, 1, KELLY_FRAC)

    # Find losing streaks
    streak = 0
    max_streak = 0
    streaks = []
    for _, _, size, pnl, _ in r["trade_log"]:
        if pnl < 0:
            streak += 1
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0
    if streak > 0:
        streaks.append(streak)

    streaks.sort(reverse=True)
    print(f"  Total trades: {r['total']}")
    print(f"  Longest losing streak: {streaks[0] if streaks else 0}")
    print(f"  Top 10 streaks: {streaks[:10]}")
    print(f"  Streaks >= 5: {sum(1 for s in streaks if s >= 5)}")
    print(f"  Streaks >= 10: {sum(1 for s in streaks if s >= 10)}")

    # Bankroll at worst point
    min_bank = min(b for _, _, _, _, b in r["trade_log"])
    min_idx = next(i for i, (_, _, _, _, b) in enumerate(r["trade_log"]) if b == min_bank)
    print(f"  Lowest bankroll: ${fmt(min_bank)} (trade #{min_idx + 1})")
    print(f"  Max drawdown: {r['max_dd']:.1f}%")

    # Show a few sample trades
    print(f"\n  Sample trades (first 15):")
    print(f"  {'#':>4} {'Before':>12} {'Size':>10} {'PnL':>11} {'After':>12}")
    for i, (wn, before, size, pnl, after) in enumerate(r["trade_log"][:15]):
        print(f"  {i+1:>4} ${fmt(before):>11} ${fmt(size):>9} ${fmt(pnl, sign=True):>10} ${fmt(after):>11}")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    run()
