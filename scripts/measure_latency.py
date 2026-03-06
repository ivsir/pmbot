"""Measure real-time latency between Binance CEX and Polymarket 5-min Up/Down markets.

Captures timestamps from both feeds simultaneously and computes:
1. How fast does PM reprice after a BTC move on Binance?
2. Is there a consistent lag we can exploit?
3. What's the actual PM price efficiency vs CEX displacement?

Runs for N minutes, collecting paired observations.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import numpy as np


@dataclass
class Observation:
    """A paired CEX + PM observation at a point in time."""
    timestamp_ms: int
    btc_price: float
    btc_change_pct: float  # vs 5s ago
    pm_mid: float  # PM Up token mid price
    pm_server_ts: int  # PM server timestamp
    pm_local_ts: int  # when we received PM update
    cex_local_ts: int  # when we received CEX update
    lag_ms: int  # pm_local_ts - cex_local_ts for same price move


@dataclass
class LatencyStats:
    observations: list = field(default_factory=list)
    cex_ticks: deque = field(default_factory=lambda: deque(maxlen=1000))
    pm_updates: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_pm_mid: float = 0.5
    last_pm_update_ms: int = 0
    last_cex_price: float = 0.0
    last_cex_update_ms: int = 0
    cex_5s_ago: deque = field(default_factory=lambda: deque(maxlen=100))


async def binance_ws(stats: LatencyStats, duration_s: int):
    """Connect to Binance BTC websocket and record ticks."""
    url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            start = time.time()
            async for msg in ws:
                if time.time() - start > duration_s:
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                now_ms = int(time.time() * 1000)
                bid = float(data.get("b", 0))
                ask = float(data.get("a", 0))
                mid = (bid + ask) / 2
                if mid <= 0:
                    continue

                stats.last_cex_price = mid
                stats.last_cex_update_ms = now_ms
                stats.cex_ticks.append({"price": mid, "ts": now_ms})

                # Track 5-second-ago price for change calculation
                stats.cex_5s_ago.append({"price": mid, "ts": now_ms})
                # Prune old entries
                while stats.cex_5s_ago and stats.cex_5s_ago[0]["ts"] < now_ms - 5000:
                    stats.cex_5s_ago.popleft()


async def polymarket_ws(stats: LatencyStats, token_ids: list[str], duration_s: int):
    """Connect to Polymarket orderbook WS and record updates."""
    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            # Subscribe to the Up token orderbooks
            sub_msg = {
                "type": "market",
                "assets_ids": token_ids,
            }
            await ws.send_json(sub_msg)
            print(f"  Subscribed to {len(token_ids)} PM tokens")

            start = time.time()
            async for msg in ws:
                if time.time() - start > duration_s:
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                now_ms = int(time.time() * 1000)

                # Handle list of updates
                updates = data if isinstance(data, list) else [data]
                for update in updates:
                    if not isinstance(update, dict):
                        continue

                    # Full book snapshot
                    if "bids" in update or "asks" in update:
                        bids = update.get("bids", [])
                        asks = update.get("asks", [])
                        best_bid = max((float(b["price"]) for b in bids), default=0)
                        best_ask = min((float(a["price"]) for a in asks), default=1)
                        pm_mid = (best_bid + best_ask) / 2
                        server_ts = int(update.get("timestamp", 0))

                        stats.last_pm_mid = pm_mid
                        stats.last_pm_update_ms = now_ms
                        stats.pm_updates.append({
                            "mid": pm_mid,
                            "local_ts": now_ms,
                            "server_ts": server_ts,
                        })

                    # Price change updates
                    elif "price_changes" in update:
                        for pc in update.get("price_changes", []):
                            if "price" in pc:
                                # Approximate mid from price change
                                stats.last_pm_mid = float(pc["price"])
                                stats.last_pm_update_ms = now_ms
                                stats.pm_updates.append({
                                    "mid": stats.last_pm_mid,
                                    "local_ts": now_ms,
                                    "server_ts": 0,
                                })


async def sampler(stats: LatencyStats, duration_s: int, interval_ms: int = 500):
    """Sample paired CEX+PM observations every interval_ms."""
    start = time.time()
    while time.time() - start < duration_s:
        await asyncio.sleep(interval_ms / 1000)

        if stats.last_cex_price <= 0 or stats.last_pm_mid <= 0:
            continue

        now_ms = int(time.time() * 1000)

        # Calculate BTC change vs 5 seconds ago
        btc_change = 0.0
        if stats.cex_5s_ago:
            old_price = stats.cex_5s_ago[0]["price"]
            if old_price > 0:
                btc_change = (stats.last_cex_price - old_price) / old_price * 100

        # Lag = how much later did PM update vs CEX
        lag = stats.last_pm_update_ms - stats.last_cex_update_ms

        obs = Observation(
            timestamp_ms=now_ms,
            btc_price=stats.last_cex_price,
            btc_change_pct=btc_change,
            pm_mid=stats.last_pm_mid,
            pm_server_ts=0,
            pm_local_ts=stats.last_pm_update_ms,
            cex_local_ts=stats.last_cex_update_ms,
            lag_ms=lag,
        )
        stats.observations.append(obs)


async def fetch_updown_tokens() -> list[str]:
    """Fetch current 5-min Up/Down market token IDs via deterministic slug."""
    import json as _json
    now = int(time.time())
    interval = 300
    base_ts = (now // interval) * interval
    tokens = []
    events_url = "https://gamma-api.polymarket.com/events/slug"

    async with aiohttp.ClientSession() as session:
        for i in range(-1, 5):
            ts = base_ts + (i * interval)
            slug = f"btc-updown-5m-{ts}"
            try:
                async with session.get(f"{events_url}/{slug}") as resp:
                    if resp.status != 200:
                        continue
                    event = await resp.json()
            except Exception:
                continue

            for market in event.get("markets", []):
                if not isinstance(market, dict):
                    continue
                raw = market.get("clobTokenIds", "[]")
                if isinstance(raw, str):
                    try:
                        raw = _json.loads(raw)
                    except (ValueError, TypeError):
                        raw = []
                tokens.extend(raw)
                q = market.get("question", "")
                if q:
                    print(f"    {q[:55]} ({len(raw)} tokens)")

    return tokens[:20]


async def main():
    duration_s = 300  # 5 minutes of observation
    print(f"Measuring Binance vs Polymarket latency for {duration_s}s...")
    print()

    # Get PM token IDs
    print("Fetching active Up/Down market tokens...")
    tokens = await fetch_updown_tokens()
    if not tokens:
        print("  No tokens found, using generic subscription")
        tokens = []
    else:
        print(f"  Found {len(tokens)} tokens")

    stats = LatencyStats()

    # Run all three tasks concurrently
    tasks = [
        binance_ws(stats, duration_s),
        sampler(stats, duration_s, interval_ms=200),
    ]
    if tokens:
        tasks.append(polymarket_ws(stats, tokens, duration_s))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"  Error: {e}")

    # ── Analyze results ──
    print()
    print("=" * 70)
    print("  LATENCY MEASUREMENT RESULTS")
    print("=" * 70)

    obs = stats.observations
    if not obs:
        print("  No observations collected!")
        return

    print(f"\n  Total observations: {len(obs)}")
    print(f"  CEX ticks received: {len(stats.cex_ticks)}")
    print(f"  PM updates received: {len(stats.pm_updates)}")

    # CEX update frequency
    if len(stats.cex_ticks) >= 2:
        cex_intervals = []
        ticks = list(stats.cex_ticks)
        for i in range(1, len(ticks)):
            cex_intervals.append(ticks[i]["ts"] - ticks[i-1]["ts"])
        print(f"\n  CEX update interval: median={np.median(cex_intervals):.0f}ms, "
              f"mean={np.mean(cex_intervals):.0f}ms, p95={np.percentile(cex_intervals, 95):.0f}ms")

    # PM update frequency
    if len(stats.pm_updates) >= 2:
        pm_intervals = []
        pms = list(stats.pm_updates)
        for i in range(1, len(pms)):
            pm_intervals.append(pms[i]["local_ts"] - pms[i-1]["local_ts"])
        print(f"  PM  update interval: median={np.median(pm_intervals):.0f}ms, "
              f"mean={np.mean(pm_intervals):.0f}ms, p95={np.percentile(pm_intervals, 95):.0f}ms")

    # Lag analysis
    lags = [o.lag_ms for o in obs]
    print(f"\n  PM lag vs CEX (last-update timing):")
    print(f"    Median: {np.median(lags):.0f}ms")
    print(f"    Mean:   {np.mean(lags):.0f}ms")
    print(f"    P25:    {np.percentile(lags, 25):.0f}ms")
    print(f"    P75:    {np.percentile(lags, 75):.0f}ms")
    print(f"    P95:    {np.percentile(lags, 95):.0f}ms")
    print(f"    Max:    {np.max(lags):.0f}ms")

    # Price efficiency: when BTC moves, how quickly does PM mid reflect it?
    print(f"\n  Price displacement analysis:")
    btc_changes = [abs(o.btc_change_pct) for o in obs]
    print(f"    Avg |BTC 5s change|: {np.mean(btc_changes):.4f}%")
    print(f"    Max |BTC 5s change|: {np.max(btc_changes):.4f}%")

    # Correlation between BTC move and PM mid
    # If PM is efficient, pm_mid should track displacement
    # If PM lags, there's a window to exploit
    big_moves = [o for o in obs if abs(o.btc_change_pct) > 0.01]  # >0.01% moves
    if big_moves:
        print(f"\n  During significant BTC moves (>{0.01}%):")
        print(f"    Count: {len(big_moves)}")
        move_lags = [o.lag_ms for o in big_moves]
        print(f"    PM lag median: {np.median(move_lags):.0f}ms")
        print(f"    PM lag mean:   {np.mean(move_lags):.0f}ms")
        print(f"    PM lag max:    {np.max(move_lags):.0f}ms")

        # Check if PM mid moves with BTC
        pm_changes = []
        for o in big_moves:
            # Expected PM mid from displacement
            # If BTC is up 0.05% from some reference, PM Up should be ~0.51+
            pm_changes.append(o.pm_mid)
        pm_std = np.std(pm_changes)
        print(f"    PM mid range: {min(pm_changes):.3f} - {max(pm_changes):.3f}")
        print(f"    PM mid std: {pm_std:.4f}")

    # Exploitability assessment
    print(f"\n  ── EXPLOITABILITY ASSESSMENT ──")
    median_lag = np.median(lags)
    if median_lag > 500:
        print(f"    PM lags CEX by {median_lag:.0f}ms median — EXPLOITABLE")
        print(f"    Can place orders before PM reprices")
    elif median_lag > 200:
        print(f"    PM lags CEX by {median_lag:.0f}ms median — MARGINAL")
        print(f"    Possible edge but tight execution needed")
    else:
        print(f"    PM lags CEX by {median_lag:.0f}ms median — NOT EXPLOITABLE")
        print(f"    PM updates too fast for latency arb")
        print(f"    Displacement model (current) is the better approach")

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
