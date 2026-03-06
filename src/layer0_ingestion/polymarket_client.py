"""Polymarket CLOB API client — real-time orderbook + trade stream."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import aiohttp
import orjson
import structlog
import websockets
from websockets.client import WebSocketClientProtocol

from config.settings import get_settings

if TYPE_CHECKING:
    from src.layer4_execution.clob_adapter import ClobAdapter

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    market_id: str
    timestamp_ms: int
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    local_receive_ms: int = 0  # local clock when message was received

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def total_bid_depth(self) -> float:
        return sum(l.size for l in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(l.size for l in self.asks)


class PolymarketClient:
    """Connects to Polymarket CLOB for BTC 5-min markets."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: aiohttp.ClientSession | None = None
        self._ws: WebSocketClientProtocol | None = None
        self._orderbooks: dict[str, OrderBook] = {}
        self._callbacks: list[Callable[[OrderBook], Coroutine[Any, Any, None]]] = []
        self._running = False
        self._live_mode = self._settings.live_trading_enabled
        self._clob_adapter: ClobAdapter | None = None
        self._wallet_balance: float = 0.0
        self._standby: bool = False
        self._min_balance_usd: float = 1.0  # minimum USDC to trade
        # Maps condition_id → [yes_token_id, no_token_id]
        self._token_map: dict[str, list[str]] = {}
        # Maps condition_id → neg_risk flag
        self._neg_risk_map: dict[str, bool] = {}

    # ── Public API ──

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            headers=self._auth_headers(),
            json_serialize=lambda x: orjson.dumps(x).decode(),
        )
        self._running = True

        if self._live_mode:
            from src.layer4_execution.clob_adapter import ClobAdapter
            self._clob_adapter = ClobAdapter()
            self._clob_adapter.initialize()
            logger.info("polymarket_client.live_mode_enabled")
        else:
            logger.info("polymarket_client.paper_mode")

        logger.info("polymarket_client.started")

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        if self._clob_adapter:
            self._clob_adapter.shutdown()
        logger.info("polymarket_client.stopped")

    def on_orderbook_update(
        self, cb: Callable[[OrderBook], Coroutine[Any, Any, None]]
    ) -> None:
        self._callbacks.append(cb)

    async def fetch_updown_markets(self, lookahead_minutes: int = 1500) -> list[dict[str, Any]]:
        """Fetch current and upcoming 5-min BTC Up/Down markets via deterministic slug.

        Markets follow the slug pattern: btc-updown-5m-{unix_timestamp}
        where timestamp = floor(time / 300) * 300 (aligned to 5-min boundaries).
        This runs 24/7 — no restricted hours.
        """
        if not self._session:
            raise RuntimeError("Client not started")

        now = int(time.time())
        now_ms = now * 1000
        interval = 300  # 5 minutes
        base_ts = (now // interval) * interval

        # Fetch current + next 12 intervals (1 hour ahead)
        # Rotation loop re-discovers every 60s, so we don't need more
        num_ahead = 13

        updown_markets: list[dict[str, Any]] = []
        events_url = "https://gamma-api.polymarket.com/events/slug"
        miss_count = 0

        for i in range(-1, num_ahead):
            ts = base_ts + (i * interval)
            slug = f"btc-updown-5m-{ts}"

            try:
                async with self._session.get(f"{events_url}/{slug}") as resp:
                    if resp.status == 404:
                        miss_count += 1
                        # Stop if we hit 5 consecutive 404s (no more markets ahead)
                        if miss_count >= 5:
                            break
                        continue
                    if resp.status != 200:
                        continue
                    event = await resp.json()
            except Exception as exc:
                logger.warning("polymarket.slug_fetch_error", slug=slug, error=str(exc))
                continue

            miss_count = 0  # reset on success

            if not isinstance(event, dict) or "markets" not in event:
                continue

            for market in event.get("markets", []):
                if not isinstance(market, dict):
                    continue

                question = market.get("question", "")
                condition_id = market.get("conditionId", "")
                if not condition_id:
                    continue

                event_start = market.get("eventStartTime", event.get("startDate", ""))
                end_date = market.get("endDate", "")
                start_ms = _iso_to_epoch_ms(event_start) if event_start else ts * 1000
                end_ms = _iso_to_epoch_ms(end_date) if end_date else (ts + interval) * 1000

                if end_ms < now_ms:
                    continue  # already resolved

                # Extract clobTokenIds from market
                raw_tokens = market.get("clobTokenIds", "[]")
                if isinstance(raw_tokens, str):
                    import json as _json
                    try:
                        raw_tokens = _json.loads(raw_tokens)
                    except (ValueError, TypeError):
                        raw_tokens = []

                neg_risk = market.get("negRisk", False)
                if isinstance(neg_risk, str):
                    neg_risk = neg_risk.lower() == "true"

                market["_start_ms"] = start_ms
                market["_end_ms"] = end_ms
                market["_seconds_until_start"] = max(0, (start_ms - now_ms) / 1000)
                market["_seconds_until_end"] = max(0, (end_ms - now_ms) / 1000)
                market["conditionId"] = condition_id
                market["clobTokenIds"] = raw_tokens
                market["negRisk"] = neg_risk
                market["question"] = question
                updown_markets.append(market)

        updown_markets.sort(key=lambda x: x["_start_ms"])
        logger.info(
            "polymarket.updown_markets_found",
            count=len(updown_markets),
            method="deterministic_slug",
            first_market=updown_markets[0].get("question", "")[:60] if updown_markets else "none",
        )
        return updown_markets

    async def fetch_btc_markets(self) -> list[dict[str, Any]]:
        """Fetch active BTC binary markets from Gamma API (discovery endpoint)."""
        if not self._session:
            raise RuntimeError("Client not started")
        # Gamma API is the correct discovery endpoint for current markets
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "active": "true",
            "closed": "false",
            "limit": "200",
            "order": "volume24hr",
            "ascending": "false",
        }
        async with self._session.get(url, params=params) as resp:
            raw = await resp.json()
            # Gamma API returns a flat list
            if isinstance(raw, list):
                markets = raw
            elif isinstance(raw, dict):
                markets = raw.get("data", [])
            else:
                markets = []
            # Filter for Bitcoin price prediction markets
            btc_markets = [
                m
                for m in markets
                if isinstance(m, dict)
                and any(
                    kw in m.get("question", "").upper()
                    for kw in ["BITCOIN", "BTC"]
                )
                and any(
                    kw in m.get("question", "").upper()
                    for kw in ["REACH", "ABOVE", "DIP", "HIT", "PRICE", "BELOW"]
                )
            ]
            logger.info(
                "polymarket.btc_markets_found",
                count=len(btc_markets),
                total_markets=len(markets),
            )
            return btc_markets

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """REST snapshot of a single market orderbook."""
        if not self._session:
            raise RuntimeError("Client not started")
        url = f"{self._settings.polymarket_clob_url}/book"
        params = {"token_id": token_id}
        async with self._session.get(url, params=params) as resp:
            data = await resp.json()
            return self._parse_orderbook(token_id, data)

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
        tick_size: str | None = None,
        neg_risk: bool = False,
    ) -> dict[str, Any]:
        """Place an order — routes to SDK in live mode, REST in paper mode."""
        if self._live_mode and self._clob_adapter:
            ts = tick_size or self._settings.poly_tick_size
            result = await self._clob_adapter.place_limit_order(
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                tick_size=ts,
                neg_risk=neg_risk,
            )
            return {
                "orderID": result.order_id,
                "success": result.success,
                "errorMsg": result.error_msg,
                "status": result.status,
            }

        # Paper mode: existing REST stub
        if not self._session:
            raise RuntimeError("Client not started")
        url = f"{self._settings.polymarket_clob_url}/order"
        payload = {
            "tokenID": token_id,
            "side": side.upper(),
            "price": str(price),
            "size": str(size),
            "type": order_type,
        }
        async with self._session.post(url, json=payload) as resp:
            result = await resp.json()
            logger.info(
                "polymarket.order_placed",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                result=result,
            )
            return result

    async def place_maker_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        expiration_s: int,
        neg_risk: bool = False,
    ) -> dict[str, Any]:
        """Place a GTD post-only maker order — zero fees.

        Routes to ClobAdapter.place_maker_order() in live mode.
        In paper mode, simulates as a normal order.
        """
        if self._live_mode and self._clob_adapter:
            ts = self._settings.poly_tick_size
            result = await self._clob_adapter.place_maker_order(
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                expiration_s=expiration_s,
                tick_size=ts,
                neg_risk=neg_risk,
            )
            return {
                "orderID": result.order_id,
                "success": result.success,
                "errorMsg": result.error_msg,
                "status": result.status,
            }

        # Paper mode fallback
        logger.info(
            "polymarket.maker_order_paper",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            expiration_s=expiration_s,
        )
        return {"orderID": f"paper-maker-{int(time.time())}", "success": True}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        if self._live_mode and self._clob_adapter:
            return await self._clob_adapter.cancel_order(order_id)

        if not self._session:
            raise RuntimeError("Client not started")
        url = f"{self._settings.polymarket_clob_url}/order/{order_id}"
        async with self._session.delete(url) as resp:
            return await resp.json()

    async def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """Get order status from CLOB. Only works in live mode."""
        if self._live_mode and self._clob_adapter:
            try:
                return await self._clob_adapter.get_order(order_id)
            except Exception as exc:
                logger.warning(
                    "polymarket.get_order_failed",
                    order_id=order_id,
                    error=str(exc),
                )
                return None
        return None

    def set_token_mapping(
        self, condition_id: str, token_ids: list[str], neg_risk: bool = False
    ) -> None:
        """Store condition_id → [yes_token_id, no_token_id] mapping."""
        self._token_map[condition_id] = token_ids
        self._neg_risk_map[condition_id] = neg_risk
        logger.debug(
            "polymarket.token_mapping_set",
            condition_id=condition_id[:20] + "...",
            tokens=len(token_ids),
            neg_risk=neg_risk,
        )

    def get_clob_token_id(self, condition_id: str, direction: str) -> str | None:
        """Look up the CLOB token ID for a given condition_id and direction.

        Polymarket convention: clobTokenIds[0] = YES, clobTokenIds[1] = NO.
        """
        tokens = self._token_map.get(condition_id)
        if not tokens:
            return None
        if direction == "BUY_YES":
            return tokens[0] if len(tokens) > 0 else None
        elif direction == "BUY_NO":
            return tokens[1] if len(tokens) > 1 else None
        return tokens[0] if tokens else None

    async def check_market_resolution(
        self, condition_id: str, window_start_ms: int = 0
    ) -> dict[str, Any] | None:
        """Check if a market has resolved and which outcome won.

        Returns dict with {resolved, winner_token, winning_outcome} or None on error.
        Tries CLOB API first, then gamma API slug for 5-min markets (CLOB drops
        resolved 5-min markets from its API, returning 404).
        """
        # Try CLOB API first (works for active/recently-closed markets)
        if self._clob_adapter:
            try:
                market = await self._clob_adapter._run_sync(
                    self._clob_adapter.client.get_market, condition_id
                )
                if isinstance(market, dict):
                    is_closed = market.get("closed", False)
                    if not is_closed:
                        pass  # fall through to gamma check
                    else:
                        tokens = market.get("tokens", [])
                        for t in tokens:
                            if t.get("winner", False):
                                return {
                                    "resolved": True,
                                    "winner_token": t.get("token_id", ""),
                                    "winning_outcome": t.get("outcome", ""),
                                }
                        return {"resolved": False}
            except Exception as exc:
                logger.debug("polymarket.clob_resolution_failed", error=str(exc))

        # Gamma API fallback for 5-min markets (CLOB returns 404 after close)
        if window_start_ms > 0 and self._session:
            return await self._check_resolution_gamma(condition_id, window_start_ms)

        return {"resolved": False}

    async def _check_resolution_gamma(
        self, condition_id: str, window_start_ms: int
    ) -> dict[str, Any] | None:
        """Check resolution via gamma API slug endpoint.

        5-min BTC Up/Down markets use deterministic slugs: btc-updown-5m-{unix_ts}
        The gamma API retains these after resolution, unlike the CLOB API.
        """
        window_ts = window_start_ms // 1000
        slug = f"btc-updown-5m-{window_ts}"
        try:
            url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return {"resolved": False}
                event = await resp.json()

            if not isinstance(event, dict):
                return {"resolved": False}

            for market in event.get("markets", []):
                mkt_cid = market.get("conditionId", "")
                if mkt_cid != condition_id:
                    continue

                if not market.get("closed", False):
                    return {"resolved": False}

                # outcomePrices is a JSON string: '["1","0"]' means Up won
                outcome_prices = market.get("outcomePrices", "[]")
                if isinstance(outcome_prices, str):
                    import json
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except (json.JSONDecodeError, TypeError):
                        outcome_prices = []
                if len(outcome_prices) >= 2:
                    if outcome_prices[0] == "1":
                        return {
                            "resolved": True,
                            "winner_token": "",
                            "winning_outcome": "Up",
                        }
                    elif outcome_prices[1] == "1":
                        return {
                            "resolved": True,
                            "winner_token": "",
                            "winning_outcome": "Down",
                        }

                return {"resolved": False}

            # condition_id not found in event — try matching any closed market
            return {"resolved": False}

        except Exception as exc:
            logger.debug(
                "polymarket.gamma_resolution_failed",
                slug=slug,
                error=str(exc),
            )
            return {"resolved": False}

    async def fetch_redeemable_positions(self) -> list[dict]:
        """Fetch positions that are ready to claim via Data API."""
        if not self._session or not self._settings.poly_funder_address:
            return []
        try:
            url = (
                f"https://data-api.polymarket.com/positions"
                f"?user={self._settings.poly_funder_address}"
                f"&redeemable=true&limit=100"
            )
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                if isinstance(data, list) and data:
                    winners = sum(1 for p in data if float(p.get("curPrice", 0) or 0) > 0)
                    if winners > 0:
                        logger.info(
                            "polymarket.redeemable_found",
                            total=len(data),
                            winners=winners,
                        )
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug("polymarket.redeemable_check_failed", error=str(exc))
            return []

    async def redeem_winning_position(self, condition_id: str) -> str | None:
        """Redeem a winning position via CTF contract through proxy wallet.

        Returns tx hash on success, None on failure.
        """
        if not self._clob_adapter:
            return None
        return await self._clob_adapter.redeem_positions(condition_id)

    def is_neg_risk(self, condition_id: str) -> bool:
        """Check if a market uses the negative risk exchange."""
        return self._neg_risk_map.get(condition_id, False)

    @property
    def is_standby(self) -> bool:
        return self._standby

    @property
    def wallet_balance(self) -> float:
        return self._wallet_balance

    async def check_wallet_balance(self) -> float:
        """Check USDC balance and update standby state.

        Returns balance in USD. Sets standby=True if below minimum.
        """
        if not self._live_mode or not self._clob_adapter:
            # Paper mode — no real balance to check
            self._standby = False
            return 0.0

        balance = await self._clob_adapter.get_collateral_balance()
        self._wallet_balance = balance

        was_standby = self._standby
        self._standby = balance < self._min_balance_usd

        if self._standby and not was_standby:
            logger.warning(
                "polymarket.STANDBY_MODE",
                msg="Insufficient funds — bot entering standby",
                balance_usd=round(balance, 2),
                min_required=self._min_balance_usd,
            )
        elif not self._standby and was_standby:
            logger.info(
                "polymarket.STANDBY_LIFTED",
                msg="Funds detected — resuming trading",
                balance_usd=round(balance, 2),
            )

        return balance

    async def stream_orderbook(self, token_ids: list[str]) -> None:
        """Subscribe to real-time orderbook updates via WebSocket.

        Args:
            token_ids: CLOB token IDs (asset IDs) to subscribe to.
        """
        while self._running:
            try:
                async with websockets.connect(
                    self._settings.polymarket_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    # Polymarket WS uses "market" type with assets_ids array
                    subscribe_msg = {
                        "type": "market",
                        "assets_ids": token_ids,
                    }
                    await ws.send(orjson.dumps(subscribe_msg).decode())
                    logger.info(
                        "polymarket.ws_subscribed",
                        assets=len(token_ids),
                    )

                    async for raw in ws:
                        local_now = int(time.time() * 1000)
                        parsed = orjson.loads(raw)
                        # WS can send a list (book snapshots) or dict (price_change)
                        msgs = parsed if isinstance(parsed, list) else [parsed]
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            market_id = msg.get("market", "")
                            if "bids" in msg or "asks" in msg:
                                # Full book snapshot
                                asset_id = msg.get("asset_id", "")
                                server_ts = int(msg.get("timestamp", local_now))
                                # Parse with asset_id as internal market_id for token-level storage
                                ob = self._parse_orderbook(asset_id or market_id, msg, server_ts)
                                ob.local_receive_ms = local_now
                                # Store by asset_id (token_id) for execution lookups
                                if asset_id:
                                    self._orderbooks[asset_id] = ob
                                # Also store under condition_id for research pipeline
                                # Use a copy with condition_id as market_id
                                if market_id:
                                    ob_cond = self._parse_orderbook(market_id, msg, server_ts)
                                    ob_cond.local_receive_ms = local_now
                                    yes_token = self._token_map.get(market_id, [None])[0] if market_id in self._token_map else None
                                    if yes_token is None or asset_id == yes_token:
                                        self._orderbooks[market_id] = ob_cond
                                for cb in self._callbacks:
                                    asyncio.create_task(cb(ob))
                            elif "price_changes" in msg:
                                # Incremental price update — apply to each affected asset
                                for ch in msg.get("price_changes", []):
                                    ch_asset = ch.get("asset_id", "")
                                    ob = self._orderbooks.get(ch_asset)
                                    if ob:
                                        self._apply_price_changes(ob, [ch])
                                        ob.timestamp_ms = local_now
                                        ob.local_receive_ms = local_now
                                # Also update condition_id entry if it exists
                                ob = self._orderbooks.get(market_id)
                                if ob:
                                    ob.timestamp_ms = local_now
                                    ob.local_receive_ms = local_now
                                    for cb in self._callbacks:
                                        asyncio.create_task(cb(ob))

            except (
                websockets.ConnectionClosed,
                ConnectionError,
                OSError,
            ) as exc:
                logger.warning(
                    "polymarket.ws_reconnecting", error=str(exc)
                )
                await asyncio.sleep(1)

    def get_cached_orderbook(self, market_id: str) -> OrderBook | None:
        return self._orderbooks.get(market_id)

    # ── Private helpers ──

    def _auth_headers(self) -> dict[str, str]:
        return {
            "POLY_API_KEY": self._settings.polymarket_api_key,
            "POLY_API_SECRET": self._settings.polymarket_api_secret,
            "POLY_PASSPHRASE": self._settings.polymarket_api_passphrase,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_orderbook(
        market_id: str,
        data: dict[str, Any],
        ts: int | None = None,
    ) -> OrderBook:
        ts = ts or int(time.time() * 1000)
        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        return OrderBook(
            market_id=market_id,
            timestamp_ms=ts,
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _apply_price_changes(ob: OrderBook, changes: list[dict[str, Any]]) -> None:
        """Apply incremental price_change updates to a cached orderbook."""
        for ch in changes:
            price = float(ch.get("price", 0))
            size = float(ch.get("size", 0))
            side = ch.get("side", "").upper()
            levels = ob.bids if side == "BUY" else ob.asks
            # Remove existing level at this price
            levels[:] = [lv for lv in levels if abs(lv.price - price) > 1e-9]
            # Add back if size > 0
            if size > 0:
                levels.append(OrderBookLevel(price=price, size=size))
            # Re-sort
            if side == "BUY":
                levels.sort(key=lambda x: x.price, reverse=True)
            else:
                levels.sort(key=lambda x: x.price)


def _iso_to_epoch_ms(iso_str: str) -> int:
    """Convert ISO 8601 datetime string to epoch milliseconds."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, AttributeError):
        return 0
