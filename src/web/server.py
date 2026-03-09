"""FastAPI dashboard server — REST endpoints + WebSocket real-time hub."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import orjson
import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from src.layer0_ingestion.event_bus import Event, EventType

if TYPE_CHECKING:
    from src.main import ArbitrageSystem

logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ── WebSocket Connection Manager ──────────────────────────────────────────────


class ConnectionManager:
    """Fan-out WebSocket broadcaster for all connected dashboard clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._tick_throttle_ms: int = 250
        self._last_tick_broadcast: int = 0

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("ws.client_connected", clients=self.client_count)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("ws.client_disconnected", clients=self.client_count)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        data = orjson.dumps(message).decode()
        dead: set[WebSocket] = set()
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        self._connections -= dead

    def should_throttle_tick(self) -> bool:
        now = int(time.time() * 1000)
        if now - self._last_tick_broadcast < self._tick_throttle_ms:
            return True
        self._last_tick_broadcast = now
        return False


manager = ConnectionManager()


# ── Snapshot builder ──────────────────────────────────────────────────────────


async def build_snapshot(system: Any) -> dict[str, Any]:
    """Aggregate full dashboard state from all system components.

    Works with both ArbitrageSystem and PaperTradingSystem.
    """
    portfolio_state = await system._portfolio.get_portfolio_state()

    bt = system._signal_validator.backtester.latest_result
    fills = system._executor.fill_monitor.get_metrics()
    platform = system._platform_risk.status.to_dict()

    # CEX ticks
    ticks: dict[str, Any] = {}
    for exchange, tick in system._cex_manager.get_all_ticks().items():
        ticks[exchange.value] = {
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": tick.mid,
            "last": tick.last,
            "timestamp_ms": tick.timestamp_ms,
        }

    # Active positions
    active_pos_list = system._portfolio.active_positions
    active_positions = [p.to_dict() for p in active_pos_list]

    # Closed positions (last 50) — handle both system types
    closed_list = getattr(system._portfolio, '_closed_positions', [])
    closed_positions = [
        p.to_dict() for p in closed_list[-50:]
    ]

    # Live trade stats from actual closed positions (not backtest)
    live_wins = sum(1 for p in closed_list if (getattr(p, 'pnl_usd', 0) or 0) > 0)
    live_losses = sum(1 for p in closed_list if (getattr(p, 'pnl_usd', 0) or 0) < 0 and getattr(p, 'status', '') == 'closed')
    live_total = live_wins + live_losses
    live_win_rate = live_wins / live_total if live_total > 0 else None
    live_total_pnl = sum(getattr(p, 'pnl_usd', 0) or 0 for p in closed_list)

    # Portfolio value = wallet cash + cost basis of open positions
    wallet = system._polymarket.wallet_balance
    open_position_value = sum(getattr(p, 'size_usd', 0) or 0 for p in active_pos_list)
    portfolio_value = round(wallet + open_position_value, 2)

    # Recent signals
    recent_signals = [s.to_dict() for s in system._signal_validator.get_recent(20)]

    # Recent spreads
    try:
        recent_spreads = await system._data_store.get_recent_spreads(50)
    except Exception:
        recent_spreads = []

    # Risk alerts
    alerts = []
    try:
        for a in system._tail_risk.get_recent_alerts(10):
            alerts.append(
                {
                    "level": a.level.value,
                    "type": a.alert_type,
                    "message": a.message,
                    "action": a.recommended_action,
                    "timestamp_ms": a.timestamp_ms,
                }
            )
    except Exception:
        pass

    return {
        "timestamp_ms": int(time.time() * 1000),
        "live_mode": get_settings().live_trading_enabled,
        "standby": system._polymarket.is_standby,
        "wallet_balance_usd": round(wallet, 2),
        "portfolio_value_usd": portfolio_value,
        "portfolio": portfolio_state,
        "live_stats": {
            "wins": live_wins,
            "losses": live_losses,
            "total_trades": live_total,
            "win_rate": round(live_win_rate, 4) if live_win_rate is not None else None,
            "total_pnl": round(live_total_pnl, 2),
        },
        "backtest": bt.to_dict() if bt else None,
        "fill_metrics": fills,
        "platform": platform,
        "trade_rate": round(system._signal_validator.get_trade_rate(), 4),
        "ticks": ticks,
        "active_positions": active_positions,
        "closed_positions": closed_positions,
        "recent_spreads": recent_spreads,
        "recent_signals": recent_signals,
        "alerts": alerts,
    }


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(system: Any) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup: EventBus bridge + snapshot broadcaster ──

        # Subscribe to EventBus → broadcast to all WS clients
        async def _on_event(event: Event) -> None:
            if event.event_type == EventType.PRICE_TICK:
                if manager.should_throttle_tick():
                    return
            try:
                await manager.broadcast(
                    {
                        "type": "event",
                        "event_type": event.event_type.value,
                        "data": event.data,
                        "timestamp_ms": event.timestamp_ms,
                        "source": event.source,
                    }
                )
            except Exception:
                pass

        system._event_bus.subscribe_all(_on_event)

        # Background: push full snapshot every second
        async def _snapshot_loop() -> None:
            while True:
                try:
                    if manager.client_count > 0:
                        snapshot = await build_snapshot(system)
                        await manager.broadcast(
                            {"type": "snapshot", "data": snapshot}
                        )
                except Exception as exc:
                    logger.debug("ws.snapshot_error", error=str(exc))
                await asyncio.sleep(1.0)

        snapshot_task = asyncio.create_task(_snapshot_loop())
        logger.info("dashboard.event_bridge_started")

        yield

        snapshot_task.cancel()

    app = FastAPI(
        title="PM Arbitrage Dashboard",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Disable caching for static assets during development
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class NoCacheMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response: Response = await call_next(request)
            if request.url.path.endswith(('.js', '.css', '.html')) or request.url.path == '/':
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return response

    app.add_middleware(NoCacheMiddleware)

    # ── REST Endpoints ──

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "standby" if system._polymarket.is_standby else ("running" if system._running else "stopped"),
            "live_mode": get_settings().live_trading_enabled,
            "standby": system._polymarket.is_standby,
            "wallet_balance_usd": round(system._polymarket.wallet_balance, 2),
            "ws_clients": manager.client_count,
            "timestamp_ms": int(time.time() * 1000),
        }

    @app.get("/api/overview")
    async def overview() -> Dict[str, Any]:
        return await build_snapshot(system)

    @app.get("/api/positions/active")
    async def positions_active() -> List[Dict[str, Any]]:
        return [p.to_dict() for p in system._portfolio.active_positions]

    @app.get("/api/positions/closed")
    async def positions_closed(
        limit: int = Query(default=100, le=500),
    ) -> List[Dict[str, Any]]:
        closed_list = getattr(system._portfolio, '_closed_positions', [])
        return [p.to_dict() for p in closed_list[-limit:]]

    @app.get("/api/spreads")
    async def spreads(
        limit: int = Query(default=100, le=500),
    ) -> List[Dict[str, Any]]:
        try:
            return await system._data_store.get_recent_spreads(limit)
        except Exception:
            return []

    @app.get("/api/signals")
    async def signals(
        limit: int = Query(default=50, le=200),
    ) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in system._signal_validator.get_recent(limit)]

    @app.get("/api/risk/alerts")
    async def risk_alerts() -> List[Dict[str, Any]]:
        return [
            {
                "level": a.level.value,
                "type": a.alert_type,
                "message": a.message,
                "action": a.recommended_action,
                "timestamp_ms": a.timestamp_ms,
            }
            for a in system._tail_risk.get_recent_alerts(20)
        ]

    @app.get("/api/risk/platform")
    async def risk_platform() -> Dict[str, Any]:
        return system._platform_risk.status.to_dict()

    @app.get("/api/performance/backtest")
    async def perf_backtest() -> Optional[Dict[str, Any]]:
        bt = system._signal_validator.backtester.latest_result
        return bt.to_dict() if bt else None

    @app.get("/api/performance/fills")
    async def perf_fills() -> Dict[str, Any]:
        return {
            "metrics": system._executor.fill_monitor.get_metrics(),
            "recent": [
                f.to_dict()
                for f in system._executor.fill_monitor.get_recent_fills(50)
            ],
        }

    @app.get("/api/ticks")
    async def ticks() -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for exchange, tick in system._cex_manager.get_all_ticks().items():
            result[exchange.value] = {
                "bid": tick.bid,
                "ask": tick.ask,
                "mid": tick.mid,
                "last": tick.last,
                "spread_bps": tick.spread_bps,
                "timestamp_ms": tick.timestamp_ms,
            }
        return result

    # ── WebSocket ──

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            # Send initial snapshot
            snapshot = await build_snapshot(system)
            await websocket.send_text(
                orjson.dumps({"type": "snapshot", "data": snapshot}).decode()
            )
            # Keep alive loop
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30
                    )
                    msg = orjson.loads(raw)
                    if msg.get("type") == "ping":
                        await websocket.send_text(
                            orjson.dumps({"type": "pong"}).decode()
                        )
                except asyncio.TimeoutError:
                    await websocket.send_text(
                        orjson.dumps({"type": "ping"}).decode()
                    )
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            manager.disconnect(websocket)

    # ── Static files (must be last) ──

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


# ── Server runner (for asyncio.create_task integration) ───────────────────────


async def start_server(
    app: FastAPI, host: str = "0.0.0.0", port: int = 8080
) -> None:
    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()
    except (OSError, SystemExit) as exc:
        import structlog
        structlog.get_logger().warning("dashboard.port_unavailable", port=port, error=str(exc))
