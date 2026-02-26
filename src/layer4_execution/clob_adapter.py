"""Async adapter for the synchronous py-clob-client SDK.

Wraps ClobClient in executor threads to preserve the bot's async architecture.
All SDK calls are dispatched to a ThreadPoolExecutor to avoid blocking the
asyncio event loop.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import structlog

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
    MarketOrderArgs,
    OpenOrderParams,
    PartialCreateOrderOptions,
)

from config.settings import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class ClobOrderResult:
    """Normalized result from a CLOB order submission."""

    success: bool
    order_id: str
    error_msg: str = ""
    status: str = ""


class ClobAdapter:
    """Async wrapper around the synchronous ClobClient.

    All SDK calls are dispatched to a thread pool to avoid blocking
    the asyncio event loop.
    """

    MAX_WORKERS = 4

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: ClobClient | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS,
            thread_name_prefix="clob-sdk",
        )
        self._initialized = False

    def initialize(self) -> None:
        """Build ClobClient from settings. Call once at startup."""
        if self._initialized:
            return

        private_key = self._settings.polygon_private_key
        if not private_key:
            raise ValueError(
                "POLYGON_PRIVATE_KEY not set. "
                "Run: python scripts/generate_poly_creds.py"
            )

        creds = ApiCreds(
            api_key=self._settings.polymarket_api_key,
            api_secret=self._settings.polymarket_api_secret,
            api_passphrase=self._settings.polymarket_api_passphrase,
        )

        funder = self._settings.poly_funder_address or None

        self._client = ClobClient(
            self._settings.polymarket_clob_url,
            key=private_key,
            chain_id=self._settings.poly_chain_id,
            signature_type=self._settings.poly_signature_type,
            funder=funder,
            creds=creds,
        )

        # Validate connectivity
        try:
            ok = self._client.get_ok()
            if ok != "OK":
                raise ConnectionError(f"CLOB health check failed: {ok}")
        except Exception as exc:
            raise ConnectionError(f"Cannot reach CLOB API: {exc}") from exc

        self._initialized = True
        logger.info(
            "clob_adapter.initialized",
            host=self._settings.polymarket_clob_url,
            chain_id=self._settings.poly_chain_id,
            sig_type=self._settings.poly_signature_type,
        )

    @property
    def client(self) -> ClobClient:
        if not self._client:
            raise RuntimeError("ClobAdapter not initialized. Call initialize() first.")
        return self._client

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous SDK call in the thread pool."""
        loop = asyncio.get_running_loop()
        bound = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, bound)

    # ── Order Placement ──

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> ClobOrderResult:
        """Place a GTC limit order via the SDK.

        The SDK handles EIP-712 signing and HMAC auth internally.

        Args:
            token_id: The outcome token ID
            side: "BUY" or "SELL"
            price: Limit price (must conform to tick_size)
            size: Number of shares (NOT USD)
            tick_size: Price granularity ("0.1", "0.01", "0.001", "0.0001")
            neg_risk: Whether this is a negative risk market
        """
        # Snap price to tick grid
        tick = float(tick_size)
        decimals = len(tick_size.split(".")[-1]) if "." in tick_size else 0
        price = round(round(price / tick) * tick, decimals)
        price = max(tick, min(1.0 - tick, price))  # clamp to valid range

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side.upper(),
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        try:
            # create_and_post_order: signs the order + posts it
            signed_order = await self._run_sync(
                self.client.create_order, order_args, options
            )
            resp = await self._run_sync(
                self.client.post_order, signed_order, OrderType.GTC
            )

            success = resp.get("success", False) if isinstance(resp, dict) else False
            order_id = resp.get("orderID", "") if isinstance(resp, dict) else ""
            error_msg = resp.get("errorMsg", "") if isinstance(resp, dict) else str(resp)

            logger.info(
                "clob_adapter.order_placed",
                order_id=order_id,
                side=side,
                price=price,
                size=size,
                success=success,
            )

            return ClobOrderResult(
                success=success,
                order_id=order_id,
                error_msg=error_msg,
                status=resp.get("status", "") if isinstance(resp, dict) else "",
            )

        except Exception as exc:
            logger.error(
                "clob_adapter.order_failed",
                error=str(exc),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
            )
            return ClobOrderResult(
                success=False, order_id="", error_msg=str(exc)
            )

    async def place_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> ClobOrderResult:
        """Place a FOK market order.

        Args:
            token_id: The outcome token ID
            side: "BUY" or "SELL"
            amount: For BUY = USD amount; for SELL = number of shares
            tick_size: Price granularity
            neg_risk: Whether this is a negative risk market
        """
        market_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=side.upper(),
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        try:
            signed_order = await self._run_sync(
                self.client.create_market_order, market_args, options
            )
            resp = await self._run_sync(
                self.client.post_order, signed_order, OrderType.FOK
            )

            success = resp.get("success", False) if isinstance(resp, dict) else False
            order_id = resp.get("orderID", "") if isinstance(resp, dict) else ""

            return ClobOrderResult(
                success=success,
                order_id=order_id,
                error_msg=resp.get("errorMsg", "") if isinstance(resp, dict) else "",
            )
        except Exception as exc:
            logger.error("clob_adapter.market_order_failed", error=str(exc))
            return ClobOrderResult(success=False, order_id="", error_msg=str(exc))

    # ── Maker Order Placement ──

    async def place_maker_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        expiration_s: int,
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> ClobOrderResult:
        """Place a GTD post-only maker order.

        Uses OrderType.GTD with post_only=True to guarantee the order sits
        on the book and never crosses (zero taker fees). Auto-cancels at
        expiration_s.

        Args:
            token_id: The outcome token ID
            side: "BUY" or "SELL"
            price: Limit price (must be below best ask for BUY)
            size: Number of shares
            expiration_s: Unix timestamp (seconds) when order expires
            tick_size: Price granularity
            neg_risk: Whether this is a negative risk market
        """
        tick = float(tick_size)
        decimals = len(tick_size.split(".")[-1]) if "." in tick_size else 0
        price = round(round(price / tick) * tick, decimals)
        price = max(tick, min(1.0 - tick, price))

        # Resolve fee rate from the CLOB for this token
        try:
            fee_rate_bps = await self._run_sync(
                self.client.get_fee_rate_bps, token_id
            )
            if fee_rate_bps is None:
                fee_rate_bps = 0
        except Exception:
            fee_rate_bps = 0

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side.upper(),
            fee_rate_bps=fee_rate_bps,
            expiration=str(expiration_s),
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        try:
            signed_order = await self._run_sync(
                self.client.create_order, order_args, options
            )
            resp = await self._run_sync(
                self.client.post_order,
                signed_order,
                OrderType.GTD,
                True,  # post_only=True → maker only
            )

            success = resp.get("success", False) if isinstance(resp, dict) else False
            order_id = resp.get("orderID", "") if isinstance(resp, dict) else ""
            error_msg = resp.get("errorMsg", "") if isinstance(resp, dict) else str(resp)

            logger.info(
                "clob_adapter.maker_order_placed",
                order_id=order_id,
                side=side,
                price=price,
                size=size,
                expiration_s=expiration_s,
                post_only=True,
                fee_rate_bps=fee_rate_bps,
                success=success,
            )

            return ClobOrderResult(
                success=success,
                order_id=order_id,
                error_msg=error_msg,
                status=resp.get("status", "") if isinstance(resp, dict) else "",
            )

        except Exception as exc:
            logger.error(
                "clob_adapter.maker_order_failed",
                error=str(exc),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
            )
            return ClobOrderResult(
                success=False, order_id="", error_msg=str(exc)
            )

    # ── Order Status ──

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch order status from the CLOB."""
        return await self._run_sync(self.client.get_order, order_id)

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch all open orders."""
        return await self._run_sync(self.client.get_orders, OpenOrderParams())

    # ── Order Cancellation ──

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a single order."""
        return await self._run_sync(self.client.cancel, order_id)

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open orders."""
        return await self._run_sync(self.client.cancel_all)

    # ── Balance ──

    async def get_collateral_balance(self) -> float:
        """Fetch USDC collateral balance from the CLOB API.

        Returns balance in USD (USDC). Returns 0.0 on any error.
        """
        try:
            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self._settings.poly_signature_type,
            )
            resp = await self._run_sync(
                self.client.get_balance_allowance, params
            )
            if isinstance(resp, dict):
                # Balance is in raw USDC units (6 decimals)
                raw = float(resp.get("balance", 0))
                balance = raw / 1e6
                logger.info(
                    "clob_adapter.balance_fetched",
                    balance_usd=round(balance, 2),
                )
                return balance
            return 0.0
        except Exception as exc:
            logger.warning(
                "clob_adapter.balance_check_failed",
                error=str(exc),
            )
            return 0.0

    # ── Redemption ──

    async def redeem_positions(self, condition_id: str) -> str | None:
        """Redeem winning positions for a resolved market.

        Calls redeemPositions on the CTF contract through the Polymarket
        proxy wallet. Returns tx hash on success, None on failure.

        Requires:
        - EOA to have MATIC for gas
        - Proxy wallet to be initialized with EOA as owner
        """
        try:
            from web3 import Web3
            from web3.constants import HASH_ZERO

            rpc_urls = [
                "https://rpc-mainnet.matic.quiknode.pro",
                "https://polygon-bor-rpc.publicnode.com",
                "https://polygon.llamarpc.com",
            ]

            w3 = None
            for rpc in rpc_urls:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
                    if w3.is_connected():
                        break
                except Exception:
                    continue

            if not w3 or not w3.is_connected():
                logger.warning("clob_adapter.redeem_no_rpc")
                return None

            pk = self._settings.polygon_private_key
            acct = w3.eth.account.from_key(pk)

            # Check EOA has MATIC for gas
            matic_balance = w3.eth.get_balance(acct.address)
            if matic_balance < w3.to_wei(0.001, "ether"):
                logger.warning(
                    "clob_adapter.redeem_no_matic",
                    eoa=acct.address,
                    matic=w3.from_wei(matic_balance, "ether"),
                )
                return None

            funder = self._settings.poly_funder_address
            if not funder:
                return None

            proxy_addr = w3.to_checksum_address(funder)
            ctf_addr = w3.to_checksum_address(
                "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            )
            usdc_addr = w3.to_checksum_address(
                "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            )

            # Encode redeemPositions call on CTF
            redeem_abi = [
                {
                    "inputs": [
                        {"name": "collateralToken", "type": "address"},
                        {"name": "parentCollectionId", "type": "bytes32"},
                        {"name": "conditionId", "type": "bytes32"},
                        {"name": "indexSets", "type": "uint256[]"},
                    ],
                    "name": "redeemPositions",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                }
            ]
            ctf = w3.eth.contract(address=ctf_addr, abi=redeem_abi)

            # Convert condition_id hex string to bytes32
            cid_hex = condition_id
            if cid_hex.startswith("0x"):
                cid_hex = cid_hex[2:]
            condition_bytes = bytes.fromhex(cid_hex)

            redeem_data = ctf.encode_abi(
                "redeemPositions",
                [usdc_addr, HASH_ZERO, condition_bytes, [1, 2]],
            )

            # Call through Polymarket proxy wallet
            proxy_abi = [
                {
                    "inputs": [
                        {
                            "components": [
                                {"name": "callType", "type": "uint8"},
                                {"name": "target", "type": "address"},
                                {"name": "value", "type": "uint256"},
                                {"name": "data", "type": "bytes"},
                            ],
                            "name": "calls",
                            "type": "tuple[]",
                        }
                    ],
                    "name": "proxy",
                    "outputs": [{"name": "returnValues", "type": "bytes[]"}],
                    "stateMutability": "payable",
                    "type": "function",
                }
            ]
            proxy = w3.eth.contract(address=proxy_addr, abi=proxy_abi)

            # callType 0 = CALL
            calls = [(0, ctf_addr, 0, bytes.fromhex(redeem_data[2:]))]

            # Try to initialize proxy first (no-op if already initialized)
            init_abi = [
                {
                    "inputs": [],
                    "name": "initialize",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                }
            ]
            init_contract = w3.eth.contract(address=proxy_addr, abi=init_abi)
            try:
                init_tx = init_contract.functions.initialize().build_transaction(
                    {
                        "from": acct.address,
                        "nonce": w3.eth.get_transaction_count(acct.address),
                        "gas": 100_000,
                        "maxFeePerGas": w3.eth.gas_price * 2,
                        "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
                    }
                )
                signed_init = w3.eth.account.sign_transaction(init_tx, pk)
                init_hash = w3.eth.send_raw_transaction(signed_init.raw_transaction)
                w3.eth.wait_for_transaction_receipt(init_hash, timeout=30)
                logger.info("clob_adapter.proxy_initialized", tx=init_hash.hex())
            except Exception:
                pass  # Already initialized — that's fine

            # Build and send the redeem transaction
            nonce = w3.eth.get_transaction_count(acct.address)
            tx = proxy.functions.proxy(calls).build_transaction(
                {
                    "from": acct.address,
                    "nonce": nonce,
                    "gas": 300_000,
                    "maxFeePerGas": w3.eth.gas_price * 2,
                    "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
                }
            )
            signed = w3.eth.account.sign_transaction(tx, pk)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt["status"] == 1:
                logger.info(
                    "clob_adapter.redeem_success",
                    condition_id=condition_id[:20] + "...",
                    tx_hash=tx_hash.hex(),
                )
                return tx_hash.hex()
            else:
                logger.warning(
                    "clob_adapter.redeem_reverted",
                    condition_id=condition_id[:20] + "...",
                    tx_hash=tx_hash.hex(),
                )
                return None

        except Exception as exc:
            logger.debug(
                "clob_adapter.redeem_failed",
                condition_id=condition_id[:20] + "..." if condition_id else "",
                error=str(exc),
            )
            return None

    # ── Cleanup ──

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=False)
        logger.info("clob_adapter.shutdown")
