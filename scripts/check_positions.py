"""Quick script to check order status and wallet balance."""
import asyncio
import sys
sys.path.insert(0, ".")

from src.layer0_ingestion.polymarket_client import PolymarketClient


async def main():
    client = PolymarketClient()
    await client.start()

    # Check balance
    balance = await client.check_wallet_balance()
    print(f"\n=== WALLET BALANCE: ${balance:.2f} ===\n")

    # Check the order that was placed
    order_id = "0x4f4e3acb723fb89cd615c9270e741cbff06838a63ab9f02ff8f5336363bae252"
    try:
        order = await client._clob_adapter.get_order(order_id)
        print(f"=== ORDER STATUS ===")
        print(f"  Order ID: {order_id[:20]}...")
        print(f"  Status: {order.get('status', 'unknown')}")
        print(f"  Size matched: {order.get('size_matched', 'N/A')}")
        print(f"  Original size: {order.get('original_size', 'N/A')}")
        print(f"  Price: {order.get('price', 'N/A')}")
        print(f"  Side: {order.get('side', 'N/A')}")
    except Exception as e:
        print(f"Order lookup failed: {e}")

    # Check open orders
    try:
        open_orders = await client._clob_adapter.get_open_orders()
        print(f"\n=== OPEN ORDERS: {len(open_orders)} ===")
        for o in open_orders:
            print(f"  {o.get('id', '')[:20]}... | {o.get('side')} | size={o.get('original_size')} | matched={o.get('size_matched')} | status={o.get('status')}")
    except Exception as e:
        print(f"Open orders check failed: {e}")

    print()


asyncio.run(main())
