"""Check the filled order's market resolution and attempt redemption."""
import asyncio
import sys
sys.path.insert(0, ".")

from src.layer0_ingestion.polymarket_client import PolymarketClient


async def main():
    client = PolymarketClient()
    await client.start()

    condition_id = "0x5d560100997de5141673ecebf75998bc7d14340e43a3fa01393e1b72b485807f"

    print(f"Condition ID: {condition_id}")
    print(f"Market: Bitcoin Up or Down - March 2, 5:10PM-5:15PM ET")
    print(f"Position: 6.75 NO shares bought at $0.51")
    print()

    # Use the clob_adapter directly
    adapter = client._clob_adapter
    if not adapter:
        print("No CLOB adapter — live mode not initialized")
        return

    print("Attempting redemption via clob_adapter...")
    try:
        tx = await adapter.redeem_positions(condition_id)
        if tx:
            print(f"Redemption TX: {tx}")
            print("Waiting for balance update...")
            await asyncio.sleep(10)
            balance = await adapter.get_collateral_balance()
            print(f"New balance: ${balance:.2f}")
        else:
            print("Redemption returned None — possible reasons:")
            print("  - Market not yet resolved")
            print("  - No MATIC for gas on your EOA")
            print("  - Position was on losing side")
    except Exception as e:
        print(f"Redemption failed: {e}")

    balance = await adapter.get_collateral_balance()
    print(f"\nCurrent wallet balance: ${balance:.2f}")


asyncio.run(main())
