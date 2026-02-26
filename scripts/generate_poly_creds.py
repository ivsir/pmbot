"""Generate Polymarket API credentials from your Polygon private key.

Usage:
    python scripts/generate_poly_creds.py

Reads POLYGON_PRIVATE_KEY from .env file.
Outputs the API key, secret, and passphrase to copy into .env.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    private_key = os.getenv("POLYGON_PRIVATE_KEY", "")
    if not private_key:
        print("ERROR: Set POLYGON_PRIVATE_KEY in .env first")
        sys.exit(1)

    funder = os.getenv("POLY_FUNDER_ADDRESS", "")
    sig_type = int(os.getenv("POLY_SIGNATURE_TYPE", "1"))
    chain_id = int(os.getenv("POLY_CHAIN_ID", "137"))
    host = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")

    from py_clob_client.client import ClobClient

    client = ClobClient(
        host,
        key=private_key,
        chain_id=chain_id,
        signature_type=sig_type,
        funder=funder if funder else None,
    )

    print("Deriving API credentials from private key...")
    creds = client.create_or_derive_api_creds()

    print("\n=== Add these to your .env file ===\n")
    print(f"POLYMARKET_API_KEY={creds.api_key}")
    print(f"POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
    print()
    print("Done. These credentials are derived deterministically")
    print("from your private key, so you can re-run this anytime.")


if __name__ == "__main__":
    main()
