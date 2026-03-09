"""Trade Logger — logs ML features + outcomes to CSV for model retraining.

Captures the 24 feature vector at trade time, plus trade metadata and outcome.
Appends to data/live_trades.csv. This data is used by the retrain script to
improve the ML model with real trading experience.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import structlog
import numpy as np

from src.layer1_research.feature_engine import FEATURE_NAMES

logger = structlog.get_logger(__name__)

CSV_DIR = Path("data")
CSV_PATH = CSV_DIR / "live_trades.csv"

METADATA_COLS = [
    "timestamp_ms",
    "position_id",
    "market_id",
    "direction",
    "displacement_pct",
    "velocity_pct",
    "z_displacement",
    "pm_obi",
    "pm_mid",
    "fair_up_prob",
    "entry_price",
    "fill_price",
    "size_usd",
    "kelly_fraction",
    "edge_pct",
    "confidence",
    # Outcome (filled after resolution)
    "outcome",  # 1=won, 0=lost
    "pnl_usd",
]

ALL_COLS = METADATA_COLS + list(FEATURE_NAMES)


class TradeLogger:
    """Logs trade features and outcomes to CSV for ML retraining."""

    def __init__(self) -> None:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_header()
        self._pending: dict[str, dict] = {}  # position_id -> row dict

    def _ensure_header(self) -> None:
        if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLS)
                writer.writeheader()

    def log_entry(
        self,
        position_id: str,
        market_id: str,
        direction: str,
        research_output,
        signal,
        features: np.ndarray | None,
    ) -> None:
        """Log trade entry — features are stored pending outcome."""
        row = {
            "timestamp_ms": int(time.time() * 1000),
            "position_id": position_id,
            "market_id": market_id,
            "direction": direction,
            "outcome": "",
            "pnl_usd": "",
        }

        # Extract signal metadata
        spread_opp = getattr(research_output, "spread_opp", None)
        if spread_opp:
            row["displacement_pct"] = getattr(spread_opp, "displacement_pct", "")
            row["velocity_pct"] = getattr(spread_opp, "velocity_pct", "")
            row["z_displacement"] = getattr(spread_opp, "z_displacement", "")
            row["pm_obi"] = getattr(spread_opp, "pm_obi", "")
            row["pm_mid"] = getattr(spread_opp, "implied_up_prob", "")
            row["fair_up_prob"] = getattr(spread_opp, "fair_up_prob", "")
            row["entry_price"] = getattr(spread_opp, "pm_yes_price", "")

        if signal:
            row["fill_price"] = getattr(signal, "fill_price", "")
            row["size_usd"] = getattr(signal, "optimal_size_usd", "")
            row["kelly_fraction"] = getattr(signal, "kelly_fraction", "")
            row["edge_pct"] = getattr(signal, "edge_pct", getattr(research_output, "edge_pct", ""))
            row["confidence"] = getattr(signal, "confidence", getattr(research_output, "confidence", ""))

        # Store features
        if features is not None:
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = round(float(features[i]), 6) if i < len(features) else ""
        else:
            for name in FEATURE_NAMES:
                row[name] = ""

        self._pending[position_id] = row
        logger.debug("trade_logger.entry", position_id=position_id[:12])

    def log_outcome(self, position_id: str, won: bool, pnl_usd: float) -> None:
        """Log trade outcome and flush to CSV."""
        row = self._pending.pop(position_id, None)
        if row is None:
            logger.debug("trade_logger.no_pending", position_id=position_id[:12])
            return

        row["outcome"] = 1 if won else 0
        row["pnl_usd"] = round(pnl_usd, 4)

        try:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLS)
                writer.writerow(row)
            logger.info(
                "trade_logger.logged",
                position_id=position_id[:12],
                won=won,
                pnl=round(pnl_usd, 2),
                pending=len(self._pending),
            )
        except Exception as e:
            logger.warning("trade_logger.write_error", error=str(e))

    @property
    def pending_count(self) -> int:
        return len(self._pending)
