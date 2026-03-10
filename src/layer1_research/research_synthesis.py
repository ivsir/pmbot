"""Research Synthesis — Bayesian fusion of spread, latency, and liquidity signals."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from config.settings import get_settings
from src.layer1_research.spread_detector import SpreadOpportunity
from src.layer1_research.latency_arb import LatencySignal
from src.layer1_research.liquidity_scanner import LiquidityProfile

logger = structlog.get_logger(__name__)


@dataclass
class ResearchOutput:
    """Fused output from all research agents — input to signal generation."""

    market_id: str
    direction: str  # "BUY_YES" or "BUY_NO"

    # Component scores
    spread_score: float  # 0-1, from spread detector
    latency_score: float  # 0-1, from latency arb
    liquidity_score: float  # 0-1, from liquidity scanner

    # Bayesian posterior
    combined_probability: float  # P(profitable | all signals)
    confidence: float  # meta-confidence in the estimate

    # Raw data
    spread_opp: SpreadOpportunity | None = None
    latency_sig: LatencySignal | None = None
    liquidity_prof: LiquidityProfile | None = None

    # Metadata
    edge_pct: float = 0.0
    max_safe_size_usd: float = 0.0
    window_end_ms: int = 0  # for GTD expiration on maker orders
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def is_actionable(self) -> bool:
        return self.edge_pct > 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "direction": self.direction,
            "spread_score": self.spread_score,
            "latency_score": self.latency_score,
            "liquidity_score": self.liquidity_score,
            "combined_probability": self.combined_probability,
            "confidence": self.confidence,
            "edge_pct": self.edge_pct,
            "max_safe_size_usd": self.max_safe_size_usd,
            "timestamp_ms": self.timestamp_ms,
        }


class ResearchSynthesis:
    """Bayesian fusion engine — combines signals from all research agents.

    Uses a naive Bayes framework with calibrated priors:
    P(profitable | spread, latency, liquidity)
    ∝ P(spread | profitable) * P(latency | profitable) * P(liquidity | profitable) * P(profitable)
    """

    # Calibrated priors — tuned from paper trader's 85-90% win rate config.
    # Momentum/spread gets 50% weight (primary directional signal),
    # latency arb gets 30% (CEX/PM lag confirmation),
    # liquidity gets 20% (execution safety filter).
    SPREAD_WEIGHT = 0.50
    LATENCY_WEIGHT = 0.30
    LIQUIDITY_WEIGHT = 0.20

    STATE_FILE = "data/bayesian_state.json"

    def __init__(self) -> None:
        settings = get_settings()
        self.BASE_PRIOR = 0.55
        self._history: list[ResearchOutput] = []
        # Default calibrated values
        self._spread_tp_rate = 0.85
        self._spread_fp_rate = 0.30
        self._latency_tp_rate = 0.80
        self._latency_fp_rate = 0.25
        self._liquidity_tp_rate = 0.90
        self._liquidity_fp_rate = 0.50
        self._trade_count = 0
        # Load persisted state if available (survives restarts)
        self._load_state()

    def synthesize(
        self,
        spread_opp: SpreadOpportunity | None,
        latency_sig: LatencySignal | None,
        liquidity_prof: LiquidityProfile | None,
    ) -> ResearchOutput | None:
        """Fuse all research signals into a single actionability score."""

        # Need at least spread OR latency signal
        if spread_opp is None and latency_sig is None:
            return None

        # Determine market and direction
        market_id = (
            spread_opp.market_id
            if spread_opp
            else latency_sig.market_id  # type: ignore
        )
        direction = (
            spread_opp.direction
            if spread_opp
            else latency_sig.direction  # type: ignore
        )

        # Check direction agreement if both signals present
        if spread_opp and latency_sig:
            if spread_opp.direction != latency_sig.direction:
                logger.debug(
                    "research.direction_conflict",
                    spread_dir=spread_opp.direction,
                    latency_dir=latency_sig.direction,
                )
                return None

        # ── Score each component (0-1) ──
        spread_score = self._score_spread(spread_opp)
        latency_score = self._score_latency(latency_sig)
        liquidity_score = self._score_liquidity(liquidity_prof)

        # ── Bayesian fusion ──
        # P(profitable | observations) using naive Bayes
        prior = self.BASE_PRIOR

        # Likelihood ratio for each signal
        lr_spread = self._likelihood_ratio(
            spread_score, self._spread_tp_rate, self._spread_fp_rate
        )
        lr_latency = self._likelihood_ratio(
            latency_score, self._latency_tp_rate, self._latency_fp_rate
        )
        lr_liquidity = self._likelihood_ratio(
            liquidity_score, self._liquidity_tp_rate, self._liquidity_fp_rate
        )

        # Weighted log-odds fusion
        log_prior = np.log(prior / (1 - prior))
        log_posterior = (
            log_prior
            + self.SPREAD_WEIGHT * np.log(lr_spread + 1e-10)
            + self.LATENCY_WEIGHT * np.log(lr_latency + 1e-10)
            + self.LIQUIDITY_WEIGHT * np.log(lr_liquidity + 1e-10)
        )

        combined_prob = 1.0 / (1.0 + np.exp(-log_posterior))
        combined_prob = float(np.clip(combined_prob, 0.01, 0.99))

        # Meta-confidence: how reliable is our estimate
        n_signals = sum(
            1
            for s in [spread_opp, latency_sig, liquidity_prof]
            if s is not None
        )
        signal_agreement = self._signal_agreement(
            spread_score, latency_score, liquidity_score
        )
        confidence = (n_signals / 3) * 0.5 + signal_agreement * 0.5

        # Edge
        edge_pct = 0.0
        if spread_opp:
            edge_pct = max(edge_pct, spread_opp.spread_pct)
        if latency_sig:
            edge_pct = max(edge_pct, latency_sig.expected_edge_pct * 100)

        # Safe order size
        max_safe = (
            liquidity_prof.max_safe_order_usd if liquidity_prof else 1000.0
        )

        output = ResearchOutput(
            market_id=market_id,
            direction=direction,
            spread_score=round(spread_score, 4),
            latency_score=round(latency_score, 4),
            liquidity_score=round(liquidity_score, 4),
            combined_probability=round(combined_prob, 4),
            confidence=round(confidence, 4),
            spread_opp=spread_opp,
            latency_sig=latency_sig,
            liquidity_prof=liquidity_prof,
            edge_pct=round(edge_pct, 3),
            max_safe_size_usd=round(max_safe, 2),
        )

        self._history.append(output)
        if len(self._history) > 2000:
            self._history = self._history[-1000:]

        logger.info(
            "research.synthesized",
            market=market_id,
            direction=direction,
            combined_prob=round(combined_prob, 3),
            confidence=round(confidence, 3),
            edge=round(edge_pct, 3),
        )

        return output

    def update_likelihoods(self, was_profitable: bool, output: ResearchOutput) -> None:
        """Feedback loop — update likelihood estimates based on trade outcome."""
        alpha = 0.05  # learning rate
        if was_profitable:
            self._spread_tp_rate += alpha * (output.spread_score - self._spread_tp_rate)
            self._latency_tp_rate += alpha * (output.latency_score - self._latency_tp_rate)
            self._liquidity_tp_rate += alpha * (output.liquidity_score - self._liquidity_tp_rate)
        else:
            self._spread_fp_rate += alpha * (output.spread_score - self._spread_fp_rate)
            self._latency_fp_rate += alpha * (output.latency_score - self._latency_fp_rate)
            self._liquidity_fp_rate += alpha * (output.liquidity_score - self._liquidity_fp_rate)
        self._trade_count += 1
        self._save_state()

    def _save_state(self) -> None:
        """Persist Bayesian priors to disk so they survive restarts."""
        state = {
            "spread_tp_rate": self._spread_tp_rate,
            "spread_fp_rate": self._spread_fp_rate,
            "latency_tp_rate": self._latency_tp_rate,
            "latency_fp_rate": self._latency_fp_rate,
            "liquidity_tp_rate": self._liquidity_tp_rate,
            "liquidity_fp_rate": self._liquidity_fp_rate,
            "base_prior": self.BASE_PRIOR,
            "trade_count": self._trade_count,
            "saved_at": time.time(),
        }
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.warning("bayesian.save_failed", error=str(exc))

    def _load_state(self) -> None:
        """Load persisted Bayesian priors from disk."""
        try:
            if not os.path.exists(self.STATE_FILE):
                logger.info("bayesian.no_saved_state_using_defaults")
                return
            with open(self.STATE_FILE) as f:
                state = json.load(f)
            self._spread_tp_rate = state["spread_tp_rate"]
            self._spread_fp_rate = state["spread_fp_rate"]
            self._latency_tp_rate = state["latency_tp_rate"]
            self._latency_fp_rate = state["latency_fp_rate"]
            self._liquidity_tp_rate = state["liquidity_tp_rate"]
            self._liquidity_fp_rate = state["liquidity_fp_rate"]
            self.BASE_PRIOR = state.get("base_prior", 0.55)
            self._trade_count = state.get("trade_count", 0)
            logger.info(
                "bayesian.state_restored",
                trade_count=self._trade_count,
                spread_tp=round(self._spread_tp_rate, 3),
                spread_fp=round(self._spread_fp_rate, 3),
            )
        except Exception as exc:
            logger.warning("bayesian.load_failed_using_defaults", error=str(exc))

    # ── Scoring helpers ──

    @staticmethod
    def _score_spread(opp: SpreadOpportunity | None) -> float:
        if opp is None:
            return 0.3  # neutral prior
        # Normalize spread_pct: 2% → 0.5, 5% → 0.85, 10% → 0.95
        return float(np.clip(1 - np.exp(-opp.spread_pct / 5), 0, 1))

    @staticmethod
    def _score_latency(sig: LatencySignal | None) -> float:
        if sig is None:
            return 0.3
        return float(np.clip(sig.confidence, 0, 1))

    @staticmethod
    def _score_liquidity(prof: LiquidityProfile | None) -> float:
        if prof is None:
            return 0.2  # penalize unknown liquidity
        if not prof.is_sufficient:
            return 0.1
        # Score based on depth relative to minimum
        ratio = prof.total_depth_usd / get_settings().liquidity_minimum_usd
        return float(np.clip(ratio / 5, 0.3, 1.0))

    @staticmethod
    def _likelihood_ratio(score: float, tp_rate: float, fp_rate: float) -> float:
        p_obs_given_true = tp_rate * score + (1 - tp_rate) * (1 - score)
        p_obs_given_false = fp_rate * score + (1 - fp_rate) * (1 - score)
        if p_obs_given_false < 1e-10:
            return 10.0
        return p_obs_given_true / p_obs_given_false

    @staticmethod
    def _signal_agreement(s1: float, s2: float, s3: float) -> float:
        scores = [s1, s2, s3]
        mean = np.mean(scores)
        std = np.std(scores)
        # Low std = high agreement
        return float(np.clip(1 - std * 2, 0, 1))
