"""Feature engineering for displacement ML model.

Computes 24-feature vectors from BTC price data for predicting 5-min window outcome.
Two entry points:
  - compute_from_candles() — for backtest (from Binance Candle objects)
  - compute_from_ticks()   — for live trading (from CEXTick objects via RollingCandleBuffer)

Both produce identical feature vectors to ensure training/inference parity.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    # Core displacement (1)
    "displacement_pct",
    # Multi-timeframe returns (4)
    "return_1m",
    "return_2m",
    "return_3m",
    "return_5m",
    # Velocity & acceleration (3)
    "velocity_15s",
    "acceleration",
    "displacement_vs_1m",
    # Volatility regime (3)
    "rolling_stdev_5m",
    "rolling_stdev_15m",
    "z_displacement",
    # Volume features (3)
    "volume_ratio_1m",
    "volume_ratio_5m",
    "volume_trend",
    # Candle microstructure (3)
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # Trend alignment (3)
    "trend_15m",
    "trend_30m",
    "trend_60m",
    # Time-of-day (3)
    "hour_sin",
    "hour_cos",
    "is_us_session",
    # Window timing (1)
    "secs_into_window",
]

N_FEATURES = len(FEATURE_NAMES)  # 24


@dataclass
class SimpleCandle:
    """Minimal candle representation used by both backtest and live paths."""
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class RollingCandleBuffer:
    """Aggregates live CEXTick data into 1-min candles for feature computation.

    Keeps up to max_candles completed candles. The current (incomplete) minute
    is tracked separately and finalized when a new minute boundary is crossed.
    """

    def __init__(self, max_candles: int = 120) -> None:
        self._candles: deque[SimpleCandle] = deque(maxlen=max_candles)
        self._current_minute_ms: int = 0
        self._current: dict | None = None

    def update(self, tick) -> None:
        """Ingest a CEXTick and update the rolling candle buffer."""
        minute_ms = (tick.timestamp_ms // 60_000) * 60_000
        if minute_ms != self._current_minute_ms:
            # Finalize previous candle
            if self._current is not None:
                self._candles.append(SimpleCandle(
                    time_ms=self._current["time"],
                    open=self._current["open"],
                    high=self._current["high"],
                    low=self._current["low"],
                    close=self._current["close"],
                    volume=self._current["volume"],
                ))
            self._current_minute_ms = minute_ms
            self._current = {
                "time": minute_ms,
                "open": tick.mid,
                "high": tick.mid,
                "low": tick.mid,
                "close": tick.mid,
                "volume": 0.0,
            }
        else:
            if self._current is not None:
                self._current["high"] = max(self._current["high"], tick.mid)
                self._current["low"] = min(self._current["low"], tick.mid)
                self._current["close"] = tick.mid

    def get_candles(self, n: int | None = None) -> list[SimpleCandle]:
        """Return the last n completed 1-min candles (or all if n is None)."""
        candles = list(self._candles)
        if n is not None:
            candles = candles[-n:]
        return candles

    @property
    def count(self) -> int:
        return len(self._candles)


class FeatureEngine:
    """Stateless feature computation from price data."""

    @staticmethod
    def compute_from_candles(
        candle_map: dict,
        window_start_ms: int,
        entry_time_ms: int,
    ) -> np.ndarray | None:
        """Compute feature vector from backtest candle data.

        Args:
            candle_map: open_time_ms -> Candle lookup (Candle has open/high/low/close/volume)
            window_start_ms: 5-min window start timestamp
            entry_time_ms: when we enter the trade

        Returns:
            np.ndarray of shape (24,) or None if insufficient data.
        """
        # Get window open candle
        window_open_candle = candle_map.get(window_start_ms)
        if not window_open_candle:
            return None
        window_open_price = window_open_candle.open

        # Get entry candle
        entry_candle = candle_map.get(entry_time_ms)
        if not entry_candle or window_open_price <= 0:
            return None
        entry_price = entry_candle.close

        # Build list of candles leading up to entry for feature computation
        # We need up to 60 minutes of lookback
        candles_before: list[SimpleCandle] = []
        for offset_min in range(60, -1, -1):
            t = entry_time_ms - offset_min * 60_000
            c = candle_map.get(t)
            if c:
                candles_before.append(SimpleCandle(
                    time_ms=t,
                    open=getattr(c, 'open', 0),
                    high=getattr(c, 'high', 0),
                    low=getattr(c, 'low', 0),
                    close=getattr(c, 'close', 0),
                    volume=getattr(c, 'volume', 0),
                ))

        if len(candles_before) < 6:
            return None

        secs_into_window = (entry_time_ms - window_start_ms) / 1000.0

        return FeatureEngine._compute_features(
            candles_before, entry_price, window_open_price,
            entry_time_ms, secs_into_window,
        )

    @staticmethod
    def compute_from_ticks(
        price_history: list,
        cex_tick,
        window_start_ms: int,
        window_open_price: float,
        candle_buffer: RollingCandleBuffer,
    ) -> np.ndarray | None:
        """Compute feature vector from live CEX tick data.

        Args:
            price_history: list[CEXTick] recent ticks
            cex_tick: current CEXTick
            window_start_ms: 5-min window start timestamp
            window_open_price: BTC price at window open
            candle_buffer: RollingCandleBuffer with aggregated 1-min candles

        Returns:
            np.ndarray of shape (24,) or None if insufficient data.
        """
        if window_open_price <= 0:
            return None

        candles = candle_buffer.get_candles()
        if len(candles) < 6:
            return None

        entry_price = cex_tick.mid
        now_ms = cex_tick.timestamp_ms or int(time.time() * 1000)
        secs_into_window = max(0, (now_ms - window_start_ms) / 1000.0)

        return FeatureEngine._compute_features(
            candles, entry_price, window_open_price,
            now_ms, secs_into_window,
        )

    @staticmethod
    def _compute_features(
        candles: list[SimpleCandle],
        entry_price: float,
        window_open_price: float,
        entry_time_ms: int,
        secs_into_window: float,
    ) -> np.ndarray:
        """Core feature computation from a list of SimpleCandle objects.

        candles should be time-ordered, with the last candle being the most recent
        (closest to entry time). Need at least 6 candles.
        """
        features = np.zeros(N_FEATURES, dtype=np.float64)

        # ── 1. Core displacement ──
        displacement_pct = (entry_price - window_open_price) / window_open_price * 100
        features[0] = displacement_pct

        # ── 2. Multi-timeframe returns ──
        # return_Xm = (entry_price - price_X_minutes_ago) / price_X_minutes_ago * 100
        for i, lookback in enumerate([1, 2, 3, 5]):
            idx = max(0, len(candles) - 1 - lookback)
            ref_price = candles[idx].close
            if ref_price > 0:
                features[1 + i] = (entry_price - ref_price) / ref_price * 100
            else:
                features[1 + i] = 0.0

        # ── 3. Velocity & acceleration ──
        # velocity_15s: approximate with half-minute candle return
        if len(candles) >= 2:
            prev = candles[-2].close
            if prev > 0:
                features[5] = (entry_price - prev) / prev * 100  # ~1min velocity as proxy
        else:
            features[5] = 0.0

        # acceleration: difference between recent velocity and older velocity
        if len(candles) >= 3:
            p0 = candles[-3].close
            p1 = candles[-2].close
            p2 = entry_price
            if p0 > 0 and p1 > 0:
                vel_old = (p1 - p0) / p0 * 100
                vel_new = (p2 - p1) / p1 * 100
                features[6] = vel_new - vel_old
        else:
            features[6] = 0.0

        # displacement_vs_1m: ratio of displacement at entry vs displacement 1 min ago
        if len(candles) >= 2 and window_open_price > 0:
            disp_1m_ago = (candles[-2].close - window_open_price) / window_open_price * 100
            if abs(disp_1m_ago) > 1e-8:
                features[7] = displacement_pct / disp_1m_ago
            else:
                features[7] = 0.0 if abs(displacement_pct) < 1e-8 else 10.0  # large if new displacement
        else:
            features[7] = 0.0

        # ── 4. Volatility regime ──
        # rolling_stdev_5m: stdev of 1-min returns over last 5 candles
        features[8] = _rolling_stdev(candles, 5)

        # rolling_stdev_15m: stdev of 1-min returns over last 15 candles
        features[9] = _rolling_stdev(candles, 15)

        # z_displacement: displacement / rolling_stdev_5m
        stdev_5m = features[8]
        features[10] = displacement_pct / stdev_5m if stdev_5m > 1e-8 else 0.0

        # ── 5. Volume features ──
        features[11], features[12], features[13] = _volume_features(candles)

        # ── 6. Candle microstructure (of the entry candle) ──
        entry_candle = candles[-1]
        features[14], features[15], features[16] = _candle_structure(entry_candle)

        # ── 7. Trend alignment ──
        # trend_Xm = (entry_price - price_X_minutes_ago) / price_X_minutes_ago * 100
        for i, lookback in enumerate([15, 30, 60]):
            idx = max(0, len(candles) - 1 - lookback)
            ref_price = candles[idx].close
            if ref_price > 0:
                features[17 + i] = (entry_price - ref_price) / ref_price * 100
            else:
                features[17 + i] = 0.0

        # ── 8. Time-of-day ──
        features[20], features[21], features[22] = _time_of_day_features(entry_time_ms)

        # ── 9. Window timing ──
        features[23] = secs_into_window / 300.0  # normalized to [0, 1]

        return features


# ── Helper functions ──

def _rolling_stdev(candles: list[SimpleCandle], n: int) -> float:
    """Stdev of 1-period returns (in %) over the last n candles."""
    recent = candles[-n:] if len(candles) >= n else candles
    if len(recent) < 3:
        return 0.01  # default small stdev
    prices = [c.close for c in recent if c.close > 0]
    if len(prices) < 3:
        return 0.01
    returns = []
    for i in range(1, len(prices)):
        r = (prices[i] - prices[i - 1]) / prices[i - 1] * 100
        returns.append(r)
    if not returns:
        return 0.01
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return max(var ** 0.5, 1e-6)


def _volume_features(candles: list[SimpleCandle]) -> tuple[float, float, float]:
    """Compute volume_ratio_1m, volume_ratio_5m, volume_trend."""
    if len(candles) < 2:
        return 0.0, 0.0, 0.0

    # volume_ratio_1m: last candle volume / median of prior 5 candles
    prior_vols = [c.volume for c in candles[-6:-1]] if len(candles) >= 6 else [c.volume for c in candles[:-1]]
    median_vol = float(np.median(prior_vols)) if prior_vols else 1.0
    last_vol = candles[-1].volume
    vol_ratio_1m = last_vol / median_vol if median_vol > 0 else 1.0

    # volume_ratio_5m: sum of last 5 candle volumes / sum of prior 5
    last_5 = [c.volume for c in candles[-5:]] if len(candles) >= 5 else [c.volume for c in candles]
    prior_5 = [c.volume for c in candles[-10:-5]] if len(candles) >= 10 else []
    sum_last = sum(last_5)
    sum_prior = sum(prior_5) if prior_5 else sum_last  # fallback to same
    vol_ratio_5m = sum_last / sum_prior if sum_prior > 0 else 1.0

    # volume_trend: linear regression slope of volume over last 5 candles
    recent_vols = [c.volume for c in candles[-5:]] if len(candles) >= 5 else [c.volume for c in candles]
    if len(recent_vols) >= 3:
        x = np.arange(len(recent_vols), dtype=np.float64)
        y = np.array(recent_vols, dtype=np.float64)
        mean_y = y.mean()
        if mean_y > 0:
            # Normalized slope: slope / mean_volume
            slope = np.polyfit(x, y, 1)[0]
            vol_trend = slope / mean_y
        else:
            vol_trend = 0.0
    else:
        vol_trend = 0.0

    return vol_ratio_1m, vol_ratio_5m, vol_trend


def _candle_structure(candle: SimpleCandle) -> tuple[float, float, float]:
    """body_ratio, upper_wick_ratio, lower_wick_ratio from a candle."""
    hl_range = candle.high - candle.low
    if hl_range < 1e-10:
        return 0.0, 0.0, 0.0

    body = abs(candle.close - candle.open)
    body_ratio = body / hl_range

    upper = candle.high - max(candle.open, candle.close)
    lower = min(candle.open, candle.close) - candle.low

    upper_wick_ratio = upper / hl_range
    lower_wick_ratio = lower / hl_range

    return body_ratio, upper_wick_ratio, lower_wick_ratio


def _time_of_day_features(timestamp_ms: int) -> tuple[float, float, float]:
    """hour_sin, hour_cos, is_us_session from a UTC timestamp."""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)
    hour = dt.hour + dt.minute / 60.0

    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)

    # US market hours: 13:30-20:00 UTC
    is_us = 1.0 if 13.5 <= hour < 20.0 else 0.0

    return hour_sin, hour_cos, is_us
