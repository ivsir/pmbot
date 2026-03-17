"""ML-based displacement predictor — replaces sigmoid for P(Up) estimation.

Loads a trained model from disk at startup. Falls back to sigmoid if
the model file does not exist or prediction fails.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class DisplacementPredictor:
    """Predicts P(BTC Up) using a trained ML model.

    Falls back to sigmoid(scale * displacement_pct) if no model available.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._sigmoid_scale = settings.displacement_sigmoid_scale
        self._model = None
        self._feature_names = None
        self._feature_indices = None
        self._using_ml = False

        if settings.ml_model_enabled:
            self._load_model(Path(settings.ml_model_path))

    def _load_model(self, model_path: Path) -> None:
        """Attempt to load the trained model."""
        if not model_path.exists():
            logger.warning(
                "displacement_predictor.no_model",
                path=str(model_path),
                msg="Using sigmoid fallback",
            )
            return

        try:
            import joblib
            artifact = joblib.load(model_path)
            self._model = artifact["model"]
            self._feature_names = artifact["feature_names"]
            self._feature_indices = artifact.get("feature_indices")
            self._using_ml = True

            logger.info(
                "displacement_predictor.model_loaded",
                path=str(model_path),
                n_features=len(self._feature_names),
                metrics=artifact.get("metrics"),
            )
        except Exception as exc:
            logger.error(
                "displacement_predictor.load_failed",
                error=str(exc),
                msg="Using sigmoid fallback",
            )

    def predict(self, features: np.ndarray | None, displacement_pct: float) -> float:
        """Predict P(Up) from feature vector.

        Args:
            features: np.ndarray of shape (24,) from FeatureEngine, or None
            displacement_pct: raw displacement (for sigmoid fallback)

        Returns:
            float in [0.01, 0.99] — calibrated P(Up)
        """
        if not self._using_ml or features is None:
            return self._sigmoid_fallback(displacement_pct)

        try:
            f = features
            if self._feature_indices is not None:
                f = features[self._feature_indices]
            X = np.nan_to_num(f.reshape(1, -1), nan=0.0, posinf=10.0, neginf=-10.0)
            prob_up = self._model.predict_proba(X)[0, 1]
            prob_up = float(np.clip(prob_up, 0.01, 0.99))
            return prob_up
        except Exception as exc:
            logger.warning(
                "displacement_predictor.predict_error",
                error=str(exc),
            )
            return self._sigmoid_fallback(displacement_pct)

    def _sigmoid_fallback(self, displacement_pct: float) -> float:
        """Original sigmoid: 1 / (1 + exp(-scale * disp))."""
        raw = 1.0 / (1.0 + math.exp(-self._sigmoid_scale * displacement_pct))
        return max(0.01, min(0.99, raw))

    @property
    def is_ml_active(self) -> bool:
        return self._using_ml
