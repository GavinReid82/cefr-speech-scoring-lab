"""Scorer evaluation: agreement with human ratings on the 0.5-step ordinal grid."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def to_grid(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Snap to the human 0.5-step grid within [lo, hi], coded as integer classes (x2)."""
    snapped = np.clip(np.round(np.asarray(scores) * 2) / 2, lo, hi)
    return (snapped * 2).astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Correlation, error, and QWK for one scorer. Grid bounds come from y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo, hi = float(y_true.min()), float(y_true.max())
    constant = np.ptp(y_pred) == 0             # mean predictor: correlation undefined
    return {
        "pearson_r": np.nan if constant else pearsonr(y_true, y_pred)[0],
        "spearman_rho": np.nan if constant else spearmanr(y_true, y_pred)[0],
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "QWK": cohen_kappa_score(to_grid(y_true, lo, hi), to_grid(y_pred, lo, hi),
                                 weights="quadratic"),
    }
