import numpy as np

from speechlab.evaluation import evaluate, to_grid


def test_to_grid_snaps_to_half_steps_and_clips():
    grid = to_grid(np.array([3.24, 3.26, 7.0, -1.0]), lo=1.0, hi=6.0)
    assert grid.tolist() == [6, 7, 12, 2]            # 3.0, 3.5, 6.0, 1.0 doubled


def test_evaluate_perfect_prediction():
    y = np.array([2.0, 3.5, 4.0, 5.0, 2.5])
    out = evaluate(y, y.copy())
    assert out["QWK"] == 1.0
    assert out["MAE"] == 0.0
    assert out["pearson_r"] == 1.0


def test_evaluate_constant_prediction_has_nan_correlation_zero_qwk():
    y = np.array([2.0, 3.5, 4.0, 5.0, 2.5])
    out = evaluate(y, np.full_like(y, y.mean()))
    assert np.isnan(out["pearson_r"])
    assert np.isnan(out["spearman_rho"])
    assert out["QWK"] == 0.0


def test_evaluate_near_constant_float_noise_still_defined():
    # np.std of a "constant" array is ~1e-16, not 0 — evaluate must use an exact test
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())
    out = evaluate(y, pred)                          # must not raise or warn
    assert np.isnan(out["pearson_r"])
