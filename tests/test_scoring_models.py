"""Synthetic regression data only — fast, deterministic fits."""

import numpy as np

from speechlab.scoring_models import TorchMLPRegressor


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, n)
    return X, y


def test_mlp_learns_linear_signal():
    X, y = _toy()
    model = TorchMLPRegressor(hidden=(16,), epochs=150, seed=42).fit(X, y)
    pred = model.predict(X)
    resid = float(np.mean((pred - y) ** 2))
    baseline = float(np.var(y))                 # MSE of the mean predictor
    assert resid < 0.2 * baseline


def test_mlp_is_deterministic_for_fixed_seed():
    X, y = _toy()
    a = TorchMLPRegressor(hidden=(8,), epochs=30, seed=7).fit(X, y).predict(X)
    b = TorchMLPRegressor(hidden=(8,), epochs=30, seed=7).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_mlp_works_in_sklearn_pipeline_cv():
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _toy(n=120)
    pipe = make_pipeline(StandardScaler(), TorchMLPRegressor(hidden=(8,), epochs=30))
    preds = cross_val_predict(pipe, X, y, cv=3)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()
