"""Scoring models beyond scikit-learn's off-the-shelf regressors."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Small fully-connected PyTorch regressor with a scikit-learn interface.

    Designed to drop into sklearn pipelines and cross_val_predict. Deterministic
    for a fixed seed. Expects standardised inputs (put a StandardScaler in front).
    """

    def __init__(self, hidden: tuple = (64, 32), epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64, weight_decay: float = 1e-4, seed: int = 42):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed

    def _build(self, n_features: int):
        import torch.nn as nn

        sizes = [n_features, *self.hidden]
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)

        self.model_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        ds = torch.utils.data.TensorDataset(X, y)
        gen = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                             shuffle=True, generator=gen)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            X = torch.as_tensor(np.asarray(X, dtype=np.float32))
            return self.model_(X).numpy().ravel()
