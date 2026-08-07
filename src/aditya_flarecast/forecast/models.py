"""Pluggable forecast model back-ends behind one interface.

All models implement :class:`BaseForecastModel`:

* :meth:`fit(X, y)` — train with automatic class-imbalance handling.
* :meth:`predict_proba(X)` — probability of a flare in the horizon.
* :meth:`save` / :meth:`load` — persist the whole estimator.
* :meth:`feature_importance` — for interpretability (where available).

Back-ends
---------
* ``hist_gbm`` — scikit-learn ``HistGradientBoostingClassifier``. Always
  available; strong tabular baseline; the default and the fallback.
* ``lightgbm`` — gradient boosting with native imbalance handling. Optional.
* ``lstm`` — a small PyTorch LSTM over the feature sequence, for teams that
  want a deep temporal model. Optional; falls back to ``hist_gbm`` if PyTorch
  is missing.

The factory :func:`build_model` resolves the requested back-end and downgrades
gracefully with a logged warning so the pipeline never hard-fails on a missing
optional dependency.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from aditya_flarecast.logging_utils import get_logger

logger = get_logger(__name__)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced sample weights that upweight the rare positive class."""
    y = np.asarray(y)
    n = len(y)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, n - n_pos)
    w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    return w


class BaseForecastModel(ABC):
    name: str = "base"

    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols = feature_cols or []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseForecastModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseForecastModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


class HistGBMModel(BaseForecastModel):
    name = "hist_gbm"

    def __init__(self, feature_cols=None, **params):
        super().__init__(feature_cols)
        from sklearn.ensemble import HistGradientBoostingClassifier

        defaults = dict(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        defaults.update(params)
        self.model = HistGradientBoostingClassifier(**defaults)

    def fit(self, X, y):
        self.feature_cols = list(X.columns)
        self.model.fit(X.to_numpy(), y, sample_weight=_sample_weights(y))
        try:
            from sklearn.inspection import permutation_importance  # noqa: F401
        except Exception:  # pragma: no cover
            pass
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X.to_numpy())[:, 1]

    def feature_importance(self):
        # HGB has no native importances; expose a lightweight proxy via the
        # training-time feature usage if available, else empty.
        return {}


class LightGBMModel(BaseForecastModel):
    name = "lightgbm"

    def __init__(self, feature_cols=None, **params):
        super().__init__(feature_cols)
        import lightgbm as lgb  # noqa: F401

        self.params = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.params.update(params)
        self.model = None

    def fit(self, X, y):
        import lightgbm as lgb

        self.feature_cols = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_cols, imp.astype(float)),
            key=lambda kv: kv[1], reverse=True,
        ))


class LSTMModel(BaseForecastModel):
    """Small LSTM over the per-timestep feature vector (optional, PyTorch)."""

    name = "lstm"

    def __init__(self, feature_cols=None, seq_len: int = 12, hidden: int = 64,
                 epochs: int = 12, lr: float = 1e-3, **_):
        super().__init__(feature_cols)
        import torch  # noqa: F401

        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.mean_ = None
        self.std_ = None
        self.net = None

    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        # Turn a (n, f) matrix into overlapping (n, seq_len, f) sequences by
        # padding the head with the first row.
        n, f = X.shape
        pad = np.repeat(X[:1], self.seq_len - 1, axis=0)
        padded = np.vstack([pad, X])
        idx = np.arange(self.seq_len)[None, :] + np.arange(n)[:, None]
        return padded[idx]

    def fit(self, X, y):
        import torch
        from torch import nn

        self.feature_cols = list(X.columns)
        Xn = X.to_numpy(dtype=np.float32)
        self.mean_ = Xn.mean(0, keepdims=True)
        self.std_ = Xn.std(0, keepdims=True) + 1e-6
        Xn = (Xn - self.mean_) / self.std_
        seqs = self._to_sequences(Xn)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        f = seqs.shape[2]

        class Net(nn.Module):
            def __init__(self, f, h):
                super().__init__()
                self.lstm = nn.LSTM(f, h, batch_first=True)
                self.head = nn.Sequential(nn.Linear(h, 1))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net(f, self.hidden).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pos_weight = torch.tensor(
            [max(1.0, (len(y) - y.sum()) / max(1, y.sum()))], device=device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xt = torch.tensor(seqs, device=device)
        yt = torch.tensor(y.astype(np.float32), device=device)
        n = len(yt)
        bs = 512
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=device)
            self.net.train()
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                logits = self.net(Xt[b])
                loss = loss_fn(logits, yt[b])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch

        Xn = (X.to_numpy(dtype=np.float32) - self.mean_) / self.std_
        seqs = torch.tensor(self._to_sequences(Xn))
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(seqs), 2048):
                logits = self.net(seqs[i:i + 2048].to(device))
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        # Move net to CPU before pickling for portability.
        if self.net is not None:
            self.net = self.net.to("cpu")
        return super().save(path)


_REGISTRY = {
    "hist_gbm": HistGBMModel,
    "lightgbm": LightGBMModel,
    "lstm": LSTMModel,
}


def build_model(name: str, feature_cols: list[str] | None = None, **params) -> BaseForecastModel:
    """Instantiate a model, falling back to hist_gbm on missing optional deps."""
    name = name.lower()
    cls = _REGISTRY.get(name, HistGBMModel)
    try:
        return cls(feature_cols=feature_cols, **params)
    except ImportError as exc:
        logger.warning(
            "Back-end %r unavailable (%s); falling back to hist_gbm.", name, exc
        )
        return HistGBMModel(feature_cols=feature_cols)
