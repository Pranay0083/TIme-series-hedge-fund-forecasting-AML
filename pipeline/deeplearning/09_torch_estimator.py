import importlib.util
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_current_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "dl_01_objective", os.path.join(_current_dir, "01_optimization_objective.py")
)
_obj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_obj)
custom_weighted_rmse_score = _obj.custom_weighted_rmse_score

_spec06 = importlib.util.spec_from_file_location(
    "dl_06_models", os.path.join(_current_dir, "06_deep_learning_models.py")
)
_mod06 = importlib.util.module_from_spec(_spec06)
_spec06.loader.exec_module(_mod06)
BottleneckAutoencoderMLP = _mod06.BottleneckAutoencoderMLP


def critical_feature_names(feature_cols: Sequence[str]) -> List[str]:
    """Columns passed as x_critical: engineered spreads, z-scores, target encoding."""
    critical = []
    for c in feature_cols:
        if c == "sub_category_target_encoded":
            critical.append(c)
        elif c.startswith("spread_") or c.startswith("ratio_") or c.startswith("imbalance_"):
            critical.append(c)
        elif c.startswith("z_score_"):
            critical.append(c)
    if not critical:
        critical = list(feature_cols[: min(8, len(feature_cols))])
    return critical


def split_all_critical_arrays(feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    crit_set = set(critical_feature_names(list(feature_cols)))
    cols_all = [c for c in feature_cols if c not in crit_set]
    cols_critical = [c for c in feature_cols if c in crit_set]
    if not cols_all:
        cols_all = list(feature_cols)
    if not cols_critical:
        cols_critical = [feature_cols[0]]
    return cols_all, cols_critical


def _to_xyw_arrays(
    X: pd.DataFrame,
    y: np.ndarray,
    w: Optional[np.ndarray],
    feature_cols: Sequence[str],
    fillna: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs = X[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(fillna).to_numpy(dtype=np.float32)
    yv = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    if w is None:
        wv = np.ones(len(yv), dtype=np.float32)
    else:
        wv = np.asarray(w, dtype=np.float32).reshape(-1)
    return Xs, yv, wv


def _skill_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    return float(
        custom_weighted_rmse_score(
            y_true=y_true.reshape(-1),
            y_pred=y_pred.reshape(-1),
            sample_weight=w,
        )
    )


class HorizonTorchEstimator:
    def __init__(
        self,
        horizon: int,
        random_seed: int = 42,
        latent_dim: int = 32,
        drop_rate: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        max_epochs: int = 200,
        patience: int = 25,
        val_frac_timesteps: float = 0.12,
        loss: str = "mse",
        huber_delta: float = 1.0,
        lr_scheduler: str = "none",
        cosine_min_lr: float = 1e-5,
    ):
        self.horizon = horizon
        self.random_seed = random_seed
        self.latent_dim = latent_dim
        self.drop_rate = drop_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac_timesteps = val_frac_timesteps
        self.loss = loss
        self.huber_delta = huber_delta
        self.lr_scheduler = lr_scheduler
        self.cosine_min_lr = cosine_min_lr

        self.feature_cols: Optional[List[str]] = None
        self.cols_all: Optional[List[str]] = None
        self.cols_critical: Optional[List[str]] = None
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, n_all: int, n_crit: int) -> nn.Module:
        return BottleneckAutoencoderMLP(
            input_dim=n_all,
            keep_original_dim=n_crit,
            latent_dim=self.latent_dim,
            drop_rate=self.drop_rate,
        ).to(self.device)

    def _temporal_val_mask(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        uniq = np.sort(np.unique(ts))
        n_ts = len(uniq)
        if n_ts < 3:
            n = len(ts)
            n_val = max(1, min(int(np.ceil(n * 0.1)), n - 1))
            val = np.zeros(n, dtype=bool)
            val[-n_val:] = True
            train = ~val
            return train, val
        n_val = max(1, int(np.ceil(n_ts * self.val_frac_timesteps)))
        if n_val >= n_ts:
            n_val = max(1, n_ts // 5)
        cutoff = uniq[-n_val]
        val = ts >= cutoff
        train = ~val
        if not train.any() or not val.any():
            n = len(ts)
            n_tail = max(1, min(int(np.ceil(n * 0.1)), n - 1))
            val = np.zeros(n, dtype=bool)
            val[-n_tail:] = True
            train = ~val
        return train, val

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        w_train: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
        ts_index: Optional[pd.Series] = None,
    ) -> "HorizonTorchEstimator":
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        exclude_meta = {"y_target", "weight", "ts_index", "horizon", "code", "id", "sub_code", "sub_category"}
        numeric_cols = [
            c
            for c in X_train.columns
            if c not in exclude_meta and pd.api.types.is_numeric_dtype(X_train[c])
        ]
        self.feature_cols = sorted(numeric_cols)
        self.cols_all, self.cols_critical = split_all_critical_arrays(self.feature_cols)

        if X_val is None and y_val is None and ts_index is not None:
            ts = ts_index.values
            tr_m, va_m = self._temporal_val_mask(ts)
            X_val = X_train.loc[X_train.index[va_m]].copy()
            y_val = y_train[va_m]
            w_train_arr = w_train if w_train is not None else None
            w_val = w_train_arr[va_m] if w_train_arr is not None else None
            X_train = X_train.loc[X_train.index[tr_m]].copy()
            y_train = y_train[tr_m]
            if w_train_arr is not None:
                w_train = w_train_arr[tr_m]

        Xa_tr, y_tr, w_tr = _to_xyw_arrays(X_train, y_train, w_train, self.feature_cols)
        idx_all = [self.feature_cols.index(c) for c in self.cols_all]
        idx_crit = [self.feature_cols.index(c) for c in self.cols_critical]
        xa_tr = Xa_tr[:, idx_all]
        xc_tr = Xa_tr[:, idx_crit]

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        if use_val:
            Xa_va, y_va, w_va = _to_xyw_arrays(X_val, y_val, w_val, self.feature_cols)
            xa_va = Xa_va[:, idx_all]
            xc_va = Xa_va[:, idx_crit]
        else:
            xa_va = xc_va = y_va = w_va = None

        self.model = self._build_model(xa_tr.shape[1], xc_tr.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        if str(self.lr_scheduler).lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs, eta_min=self.cosine_min_lr
            )

        Xat = torch.from_numpy(xa_tr)
        Xct = torch.from_numpy(xc_tr)
        yt = torch.from_numpy(y_tr)
        wt = torch.from_numpy(w_tr)
        ds = TensorDataset(Xat, Xct, yt, wt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_state = None
        best_val = -1.0
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb_a, xb_c, yb, wb in dl:
                xb_a = xb_a.to(self.device)
                xb_c = xb_c.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb_a, xb_c)
                if str(self.loss).lower() == "huber":
                    per_ex = F.smooth_l1_loss(pred, yb, beta=self.huber_delta, reduction="none")
                else:
                    per_ex = (pred - yb) ** 2
                loss = (wb.unsqueeze(1) * per_ex).sum() / (wb.sum() + 1e-8)
                loss.backward()
                opt.step()

            if scheduler is not None:
                scheduler.step()

            if use_val:
                self.model.eval()
                with torch.no_grad():
                    pv_chunks = []
                    bs = self.batch_size
                    for i in range(0, len(xa_va), bs):
                        sl = slice(i, i + bs)
                        pa = torch.from_numpy(xa_va[sl]).to(self.device)
                        pc = torch.from_numpy(xc_va[sl]).to(self.device)
                        pv_chunks.append(self.model(pa, pc).cpu().numpy())
                    preds_val = np.concatenate(pv_chunks, axis=0)
                score = _skill_score(y_va.reshape(-1), preds_val.reshape(-1), w_va)
                if score > best_val + 1e-6:
                    best_val = score
                    bad_epochs = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_cols is None:
            raise ValueError("Estimator is not fitted.")
        self.model.eval()
        Xa, _, _ = _to_xyw_arrays(X, np.zeros(len(X)), None, self.feature_cols)
        idx_all = [self.feature_cols.index(c) for c in self.cols_all]
        idx_crit = [self.feature_cols.index(c) for c in self.cols_critical]
        xa = Xa[:, idx_all]
        xc = Xa[:, idx_crit]
        outs = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(xa), bs):
                sl = slice(i, i + bs)
                pa = torch.from_numpy(xa[sl]).to(self.device)
                pc = torch.from_numpy(xc[sl]).to(self.device)
                outs.append(self.model(pa, pc).cpu().numpy())
        return np.concatenate(outs, axis=0).reshape(-1)

    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Nothing to save.")
        payload = {
            "horizon": self.horizon,
            "random_seed": self.random_seed,
            "latent_dim": self.latent_dim,
            "drop_rate": self.drop_rate,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "lr_scheduler": self.lr_scheduler,
            "cosine_min_lr": self.cosine_min_lr,
            "feature_cols": self.feature_cols,
            "cols_all": self.cols_all,
            "cols_critical": self.cols_critical,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, filepath)

    def load(self, filepath: str) -> None:
        payload = torch.load(filepath, map_location=self.device)
        self.horizon = payload["horizon"]
        self.feature_cols = payload["feature_cols"]
        self.cols_all = payload["cols_all"]
        self.cols_critical = payload["cols_critical"]
        self.latent_dim = payload.get("latent_dim", 32)
        self.drop_rate = payload.get("drop_rate", 0.3)
        n_all = len(self.cols_all)
        n_crit = len(self.cols_critical)
        self.model = self._build_model(n_all, n_crit)
        self.model.load_state_dict(payload["state_dict"])
