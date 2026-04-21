from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .flops import effective_params


class TrainingScalingLaw:
    def __init__(
        self,
        huber_delta: float = 1e-3,
        n_restarts: int = 500,
        max_iter: int = 10000,
    ):
        self.huber_delta = huber_delta
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.params: Optional[np.ndarray] = None

    def predict(
        self,
        N: np.ndarray,
        D: np.ndarray,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted. Call fit() first.")
        E, X, x_exp, Y, y_exp = params
        x = np.exp(x_exp)
        y = np.exp(y_exp)
        N = np.asarray(N, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        N_safe = np.maximum(N, 1.0)
        D_safe = np.maximum(D, 1.0)
        return E + X * N_safe ** (-x) + Y * D_safe ** (-y)

    def huber_loss(
        self,
        params: np.ndarray,
        N_data: np.ndarray,
        D_data: np.ndarray,
        loss_data: np.ndarray,
    ) -> float:
        predicted = self.predict(N_data, D_data, params)
        residual = np.log(np.maximum(predicted, 1e-12)) - np.log(np.maximum(loss_data, 1e-12))
        abs_res = np.abs(residual)
        delta = self.huber_delta
        quadratic = 0.5 * residual ** 2
        linear = delta * (abs_res - 0.5 * delta)
        loss = np.sum(np.where(abs_res < delta, quadratic, linear))
        return float(loss)

    def fit(
        self,
        data: List[Tuple[float, float, float, nn.Module]],
    ) -> np.ndarray:
        N_vals = []
        D_vals = []
        loss_vals = []
        for mu_rec, D, val_loss, model in data:
            _, _, N_eff = effective_params(model, mu_rec, self._infer_mu_bwd(mu_rec))
            N_vals.append(float(N_eff))
            D_vals.append(float(D))
            loss_vals.append(float(val_loss))

        N_arr = np.array(N_vals, dtype=np.float64)
        D_arr = np.array(D_vals, dtype=np.float64)
        loss_arr = np.array(loss_vals, dtype=np.float64)

        log_N = np.log(np.maximum(N_arr, 1.0))
        log_D = np.log(np.maximum(D_arr, 1.0))
        log_loss = np.log(np.maximum(loss_arr, 1e-12))

        median_loss = np.median(loss_arr)
        init_E = median_loss * 0.8

        def objective(params_exp: np.ndarray) -> float:
            return self.huber_loss(params_exp, N_arr, D_arr, loss_arr)

        best_loss = np.inf
        best_params = None

        for _ in range(self.n_restarts):
            E_init = np.random.uniform(0.5, 3.0)
            X_init = np.random.uniform(1.0, 100.0)
            x_exp_init = np.log(np.random.uniform(0.1, 1.0))
            Y_init = np.random.uniform(1.0, 100.0)
            y_exp_init = np.log(np.random.uniform(0.1, 1.0))
            x0 = np.array([E_init, X_init, x_exp_init, Y_init, y_exp_init])

            try:
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-12},
                    bounds=[
                        (0.01, 50.0),
                        (1e-6, 1e6),
                        (np.log(1e-4), np.log(5.0)),
                        (1e-6, 1e6),
                        (np.log(1e-4), np.log(5.0)),
                    ],
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue

        if best_params is not None:
            guided_inits = [
                np.array([median_loss, 50.0, np.log(0.3), 50.0, np.log(0.5)]),
                np.array([median_loss, 10.0, np.log(0.5), 10.0, np.log(0.7)]),
                np.array([median_loss * 0.5, 100.0, np.log(0.2), 100.0, np.log(0.3)]),
            ]
            for x0 in guided_inits:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method="L-BFGS-B",
                        options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-12},
                        bounds=[
                            (0.01, 50.0),
                            (1e-6, 1e6),
                            (np.log(1e-4), np.log(5.0)),
                            (1e-6, 1e6),
                            (np.log(1e-4), np.log(5.0)),
                        ],
                    )
                    if result.fun < best_loss:
                        best_loss = result.fun
                        best_params = result.x.copy()
                except Exception:
                    continue

        self.params = best_params
        return best_params

    def evaluate_fit(
        self,
        data: List[Tuple[float, float, float, nn.Module]],
        params: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted.")

        N_vals = []
        D_vals = []
        loss_vals = []
        for mu_rec, D, val_loss, model in data:
            _, _, N_eff = effective_params(model, mu_rec, self._infer_mu_bwd(mu_rec))
            N_vals.append(float(N_eff))
            D_vals.append(float(D))
            loss_vals.append(float(val_loss))

        N_arr = np.array(N_vals, dtype=np.float64)
        D_arr = np.array(D_vals, dtype=np.float64)
        loss_arr = np.array(loss_vals, dtype=np.float64)
        pred = self.predict(N_arr, D_arr, params)

        mape = np.mean(np.abs((pred - loss_arr) / loss_arr)) * 100.0
        rmse = np.sqrt(np.mean((pred - loss_arr) ** 2))
        mae = np.mean(np.abs(pred - loss_arr))
        max_err = np.max(np.abs(pred - loss_arr))

        return {
            "mape": float(mape),
            "rmse": float(rmse),
            "mae": float(mae),
            "max_error": float(max_err),
            "huber_loss": float(self.huber_loss(params, N_arr, D_arr, loss_arr)),
        }

    def get_named_params(
        self, params: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted.")
        E, X, x_exp, Y, y_exp = params
        return {
            "E": float(E),
            "X": float(X),
            "x": float(np.exp(x_exp)),
            "Y": float(Y),
            "y": float(np.exp(y_exp)),
        }

    def _infer_mu_bwd(self, mu_rec: float) -> float:
        if hasattr(mu_rec, "mu_bwd"):
            return float(mu_rec.mu_bwd)
        return float(mu_rec) * 0.5
