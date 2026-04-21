from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .flops import effective_params


class UnifiedScalingLaw:
    def __init__(
        self,
        n_restarts: int = 1000,
        max_iter: int = 10000,
        huber_delta: float = 1e-3,
    ):
        self.huber_delta = huber_delta
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.params: Optional[np.ndarray] = None

    def predict(
        self,
        N: np.ndarray,
        D: np.ndarray,
        T: np.ndarray,
        mu_rec: np.ndarray,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted. Call fit() first.")
        E, X, x_exp, Y, y_exp, Z, z_exp, log_Z_exp = params
        x = np.exp(x_exp)
        y = np.exp(y_exp)
        z = np.exp(z_exp)
        Z_val = np.exp(log_Z_exp)

        N = np.asarray(N, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        mu_rec = np.asarray(mu_rec, dtype=np.float64)

        N_safe = np.maximum(N, 1.0)
        D_safe = np.maximum(D, 1.0)
        T_ratio = T / np.maximum(mu_rec, 1.0)

        return E + X * N_safe ** (-x) + Y * D_safe ** (-y) + Z_val * np.exp(-z * T_ratio)

    def huber_loss_log(
        self,
        params: np.ndarray,
        data_points: np.ndarray,
    ) -> float:
        N_vals = data_points[:, 0]
        D_vals = data_points[:, 1]
        T_vals = data_points[:, 2]
        mu_rec_vals = data_points[:, 3]
        actual_loss = data_points[:, 4]

        predicted = self.predict(N_vals, D_vals, T_vals, mu_rec_vals, params)
        residual = np.log(np.maximum(predicted, 1e-12)) - np.log(np.maximum(actual_loss, 1e-12))
        abs_res = np.abs(residual)
        delta = self.huber_delta
        quadratic = 0.5 * residual ** 2
        linear = delta * (abs_res - 0.5 * delta)
        return float(np.sum(np.where(abs_res < delta, quadratic, linear)))

    def fit(
        self,
        data: List[Tuple[nn.Module, float, float, float, float]],
    ) -> np.ndarray:
        data_matrix = []
        for model, mu_rec, D, T, val_loss in data:
            _, _, N_eff = effective_params(model, mu_rec, mu_rec * 0.5)
            data_matrix.append([float(N_eff), float(D), float(T), float(mu_rec), float(val_loss)])

        data_matrix = np.array(data_matrix, dtype=np.float64)

        median_loss = np.median(data_matrix[:, 4])
        mean_N = np.mean(data_matrix[:, 0])
        mean_D = np.mean(data_matrix[:, 1])

        def objective(params_exp: np.ndarray) -> float:
            return self.huber_loss_log(params_exp, data_matrix)

        best_loss = np.inf
        best_params = None

        rng = np.random.RandomState(42)

        for i in range(self.n_restarts):
            E_init = rng.uniform(1.0, 5.0)
            X_init = rng.uniform(1.0, 200.0)
            x_exp_init = np.log(rng.uniform(0.05, 1.5))
            Y_init = rng.uniform(1.0, 200.0)
            y_exp_init = np.log(rng.uniform(0.05, 1.5))
            Z_init = rng.uniform(0.01, 10.0)
            z_exp_init = np.log(rng.uniform(0.01, 10.0))
            log_Z_exp_init = np.log(max(Z_init, 1e-8))

            x0 = np.array([
                E_init, X_init, x_exp_init, Y_init,
                y_exp_init, Z_init, z_exp_init, log_Z_exp_init,
            ])

            try:
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-12},
                    bounds=[
                        (0.01, 100.0),
                        (1e-8, 1e8),
                        (np.log(1e-4), np.log(5.0)),
                        (1e-8, 1e8),
                        (np.log(1e-4), np.log(5.0)),
                        (1e-8, 1e4),
                        (np.log(1e-4), np.log(50.0)),
                        (np.log(1e-8), np.log(1e4)),
                    ],
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue

        if best_params is not None:
            guided_inits = [
                np.array([
                    median_loss, 50.0, np.log(0.3), 50.0,
                    np.log(0.5), 1.0, np.log(1.0), np.log(1.0),
                ]),
                np.array([
                    median_loss * 0.8, 100.0, np.log(0.5), 100.0,
                    np.log(0.3), 0.5, np.log(0.5), np.log(0.5),
                ]),
                np.array([
                    median_loss * 1.2, 20.0, np.log(0.2), 20.0,
                    np.log(0.7), 5.0, np.log(2.0), np.log(5.0),
                ]),
                np.array([
                    median_loss, 80.0, np.log(0.4), 80.0,
                    np.log(0.4), 2.0, np.log(1.5), np.log(2.0),
                ]),
            ]
            for x0 in guided_inits:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method="L-BFGS-B",
                        options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-12},
                        bounds=[
                            (0.01, 100.0),
                            (1e-8, 1e8),
                            (np.log(1e-4), np.log(5.0)),
                            (1e-8, 1e8),
                            (np.log(1e-4), np.log(5.0)),
                            (1e-8, 1e4),
                            (np.log(1e-4), np.log(50.0)),
                            (np.log(1e-8), np.log(1e4)),
                        ],
                    )
                    if result.fun < best_loss:
                        best_loss = result.fun
                        best_params = result.x.copy()
                except Exception:
                    continue

            for _ in range(5):
                perturbation = best_params * (1.0 + 0.01 * rng.randn(len(best_params)))
                perturbation = np.maximum(perturbation, 1e-8)
                try:
                    result = minimize(
                        objective,
                        perturbation,
                        method="L-BFGS-B",
                        options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-12},
                        bounds=[
                            (0.01, 100.0),
                            (1e-8, 1e8),
                            (np.log(1e-4), np.log(5.0)),
                            (1e-8, 1e8),
                            (np.log(1e-4), np.log(5.0)),
                            (1e-8, 1e4),
                            (np.log(1e-4), np.log(50.0)),
                            (np.log(1e-8), np.log(1e4)),
                        ],
                    )
                    if result.fun < best_loss:
                        best_loss = result.fun
                        best_params = result.x.copy()
                except Exception:
                    continue

        self.params = best_params
        return best_params

    def evaluate(
        self,
        held_out_data: List[Tuple[nn.Module, float, float, float, float]],
        params: Optional[np.ndarray] = None,
        use_oracle_floor: bool = False,
    ) -> Dict[str, float]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted.")

        oracle_floors: Dict[Tuple, float] = {}
        for model, mu_rec, D, T, val_loss in held_out_data:
            key = (id(model), mu_rec, D)
            if T == mu_rec and key not in oracle_floors:
                oracle_floors[key] = float(val_loss)

        N_vals = []
        D_vals = []
        T_vals = []
        mu_rec_vals = []
        actual_vals = []
        oracle_floor_vals = []

        for model, mu_rec, D, T, val_loss in held_out_data:
            _, _, N_eff = effective_params(model, mu_rec, mu_rec * 0.5)
            N_vals.append(float(N_eff))
            D_vals.append(float(D))
            T_vals.append(float(T))
            mu_rec_vals.append(float(mu_rec))
            actual_vals.append(float(val_loss))

            key = (id(model), mu_rec, D)
            if use_oracle_floor and key in oracle_floors:
                oracle_floor_vals.append(oracle_floors[key])
            else:
                oracle_floor_vals.append(None)

        N_arr = np.array(N_vals)
        D_arr = np.array(D_vals)
        T_arr = np.array(T_vals)
        mu_rec_arr = np.array(mu_rec_vals)
        actual_arr = np.array(actual_vals)

        predicted = self.predict(N_arr, D_arr, T_arr, mu_rec_arr, params)

        if use_oracle_floor:
            E, X, x_exp, Y, y_exp, Z, z_exp, log_Z_exp = params
            x = np.exp(x_exp)
            y = np.exp(y_exp)
            z = np.exp(z_exp)
            Z_val = np.exp(log_Z_exp)
            N_safe = np.maximum(N_arr, 1.0)
            D_safe = np.maximum(D_arr, 1.0)
            T_ratio = T_arr / np.maximum(mu_rec_arr, 1.0)

            adjusted = np.copy(predicted)
            for i in range(len(oracle_floor_vals)):
                if oracle_floor_vals[i] is not None:
                    floor = oracle_floor_vals[i]
                    adjusted[i] = floor + Z_val * np.exp(-z * T_ratio[i])
            predicted = adjusted

        mape = np.mean(np.abs((predicted - actual_arr) / actual_arr)) * 100.0
        rmse = np.sqrt(np.mean((predicted - actual_arr) ** 2))
        mae = np.mean(np.abs(predicted - actual_arr))
        max_ae = np.max(np.abs(predicted - actual_arr))

        return {
            "mape": float(mape),
            "rmse": float(rmse),
            "mae": float(mae),
            "max_error": float(max_ae),
            "n_points": len(actual_arr),
            "use_oracle_floor": use_oracle_floor,
        }

    def get_named_params(
        self, params: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("Parameters not fitted.")
        E, X, x_exp, Y, y_exp, Z, z_exp, log_Z_exp = params
        return {
            "E": float(E),
            "X": float(X),
            "x": float(np.exp(x_exp)),
            "Y": float(Y),
            "y": float(np.exp(y_exp)),
            "Z": float(Z),
            "z": float(np.exp(z_exp)),
            "Z_exp": float(np.exp(log_Z_exp)),
        }
