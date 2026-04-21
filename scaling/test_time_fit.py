from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple


class TestTimeSaturationFit:
    def __init__(self, huber_delta: float = 1e-3):
        self.huber_delta = huber_delta

    def _compute_huber(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> float:
        residual = np.log(np.maximum(predicted, 1e-12)) - np.log(np.maximum(actual, 1e-12))
        abs_res = np.abs(residual)
        delta = self.huber_delta
        quadratic = 0.5 * residual ** 2
        linear = delta * (abs_res - 0.5 * delta)
        return float(np.sum(np.where(abs_res < delta, quadratic, linear)))

    def _split_id_ood(
        self, T_values: np.ndarray, loss_values: np.ndarray, mu_rec: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)
        mask_id = T_values <= mu_rec
        mask_ood = T_values > mu_rec
        T_id, loss_id = T_values[mask_id], loss_values[mask_id]
        T_ood, loss_ood = T_values[mask_ood], loss_values[mask_ood]
        return T_id, loss_id, T_ood, loss_ood

    def fit_exponential(
        self, T_values: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)

        def model(T, L_inf, Z, z):
            return L_inf + Z * np.exp(-z * T)

        valid = T_values > 0
        T_v = T_values[valid]
        L_v = loss_values[valid]

        L_inf_init = np.min(L_v) * 0.95
        Z_init = float(np.max(L_v) - L_inf_init)
        z_init = 0.5 / float(np.median(T_v))

        best_result = None
        best_cost = np.inf

        init_sets = [
            (L_inf_init, Z_init, z_init),
            (np.min(L_v), Z_init * 1.2, z_init * 2.0),
            (np.min(L_v) * 0.9, Z_init * 0.8, z_init * 0.5),
            (np.min(L_v) * 1.05, Z_init * 2.0, z_init * 0.2),
            (np.percentile(L_v, 10), Z_init, 1.0 / float(np.mean(T_v))),
        ]

        for L0, Z0, z0 in init_sets:
            try:
                popt, _ = curve_fit(
                    model,
                    T_v,
                    L_v,
                    p0=[L0, Z0, z0],
                    bounds=([0.0, 1e-8, 1e-8], [np.inf, np.inf, np.inf]),
                    maxfev=50000,
                    method="trf",
                )
                pred = model(T_v, *popt)
                cost = np.sum((pred - L_v) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_result = popt
            except Exception:
                continue

        if best_result is None:
            return (float(L_inf_init), float(Z_init), float(z_init), np.inf, np.inf)

        L_inf, Z, z = best_result

        mu_rec_est = float(np.max(T_values))
        T_id, loss_id, T_ood, loss_ood = self._split_id_ood(T_values, loss_values, mu_rec_est)

        huber_id = self._compute_huber(model(T_id, L_inf, Z, z), loss_id) if len(T_id) > 0 else np.inf
        huber_ood = self._compute_huber(model(T_ood, L_inf, Z, z), loss_ood) if len(T_ood) > 0 else np.inf

        return (float(L_inf), float(Z), float(z), float(huber_id), float(huber_ood))

    def fit_stretched_power(
        self, T_values: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)

        def model(T, L_inf, Z, z):
            return L_inf + Z * np.power(1.0 + T, -z)

        valid = T_values >= 0
        T_v = T_values[valid]
        L_v = loss_values[valid]

        L_inf_init = np.min(L_v) * 0.95
        Z_init = float(np.max(L_v) - L_inf_init)
        z_init = 0.5

        best_result = None
        best_cost = np.inf

        init_sets = [
            (L_inf_init, Z_init, z_init),
            (np.min(L_v), Z_init * 1.2, z_init * 2.0),
            (np.min(L_v) * 0.9, Z_init * 0.8, z_init * 0.3),
            (np.percentile(L_v, 10), Z_init, 1.0),
        ]

        for L0, Z0, z0 in init_sets:
            try:
                popt, _ = curve_fit(
                    model,
                    T_v,
                    L_v,
                    p0=[L0, Z0, z0],
                    bounds=([0.0, 1e-8, 1e-8], [np.inf, np.inf, np.inf]),
                    maxfev=50000,
                    method="trf",
                )
                pred = model(T_v, *popt)
                cost = np.sum((pred - L_v) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_result = popt
            except Exception:
                continue

        if best_result is None:
            return (float(L_inf_init), float(Z_init), float(z_init), np.inf, np.inf)

        L_inf, Z, z = best_result

        mu_rec_est = float(np.max(T_values))
        T_id, loss_id, T_ood, loss_ood = self._split_id_ood(T_values, loss_values, mu_rec_est)

        huber_id = self._compute_huber(model(T_id, L_inf, Z, z), loss_id) if len(T_id) > 0 else np.inf
        huber_ood = self._compute_huber(model(T_ood, L_inf, Z, z), loss_ood) if len(T_ood) > 0 else np.inf

        return (float(L_inf), float(Z), float(z), float(huber_id), float(huber_ood))

    def fit_power(
        self, T_values: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)

        def model(T, L_inf, Z, z):
            return L_inf + Z * np.power(np.maximum(T, 1e-8), -z)

        valid = T_values > 0
        T_v = T_values[valid]
        L_v = loss_values[valid]

        L_inf_init = np.min(L_v) * 0.95
        Z_init = float(np.max(L_v) - L_inf_init)
        z_init = 0.5

        best_result = None
        best_cost = np.inf

        init_sets = [
            (L_inf_init, Z_init, z_init),
            (np.min(L_v), Z_init * 1.2, z_init * 2.0),
            (np.min(L_v) * 0.9, Z_init * 0.8, z_init * 0.3),
            (np.percentile(L_v, 10), Z_init, 1.0),
        ]

        for L0, Z0, z0 in init_sets:
            try:
                popt, _ = curve_fit(
                    model,
                    T_v,
                    L_v,
                    p0=[L0, Z0, z0],
                    bounds=([0.0, 1e-8, 1e-8], [np.inf, np.inf, np.inf]),
                    maxfev=50000,
                    method="trf",
                )
                pred = model(T_v, *popt)
                cost = np.sum((pred - L_v) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_result = popt
            except Exception:
                continue

        if best_result is None:
            return (float(L_inf_init), float(Z_init), float(z_init), np.inf, np.inf)

        L_inf, Z, z = best_result

        mu_rec_est = float(np.max(T_values))
        T_id, loss_id, T_ood, loss_ood = self._split_id_ood(T_values, loss_values, mu_rec_est)

        huber_id = self._compute_huber(model(T_id, L_inf, Z, z), loss_id) if len(T_id) > 0 else np.inf
        huber_ood = self._compute_huber(model(T_ood, L_inf, Z, z), loss_ood) if len(T_ood) > 0 else np.inf

        return (float(L_inf), float(Z), float(z), float(huber_id), float(huber_ood))

    def fit_pure_power(
        self, T_values: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[float, float, float, float]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)

        def model(T, Z, z):
            return Z * np.power(np.maximum(T, 1e-8), -z)

        valid = T_values > 0
        T_v = T_values[valid]
        L_v = loss_values[valid]

        log_T = np.log(T_v)
        log_L = np.log(np.maximum(L_v, 1e-12))

        A = np.column_stack([np.ones_like(log_T), -log_T])
        coeffs, _, _, _ = np.linalg.lstsq(A, log_L, rcond=None)
        log_Z, z = coeffs[0], coeffs[1]
        Z = np.exp(log_Z)

        mu_rec_est = float(np.max(T_values))
        T_id, loss_id, T_ood, loss_ood = self._split_id_ood(T_values, loss_values, mu_rec_est)

        huber_id = self._compute_huber(model(T_id, Z, z), loss_id) if len(T_id) > 0 else np.inf
        huber_ood = self._compute_huber(model(T_ood, Z, z), loss_ood) if len(T_ood) > 0 else np.inf

        return (float(Z), float(z), float(huber_id), float(huber_ood))

    def compare_all_forms(
        self,
        T_values: np.ndarray,
        loss_values: np.ndarray,
        mu_rec: float,
    ) -> Dict[str, Dict]:
        T_values = np.asarray(T_values, dtype=np.float64)
        loss_values = np.asarray(loss_values, dtype=np.float64)

        L_inf_exp, Z_exp, z_exp, hid_exp, hood_exp = self.fit_exponential(T_values, loss_values)
        L_inf_sp, Z_sp, z_sp, hid_sp, hood_sp = self.fit_stretched_power(T_values, loss_values)
        L_inf_pw, Z_pw, z_pw, hid_pw, hood_pw = self.fit_power(T_values, loss_values)
        Z_pp, z_pp, hid_pp, hood_pp = self.fit_pure_power(T_values, loss_values)

        floor_error_exp = abs(L_inf_exp - loss_values[T_values <= mu_rec].min()) / loss_values[T_values <= mu_rec].min() * 100 if np.any(T_values <= mu_rec) else np.inf
        floor_error_sp = abs(L_inf_sp - loss_values[T_values <= mu_rec].min()) / loss_values[T_values <= mu_rec].min() * 100 if np.any(T_values <= mu_rec) else np.inf
        floor_error_pw = abs(L_inf_pw - loss_values[T_values <= mu_rec].min()) / loss_values[T_values <= mu_rec].min() * 100 if np.any(T_values <= mu_rec) else np.inf

        return {
            "exponential": {
                "params": {"L_inf": L_inf_exp, "Z": Z_exp, "z": z_exp},
                "huber_id": hid_exp,
                "huber_ood": hood_exp,
                "floor_error_pct": floor_error_exp,
            },
            "stretched_power": {
                "params": {"L_inf": L_inf_sp, "Z": Z_sp, "z": z_sp},
                "huber_id": hid_sp,
                "huber_ood": hood_sp,
                "floor_error_pct": floor_error_sp,
            },
            "power": {
                "params": {"L_inf": L_inf_pw, "Z": Z_pw, "z": z_pw},
                "huber_id": hid_pw,
                "huber_ood": hood_pw,
                "floor_error_pct": floor_error_pw,
            },
            "pure_power": {
                "params": {"Z": Z_pp, "z": z_pp},
                "huber_id": hid_pp,
                "huber_ood": hood_pp,
                "floor_error_pct": None,
            },
        }
