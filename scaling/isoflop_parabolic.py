from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .flops import training_flops


class IsoFLOPParabolicFit:
    def __init__(self):
        self.mu_rec_parabolas: Dict[float, Dict] = {}
        self.token_parabolas: Dict[float, Dict] = {}
        self.mu_rec_power_law: Optional[Tuple[float, float]] = None
        self.token_power_law: Optional[Tuple[float, float]] = None

    def fit_mu_rec_parabola(
        self,
        mu_rec_values: np.ndarray,
        val_losses: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        mu_rec_values = np.asarray(mu_rec_values, dtype=np.float64)
        val_losses = np.asarray(val_losses, dtype=np.float64)
        mu_rec_values = mu_rec_values[mu_rec_values > 0]
        val_losses = val_losses[: len(mu_rec_values)]
        val_losses = val_losses[val_losses > 0]

        log_mu = np.log(mu_rec_values)
        log_L = np.log(val_losses)

        A = np.column_stack([log_mu ** 2, log_mu, np.ones_like(log_mu)])
        result = np.linalg.lstsq(A, log_L, rcond=None)
        coeffs = result[0]
        a, b, c = coeffs

        if a > 0:
            log_mu_star = -b / (2.0 * a)
            mu_star = np.exp(log_mu_star)
        else:
            log_mu_min = np.log(mu_rec_values.min())
            log_mu_max = np.log(mu_rec_values.max())
            log_mu_star = 0.5 * (log_mu_min + log_mu_max)
            mu_star = np.exp(log_mu_star)

        predicted_log = a * log_mu ** 2 + b * log_mu + c
        residuals = log_L - predicted_log
        rmse = np.sqrt(np.mean(residuals ** 2))

        return (float(a), float(b), float(c), float(mu_star))

    def fit_token_parabola(
        self,
        token_values: np.ndarray,
        val_losses: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        token_values = np.asarray(token_values, dtype=np.float64)
        val_losses = np.asarray(val_losses, dtype=np.float64)
        token_values = token_values[token_values > 0]
        val_losses = val_losses[: len(token_values)]
        val_losses = val_losses[val_losses > 0]

        log_D = np.log(token_values)
        log_L = np.log(val_losses)

        A = np.column_stack([log_D ** 2, log_D, np.ones_like(log_D)])
        result = np.linalg.lstsq(A, log_L, rcond=None)
        coeffs = result[0]
        a, b, c = coeffs

        if a > 0:
            log_D_star = -b / (2.0 * a)
            D_star = np.exp(log_D_star)
        else:
            log_D_min = np.log(token_values.min())
            log_D_max = np.log(token_values.max())
            log_D_star = 0.5 * (log_D_min + log_D_max)
            D_star = np.exp(log_D_star)

        predicted_log = a * log_D ** 2 + b * log_D + c
        residuals = log_L - predicted_log
        rmse = np.sqrt(np.mean(residuals ** 2))

        return (float(a), float(b), float(c), float(D_star))

    def fit_power_law(
        self,
        flop_budgets: np.ndarray,
        optimal_values: np.ndarray,
    ) -> Tuple[float, float]:
        flop_budgets = np.asarray(flop_budgets, dtype=np.float64)
        optimal_values = np.asarray(optimal_values, dtype=np.float64)
        valid = (flop_budgets > 0) & (optimal_values > 0)
        flop_budgets = flop_budgets[valid]
        optimal_values = optimal_values[valid]

        log_C = np.log(flop_budgets)
        log_opt = np.log(optimal_values)

        A = np.column_stack([log_C, np.ones_like(log_C)])
        result = np.linalg.lstsq(A, log_opt, rcond=None)
        coeffs = result[0]
        exponent = float(coeffs[0])
        intercept = float(coeffs[1])

        predicted = exponent * log_C + intercept
        residuals = log_opt - predicted
        rmse = np.sqrt(np.mean(residuals ** 2))
        r_squared = 1.0 - np.sum(residuals ** 2) / np.sum((log_opt - np.mean(log_opt)) ** 2)

        return (exponent, intercept)

    def compute_iso_flop_budget(
        self,
        model,
        mu_rec: float,
        mu_bwd: float,
        D: float,
    ) -> float:
        result = training_flops(model, mu_rec, mu_bwd, D)
        return float(result["total_flops"])

    def fit_all_iso_flop_parabolas(
        self,
        iso_flop_data: Dict[float, List[Tuple[float, float]]],
    ) -> Dict[float, Dict]:
        all_results = {}
        for budget, points in iso_flop_data.items():
            mu_recs = np.array([p[0] for p in points])
            losses = np.array([p[1] for p in points])
            a, b, c, mu_star = self.fit_mu_rec_parabola(mu_recs, losses)
            all_results[budget] = {
                "a": a,
                "b": b,
                "c": c,
                "mu_star": mu_star,
                "flop_budget": budget,
            }
            self.mu_rec_parabolas[budget] = all_results[budget]
        return all_results

    def extract_power_laws_from_iso_flops(
        self,
        iso_flop_data: Dict[float, List[Tuple[float, float]]],
    ) -> Dict[str, Dict]:
        fit_results = self.fit_all_iso_flop_parabolas(iso_flop_data)

        budgets = []
        mu_stars = []
        for budget, result in sorted(fit_results.items()):
            budgets.append(budget)
            mu_stars.append(result["mu_star"])

        if len(budgets) >= 2:
            mu_exp, mu_int = self.fit_power_law(np.array(budgets), np.array(mu_stars))
            self.mu_rec_power_law = (mu_exp, mu_int)

        return {
            "mu_rec_exponent": self.mu_rec_power_law[0] if self.mu_rec_power_law else None,
            "mu_rec_intercept": self.mu_rec_power_law[1] if self.mu_rec_power_law else None,
            "iso_flop_fits": fit_results,
        }

    def predict_optimal_mu_rec(self, flop_budget: float) -> Optional[float]:
        if self.mu_rec_power_law is None:
            return None
        exponent, intercept = self.mu_rec_power_law
        return float(np.exp(exponent * np.log(flop_budget) + intercept))

    def predict_optimal_tokens(self, flop_budget: float) -> Optional[float]:
        if self.token_power_law is None:
            return None
        exponent, intercept = self.token_power_law
        return float(np.exp(exponent * np.log(flop_budget) + intercept))
