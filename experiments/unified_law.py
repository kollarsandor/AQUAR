import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import os
import math
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from scipy.optimize import minimize, curve_fit
from matplotlib import pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer
from scaling.flops import effective_params, training_flops

T_EVAL_RANGE = list(range(1, 17))
MAPE_ORACLE_THRESHOLD = 0.17
MAPE_PREDICTED_THRESHOLD = 1.31


def unified_law_loss(T, L_inf, alpha, beta):
    return L_inf + alpha * math.exp(-beta * T)


def eval_at_T(model, device, vocab_size, T, seq_len=256, num_batches=15, micro_bs=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            data = torch.randint(0, vocab_size, (micro_bs, seq_len), device=device)
            labels = data[:, 1:].contiguous().view(-1)
            logits = model(data, recurrence_steps=T)
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab_size), labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    return math.exp(total_loss / total_tokens)


def fit_unified_law(T_vals, loss_vals):
    T_arr = np.array(T_vals, dtype=np.float64)
    L_arr = np.array(loss_vals, dtype=np.float64)
    L_inf_init = L_arr[-1]
    alpha_init = L_arr[0] - L_arr[-1]
    beta_init = 0.5
    try:
        popt, pcov = curve_fit(
            unified_law_loss, T_arr, L_arr,
            p0=[L_inf_init, alpha_init, beta_init],
            bounds=([0, 0, 0.01], [np.inf, np.inf, 10.0]),
            maxfev=20000,
        )
        L_inf_fit, alpha_fit, beta_fit = popt
        fitted = [unified_law_loss(t, *popt) for t in T_vals]
        residuals = np.array(loss_vals) - np.array(fitted)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((L_arr - np.mean(L_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "L_inf": L_inf_fit,
            "alpha": alpha_fit,
            "beta": beta_fit,
            "r_squared": r_squared,
            "fitted_vals": fitted,
        }
    except Exception:
        return {"L_inf": None, "alpha": None, "beta": None, "r_squared": 0.0, "fitted_vals": None}


def predict_law_params(N_eff, D, a_N, b_N, a_D, b_D, c):
    L_inf = a_N / (N_eff ** b_N) + c
    alpha = a_D / (D ** b_D)
    beta = 0.5
    return L_inf, alpha, beta


def compute_mape(predicted, actual):
    if len(predicted) != len(actual):
        return float("inf")
    errors = []
    for p, a in zip(predicted, actual):
        if a > 0:
            errors.append(abs(p - a) / a * 100)
    return sum(errors) / len(errors) if errors else float("inf")


def run_unified_law_validation(isoflop_results_dir, heldout_results_dir, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    isoflop_path = Path(isoflop_results_dir)
    if not isoflop_path.exists():
        print(f"IsoFLOP results not found at {isoflop_results_dir}")
        return {}

    fitting_data = {}
    for results_file in sorted(isoflop_path.glob("isoflop_sweep_*_results.json")):
        with open(results_file) as f:
            results = json.load(f)
        config_name = results_file.stem.replace("isoflop_sweep_", "").replace("_results", "")
        fitting_data[config_name] = results

    law_fits = {}
    for config_name, results in fitting_data.items():
        law_fits[config_name] = {}
        for budget_key, budget_data in results.items():
            if not isinstance(budget_data, dict):
                continue
            mu_vals = [int(k) for k in budget_data if k not in ("baseline", "parabolic_fit")]
            if len(mu_vals) < 3:
                continue
            loss_vals = [budget_data[str(m)]["val_ppl"] for m in mu_vals]
            fit = fit_unified_law(mu_vals, loss_vals)
            law_fits[config_name][budget_key] = fit
            n1, n2, total_params = effective_params(
                LoopedTransformer(config_name, max_seq_len=128, mu_rec=8, vocab_size=32768),
                8, 4
            )
            tokens = budget_data.get(str(mu_vals[0]), {}).get("tokens", 1e9)
            print(f"  {config_name}/{budget_key}: L_inf={fit['L_inf']:.4f}, alpha={fit.get('alpha', 0):.4f}, beta={fit.get('beta', 0):.4f}, R²={fit['r_squared']:.4f}")

    heldout_path = Path(heldout_results_dir)
    validation_results = {}
    if heldout_path.exists():
        for results_file in sorted(heldout_path.glob("quality_vs_fixed_*_results.json")):
            with open(results_file) as f:
                heldout = json.load(f)

    oracle_mapes = []
    predicted_mapes = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    validation_results = {
        "oracle_mape_per_model": {},
        "predicted_mape_per_model": {},
        "summary": {},
    }

    for config_name, fits in law_fits.items():
        for budget_key, fit in fits.items():
            if fit["L_inf"] is None:
                continue
            mu_vals = [int(k) for k in fitting_data[config_name][budget_key] if k not in ("baseline", "parabolic_fit")]
            loss_vals = [fitting_data[config_name][budget_key][str(m)]["val_ppl"] for m in mu_vals]
            if fit["fitted_vals"] is not None:
                oracle_mape = compute_mape(fit["fitted_vals"], loss_vals)
                oracle_mapes.append(oracle_mape)
                validation_results["oracle_mape_per_model"][f"{config_name}_{budget_key}"] = oracle_mape

    if len(oracle_mapes) > 0:
        mean_oracle_mape = sum(oracle_mapes) / len(oracle_mapes)
        max_oracle_mape = max(oracle_mapes)
        print(f"\n=== Unified Law Validation ===")
        print(f"  Oracle floor MAPE: mean={mean_oracle_mape:.4f}%, max={max_oracle_mape:.4f}%")
        print(f"  Oracle threshold (≤0.17%): {'PASS' if mean_oracle_mape <= MAPE_ORACLE_THRESHOLD else 'FAIL'}")
        validation_results["summary"] = {
            "mean_oracle_mape": mean_oracle_mape,
            "max_oracle_mape": max_oracle_mape,
            "oracle_pass": mean_oracle_mape <= MAPE_ORACLE_THRESHOLD,
            "oracle_threshold": MAPE_ORACLE_THRESHOLD,
        }

    results_path = Path(output_dir) / "unified_law_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return validation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoflop-results-dir", type=str, required=True)
    parser.add_argument("--heldout-results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_unified_law_validation(args.isoflop_results_dir, args.heldout_results_dir, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
