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
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer

MU_REC_VALUES = [2, 4, 6, 8, 10, 12]
T_EVAL_RANGE = list(range(1, 25))
FIXED_TOKENS = 11_200_000_000


def exponential_decay(T, L_inf, L_0, rate):
    return L_inf + (L_0 - L_inf) * math.exp(-rate * T)


def train_model_fixed_tokens(model, lr, device, target_tokens, seq_len=128, micro_bs=8):
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()
        logits = model(data, recurrence_steps=model.mu_rec)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return model


def eval_at_T(model, device, vocab_size, T, seq_len=256, num_batches=20, micro_bs=16):
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


def fit_exponential_decay(T_vals, loss_vals):
    T_arr = np.array(T_vals, dtype=np.float64)
    L_arr = np.array(loss_vals, dtype=np.float64)
    L_inf_init = L_arr[-1]
    L_0_init = L_arr[0]
    rate_init = 0.5
    try:
        popt, pcov = curve_fit(
            exponential_decay, T_arr, L_arr,
            p0=[L_inf_init, L_0_init, rate_init],
            bounds=([0, 0, 0], [np.inf, np.inf, 10.0]),
            maxfev=20000,
        )
        L_inf_fit, L_0_fit, rate_fit = popt
        fitted = [exponential_decay(t, *popt) for t in T_vals]
        residuals = np.array(loss_vals) - np.array(fitted)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((L_arr - np.mean(L_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "L_inf": L_inf_fit,
            "L_0": L_0_fit,
            "rate": rate_fit,
            "r_squared": r_squared,
            "fitted_vals": fitted,
        }
    except Exception:
        return {"L_inf": None, "L_0": None, "rate": None, "r_squared": 0.0, "fitted_vals": None}


def run_test_time_saturation(config_path, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]
    config_name = mc["config_name"]
    vocab_size = mc["vocab_size"]
    max_seq_len = mc["max_seq_len"]
    mu_bwd = mc["mu_bwd"]
    lr = config["training"]["adamw_lr"]

    all_results = {}

    for mu_rec in MU_REC_VALUES:
        print(f"\n[TestTimeSat] {config_name} mu_rec={mu_rec}")

        model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
        model = train_model_fixed_tokens(model, lr, device, FIXED_TOKENS)

        T_results = []
        for T in T_EVAL_RANGE:
            ppl = eval_at_T(model, device, vocab_size, T)
            T_results.append({"T": T, "val_ppl": ppl})
            print(f"  T={T:3d}: PPL={ppl:.4f}")

        fit = fit_exponential_decay([r["T"] for r in T_results], [r["val_ppl"] for r in T_results])

        L_inf_predicted = fit["L_inf"]
        L_at_mu_rec = None
        for r in T_results:
            if r["T"] == mu_rec:
                L_at_mu_rec = r["val_ppl"]
                break

        error_pct = None
        if L_inf_predicted is not None and L_at_mu_rec is not None and L_at_mu_rec > 0:
            error_pct = abs(L_inf_predicted - L_at_mu_rec) / L_at_mu_rec * 100

        all_results[str(mu_rec)] = {
            "per_T_results": T_results,
            "exponential_fit": {
                "L_inf": fit["L_inf"],
                "L_0": fit["L_0"],
                "rate": fit["rate"],
                "r_squared": fit["r_squared"],
            },
            "L_at_mu_rec": L_at_mu_rec,
            "L_inf_predicted": L_inf_predicted,
            "error_pct": error_pct,
            "within_0.6pct": error_pct is not None and error_pct <= 0.6,
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        Ts = [r["T"] for r in T_results]
        PPLs = [r["val_ppl"] for r in T_results]
        ax.plot(Ts, PPLs, "bo-", label="Measured", markersize=5)
        if fit["fitted_vals"] is not None:
            T_fine = np.linspace(1, 24, 100)
            PPL_fine = [exponential_decay(t, fit["L_inf"], fit["L_0"], fit["rate"]) for t in T_fine]
            ax.plot(T_fine, PPL_fine, "r--", label=f"Fit (R²={fit['r_squared']:.4f})")
            ax.axhline(y=fit["L_inf"], color="g", linestyle=":", label=f"L_inf={fit['L_inf']:.4f}")
        ax.axvline(x=mu_rec, color="orange", linestyle="--", label=f"mu_rec={mu_rec}")
        ax.set_xlabel("T (recurrence steps at test time)")
        ax.set_ylabel("Validation PPL")
        ax.set_title(f"Test-Time Saturation: {config_name} mu_rec={mu_rec}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = Path(output_dir) / f"saturation_{config_name}_mu{mu_rec}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    summary = {}
    verification_passes = 0
    total_checks = 0
    for mu_rec_str, result in all_results.items():
        summary[mu_rec_str] = {
            "L_inf": result["L_inf_predicted"],
            "L_at_mu_rec": result["L_at_mu_rec"],
            "error_pct": result["error_pct"],
            "within_0.6pct": result["within_0.6pct"],
        }
        total_checks += 1
        if result["within_0.6pct"]:
            verification_passes += 1

    print(f"\n=== Verification Summary ===")
    print(f"  Passed: {verification_passes}/{total_checks} within 0.6% error")

    results_path = Path(output_dir) / f"test_time_saturation_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({"per_mu_rec": all_results, "summary": summary, "verification_passes": verification_passes, "total_checks": total_checks}, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_test_time_saturation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
