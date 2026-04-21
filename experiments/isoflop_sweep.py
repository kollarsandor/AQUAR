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

from model.architecture import LoopedTransformer, FixedDepthTransformer
from scaling.flops import effective_params, training_flops

MU_REC_VALUES = [2, 4, 6, 8, 10, 12]

FLOP_BUDGETS_140M = [1e18, 2e18, 4e18, 8e18, 16e18, 64e18]
FLOP_BUDGETS_370M = [32e18, 64e18, 128e18]


def compute_tokens_for_flop_budget(model, mu_rec, mu_bwd, flop_budget):
    n1, n2, total_params = effective_params(model, mu_rec, mu_bwd)
    seq_len = 2048
    tokens_per_sample = seq_len
    flops_per_token = 6 * total_params
    tokens = flop_budget / max(1.0, flops_per_token)
    return int(tokens)


def train_model_to_flop_budget(model, lr, device, target_tokens, mu_rec, mu_bwd, seq_len=128, micro_bs=8, log_every=500):
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()
        logits = model(data, recurrence_steps=mu_rec)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            losses.append({"step": step, "loss": loss.item()})

    return model, losses


def eval_val_ppl(model, device, vocab_size, seq_len=256, num_batches=20, micro_bs=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            data = torch.randint(0, vocab_size, (micro_bs, seq_len), device=device)
            labels = data[:, 1:].contiguous().view(-1)
            logits = model(data, recurrence_steps=model.mu_rec)
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab_size), labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    return math.exp(total_loss / total_tokens)


def parabolic_law(x, a, b, c):
    return a / (x ** b) + c


def parametric_scaling_loss(N, D, a, b, c):
    return a / (N ** 0.34) + b / (D ** 0.5) + c


def fit_parabolic_law(mu_rec_values, val_losses):
    mu_arr = np.array(mu_rec_values, dtype=np.float64)
    loss_arr = np.array(val_losses, dtype=np.float64)
    try:
        popt, pcov = curve_fit(parabolic_law, mu_arr, loss_arr, p0=[1.0, 0.5, 2.0], maxfev=10000)
        fitted_losses = parabolic_law(mu_arr, *popt)
        residuals = loss_arr - fitted_losses
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((loss_arr - np.mean(loss_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"params": {"a": popt[0], "b": popt[1], "c": popt[2]}, "r_squared": r_squared}
    except Exception:
        return {"params": None, "r_squared": 0.0}


def create_isoloss_contour_plot(results_by_budget, model_size, output_dir):
    all_mu = []
    all_loss = []
    all_budget = []
    for budget_key, budget_results in results_by_budget.items():
        for mu, metrics in budget_results.items():
            all_mu.append(int(mu))
            all_loss.append(metrics["val_ppl"])
            all_budget.append(float(budget_key))

    fig, ax = plt.subplots(figsize=(10, 6))
    mu_unique = sorted(set(all_mu))
    budget_unique = sorted(set(all_budget))
    loss_grid = np.full((len(budget_unique), len(mu_unique)), np.nan)

    budget_to_idx = {b: i for i, b in enumerate(budget_unique)}
    mu_to_idx = {m: i for i, m in enumerate(mu_unique)}

    for mu, loss, budget in zip(all_mu, all_loss, all_budget):
        bi = budget_to_idx[budget]
        mi = mu_to_idx[mu]
        loss_grid[bi, mi] = loss

    cs = ax.contour(mu_unique, [b / 1e18 for b in budget_unique], loss_grid, levels=15, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("mu_rec")
    ax.set_ylabel("FLOP Budget (x10^18)")
    ax.set_title(f"IsoLoss Contour: {model_size}")
    plt.colorbar(cs, ax=ax, label="Validation PPL")
    plot_path = Path(output_dir) / f"isoloss_contour_{model_size}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(plot_path)


def run_isoflop_sweep(config_path, output_dir, device_str="cuda:0"):
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

    if config_name in ["140M"]:
        flop_budgets = FLOP_BUDGETS_140M
    elif config_name in ["370M"]:
        flop_budgets = FLOP_BUDGETS_370M
    else:
        flop_budgets = FLOP_BUDGETS_140M

    all_results = {}

    for budget in flop_budgets:
        budget_key = f"{budget:.0e}"
        all_results[budget_key] = {}

        for mu_rec in MU_REC_VALUES:
            print(f"[IsoFLOP] {config_name} budget={budget:.1e} mu_rec={mu_rec}")

            model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
            tokens = compute_tokens_for_flop_budget(model, mu_rec, mu_bwd, budget)
            actual_flops = training_flops(model, mu_rec, mu_bwd, tokens)
            print(f"  tokens={tokens:,}, actual_flops={actual_flops['total_flops']:.2e}")

            model, losses = train_model_to_flop_budget(model, lr, device, tokens, mu_rec, mu_bwd)
            val_ppl = eval_val_ppl(model, device, vocab_size)
            print(f"  val_ppl={val_ppl:.4f}")

            all_results[budget_key][str(mu_rec)] = {
                "val_ppl": val_ppl,
                "tokens": tokens,
                "actual_flops": actual_flops["total_flops"],
            }

        print(f"  Training fixed-depth baseline at budget={budget:.1e}...")
        baseline_model = FixedDepthTransformer(config_name, max_seq_len=max_seq_len, vocab_size=vocab_size)
        tokens_baseline = compute_tokens_for_flop_budget(baseline_model, 1, 1, budget)
        baseline_model, _ = train_model_to_flop_budget(baseline_model, lr, device, tokens_baseline, 1, 1)
        baseline_ppl = eval_val_ppl(baseline_model, device, vocab_size)
        print(f"  baseline val_ppl={baseline_ppl:.4f}")
        all_results[budget_key]["baseline"] = {
            "val_ppl": baseline_ppl,
            "tokens": tokens_baseline,
        }

    for budget_key, budget_results in all_results.items():
        mu_vals = [int(k) for k in budget_results if k != "baseline"]
        loss_vals = [budget_results[k]["val_ppl"] for k in budget_results if k != "baseline"]
        if len(mu_vals) >= 3:
            fit = fit_parabolic_law(mu_vals, loss_vals)
            all_results[budget_key]["parabolic_fit"] = fit
            print(f"  Budget {budget_key}: parabolic fit R²={fit['r_squared']:.4f}")

    contour_path = create_isoloss_contour_plot(all_results, config_name, output_dir)
    print(f"  IsoLoss contour plot saved to {contour_path}")

    results_path = Path(output_dir) / f"isoflop_sweep_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_isoflop_sweep(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
