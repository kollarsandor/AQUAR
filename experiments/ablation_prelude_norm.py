import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import os
import math
from pathlib import Path
from torch.optim import AdamW
from matplotlib import pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer
from model.blocks import RMSNorm

FIXED_TOKENS = 104_000_000_000


def log_spectral_norm(matrix):
    if matrix.ndim == 1:
        return torch.max(torch.abs(matrix)).item()
    try:
        s = torch.linalg.svdvals(matrix.float())
        return s[0].item()
    except Exception:
        return float("inf")


def train_with_logging(model, lr, device, target_tokens, use_prelude_norm, seq_len=128, micro_bs=8):
    model = model.to(device).train()
    if not use_prelude_norm and hasattr(model, "prelude_norm"):
        model.prelude_norm = nn.Identity()

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(4000, steps_needed // 10)

    diagnostics = {
        "step": [],
        "loss": [],
        "spectral_norm_A_bar": [],
        "spectral_norm_B_bar": [],
        "spectral_norm_C_proj": [],
        "activation_norm": [],
        "residual_norm": [],
    }

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()
        logits = model(data, recurrence_steps=model.mu_rec)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))

        if step % 100 == 0:
            with torch.no_grad():
                A_bar_vec, B_bar = model.injection()
                sn_A_bar = log_spectral_norm(A_bar_vec)
                sn_B_bar = log_spectral_norm(B_bar)
                sn_C_proj = log_spectral_norm(model.lm_head.weight.float())
                h = model.embedding(data)
                if hasattr(model, "prelude_norm") and not isinstance(model.prelude_norm, nn.Identity):
                    h = model.prelude_norm(h)
                act_norm = torch.norm(h.float(), dim=-1).mean().item()
                res_norm = torch.norm(h.float(), dim=-1).max().item()

            diagnostics["step"].append(step)
            diagnostics["loss"].append(loss.item())
            diagnostics["spectral_norm_A_bar"].append(sn_A_bar)
            diagnostics["spectral_norm_B_bar"].append(sn_B_bar)
            diagnostics["spectral_norm_C_proj"].append(sn_C_proj)
            diagnostics["activation_norm"].append(act_norm)
            diagnostics["residual_norm"].append(res_norm)

            if step % 10000 == 0:
                print(f"  step={step}, loss={loss.item():.4f}, sn_A={sn_A_bar:.4f}, act_norm={act_norm:.4f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return model, diagnostics


def plot_diagnostics(diag_with, diag_without, output_dir, config_name):
    metrics = [
        ("loss", "Training Loss"),
        ("spectral_norm_A_bar", "Spectral Norm A_bar"),
        ("spectral_norm_B_bar", "Spectral Norm B_bar"),
        ("activation_norm", "Activation Norm"),
        ("residual_norm", "Residual Norm (max)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(diag_without["step"], diag_without[metric], "r-", alpha=0.7, label="Without Prelude-Norm")
        ax.plot(diag_with["step"], diag_with[metric], "b-", alpha=0.7, label="With Prelude-Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: {config_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"prelude_norm_ablation_{config_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(plot_path)


def run_prelude_norm_ablation(config_path, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]
    config_name = mc["config_name"]
    vocab_size = mc["vocab_size"]
    max_seq_len = mc["max_seq_len"]
    mu_rec = mc["mu_rec"]
    lr = config["training"]["adamw_lr"]

    print(f"[PreludeNormAblation] {config_name}")

    print("  Training WITHOUT prelude-norm...")
    model_without = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
    _, diag_without = train_with_logging(model_without, lr, device, FIXED_TOKENS, use_prelude_norm=False)

    late_sn_A = diag_without["spectral_norm_A_bar"][-10:]
    late_act_norm = diag_without["activation_norm"][-10:]
    late_res_norm = diag_without["residual_norm"][-10:]

    print("  Training WITH prelude-norm...")
    model_with = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
    _, diag_with = train_with_logging(model_with, lr, device, FIXED_TOKENS, use_prelude_norm=True)

    plot_path = plot_diagnostics(diag_with, diag_without, output_dir, config_name)

    late_sn_A_with = diag_with["spectral_norm_A_bar"][-10:]
    late_act_norm_with = diag_with["activation_norm"][-10:]
    late_res_norm_with = diag_with["residual_norm"][-10:]

    def exploded(values):
        if not values:
            return False
        recent = values[-10:]
        if len(recent) < 2:
            return False
        return recent[-1] > recent[0] * 2

    without_stable = not exploded(late_res_norm)
    with_stable = not exploded(late_res_norm_with)

    results = {
        "config_name": config_name,
        "without_prelude_norm": {
            "final_loss": diag_without["loss"][-1],
            "final_spectral_norm_A_bar": late_sn_A,
            "final_activation_norm": late_act_norm,
            "final_residual_norm": late_res_norm,
            "late_training_residual_explosion": exploded(late_res_norm),
        },
        "with_prelude_norm": {
            "final_loss": diag_with["loss"][-1],
            "final_spectral_norm_A_bar": late_sn_A_with,
            "final_activation_norm": late_act_norm_with,
            "final_residual_norm": late_res_norm_with,
            "late_training_residual_explosion": exploded(late_res_norm_with),
        },
        "verification": {
            "without_prelude_norm_explosion_detected": not without_stable,
            "with_prelude_norm_stable": with_stable,
            "prelude_norm_prevents_explosion": (not without_stable) and with_stable,
        },
    }

    print(f"\n=== Verification ===")
    print(f"  Without prelude-norm explosion: {results['verification']['without_prelude_norm_explosion_detected']}")
    print(f"  With prelude-norm stable: {results['verification']['with_prelude_norm_stable']}")
    print(f"  Prelude-norm prevents explosion: {results['verification']['prelude_norm_prevents_explosion']}")

    results_path = Path(output_dir) / f"prelude_norm_ablation_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_prelude_norm_ablation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
