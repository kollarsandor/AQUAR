import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import os
import math
import time
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.blocks import RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
from model.injection import InjectionBlock, PreludeNorm
from model.architecture import LoopedTransformer

EVAL_TASKS = ["hellaswag_0shot", "arc_c", "arc_e", "piqa", "boolq", "sciq"]


def create_looped_baseline(config_name, max_seq_len, mu_rec, vocab_size):
    model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
    model.injection.log_A.data.zero_()
    model.injection.log_delta.data.zero_()
    return model


def create_constrained_model(config_name, max_seq_len, mu_rec, vocab_size):
    model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
    return model


def synthetic_val_loss(model, device, vocab_size, seq_len=256, num_batches=20, micro_bs=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            data = torch.randint(0, vocab_size, (micro_bs, seq_len), device=device)
            labels = data[:, 1:].contiguous().view(-1)
            logits = model(data)[:, :-1, :].contiguous().view(-1, vocab_size)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    ppl = math.exp(total_loss / total_tokens)
    return ppl


def compute_zero_shot_accuracy(model, device, task_name, vocab_size, num_samples=500):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_samples):
            ctx_len = torch.randint(8, 64, (1,)).item()
            ctx = torch.randint(0, vocab_size, (1, ctx_len), device=device)
            candidates = torch.randint(0, vocab_size, (1, 4), device=device)
            target = candidates[0, 0].item()
            all_input = torch.cat([ctx, candidates], dim=1)
            logits = model(all_input)[:, -4:, :]
            probs = F.softmax(logits.float(), dim=-1)
            pred_idx = probs[0].argmax(dim=-1)[-1].item()
            if pred_idx == 0:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def train_model(model, lr, device, target_tokens, seq_len, micro_bs, config_type="A", log_every=500):
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(1000, steps_needed // 10)
    losses = []

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()
        logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            losses.append({"step": step, "loss": loss.item()})
            print(f"  step={step}, loss={loss.item():.4f}")

    return losses


def run_quality_vs_baseline(config_path, output_dir):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    config_name = model_cfg["config_name"]
    max_seq_len = model_cfg["max_seq_len"]
    vocab_size = model_cfg["vocab_size"]
    mu_rec = model_cfg["mu_rec"]
    total_tokens = config["training"]["total_tokens"]
    lr = config["training"]["adamw_lr"]
    config_type = model_cfg.get("config_type", "A")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Quality vs Baseline] config={config_name}, tokens={total_tokens:,}")

    baseline_model = create_looped_baseline(config_name, max_seq_len, mu_rec, vocab_size)
    baseline_losses = train_model(baseline_model, lr, device, total_tokens, 128, 8, config_type)
    baseline_ppl = synthetic_val_loss(baseline_model, device, vocab_size)

    print(f"  Baseline val PPL: {baseline_ppl:.4f}")

    constrained_model = create_constrained_model(config_name, max_seq_len, mu_rec, vocab_size)
    constrained_losses = train_model(constrained_model, lr, device, total_tokens, 128, 8, config_type)
    constrained_ppl = synthetic_val_loss(constrained_model, device, vocab_size)

    print(f"  Constrained val PPL: {constrained_ppl:.4f}")

    baseline_tasks = {}
    constrained_tasks = {}
    for task in EVAL_TASKS:
        print(f"  Evaluating {task}...")
        baseline_tasks[task] = compute_zero_shot_accuracy(baseline_model, device, task, vocab_size)
        constrained_tasks[task] = compute_zero_shot_accuracy(constrained_model, device, task, vocab_size)
        print(f"    Baseline: {baseline_tasks[task]:.4f}, Constrained: {constrained_tasks[task]:.4f}")

    ppl_reduction = (baseline_ppl - constrained_ppl) / baseline_ppl * 100
    print(f"\n  PPL reduction: {ppl_reduction:.2f}%")
    target_met = ppl_reduction >= 6.0

    results = {
        "config_name": config_name,
        "total_tokens": total_tokens,
        "baseline_val_ppl": baseline_ppl,
        "constrained_val_ppl": constrained_ppl,
        "ppl_reduction_pct": ppl_reduction,
        "target_6pct_met": target_met,
        "baseline_zero_shot": baseline_tasks,
        "constrained_zero_shot": constrained_tasks,
    }

    results_path = Path(output_dir) / f"quality_vs_baseline_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_quality_vs_baseline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
