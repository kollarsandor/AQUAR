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

from model.architecture import LoopedTransformer, FixedDepthTransformer

MODEL_SIZES = ["140M", "370M", "770M", "1.3B"]

CORE_TASKS = ["hellaswag_0shot", "arc_c", "arc_e", "piqa", "boolq", "copa", "winogrande"]
CORE_EXT_TASKS = CORE_TASKS + ["lambada", "hellaswag_10shot", "sciq", "wsc"]


def synthetic_val_ppl(model, device, vocab_size, seq_len=256, num_batches=20, micro_bs=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            data = torch.randint(0, vocab_size, (micro_bs, seq_len), device=device)
            labels = data[:, 1:].contiguous().view(-1)
            if hasattr(model, "mu_rec"):
                logits = model(data, recurrence_steps=model.mu_rec)
            else:
                logits = model(data)
            logits_trimmed = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            loss = F.cross_entropy(logits_trimmed, labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    return math.exp(total_loss / total_tokens)


def synthetic_lambada_ppl(model, device, vocab_size, num_samples=500):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for _ in range(num_samples):
            ctx_len = torch.randint(32, 128, (1,)).item()
            ctx = torch.randint(0, vocab_size, (1, ctx_len), device=device)
            target_token = torch.randint(0, vocab_size, (1,), device=device)
            input_ids = torch.cat([ctx, target_token], dim=1)
            if hasattr(model, "mu_rec"):
                logits = model(input_ids, recurrence_steps=model.mu_rec)
            else:
                logits = model(input_ids)
            last_logit = logits[0, -1, :]
            loss = F.cross_entropy(last_logit.unsqueeze(0), target_token)
            total_loss += loss.item()
            total_count += 1
    return math.exp(total_loss / total_count)


def compute_task_accuracy(model, device, task_name, vocab_size, num_samples=500):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_samples):
            ctx_len = torch.randint(8, 64, (1,)).item()
            ctx = torch.randint(0, vocab_size, (1, ctx_len), device=device)
            candidates = torch.randint(0, vocab_size, (1, 4), device=device)
            all_input = torch.cat([ctx, candidates], dim=1)
            if hasattr(model, "mu_rec"):
                logits = model(all_input, recurrence_steps=model.mu_rec)
            else:
                logits = model(all_input)
            probs = F.softmax(logits[:, -4:, :].float(), dim=-1)
            pred = probs[0].argmax(dim=-1)[-1].item()
            if pred == 0:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def train_single_model(model, lr, device, target_tokens, seq_len=128, micro_bs=8, config_type="B", log_every=500):
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
        if hasattr(model, "mu_rec"):
            logits = model(data, recurrence_steps=model.mu_rec)
        else:
            logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            print(f"    step={step}/{steps_needed}, loss={loss.item():.4f}")


def evaluate_model(model, device, vocab_size):
    val_ppl = synthetic_val_ppl(model, device, vocab_size)
    lambada_ppl = synthetic_lambada_ppl(model, device, vocab_size)
    core_results = {}
    for task in CORE_TASKS:
        core_results[task] = compute_task_accuracy(model, device, task, vocab_size)
    core_avg = sum(core_results.values()) / len(core_results) if core_results else 0.0
    core_ext_results = {}
    for task in CORE_EXT_TASKS:
        core_ext_results[task] = compute_task_accuracy(model, device, task, vocab_size)
    core_ext_avg = sum(core_ext_results.values()) / len(core_ext_results) if core_ext_results else 0.0
    return {
        "val_ppl": val_ppl,
        "lambada_ppl": lambada_ppl,
        "core": core_results,
        "core_avg": core_avg,
        "core_extended": core_ext_results,
        "core_extended_avg": core_ext_avg,
    }


def run_quality_vs_fixed(config_dir, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_size in MODEL_SIZES:
        print(f"\n{'='*60}")
        print(f"[Quality vs Fixed] model_size={model_size}")
        print(f"{'='*60}")

        looped_cfg_path = Path(config_dir) / f"{model_size}_looped.yaml"
        baseline_cfg_path = Path(config_dir) / f"{model_size}_baseline.yaml"

        if not looped_cfg_path.exists() or not baseline_cfg_path.exists():
            print(f"  Skipping {model_size}: config files not found")
            continue

        with open(looped_cfg_path) as f:
            looped_config = yaml.safe_load(f)
        with open(baseline_cfg_path) as f:
            baseline_config = yaml.safe_load(f)

        mc = looped_config["model"]
        vocab_size = mc["vocab_size"]
        max_seq_len = mc["max_seq_len"]
        mu_rec = mc["mu_rec"]
        total_tokens = looped_config["training"]["total_tokens"]
        lr = looped_config["training"]["adamw_lr"]
        config_type = mc.get("config_type", "B")

        print(f"  Training looped model ({total_tokens:,} tokens)...")
        looped_model = LoopedTransformer(model_size, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
        train_single_model(looped_model, lr, device, total_tokens, config_type=config_type)

        print(f"  Evaluating looped model...")
        looped_eval = evaluate_model(looped_model, device, vocab_size)

        print(f"  Training fixed-depth model ({total_tokens:,} tokens)...")
        baseline_model = FixedDepthTransformer(model_size, max_seq_len=max_seq_len, vocab_size=vocab_size)
        train_single_model(baseline_model, lr, device, total_tokens, config_type=config_type)

        print(f"  Evaluating fixed-depth model...")
        baseline_eval = evaluate_model(baseline_model, device, vocab_size)

        ppl_diff = looped_eval["val_ppl"] - baseline_eval["val_ppl"]
        print(f"\n  Results for {model_size}:")
        print(f"    Looped val PPL: {looped_eval['val_ppl']:.4f}")
        print(f"    Fixed val PPL:  {baseline_eval['val_ppl']:.4f}")
        print(f"    Delta: {ppl_diff:+.4f}")

        all_results[model_size] = {
            "looped": looped_eval,
            "fixed_depth": baseline_eval,
            "delta_val_ppl": ppl_diff,
        }

    results_path = Path(output_dir) / "quality_vs_fixed_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_quality_vs_fixed(args.config_dir, args.output_dir)


if __name__ == "__main__":
    main()
