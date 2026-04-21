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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer

MU_REC_VALUES = [4, 8, 14, 20, 26, 32]
T_EVAL_VALUES = [1, 4, 8, 16, 64]
FIXED_TOKENS = 10_000_000_000


def train_model(model, lr, device, target_tokens, seq_len=128, micro_bs=8):
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


def run_mu_rec_ablation(config_path, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]
    config_name = mc["config_name"]
    vocab_size = mc["vocab_size"]
    max_seq_len = mc["max_seq_len"]
    mu_bwd = 4
    lr = config["training"]["adamw_lr"]

    all_results = {}

    for mu_rec in MU_REC_VALUES:
        print(f"\n[MuRecAblation] {config_name} mu_rec={mu_rec}")

        model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
        model = train_model(model, lr, device, FIXED_TOKENS)

        T_results = {}
        for T in T_EVAL_VALUES:
            ppl = eval_at_T(model, device, vocab_size, T)
            T_results[str(T)] = ppl
            print(f"  T={T:3d}: PPL={ppl:.4f}")

        all_results[str(mu_rec)] = {
            "mu_rec": mu_rec,
            "mu_bwd": mu_bwd,
            "per_T_ppl": T_results,
        }

    results_path = Path(output_dir) / f"ablation_mu_rec_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_mu_rec_ablation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
