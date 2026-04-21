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
from torch.distributions import Poisson

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer
from training.sampler import RecurrenceDepthSampler, TruncatedBaselineSampler

FIXED_TOKENS = 10_000_000_000
MU_REC = 8
MU_BWD = 8


def train_full_backprop(model, lr, device, target_tokens, seq_len=128, micro_bs=8):
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []
    wall_time = 0.0

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        t0 = time.time()
        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()
        logits = model(data, recurrence_steps=MU_REC)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        wall_time += time.time() - t0

        if step % 500 == 0:
            losses.append({"step": step, "loss": loss.item()})

    return model, losses, wall_time


def train_truncated_baseline(model, lr, device, target_tokens, seq_len=128, micro_bs=8):
    sampler = TruncatedBaselineSampler(mu_rec=MU_REC, mu_bwd=MU_BWD)
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []
    wall_time = 0.0

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        t0 = time.time()
        T, nograd, grad = sampler.sample_batch(batch_size=micro_bs, device=device)
        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()

        with torch.no_grad():
            h = model.embedding(data)
            for _ in range(MU_REC - MU_BWD):
                for block in model.recurrent_blocks:
                    norm1, attn, norm2, mlp = block
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
                A_bar_vec, B_bar = model.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        h = h.detach().requires_grad_(True)
        for _ in range(MU_BWD):
            for block in model.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            A_bar_vec, B_bar = model.injection()
            h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        for block in model.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        h = model.output_norm(h)
        logits = model.lm_head(h)

        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        wall_time += time.time() - t0

        if step % 500 == 0:
            losses.append({"step": step, "loss": loss.item()})

    return model, losses, wall_time


def train_corrected_algorithm(model, lr, device, target_tokens, seq_len=128, micro_bs=8):
    sampler = RecurrenceDepthSampler(mu_rec=MU_REC, mu_bwd=MU_BWD)
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []
    wall_time = 0.0

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        t0 = time.time()
        T, nograd, grad = sampler.sample_batch(batch_size=micro_bs, device=device)
        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()

        with torch.no_grad():
            h = model.embedding(data)
            for block in model.prelude_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            h = model.prelude_norm(h)

            max_nograd = nograd.max().item()
            for _ in range(int(max_nograd)):
                for block in model.recurrent_blocks:
                    norm1, attn, norm2, mlp = block
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
                A_bar_vec, B_bar = model.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        h = h.detach().requires_grad_(True)
        max_grad = grad.max().item()
        for _ in range(int(max_grad)):
            for block in model.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            A_bar_vec, B_bar = model.injection()
            h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        for block in model.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        h = model.output_norm(h)
        logits = model.lm_head(h)

        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        wall_time += time.time() - t0

        if step % 500 == 0:
            losses.append({"step": step, "loss": loss.item()})

    return model, losses, wall_time


def eval_val_ppl(model, device, vocab_size, seq_len=256, num_batches=20, micro_bs=16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            data = torch.randint(0, vocab_size, (micro_bs, seq_len), device=device)
            labels = data[:, 1:].contiguous().view(-1)
            logits = model(data, recurrence_steps=MU_REC)
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab_size), labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.numel()
    return math.exp(total_loss / total_tokens)


def run_sampling_algo_ablation(config_path, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]
    config_name = mc["config_name"]
    vocab_size = mc["vocab_size"]
    max_seq_len = mc["max_seq_len"]
    lr = config["training"]["adamw_lr"]

    all_results = {}

    print("[SamplingAlgo] Training full backprop model...")
    model_a = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=MU_REC, vocab_size=vocab_size)
    model_a, losses_a, time_a = train_full_backprop(model_a, lr, device, FIXED_TOKENS)
    ppl_a = eval_val_ppl(model_a, device, vocab_size)
    print(f"  Full backprop: val_ppl={ppl_a:.4f}, time={time_a:.1f}s")

    print("[SamplingAlgo] Training truncated baseline model...")
    model_b = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=MU_REC, vocab_size=vocab_size)
    model_b, losses_b, time_b = train_truncated_baseline(model_b, lr, device, FIXED_TOKENS)
    ppl_b = eval_val_ppl(model_b, device, vocab_size)
    print(f"  Truncated baseline: val_ppl={ppl_b:.4f}, time={time_b:.1f}s")

    print("[SamplingAlgo] Training corrected algorithm model...")
    model_c = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=MU_REC, vocab_size=vocab_size)
    model_c, losses_c, time_c = train_corrected_algorithm(model_c, lr, device, FIXED_TOKENS)
    ppl_c = eval_val_ppl(model_c, device, vocab_size)
    print(f"  Corrected algorithm: val_ppl={ppl_c:.4f}, time={time_c:.1f}s")

    corrected_matches_full = abs(ppl_c - ppl_a) / ppl_a < 0.05
    corrected_outperforms_truncated = ppl_c < ppl_b

    results = {
        "full_backprop": {"val_ppl": ppl_a, "wall_time_s": time_a, "final_losses": [l["loss"] for l in losses_a[-5:]]},
        "truncated_baseline": {"val_ppl": ppl_b, "wall_time_s": time_b, "final_losses": [l["loss"] for l in losses_b[-5:]]},
        "corrected_algorithm": {"val_ppl": ppl_c, "wall_time_s": time_c, "final_losses": [l["loss"] for l in losses_c[-5:]]},
        "verification": {
            "corrected_matches_full_backprop_5pct": corrected_matches_full,
            "corrected_outperforms_truncated": corrected_outperforms_truncated,
            "ppl_gap_corrected_vs_full_pct": abs(ppl_c - ppl_a) / ppl_a * 100,
            "ppl_gap_corrected_vs_truncated_pct": (ppl_b - ppl_c) / ppl_b * 100,
        },
    }

    print(f"\n=== Verification ===")
    print(f"  Corrected matches full backprop (5%): {corrected_matches_full}")
    print(f"  Corrected outperforms truncated: {corrected_outperforms_truncated}")
    print(f"  Corrected vs Full gap: {results['verification']['ppl_gap_corrected_vs_full_pct']:.2f}%")
    print(f"  Corrected vs Truncated gap: {results['verification']['ppl_gap_corrected_vs_truncated_pct']:.2f}%")

    results_path = Path(output_dir) / f"sampling_algo_ablation_{config_name}_results.json"
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
    run_sampling_algo_ablation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
