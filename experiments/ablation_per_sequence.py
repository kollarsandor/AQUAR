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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture import LoopedTransformer
from training.sampler import RecurrenceDepthSampler

T_EVAL_VALUES = [1, 4, 8, 16]


def train_per_microbatch(model, lr, device, target_tokens, mu_rec, mu_bwd, seq_len=128, micro_bs=8):
    sampler = RecurrenceDepthSampler(mu_rec=mu_rec, mu_bwd=mu_bwd)
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []
    wall_time = 0.0
    spike_count = 0
    last_loss = float("inf")

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        t0 = time.time()
        T_mb, nograd_mb, grad_mb = sampler.sample_microbatch(batch_size=micro_bs, device=device)
        T_val = T_mb[0].item()

        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()

        nograd_steps = int(nograd_mb[0].item())
        grad_steps = int(grad_mb[0].item())

        with torch.no_grad():
            h = model.embedding(data)
            for block in model.prelude_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            h = model.prelude_norm(h)

            for _ in range(nograd_steps):
                for block in model.recurrent_blocks:
                    norm1, attn, norm2, mlp = block
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
                A_bar_vec, B_bar = model.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        h = h.detach().requires_grad_(True)
        for _ in range(grad_steps):
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

        current_loss = loss.item()
        if current_loss > last_loss * 1.5 and step > warmup_steps:
            spike_count += 1
        last_loss = current_loss

        if step % 500 == 0:
            losses.append({"step": step, "loss": current_loss})

    return model, losses, wall_time, spike_count


def train_per_sequence(model, lr, device, target_tokens, mu_rec, mu_bwd, seq_len=128, micro_bs=8):
    sampler = RecurrenceDepthSampler(mu_rec=mu_rec, mu_bwd=mu_bwd)
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    warmup_steps = min(2000, steps_needed // 10)
    losses = []
    wall_time = 0.0
    spike_count = 0
    last_loss = float("inf")

    for step in range(steps_needed):
        lr_scale = min(1.0, step / max(1, warmup_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_scale

        t0 = time.time()
        T, nograd, grad = sampler.sample_batch(batch_size=micro_bs, device=device)
        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data[:, 1:].clone()

        max_nograd = int(nograd.max().item())
        max_grad = int(grad.max().item())

        with torch.no_grad():
            h = model.embedding(data)
            for block in model.prelude_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            h = model.prelude_norm(h)

            for _ in range(max_nograd):
                for block in model.recurrent_blocks:
                    norm1, attn, norm2, mlp = block
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
                A_bar_vec, B_bar = model.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)

        h = h.detach().requires_grad_(True)
        for _ in range(max_grad):
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

        current_loss = loss.item()
        if current_loss > last_loss * 1.5 and step > warmup_steps:
            spike_count += 1
        last_loss = current_loss

        if step % 500 == 0:
            losses.append({"step": step, "loss": current_loss})

    return model, losses, wall_time, spike_count


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


def run_per_sequence_ablation(config_path, output_dir, device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]
    config_name = mc["config_name"]
    vocab_size = mc["vocab_size"]
    max_seq_len = mc["max_seq_len"]
    mu_rec = mc["mu_rec"]
    mu_bwd = mc["mu_bwd"]
    lr = config["training"]["adamw_lr"]
    total_tokens = config["training"]["total_tokens"]

    all_results = {}

    for sampling_mode in ["per_microbatch", "per_sequence"]:
        print(f"\n[PerSequenceAblation] {config_name} mode={sampling_mode}")

        model = LoopedTransformer(config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)

        if sampling_mode == "per_microbatch":
            model, losses, wall_time, spikes = train_per_microbatch(model, lr, device, total_tokens, mu_rec, mu_bwd)
        else:
            model, losses, wall_time, spikes = train_per_sequence(model, lr, device, total_tokens, mu_rec, mu_bwd)

        print(f"  Training time: {wall_time:.1f}s, Loss spikes: {spikes}")

        T_results = {}
        for T in T_EVAL_VALUES:
            ppl = eval_at_T(model, device, vocab_size, T)
            T_results[str(T)] = ppl
            print(f"  T={T}: PPL={ppl:.4f}")

        all_results[sampling_mode] = {
            "wall_time_s": wall_time,
            "loss_spike_count": spikes,
            "final_losses": [l["loss"] for l in losses[-5:]],
            "per_T_ppl": T_results,
        }

    mb_results = all_results["per_microbatch"]
    seq_results = all_results["per_sequence"]

    fewer_spikes = seq_results["loss_spike_count"] <= mb_results["loss_spike_count"]
    better_ppl = all(seq_results["per_T_ppl"][str(T)] <= mb_results["per_T_ppl"][str(T)] for T in T_EVAL_VALUES)
    overhead = (seq_results["wall_time_s"] - mb_results["wall_time_s"]) / mb_results["wall_time_s"] * 100

    summary = {
        "per_sequence_fewer_spikes": fewer_spikes,
        "per_sequence_better_ppl_at_all_T": better_ppl,
        "per_sequence_overhead_pct": overhead,
        "overhead_within_2pct": overhead <= 2.0,
    }

    print(f"\n=== Verification ===")
    print(f"  Per-sequence fewer spikes: {fewer_spikes}")
    print(f"  Per-sequence better PPL at all T: {better_ppl}")
    print(f"  Per-sequence overhead: {overhead:.2f}%")
    print(f"  Overhead within 2%: {summary['overhead_within_2pct']}")

    results_path = Path(output_dir) / f"per_sequence_ablation_{config_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({"per_mode": all_results, "summary": summary}, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_per_sequence_ablation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
