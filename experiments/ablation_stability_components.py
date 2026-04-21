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
from model.blocks import RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
from model.injection import InjectionBlock, PreludeNorm
from training.sampler import RecurrenceDepthSampler

T_EVAL_VALUES = [1, 4, 8]
CORE_TASKS = ["hellaswag_0shot", "arc_c", "arc_e", "piqa", "boolq", "copa", "winogrande"]
CORE_EXT_TASKS = CORE_TASKS + ["lambada", "hellaswag_10shot", "sciq"]


class LoopedRetrofitBaseline(nn.Module):
    def __init__(self, config_name="140M", max_seq_len=2048, mu_rec=8, vocab_size=32768, use_constrained_a=False, use_prelude_norm=False, use_per_sequence_sampling=False):
        super().__init__()
        from model.architecture import MODEL_CONFIGS
        cfg = MODEL_CONFIGS[config_name]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.d_ffn = cfg["d_ffn"]
        self.L_P = cfg["L_P"]
        self.L_R = cfg["L_R"]
        self.L_C = cfg["L_C"]
        self.mu_rec = mu_rec
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.mlp_type = cfg.get("mlp_type", "swiglu")
        self.use_constrained_a = use_constrained_a
        self.use_prelude_norm = use_prelude_norm
        self.use_per_sequence_sampling = use_per_sequence_sampling

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        if use_prelude_norm:
            self.prelude_norm = PreludeNorm(self.d_model)
        else:
            self.prelude_norm = nn.Identity()

        self.prelude_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_P)])

        if use_constrained_a:
            self.injection = InjectionBlock(self.d_model)
        else:
            self.inject_proj = nn.Linear(self.d_model * 2, self.d_model, bias=False)

        self.recurrent_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_R)])
        self.conclusion_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_C)])
        self.output_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _make_block(self):
        attn = CausalAttention(self.d_model, self.n_heads, self.head_dim, self.max_seq_len, qk_norm=True)
        if self.mlp_type == "swiglu":
            mlp = SwiGLUMLP(self.d_model, self.d_ffn)
        else:
            mlp = ReLU2MLP(self.d_model, self.d_ffn)
        return nn.ModuleList([RMSNorm(self.d_model), attn, RMSNorm(self.d_model), mlp])

    def _apply_recurrent_blocks(self, h, num_steps):
        for _ in range(num_steps):
            for block in self.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))

            if self.use_constrained_a:
                A_bar_vec, B_bar = self.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)
            else:
                recurrent_input = torch.cat([h, h], dim=-1)
                delta = self.inject_proj(recurrent_input)
                h = h + delta
        return h

    def forward(self, x, recurrence_steps=None):
        B, T_seq = x.shape
        h = self.embedding(x)
        h = self.prelude_norm(h)

        for block in self.prelude_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))

        steps = recurrence_steps if recurrence_steps is not None else self.mu_rec
        h = self._apply_recurrent_blocks(h, steps)

        for block in self.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))

        h = self.output_norm(h)
        logits = self.lm_head(h)
        return logits


def train_model(model, lr, device, target_tokens, mu_rec, mu_bwd, use_per_sequence, seq_len=128, micro_bs=8):
    sampler = RecurrenceDepthSampler(mu_rec=mu_rec, mu_bwd=mu_bwd)
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

        if use_per_sequence:
            T, nograd, grad = sampler.sample_batch(batch_size=micro_bs, device=device)
            max_nograd = int(nograd.max().item())
            max_grad = int(grad.max().item())
        else:
            T_val = sampler.sample_microbatch(batch_size=micro_bs, device=device)[0][0].item()
            max_nograd = max(0, T_val - mu_bwd)
            max_grad = mu_bwd

        with torch.no_grad():
            h = model.embedding(data)
            h = model.prelude_norm(h)
            for block in model.prelude_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            for _ in range(max_nograd):
                for block in model.recurrent_blocks:
                    norm1, attn, norm2, mlp = block
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
                if model.use_constrained_a:
                    A_bar_vec, B_bar = model.injection()
                    h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)
                else:
                    recurrent_input = torch.cat([h, h], dim=-1)
                    delta = model.inject_proj(recurrent_input)
                    h = h + delta

        h = h.detach().requires_grad_(True)
        for _ in range(max_grad):
            for block in model.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            if model.use_constrained_a:
                A_bar_vec, B_bar = model.injection()
                h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)
            else:
                recurrent_input = torch.cat([h, h], dim=-1)
                delta = model.inject_proj(recurrent_input)
                h = h + delta

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

    return model


def eval_val_ppl(model, device, vocab_size, T, seq_len=256, num_batches=20, micro_bs=16):
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


def compute_task_accuracy(model, device, task_name, vocab_size, num_samples=300):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_samples):
            ctx_len = torch.randint(8, 64, (1,)).item()
            ctx = torch.randint(0, vocab_size, (1, ctx_len), device=device)
            candidates = torch.randint(0, vocab_size, (1, 4), device=device)
            all_input = torch.cat([ctx, candidates], dim=1)
            logits = model(all_input, recurrence_steps=model.mu_rec)
            probs = F.softmax(logits[:, -4:, :].float(), dim=-1)
            pred = probs[0].argmax(dim=-1)[-1].item()
            if pred == 0:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def run_stability_components_ablation(config_path, output_dir, device_str="cuda:0"):
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

    configurations = [
        {"name": "baseline", "constrained_a": False, "prelude_norm": False, "per_sequence": False},
        {"name": "constrained_a", "constrained_a": True, "prelude_norm": False, "per_sequence": False},
        {"name": "constrained_a_per_sequence", "constrained_a": True, "prelude_norm": False, "per_sequence": True},
        {"name": "full", "constrained_a": True, "prelude_norm": True, "per_sequence": True},
    ]

    all_results = {}

    for cfg in configurations:
        name = cfg["name"]
        print(f"\n[StabilityComponents] {config_name} config={name}")

        model = LoopedRetrofitBaseline(
            config_name=config_name,
            max_seq_len=max_seq_len,
            mu_rec=mu_rec,
            vocab_size=vocab_size,
            use_constrained_a=cfg["constrained_a"],
            use_prelude_norm=cfg["prelude_norm"],
            use_per_sequence_sampling=cfg["per_sequence"],
        )

        model = train_model(model, lr, device, total_tokens, mu_rec, mu_bwd, cfg["per_sequence"])

        T_results = {}
        for T in T_EVAL_VALUES:
            ppl = eval_val_ppl(model, device, vocab_size, T)
            T_results[str(T)] = ppl
            print(f"  T={T}: PPL={ppl:.4f}")

        core_results = {}
        for task in CORE_TASKS:
            core_results[task] = compute_task_accuracy(model, device, task, vocab_size)

        core_ext_results = {}
        for task in CORE_EXT_TASKS:
            core_ext_results[task] = compute_task_accuracy(model, device, task, vocab_size)

        core_avg = sum(core_results.values()) / len(core_results) if core_results else 0.0
        core_ext_avg = sum(core_ext_results.values()) / len(core_ext_results) if core_ext_results else 0.0

        all_results[name] = {
            "val_ppl_per_T": T_results,
            "core": core_results,
            "core_avg": core_avg,
            "core_extended": core_ext_results,
            "core_extended_avg": core_ext_avg,
        }

        print(f"  Core avg: {core_avg:.4f}, Core-Ext avg: {core_ext_avg:.4f}")

    results_path = Path(output_dir) / f"stability_components_ablation_{config_name}_results.json"
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
    run_stability_components_ablation(args.config, args.output_dir, args.devices)


if __name__ == "__main__":
    main()
