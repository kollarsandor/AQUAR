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
from torch.utils.data import DataLoader, TensorDataset

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __builtins__:
    import sys as _sys
    _sys.path.insert(0, sys_path)

from model.blocks import RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
from model.injection import InjectionBlock, PreludeNorm
from model.init import init_looped_model

MODEL_CONFIGS = {
    "100M": {"d_model": 768, "n_heads": 12, "head_dim": 64, "d_ffn": 3072, "L_P": 2, "L_R": 6, "L_C": 2, "mlp_type": "relu2"},
    "140M": {"d_model": 768, "n_heads": 6, "head_dim": 128, "d_ffn": 3072, "L_P": 2, "L_R": 6, "L_C": 2, "mlp_type": "swiglu"},
    "350M": {"d_model": 1024, "n_heads": 16, "head_dim": 64, "d_ffn": 4096, "L_P": 2, "L_R": 10, "L_C": 2, "mlp_type": "relu2"},
}

GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

try:
    import regex as re_mod
except ImportError:
    import re as re_mod


class PreNormLoopedNoConstraint(nn.Module):
    def __init__(self, config_name="140M", max_seq_len=2048, mu_rec=8, vocab_size=32768):
        super().__init__()
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
        self.mlp_type = cfg["mlp_type"]
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.prelude_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_P)])
        self.A_free = nn.Parameter(torch.randn(self.d_model, self.d_model) * 0.01)
        self.B_inject = nn.Parameter(torch.randn(self.d_model, self.d_model) * (1.0 / math.sqrt(self.d_model)))
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

    def forward(self, x, recurrence_steps=None):
        B, T = x.shape
        h = self.embedding(x)
        for block in self.prelude_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        if recurrence_steps is None:
            recurrence_steps = self.mu_rec
        for _ in range(recurrence_steps):
            for block in self.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            h = torch.matmul(h, self.A_free.T) + torch.matmul(h, self.B_inject.T)
        for block in self.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        h = self.output_norm(h)
        logits = self.lm_head(h)
        return logits

    def get_A_bar_matrix(self):
        return self.A_free.detach()


class PreNormLoopedWithResidualNorm(nn.Module):
    def __init__(self, config_name="140M", max_seq_len=2048, mu_rec=8, vocab_size=32768):
        super().__init__()
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
        self.mlp_type = cfg["mlp_type"]
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.prelude_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_P)])
        self.injection = InjectionBlock(self.d_model)
        self.recurrent_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_R)])
        self.conclusion_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_C)])
        self.output_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.residual_norm = RMSNorm(self.d_model)

    def _make_block(self):
        attn = CausalAttention(self.d_model, self.n_heads, self.head_dim, self.max_seq_len, qk_norm=True)
        if self.mlp_type == "swiglu":
            mlp = SwiGLUMLP(self.d_model, self.d_ffn)
        else:
            mlp = ReLU2MLP(self.d_model, self.d_ffn)
        return nn.ModuleList([RMSNorm(self.d_model), attn, RMSNorm(self.d_model), mlp])

    def forward(self, x, recurrence_steps=None):
        B, T = x.shape
        h = self.embedding(x)
        for block in self.prelude_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        if recurrence_steps is None:
            recurrence_steps = self.mu_rec
        for step in range(recurrence_steps):
            for idx, block in enumerate(self.recurrent_blocks):
                norm1, attn, norm2, mlp = block
                if step > 0 and idx == 0:
                    h_pre = h
                    h = self.residual_norm(h) + attn(norm1(h))
                    h = self.residual_norm(h) + mlp(norm2(h))
                else:
                    h = h + attn(norm1(h))
                    h = h + mlp(norm2(h))
            A_bar_vec, B_bar = self.injection()
            h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)
        for block in self.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        h = self.output_norm(h)
        logits = self.lm_head(h)
        return logits


class ConstrainedALooped(nn.Module):
    def __init__(self, config_name="140M", max_seq_len=2048, mu_rec=8, vocab_size=32768):
        super().__init__()
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
        self.mlp_type = cfg["mlp_type"]
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.prelude_norm = PreludeNorm(self.d_model)
        self.prelude_blocks = nn.ModuleList([self._make_block() for _ in range(self.L_P)])
        self.injection = InjectionBlock(self.d_model)
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

    def forward(self, x, recurrence_steps=None):
        B, T = x.shape
        h = self.embedding(x)
        h = self.prelude_norm(h)
        for block in self.prelude_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        if recurrence_steps is None:
            recurrence_steps = self.mu_rec
        for _ in range(recurrence_steps):
            for block in self.recurrent_blocks:
                norm1, attn, norm2, mlp = block
                h = h + attn(norm1(h))
                h = h + mlp(norm2(h))
            A_bar_vec, B_bar = self.injection()
            h = A_bar_vec.unsqueeze(0).unsqueeze(0) * h + torch.matmul(h, B_bar.T)
        for block in self.conclusion_blocks:
            norm1, attn, norm2, mlp = block
            h = h + attn(norm1(h))
            h = h + mlp(norm2(h))
        h = self.output_norm(h)
        logits = self.lm_head(h)
        return logits


def spectral_radius(A):
    if A.ndim == 1:
        return torch.max(torch.abs(A)).item()
    eigs = torch.linalg.eigvals(A)
    return torch.max(torch.abs(eigs)).item()


def compute_spectral_radius_of_A_bar(model):
    if isinstance(model, PreNormLoopedNoConstraint):
        return spectral_radius(model.get_A_bar_matrix())
    elif isinstance(model, (PreNormLoopedWithResidualNorm, ConstrainedALooped)):
        A_bar_vec, _ = model.injection()
        return torch.max(torch.abs(A_bar_vec.detach())).item()
    return float("inf")


def generate_synthetic_data(vocab_size, seq_len, num_samples, device):
    data = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    labels = data.clone()
    labels[:, :-1] = data[:, 1:]
    labels[:, -1] = data[:, 0]
    dataset = TensorDataset(data, labels)
    return dataset


def train_single_run(model, lr, device, target_tokens=500_000_000, seq_len=128, micro_bs=8, log_every=100):
    model = model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_tokens = 0
    tokens_per_step = micro_bs * seq_len
    steps_needed = target_tokens // tokens_per_step
    losses = []
    state_norms = []
    spectral_radii = []
    diverged = False

    for step in range(steps_needed):
        data = torch.randint(0, model.vocab_size, (micro_bs, seq_len), device=device)
        labels = data.clone()
        labels[:, :-1] = data[:, 1:]
        labels[:, -1] = data[:, 0]

        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, model.vocab_size), labels[:, :-1].contiguous().view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += tokens_per_step

        if step % log_every == 0:
            losses.append(loss.item())
            with torch.no_grad():
                h = model.embedding(data)
                if hasattr(model, "prelude_norm"):
                    h = model.prelude_norm(h)
                state_norm = torch.norm(h.float()).item()
            state_norms.append(state_norm)
            if step % (log_every * 10) == 0:
                sr = compute_spectral_radius_of_A_bar(model)
                spectral_radii.append({"step": step, "spectral_radius": sr})

    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "diverged": diverged,
        "total_tokens": total_tokens,
        "losses": losses,
        "state_norms": state_norms,
        "spectral_radii": spectral_radii,
    }


def run_stability_sweep(config_path, output_dir):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    config_name = model_cfg["config_name"]
    max_seq_len = model_cfg["max_seq_len"]
    vocab_size = model_cfg["vocab_size"]
    mu_rec = model_cfg["mu_rec"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    learning_rates = [2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
    variant_names = ["prenorm_no_constraint", "prenorm_residual_norm", "constrained_a"]
    variant_classes = [PreNormLoopedNoConstraint, PreNormLoopedWithResidualNorm, ConstrainedALooped]

    all_results = {}
    for variant_name, variant_cls in zip(variant_names, variant_classes):
        all_results[variant_name] = {}
        for lr in learning_rates:
            print(f"[StabilitySweep] variant={variant_name}, lr={lr:.1e}")
            model = variant_cls(config_name=config_name, max_seq_len=max_seq_len, mu_rec=mu_rec, vocab_size=vocab_size)
            result = train_single_run(model, lr=lr, device=device, target_tokens=500_000_000, seq_len=128, micro_bs=8)
            run_key = f"lr_{lr:.1e}"
            all_results[variant_name][run_key] = result
            print(f"  diverged={result['diverged']}, final_loss={result['final_loss']:.4f}")

    summary = {}
    for variant_name in variant_names:
        converged_count = sum(1 for r in all_results[variant_name].values() if not r["diverged"])
        summary[variant_name] = {
            "converged_count": converged_count,
            "total_runs": len(learning_rates),
            "min_lr_converged": None,
        }
        for lr in learning_rates:
            run_key = f"lr_{lr:.1e}"
            r = all_results[variant_name][run_key]
            if not r["diverged"] and summary[variant_name]["min_lr_converged"] is None:
                summary[variant_name]["min_lr_converged"] = lr

    results_path = Path(output_dir) / "stability_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump({"all_results": {k: {k2: {kk: vv for kk, vv in v2.items() if kk != "losses" and kk != "state_norms" and kk != "spectral_radii"} for k2, v2 in v.items()} for k, v in all_results.items()}, "summary": summary}, f, indent=2)

    detailed_path = Path(output_dir) / "stability_sweep_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Stability Sweep Summary ===")
    for variant_name in variant_names:
        s = summary[variant_name]
        print(f"  {variant_name}: converged {s['converged_count']}/{s['total_runs']} runs")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    run_stability_sweep(args.config, args.output_dir)


if __name__ == "__main__":
    main()
