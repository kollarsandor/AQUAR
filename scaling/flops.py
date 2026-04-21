from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional


def _count_module_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _get_model_components(model: nn.Module) -> Dict[str, int]:
    components: Dict[str, int] = {}

    prelude = getattr(model, "prelude_blocks", None)
    if prelude is None:
        prelude = getattr(model, "prelude", None)
    components["prelude"] = _count_module_params(prelude)

    recurrent = getattr(model, "recurrent_blocks", None)
    if recurrent is None:
        recurrent = getattr(model, "recurrent", None)
    components["recurrent"] = _count_module_params(recurrent)

    coda = getattr(model, "coda_blocks", None)
    if coda is None:
        coda = getattr(model, "coda", None)
    components["coda"] = _count_module_params(coda)

    injection = getattr(model, "injection_block", None)
    if injection is None:
        injection = getattr(model, "injection", None)
    components["injection"] = _count_module_params(injection)

    embedding = getattr(model, "embedding", None)
    if embedding is None:
        embedding = getattr(model, "embed", None)
        if embedding is None:
            if hasattr(model, "token_embedding"):
                embedding = model.token_embedding
    components["embedding"] = _count_module_params(embedding)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = getattr(model, "output_head", None)
    components["lm_head"] = _count_module_params(lm_head)

    if hasattr(model, "tie_weights") and model.tie_weights:
        components["lm_head"] = 0
    elif hasattr(model, "output_embeddings") and model.output_embeddings is not None:
        components["lm_head"] = 0
    else:
        if lm_head is not None and hasattr(model, "embedding"):
            emb = getattr(model, "embedding", None)
            if emb is not None:
                try:
                    if lm_head.weight.data_ptr() == emb.weight.data_ptr():
                        components["lm_head"] = 0
                except Exception:
                    components["lm_head"] = components["lm_head"]

    components["total"] = sum(v for k, v in components.items() if k not in ("embedding", "lm_head")) + components["lm_head"]

    return components


def effective_params(
    model: nn.Module, mu_rec: float, mu_bwd: float
) -> Tuple[float, float, float]:
    components = _get_model_components(model)
    recurrent_params = components["recurrent"]
    prelude_params = components["prelude"]
    coda_params = components["coda"]
    injection_params = components["injection"]
    lm_head_params = components["lm_head"]

    expected_nograd_steps = max(mu_rec - mu_bwd, 0.0)
    expected_grad_steps = max(mu_bwd, 0.0)

    N_hat_1 = recurrent_params * expected_nograd_steps
    N_hat_2 = (
        recurrent_params * expected_grad_steps
        + prelude_params
        + coda_params
        + injection_params
        + lm_head_params
    )

    total_effective = N_hat_1 + N_hat_2

    return (N_hat_1, N_hat_2, total_effective)


def attention_flops_per_token(
    n_heads: int, d_head: int, seq_len: int, n_layers: int
) -> int:
    d_model = n_heads * d_head
    flops_per_layer = 2 * d_model * seq_len
    return flops_per_layer * n_layers


def recurrent_attention_flops(
    model: nn.Module, seq_len: int, mu_rec: float
) -> float:
    total_flops = 0.0

    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "n_heads"):
            n_heads = config.n_heads
        elif hasattr(config, "num_attention_heads"):
            n_heads = config.num_attention_heads
        else:
            n_heads = getattr(model, "n_heads", 8)
        if hasattr(config, "d_head"):
            d_head = config.d_head
        elif hasattr(config, "head_dim"):
            d_head = config.head_dim
        elif hasattr(config, "hidden_size") and n_heads > 0:
            d_head = config.hidden_size // n_heads
        else:
            d_head = getattr(model, "d_head", 64)
    else:
        n_heads = getattr(model, "n_heads", 8)
        d_head = getattr(model, "d_head", 64)

    components = _get_model_components(model)
    recurrent_params = components["recurrent"]
    prelude_params = components["prelude"]
    coda_params = components["coda"]

    if prelude_params > 0:
        d_model = n_heads * d_head
        if recurrent_params > 0:
            transformer_block_params = (2 * d_model * d_model + 2 * d_model * (4 * d_model))
            n_prelude_layers = max(round(prelude_params / transformer_block_params), 1)
        else:
            n_prelude_layers = 1
        total_flops += attention_flops_per_token(n_heads, d_head, seq_len, n_prelude_layers)

    if recurrent_params > 0:
        d_model = n_heads * d_head
        transformer_block_params = (2 * d_model * d_model + 2 * d_model * (4 * d_model))
        n_recurrent_layers = max(round(recurrent_params / transformer_block_params), 1)
        total_flops += attention_flops_per_token(n_heads, d_head, seq_len, n_recurrent_layers) * mu_rec

    if coda_params > 0:
        d_model = n_heads * d_head
        if recurrent_params > 0:
            transformer_block_params = (2 * d_model * d_model + 2 * d_model * (4 * d_model))
            n_coda_layers = max(round(coda_params / transformer_block_params), 1)
        else:
            n_coda_layers = 1
        total_flops += attention_flops_per_token(n_heads, d_head, seq_len, n_coda_layers)

    return total_flops


def training_flops(
    model: nn.Module, mu_rec: float, mu_bwd: float, tokens: float
) -> Dict[str, Any]:
    N_hat_1, N_hat_2, total_effective = effective_params(model, mu_rec, mu_bwd)
    D = tokens

    compute_flops = (2.0 * N_hat_1 + 6.0 * N_hat_2) * D

    attn_flops = recurrent_attention_flops(model, seq_len=1, mu_rec=mu_rec)
    attn_total = attn_flops * D

    C = compute_flops + attn_total

    components = _get_model_components(model)
    recurrent_params = components["recurrent"]
    prelude_params = components["prelude"]
    coda_params = components["coda"]
    injection_params = components["injection"]
    lm_head_params = components["lm_head"]

    return {
        "total_flops": C,
        "compute_flops": compute_flops,
        "attention_flops": attn_total,
        "N_hat_1": N_hat_1,
        "N_hat_2": N_hat_2,
        "total_effective_params": total_effective,
        "tokens": D,
        "mu_rec": mu_rec,
        "mu_bwd": mu_bwd,
        "nograd_coefficient": 2.0,
        "grad_coefficient": 6.0,
        "recurrent_params": recurrent_params,
        "prelude_params": prelude_params,
        "coda_params": coda_params,
        "injection_params": injection_params,
        "lm_head_params": lm_head_params,
    }
