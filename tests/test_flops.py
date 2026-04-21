import pytest
import torch
import sys
sys.path.insert(0, '/home/z/my-project/download/looped-lm')
from model.architecture import LoopedTransformer
from scaling.flops import effective_params, training_flops, attention_flops_per_token


def test_effective_params_basic():
    model = LoopedTransformer('140M')
    N1, N2, total = effective_params(model, mu_rec=8, mu_bwd=4)
    assert N1 > 0
    assert N2 > 0
    assert total == N1 + N2
    assert total > 100_000_000


def test_effective_params_excludes_embeddings():
    model = LoopedTransformer('140M')
    N1, N2, total = effective_params(model, mu_rec=8, mu_bwd=4)
    emb_params = model.embedding.numel()
    total_model_params = sum(p.numel() for p in model.parameters())
    assert total < total_model_params


def test_training_flops_returns_dict():
    model = LoopedTransformer('140M')
    result = training_flops(model, mu_rec=8, mu_bwd=4, tokens=11_200_000_000)
    assert isinstance(result, dict)
    assert 'total_flops' in result
    assert result['total_flops'] > 0


def test_attention_flops_per_token():
    flops = attention_flops_per_token(n_heads=6, d_head=128, seq_len=2048, n_layers=6)
    assert flops > 0
    expected = 2 * 6 * 128 * 2048 * 6
    assert abs(flops - expected) < 1e-6


def test_flops_scale_with_mu_rec():
    model = LoopedTransformer('140M')
    flops_4 = training_flops(model, mu_rec=4, mu_bwd=2, tokens=10_000_000_000)
    flops_8 = training_flops(model, mu_rec=8, mu_bwd=4, tokens=10_000_000_000)
    assert flops_8['total_flops'] > flops_4['total_flops']
