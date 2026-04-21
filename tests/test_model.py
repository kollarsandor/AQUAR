import pytest
import torch
import sys
sys.path.insert(0, '/home/z/my-project/download/looped-lm')
from model.architecture import LoopedTransformer, FixedDepthTransformer, MODEL_CONFIGS
from model.blocks import RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
from model.injection import InjectionBlock, PreludeNorm
from model.init import scaled_init, scaled_zero_init, init_looped_model


def test_model_forward_100M():
    model = LoopedTransformer('100M', max_seq_len=128, mu_rec=2)
    x = torch.randint(0, 65536, (2, 32))
    logits = model(x, recurrence_steps=2)
    assert logits.shape == (2, 32, 65536)


def test_model_forward_140M():
    model = LoopedTransformer('140M', max_seq_len=128, mu_rec=2)
    x = torch.randint(0, 32768, (2, 32))
    logits = model(x, recurrence_steps=2)
    assert logits.shape == (2, 32, 32768)


def test_fixed_depth_forward():
    model = FixedDepthTransformer('140M', max_seq_len=128)
    x = torch.randint(0, 32768, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 32768)


def test_rmsnorm():
    norm = RMSNorm(512)
    x = torch.randn(2, 10, 512)
    out = norm(x)
    assert out.shape == x.shape


def test_attention():
    attn = CausalAttention(d_model=768, n_heads=6, head_dim=128, max_seq_len=128, qk_norm=True)
    x = torch.randn(2, 16, 768)
    out = attn(x)
    assert out.shape == x.shape


def test_swiglu_mlp():
    mlp = SwiGLUMLP(d_model=1024, d_ffn=3520)
    x = torch.randn(2, 10, 1024)
    out = mlp(x)
    assert out.shape == x.shape


def test_relu2_mlp():
    mlp = ReLU2MLP(d_model=768, d_ffn=3072)
    x = torch.randn(2, 10, 768)
    out = mlp(x)
    assert out.shape == x.shape


def test_injection_block():
    block = InjectionBlock(512)
    A_bar, B_bar = block()
    assert A_bar.shape == (512,)
    assert B_bar.shape == (512, 512)


def test_model_configs():
    for name in ['100M', '350M', '140M', '370M', '770M', '1.3B']:
        config = MODEL_CONFIGS[name]
        assert 'L_P' in config
        assert 'L_R' in config
        assert 'L_C' in config
        assert 'd_model' in config


def test_init_routines():
    model = LoopedTransformer('140M', max_seq_len=128)
    init_looped_model(model, config_type='B')
    x = torch.randint(0, 32768, (2, 16))
    logits = model(x, recurrence_steps=1)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_gradient_flow():
    model = LoopedTransformer('140M', max_seq_len=64, mu_rec=2)
    x = torch.randint(0, 32768, (2, 16))
    logits = model(x, recurrence_steps=2)
    loss = logits.float().mean()
    loss.backward()
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad
