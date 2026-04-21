import pytest
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/z/my-project/download/looped-lm')
from model.injection import InjectionBlock, spectral_radius_A_bar, test_spectral_radius_always_below_one


def test_spectral_radius_below_one_random_init():
    for _ in range(1000):
        block = InjectionBlock(1024)
        A_bar = torch.exp(F.softplus(block.log_delta) * (-torch.exp(block.log_A)))
        sr = spectral_radius_A_bar(A_bar)
        assert sr < 1.0


def test_spectral_radius_below_one_after_many_steps():
    block = InjectionBlock(768)
    for _ in range(100):
        A_bar_vec, B_bar = block()
        sr = spectral_radius_A_bar(A_bar_vec)
        assert sr < 1.0


def test_injection_block_shapes():
    block = InjectionBlock(512)
    A_bar, B_bar = block()
    assert A_bar.shape == (512,)
    assert B_bar.shape == (512, 512)


def test_injection_block_forward_no_grad():
    block = InjectionBlock(256)
    A_bar, B_bar = block()
    assert not A_bar.requires_grad


def test_spectral_norm_computation():
    block = InjectionBlock(512)
    A_bar_vec, B_bar = block()
    sn_b = spectral_radius_A_bar(A_bar_vec)
    assert isinstance(sn_b, float) or (isinstance(sn_b, torch.Tensor) and sn_b.ndim == 0)


def test_construction_spectral_radius():
    test_spectral_radius_always_below_one()
