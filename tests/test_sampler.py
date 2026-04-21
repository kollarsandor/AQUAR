import pytest
import torch
import sys
sys.path.insert(0, '/home/z/my-project/download/looped-lm')
from training.sampler import RecurrenceDepthSampler, TruncatedBaselineSampler


def test_sampler_returns_correct_shapes():
    sampler = RecurrenceDepthSampler(mu_rec=8, mu_bwd=4)
    T, nograd, grad = sampler.sample_batch(batch_size=32, device='cpu')
    assert T.shape == (32,)
    assert nograd.shape == (32,)
    assert grad.shape == (32,)


def test_sampler_mean_approximately_mu_rec():
    sampler = RecurrenceDepthSampler(mu_rec=8, mu_bwd=4)
    samples = []
    for _ in range(10000):
        T, _, _ = sampler.sample_batch(batch_size=1, device='cpu')
        samples.append(T.item())
    mean_T = sum(samples) / len(samples)
    assert 6 < mean_T < 12


def test_sampler_minimum_one():
    sampler = RecurrenceDepthSampler(mu_rec=8, mu_bwd=4)
    for _ in range(1000):
        T, _, _ = sampler.sample_batch(batch_size=1, device='cpu')
        assert T.item() >= 1


def test_nograd_plus_grad_equals_T():
    sampler = RecurrenceDepthSampler(mu_rec=8, mu_bwd=4)
    T, nograd, grad = sampler.sample_batch(batch_size=100, device='cpu')
    for i in range(100):
        assert nograd[i].item() + grad[i].item() == T[i].item()


def test_truncated_baseline_sampler():
    sampler = TruncatedBaselineSampler(mu_rec=8, mu_bwd=4)
    T, nograd, grad = sampler.sample_batch(batch_size=100, device='cpu')
    for i in range(100):
        assert grad[i].item() == 4


def test_microbatch_sampling():
    sampler = RecurrenceDepthSampler(mu_rec=8, mu_bwd=4)
    T_mb, _, _ = sampler.sample_microbatch(batch_size=32, device='cpu')
    assert T_mb.shape == (32,)
    assert (T_mb == T_mb[0]).all()
