import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm


class InjectionBlock(nn.Module):

    def __init__(self, d_h, d_c=None):
        super().__init__()
        d_c = d_c if d_c is not None else d_h
        self.d_h = d_h
        self.d_c = d_c
        self.log_A = nn.Parameter(torch.zeros(d_h))
        self.B = nn.Parameter(torch.randn(d_h, d_h) * (1.0 / d_h ** 0.5))
        self.log_delta = nn.Parameter(torch.zeros(d_h))
        self.c_proj = nn.Linear(d_h, d_c, bias=False)

    def forward(self):
        A_diag = -torch.exp(self.log_A)
        delta = F.softplus(self.log_delta)
        A_bar = torch.exp(delta * A_diag)
        B_bar = delta.unsqueeze(-1) * self.B
        return A_bar, B_bar


def spectral_radius_A_bar(A_bar):
    return torch.max(torch.abs(A_bar)).item()


def spectral_norm_B_bar(B_bar):
    return torch.linalg.norm(B_bar, ord=2).item()


def spectral_norm_C_proj(c_proj):
    return torch.linalg.norm(c_proj.weight, ord=2).item()


def test_spectral_radius_always_below_one():
    d_h = 64
    for _ in range(1000):
        log_A = torch.randn(d_h)
        log_delta = torch.randn(d_h)
        A_diag = -torch.exp(log_A)
        delta = F.softplus(log_delta)
        A_bar = torch.exp(delta * A_diag)
        rho = torch.max(torch.abs(A_bar)).item()
        assert rho < 1.0, f"Spectral radius {rho} >= 1.0"
    return True


class PreludeNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)

    def forward(self, x):
        return self.norm(x)
