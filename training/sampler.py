import math
import torch
from typing import Tuple


class RecurrenceDepthSampler:
    def __init__(self, mu_rec: int, sigma: float = 0.5, mu_bwd: int | None = None):
        self.mu_rec = mu_rec
        self.sigma = sigma
        if mu_bwd is None:
            self.mu_bwd = math.ceil(mu_rec / 2)
        else:
            self.mu_bwd = mu_bwd
        self.log_mu_rec = math.log(mu_rec)
        self.mu_normal = self.log_mu_rec - (sigma ** 2) / 2.0

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = torch.randn(batch_size, device=device) * self.sigma + self.mu_normal
        lam = torch.exp(tau)
        T_values = torch.poisson(lam) + 1
        T_values = T_values.clamp(min=1)
        nograd_steps = torch.clamp(T_values - self.mu_bwd, min=0)
        grad_steps = torch.clamp(T_values, max=self.mu_bwd)
        return T_values, nograd_steps, grad_steps

    def sample_microbatch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau_val = torch.randn(1, device=device).item() * self.sigma + self.mu_normal
        lam_val = math.exp(tau_val)
        T_val = torch.poisson(torch.tensor([lam_val], device=device)).item() + 1
        T_val = max(1, T_val)
        nograd = max(0, T_val - self.mu_bwd)
        grad = min(T_val, self.mu_bwd)
        T_values = torch.full((batch_size,), T_val, dtype=torch.long, device=device)
        nograd_steps = torch.full((batch_size,), nograd, dtype=torch.long, device=device)
        grad_steps = torch.full((batch_size,), grad, dtype=torch.long, device=device)
        return T_values, nograd_steps, grad_steps


class TruncatedBaselineSampler:
    def __init__(self, mu_rec: int, mu_bwd: int | None = None, sigma: float = 0.5):
        self.mu_rec = mu_rec
        self.mu_bwd = mu_bwd if mu_bwd is not None else math.ceil(mu_rec / 2)
        self.sigma = sigma
        self.mu_nograd = mu_rec - self.mu_bwd
        self.mu_nograd = max(self.mu_nograd, 1)
        self.log_mu_nograd = math.log(self.mu_nograd)
        self.mu_normal = self.log_mu_nograd - (sigma ** 2) / 2.0

    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = torch.randn(batch_size, device=device) * self.sigma + self.mu_normal
        lam = torch.exp(tau)
        k_vals = torch.poisson(lam)
        k_vals = k_vals.clamp(min=1)
        T_values = k_vals + self.mu_bwd
        nograd_steps = k_vals
        grad_steps = torch.full_like(T_values, self.mu_bwd)
        return T_values, nograd_steps, grad_steps
