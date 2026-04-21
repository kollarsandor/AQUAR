import torch
from torch.optim import Optimizer
from typing import List, Optional


class AdamWNoEps(Optimizer):
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.0,
        eps: float = 0.0,
        max_update_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, max_update_norm=max_update_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            max_update_norm = group['max_update_norm']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWNoEps does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()) + eps
                step_size = lr / bias_correction1
                update = (exp_avg / denom) * step_size

                if weight_decay > 0.0:
                    update.add_(p, alpha=weight_decay * lr)

                if max_update_norm > 0.0:
                    update_norm = torch.norm(update)
                    if update_norm > max_update_norm:
                        update.mul_(max_update_norm / (update_norm + 1e-12))

                p.add_(-update)

        return loss


class ConfigBAdamW(Optimizer):
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 8e-3,
        betas: tuple = (0.8, 0.95),
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        max_update_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, max_update_norm=max_update_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            max_update_norm = group['max_update_norm']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ConfigBAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()) + eps
                step_size = lr / bias_correction1
                update = (exp_avg / denom) * step_size

                if weight_decay > 0.0:
                    update.add_(p, alpha=weight_decay * lr)

                if max_update_norm > 0.0:
                    update_norm = torch.norm(update)
                    if update_norm > max_update_norm:
                        update.mul_(max_update_norm / (update_norm + 1e-12))

                p.add_(-update)

        return loss
