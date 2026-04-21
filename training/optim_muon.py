import torch
from torch.optim import Optimizer
from typing import List, Optional
import math


def _newton_schulz(g: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    rows, cols = g.shape
    if rows < cols:
        g_t = g.t()
        x = g_t @ g
        eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
        for _ in range(n_iters):
            a = x @ x.t()
            b = 0.5 * (3.0 * eye - a)
            x = x @ b
        return x @ g_t
    else:
        x = g @ g.t()
        eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
        for _ in range(n_iters):
            a = x @ x.t()
            b = 0.5 * (3.0 * eye - a)
            x = x @ b
        return x.t() @ g


class Muon(Optimizer):
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 8e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.2,
        n_polar_iters: int = 5,
        ns_steps: int = 5,
        adamuon: bool = True,
        max_update_norm: float = 1.0,
        total_steps: Optional[int] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            n_polar_iters=n_polar_iters,
            ns_steps=ns_steps,
            adamuon=adamuon,
            max_update_norm=max_update_norm,
            total_steps=total_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            n_polar_iters = group['n_polar_iters']
            ns_steps = group['ns_steps']
            adamuon = group['adamuon']
            max_update_norm = group['max_update_norm']
            total_steps = group['total_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['velocity'] = torch.zeros_like(p)
                    if adamuon:
                        state['grad_var'] = torch.zeros_like(p)

                state['step'] += 1
                current_step = state['step']

                if total_steps is not None and total_steps > 0:
                    wd_t = weight_decay * max(0.0, 1.0 - current_step / total_steps)
                else:
                    wd_t = weight_decay

                velocity = state['velocity']
                velocity.mul_(momentum).add_(grad)

                if adamuon:
                    grad_var = state['grad_var']
                    grad_var.mul_(0.99).add_(0.01 * (grad ** 2))
                    scale = 1.0 / (grad_var.sqrt() + 1e-12)
                    velocity = velocity * scale

                if wd_t > 0.0:
                    cautious_mask = (p * velocity > 0).float()
                    velocity.add_(p * wd_t * cautious_mask)

                param_shape = p.shape
                if p.dim() >= 2:
                    flat_v = velocity.reshape(param_shape[0], -1)
                    rows, cols = flat_v.shape
                    if rows < cols:
                        padded = torch.zeros(rows, rows, dtype=flat_v.dtype, device=flat_v.device)
                        padded[:, :cols] = flat_v
                        ortho = _newton_schulz(padded, n_iters=ns_steps)
                        ortho = ortho[:, :cols]
                    else:
                        ortho = _newton_schulz(flat_v, n_iters=ns_steps)

                    if max_update_norm > 0.0:
                        update_flat = -lr * ortho
                        update_norm = torch.norm(update_flat)
                        if update_norm > max_update_norm:
                            update_flat.mul_(max_update_norm / (update_norm + 1e-12))
                        p.add_(update_flat.reshape(param_shape))
                    else:
                        p.add_(-lr * ortho.reshape(param_shape))
                else:
                    update = -lr * velocity
                    if max_update_norm > 0.0:
                        update_norm = torch.norm(update)
                        if update_norm > max_update_norm:
                            update.mul_(max_update_norm / (update_norm + 1e-12))
                    p.add_(update)

        return loss
