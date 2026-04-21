import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


class WarmupCooldownSchedule(LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int = 4096,
        total_steps: int = 100000,
        cooldown_frac: float = 0.1,
        last_epoch: int = -1,
    ):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cooldown_frac = cooldown_frac
        self.cooldown_start = int(total_steps * (1.0 - cooldown_frac))
        lr_lambda = self._lr_lambda
        super().__init__(optimizer, lr_lambda, last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            return self.peak_lr * (current_step + 1) / self.warmup_steps
        elif current_step < self.cooldown_start:
            return self.peak_lr
        else:
            progress = (current_step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
            progress = min(progress, 1.0)
            return self.peak_lr * (1.0 - progress)

    def get_lr(self, current_step: int) -> float:
        return self._lr_lambda(current_step)

    def step(self, current_step: Optional[int] = None):
        if current_step is not None:
            self.last_epoch = current_step
        else:
            self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr


class FixedCooldownSchedule(LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        total_steps: int = 100000,
        cooldown_start_frac: float = 0.5,
        last_epoch: int = -1,
    ):
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.cooldown_start_frac = cooldown_start_frac
        self.cooldown_start = int(total_steps * cooldown_start_frac)
        lr_lambda = self._lr_lambda
        super().__init__(optimizer, lr_lambda, last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        if current_step < self.cooldown_start:
            return self.peak_lr
        else:
            progress = (current_step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
            progress = min(progress, 1.0)
            return self.peak_lr * (1.0 - progress)

    def get_lr(self, current_step: int) -> float:
        return self._lr_lambda(current_step)

    def step(self, current_step: Optional[int] = None):
        if current_step is not None:
            self.last_epoch = current_step
        else:
            self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr
