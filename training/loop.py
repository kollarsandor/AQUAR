import os
import time
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LoopedTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_adamw: torch.optim.Optimizer,
        optimizer_muon: Optional[torch.optim.Optimizer],
        scheduler_adamw: Any,
        scheduler_muon: Optional[Any],
        sampler: Any,
        tokenizer: Any,
        train_dataloader,
        val_dataloader,
        config: Dict[str, Any],
        device: torch.device,
        use_ddp: bool = False,
        use_fsdp: bool = False,
        rank: int = 0,
    ):
        self.config = config
        self.device = device
        self.use_ddp = use_ddp
        self.use_fsdp = use_fsdp
        self.rank = rank
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer_adamw = optimizer_adamw
        self.optimizer_muon = optimizer_muon
        self.scheduler_adamw = scheduler_adamw
        self.scheduler_muon = scheduler_muon

        self.total_steps = config.get('total_steps', 100000)
        self.log_every = config.get('log_every', 100)
        self.val_every = config.get('val_every', 1000)
        self.save_every = config.get('save_every', 5000)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.h0_sigma = config.get('h0_sigma', 0.02)
        self.use_bf16 = config.get('use_bf16', True)
        self.use_wandb = config.get('use_wandb', False) and HAS_WANDB
        self.use_tensorboard = config.get('use_tensorboard', False)
        self.mu_bwd = config.get('mu_bwd', sampler.mu_bwd if hasattr(sampler, 'mu_bwd') else math.ceil(config.get('mu_rec', 8) / 2))
        self.microbatch_size = config.get('microbatch_size', None)

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_completed = 0

        if self.use_bf16 and device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
            self.amp_dtype = torch.bfloat16
        else:
            self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and not self.use_bf16))
            self.amp_dtype = torch.float16 if device.type == 'cuda' and not self.use_bf16 else torch.float32

        if use_fsdp:
            self.model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=None,
                device_id=device,
            )
        elif use_ddp:
            model = model.to(device)
            self.model = DDP(model, device_ids=[device] if device.type == 'cuda' else None)
        else:
            self.model = model.to(device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.use_wandb and rank == 0:
            wandb.init(
                project=config.get('wandb_project', 'looped-lm'),
                name=config.get('run_name', 'default'),
                config=config,
            )

        if self.use_tensorboard and rank == 0:
            log_dir = config.get('tensorboard_dir', 'runs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def _apply_prelude(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            if hasattr(model, 'compute_prelude'):
                e = model.compute_prelude(tokens)
            elif hasattr(model, 'prelude') and hasattr(model, 'prelude_norm'):
                e = model.prelude(tokens)
                e = model.prelude_norm(e)
            else:
                raise AttributeError("Model must have compute_prelude or (prelude, prelude_norm) methods")
        return e

    def _apply_recurrent_block(self, h: torch.Tensor) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            if hasattr(model, 'recurrent_blocks'):
                recurrent_out = model.recurrent_blocks(h)
            else:
                raise AttributeError("Model must have recurrent_blocks attribute")
        return recurrent_out

    def _get_injection_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'injection'):
            inj = model.injection
            if hasattr(inj, 'A_bar'):
                A_bar = inj.A_bar
            elif hasattr(inj, 'get_A_bar'):
                A_bar = inj.get_A_bar()
            else:
                raise AttributeError("Injection module must have A_bar attribute")

            if hasattr(inj, 'B_bar'):
                B_bar = inj.B_bar
            elif hasattr(inj, 'get_B_bar'):
                B_bar = inj.get_B_bar()
            else:
                raise AttributeError("Injection module must have B_bar attribute")
        else:
            raise AttributeError("Model must have injection module")
        return A_bar, B_bar

    def _apply_coda(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            if hasattr(model, 'coda'):
                logits = model.coda(h)
            elif hasattr(model, 'lm_head'):
                logits = model.lm_head(h)
            else:
                raise AttributeError("Model must have coda or lm_head attribute")
        return logits.float()

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        tokens = batch.to(self.device)
        B, seq_len = tokens.shape

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            e = self._apply_prelude(tokens)

        T_values, nograd_steps, grad_steps = self.sampler.sample_batch(B, self.device)
        T_max = T_values.max().item()
        T_max = max(T_max, 1)
        mu_bwd_tensor = torch.full((B,), self.mu_bwd, dtype=torch.long, device=self.device)
        mu_bwd_tensor = torch.min(mu_bwd_tensor, T_values)
        tau = T_max - T_values

        h = torch.randn(B, *e.shape[1:], device=self.device, dtype=e.dtype) * self.h0_sigma
        h = h.to(self.amp_dtype) if self.amp_dtype != torch.float32 else h

        A_bar, B_bar = self._get_injection_params()
        A_bar_diag = A_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and A_bar.dtype != self.amp_dtype else A_bar
        B_bar_mat = B_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and B_bar.dtype != self.amp_dtype else B_bar

        A_bar_expanded = A_bar_diag.view(1, 1, -1).expand(B, -1, -1)
        B_bar_e = torch.matmul(e, B_bar_mat.T)

        no_update_mask = torch.zeros(B, T_max, dtype=torch.bool, device=self.device)
        nograd_mask = torch.zeros(B, T_max, dtype=torch.bool, device=self.device)
        grad_mask = torch.zeros(B, T_max, dtype=torch.bool, device=self.device)

        for b in range(B):
            t_b = T_values[b].item()
            tau_b = tau[b].item()
            mu_bwd_b = mu_bwd_tensor[b].item()
            grad_start = T_max - mu_bwd_b

            for t in range(T_max):
                if t < tau_b:
                    no_update_mask[b, t] = True
                elif t < grad_start:
                    nograd_mask[b, t] = True
                else:
                    grad_mask[b, t] = True

        detach_boundary = T_max - self.mu_bwd
        detach_done = False

        for t in range(T_max):
            with torch.no_grad() if nograd_mask[:, t].all() and not grad_mask[:, t].any() else torch.enable_grad():
                is_nograd_only = nograd_mask[:, t].all() and not grad_mask[:, t].any()
                is_mixed = nograd_mask[:, t].any() and grad_mask[:, t].any()

                if is_nograd_only:
                    h_nograd = h.detach()
                elif is_mixed:
                    if not detach_done and t >= detach_boundary and detach_boundary >= 0:
                        h = h.detach()
                        detach_done = True
                    h_nograd = h.detach()
                else:
                    if not detach_done and t >= detach_boundary and detach_boundary >= 0:
                        h = h.detach()
                        detach_done = True
                    h_nograd = h

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                    recurrent_out = self._apply_recurrent_block(h_nograd)

                R_bar = recurrent_out - h_nograd
                h_next = A_bar_expanded * h_nograd + B_bar_e + R_bar

                h_next_expanded = h_next.unsqueeze(1) if h_next.dim() == 2 else h_next

                no_update_expanded = no_update_mask[:, t].view(B, 1, 1).expand_as(h_next)
                h = torch.where(no_update_expanded, h_nograd, h_next)

        logits = self._apply_coda(h, tokens)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

        return loss

    def _compute_gradient_norm(self) -> float:
        total_sq_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq_norm += p.grad.data.float().norm(2).item() ** 2
        return math.sqrt(total_sq_norm)

    def _compute_spectral_diagnostics(self) -> Dict[str, float]:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        diagnostics = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() >= 2 and param.numel() > 1:
                    try:
                        s = torch.linalg.svdvals(param.float())
                        if s.numel() > 0:
                            diagnostics[f'spectral/{name}/max'] = s[0].item()
                            diagnostics[f'spectral/{name}/min'] = s[-1].item()
                            diagnostics[f'spectral/{name}/cond'] = (s[0] / (s[-1] + 1e-12)).item()
                            diagnostics[f'spectral/{name}/rank_eff'] = (s.sum() / (s[0] + 1e-12)).item()
                            if s.numel() > 1:
                                diagnostics[f'spectral/{name}/top5_ratio'] = (s[:5].sum() / (s.sum() + 1e-12)).item()
                    except Exception:
                        _ = 0
        return diagnostics

    def _compute_loss_components(self, logits: torch.Tensor, tokens: torch.Tensor) -> Dict[str, float]:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        )
        per_token_loss = per_token_loss.view(tokens.size(0), -1)
        valid_mask = shift_labels != -100
        if valid_mask.any():
            pos_loss = per_token_loss[per_token_loss > 0]
            return {
                'loss/mean': per_token_loss[valid_mask].mean().item(),
                'loss/max': pos_loss.max().item() if pos_loss.numel() > 0 else 0.0,
                'loss/min': per_token_loss[valid_mask].min().item() if valid_mask.any() else 0.0,
                'loss/std': per_token_loss[valid_mask].std().item() if valid_mask.sum() > 1 else 0.0,
            }
        return {'loss/mean': 0.0, 'loss/max': 0.0, 'loss/min': 0.0, 'loss/std': 0.0}

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        if self.rank != 0:
            return
        log_str = f"Step {step}/{self.total_steps}"
        for k, v in sorted(metrics.items()):
            log_str += f" | {k}: {v:.6f}" if isinstance(v, float) else f" | {k}: {v}"
        print(log_str, flush=True)

        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)

    @torch.no_grad()
    def validate(self) -> float:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in self.val_dataloader:
            if isinstance(batch, (list, tuple)):
                tokens = batch[0]
            else:
                tokens = batch

            tokens = tokens.to(self.device)
            B, seq_len = tokens.shape

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                e = self._apply_prelude(tokens)

            T_fixed = torch.full((B,), self.sampler.mu_rec, dtype=torch.long, device=self.device)
            h = torch.zeros(B, *e.shape[1:], device=self.device, dtype=e.dtype)

            A_bar, B_bar = self._get_injection_params()
            A_bar_cast = A_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and A_bar.dtype != self.amp_dtype else A_bar
            B_bar_cast = B_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and B_bar.dtype != self.amp_dtype else B_bar
            A_bar_exp = A_bar_cast.view(1, 1, -1).expand(B, -1, -1)
            B_bar_e = torch.matmul(e, B_bar_cast.T)

            T_val = T_fixed[0].item()
            for _ in range(T_val):
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                    recurrent_out = self._apply_recurrent_block(h)
                R_bar = recurrent_out - h
                h = A_bar_exp * h + B_bar_e + R_bar

            logits = self._apply_coda(h, tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum',
            )
            valid = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid
            num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        model.train()
        return avg_loss

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        if self.rank != 0:
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'global_step': step,
            'epochs_completed': self.epochs_completed,
            'model_state_dict': model.state_dict(),
            'optimizer_adamw_state_dict': self.optimizer_adamw.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        if self.optimizer_muon is not None:
            checkpoint['optimizer_muon_state_dict'] = self.optimizer_muon.state_dict()

        path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, path)

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)

        print(f"Saved checkpoint at step {step} to {path}", flush=True)

    def load_checkpoint(self, checkpoint_path: str):
        map_location = self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw_state_dict'])

        if 'optimizer_muon_state_dict' in checkpoint and self.optimizer_muon is not None:
            self.optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.epochs_completed = checkpoint.get('epochs_completed', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {self.global_step}", flush=True)

    def train(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.train()

        start_step = self.global_step
        step_in_epoch = 0
        epoch = self.epochs_completed
        train_iter = iter(self.train_dataloader)

        while self.global_step < self.total_steps:
            if self.use_ddp and hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                if self.global_step >= self.total_steps:
                    break

                if isinstance(batch, (list, tuple)):
                    tokens = batch[0]
                else:
                    tokens = batch

                if self.microbatch_size is not None and tokens.size(0) > self.microbatch_size:
                    mb_losses = []
                    for mb_start in range(0, tokens.size(0), self.microbatch_size):
                        mb_end = min(mb_start + self.microbatch_size, tokens.size(0))
                        mb_tokens = tokens[mb_start:mb_end]
                        mb_loss = self.train_step(mb_tokens)
                        mb_loss = mb_loss / (tokens.size(0) // self.microbatch_size)
                        mb_losses.append(mb_loss)
                    loss = mb_losses[0]
                    for mb_loss in mb_losses[1:]:
                        loss = loss + mb_loss
                else:
                    loss = self.train_step(tokens)

                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad.float()

                grad_norm = self._compute_gradient_norm()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_adamw)
                    self.scaler.update()
                else:
                    self.optimizer_adamw.backward = None
                    (loss / 1.0).backward()
                    self.optimizer_adamw.step()

                if self.optimizer_muon is not None:
                    if hasattr(self.optimizer_muon, 'param_groups'):
                        for pg in self.optimizer_muon.param_groups:
                            pg['total_steps'] = self.total_steps
                    self.optimizer_muon.step()

                if self.scheduler_adamw is not None:
                    if hasattr(self.scheduler_adamw, 'step'):
                        self.scheduler_adamw.step(self.global_step)

                if self.scheduler_muon is not None and hasattr(self.scheduler_muon, 'step'):
                    self.scheduler_muon.step(self.global_step)

                self.optimizer_adamw.zero_grad(set_to_none=True)
                if self.optimizer_muon is not None:
                    self.optimizer_muon.zero_grad(set_to_none=True)

                self.global_step += 1
                step_in_epoch += 1

                if self.global_step % self.log_every == 0:
                    metrics = {
                        'train/loss': loss.item(),
                        'train/grad_norm': grad_norm,
                        'train/lr_adamw': self.optimizer_adamw.param_groups[0]['lr'],
                        'train/step': self.global_step,
                        'train/epoch': epoch,
                    }
                    if self.optimizer_muon is not None:
                        metrics['train/lr_muon'] = self.optimizer_muon.param_groups[0]['lr']

                    if self.global_step % (self.log_every * 5) == 0:
                        spec_diag = self._compute_spectral_diagnostics()
                        metrics.update(spec_diag)

                    self._log_metrics(metrics, self.global_step)

                if self.global_step % self.val_every == 0:
                    val_loss = self.validate()
                    metrics = {'val/loss': val_loss, 'train/step': self.global_step}
                    self._log_metrics(metrics, self.global_step)
                    self.save_checkpoint(self.global_step, val_loss)

                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(self.global_step)

            epoch += 1
            self.epochs_completed = epoch
            train_iter = iter(self.train_dataloader)

        self.save_checkpoint(self.global_step)

        if self.use_wandb and self.rank == 0:
            wandb.finish()

        if self.writer is not None and self.rank == 0:
            self.writer.close()

        print(f"Training complete. Total steps: {self.global_step}", flush=True)

    def _compute_loss_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        tokens = batch.to(self.device)
        B, seq_len = tokens.shape

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
            e = self._apply_prelude(tokens)

        T_values, _, _ = self.sampler.sample_batch(B, self.device)
        T_max = T_values.max().item()
        T_max = max(T_max, 1)

        h = torch.zeros(B, *e.shape[1:], device=self.device, dtype=e.dtype)

        A_bar, B_bar = self._get_injection_params()
        A_bar_cast = A_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and A_bar.dtype != self.amp_dtype else A_bar
        B_bar_cast = B_bar.to(self.amp_dtype) if self.amp_dtype != torch.float32 and B_bar.dtype != self.amp_dtype else B_bar
        A_bar_exp = A_bar_cast.view(1, 1, -1).expand(B, -1, -1)
        B_bar_e = torch.matmul(e, B_bar_cast.T)

        for t in range(T_max):
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                recurrent_out = self._apply_recurrent_block(h)
            R_bar = recurrent_out - h
            h = A_bar_exp * h + B_bar_e + R_bar

        logits = self._apply_coda(h, tokens)
        loss_components = self._compute_loss_components(logits, tokens)
        return loss_components
