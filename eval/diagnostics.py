import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class RecurrenceDiagnostics:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.history_residual_norms = []
        self.history_state_norms = []
        self.history_spectral_radii = []
        self.history_injection_norms = []

    def log_diagnostics(self, h_prev, h_current, step):
        residual = self.compute_recurrent_residual_norm(h_prev, h_current)
        self.history_residual_norms.append(residual)
        state = self.compute_state_norm(h_current)
        self.history_state_norms.append(state)
        injection_norms = self.compute_injection_spectral_norms()
        self.history_injection_norms.append(injection_norms)
        spectral_radius = self.compute_spectral_radius()
        self.history_spectral_radii.append(spectral_radius)
        return {
            "step": step,
            "residual_norm": residual,
            "state_norm": state,
            "spectral_radius": spectral_radius,
            "injection_norms": injection_norms,
        }

    def compute_recurrent_residual_norm(self, h_prev, h_current):
        return torch.norm(h_current - h_prev, p=2, dim=-1).mean().item()

    def compute_state_norm(self, h):
        return torch.norm(h, p=2, dim=-1).mean().item()

    def compute_spectral_radius(self):
        model = self.model
        spectral_radius = 1.0
        if hasattr(model, "A_bar") and model.A_bar is not None:
            A_bar = model.A_bar
            if isinstance(A_bar, nn.Parameter):
                A_mat = A_bar.data
            else:
                A_mat = A_bar
            if A_mat.dim() == 1:
                diag = torch.exp(-A_mat)
                spectral_radius = torch.max(torch.abs(diag)).item()
            elif A_mat.dim() == 2:
                eigenvalues = torch.linalg.eigvals(A_mat)
                spectral_radius = torch.max(torch.abs(eigenvalues)).real.item()
        if hasattr(model, "recurrent_block") and model.recurrent_block is not None:
            rb = model.recurrent_block
            if hasattr(rb, "A") and rb.A is not None:
                A = rb.A
                if isinstance(A, nn.Parameter):
                    A = A.data
                if hasattr(rb, "delta") and rb.delta is not None:
                    delta = rb.delta
                    if isinstance(delta, nn.Parameter):
                        delta = delta.data
                    diag_vals = torch.exp(-delta * torch.exp(A))
                    spectral_radius = torch.max(torch.abs(diag_vals)).item()
                elif A.dim() == 1:
                    diag_vals = torch.exp(-torch.abs(A))
                    spectral_radius = torch.max(torch.abs(diag_vals)).item()
                elif A.dim() == 2:
                    eigenvalues = torch.linalg.eigvals(A)
                    spectral_radius = torch.max(torch.abs(eigenvalues)).real.item()
        return spectral_radius

    def compute_injection_spectral_norms(self):
        model = self.model
        norms = {}
        if hasattr(model, "A_bar") and model.A_bar is not None:
            A_bar = model.A_bar
            if isinstance(A_bar, nn.Parameter):
                A_mat = A_bar.data
            else:
                A_mat = A_bar
            if A_mat.dim() == 2:
                norms["A_bar"] = torch.linalg.matrix_norm(A_mat, ord=2).item()
            elif A_mat.dim() == 1:
                norms["A_bar"] = torch.norm(A_mat, p=2).item()
        if hasattr(model, "B_bar") and model.B_bar is not None:
            B_bar = model.B_bar
            if isinstance(B_bar, nn.Parameter):
                B_mat = B_bar.data
            else:
                B_mat = B_bar
            if B_mat.dim() == 2:
                norms["B_bar"] = torch.linalg.matrix_norm(B_mat, ord=2).item()
            elif B_mat.dim() == 1:
                norms["B_bar"] = torch.norm(B_mat, p=2).item()
        if hasattr(model, "recurrent_block") and model.recurrent_block is not None:
            rb = model.recurrent_block
            if hasattr(rb, "B") and rb.B is not None:
                B = rb.B
                if isinstance(B, nn.Parameter):
                    B = B.data
                if B.dim() == 2:
                    norms["B_bar"] = torch.linalg.matrix_norm(B, ord=2).item()
                elif B.dim() == 1:
                    norms["B_bar"] = torch.norm(B, p=2).item()
            if hasattr(rb, "C_proj") and rb.C_proj is not None:
                C_proj = rb.C_proj
                if isinstance(C_proj, nn.Linear):
                    W = C_proj.weight.data
                    norms["C_proj"] = torch.linalg.matrix_norm(W, ord=2).item()
                elif isinstance(C_proj, nn.Parameter):
                    norms["C_proj"] = torch.linalg.matrix_norm(C_proj.data, ord=2).item()
        if hasattr(model, "C_proj") and model.C_proj is not None:
            C_proj = model.C_proj
            if isinstance(C_proj, nn.Linear):
                W = C_proj.weight.data
                norms["C_proj"] = torch.linalg.matrix_norm(W, ord=2).item()
            elif isinstance(C_proj, nn.Parameter):
                norms["C_proj"] = torch.linalg.matrix_norm(C_proj.data, ord=2).item()
        return norms

    def compute_per_layer_norms(self, inputs):
        model = self.model
        layer_norms = OrderedDict()
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                layer_norms[name] = torch.norm(out, p=2).item()
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                model(inputs)
            elif isinstance(inputs, dict):
                model(inputs["input_ids"])

        for h in hooks:
            h.remove()

        p_blocks = {k: v for k, v in layer_norms.items() if "P" in k or "project" in k or "proj" in k}
        r_blocks = {k: v for k, v in layer_norms.items() if "R" in k or "recurrent" in k}
        c_blocks = {k: v for k, v in layer_norms.items() if "C" in k or "c_proj" in k or "output" in k}
        return {
            "P_blocks": p_blocks,
            "R_blocks": r_blocks,
            "C_blocks": c_blocks,
            "all_layers": layer_norms,
        }

    def get_summary(self):
        if not self.history_residual_norms:
            return {}
        return {
            "mean_residual_norm": sum(self.history_residual_norms) / len(self.history_residual_norms),
            "final_residual_norm": self.history_residual_norms[-1] if self.history_residual_norms else 0.0,
            "residual_norm_trend": self.history_residual_norms,
            "mean_state_norm": sum(self.history_state_norms) / len(self.history_state_norms),
            "final_state_norm": self.history_state_norms[-1] if self.history_state_norms else 0.0,
            "state_norm_trend": self.history_state_norms,
            "spectral_radius": self.history_spectral_radii[-1] if self.history_spectral_radii else 1.0,
            "spectral_radius_history": self.history_spectral_radii,
            "final_injection_norms": self.history_injection_norms[-1] if self.history_injection_norms else {},
            "injection_norm_history": self.history_injection_norms,
        }

    def reset(self):
        self.history_residual_norms = []
        self.history_state_norms = []
        self.history_spectral_radii = []
        self.history_injection_norms = []
