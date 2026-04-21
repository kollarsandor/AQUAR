import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len=2048, base=50000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=None):
    if rotary_dim is not None and rotary_dim < q.shape[-1]:
        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]
        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
    return q, k


class CausalAttention(nn.Module):

    def __init__(self, d_model, n_heads, head_dim=None, max_seq_len=2048, qk_norm=False, rotary_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.d_model = d_model
        self.inner_dim = n_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.inner_dim)
            self.k_norm = RMSNorm(self.inner_dim)
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        self.rope = RotaryPositionalEmbedding(self.rotary_dim, max_seq_len)

    def forward(self, x, token_emb=None, gated_value_emb=None, apply_gated_value=False):
        B, n, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if apply_gated_value and gated_value_emb is not None and token_emb is not None:
            v = gated_value_emb(v, token_emb)

        q = q.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(n)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n, self.inner_dim)
        return self.o_proj(attn_out)


class SwiGLUMLP(nn.Module):

    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ReLU2MLP(nn.Module):

    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ffn, bias=False)
        self.proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        h = F.relu(F.relu(self.fc(x)))
        return self.proj(h)


class GatedValueEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, v_proj_output, token_emb):
        gate = torch.sigmoid(self.gate)
        return gate * v_proj_output + (1.0 - gate) * token_emb


class TransformerBlockA(nn.Module):

    def __init__(self, d_model, n_heads, d_ffn, max_seq_len=2048, head_dim=None, layer_idx=0, vocab_size=65536, rotary_dim=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            qk_norm=False,
            rotary_dim=rotary_dim,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ffn)

    def forward(self, x, token_emb=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlockB(nn.Module):

    def __init__(self, d_model, n_heads, d_ffn, max_seq_len=2048, head_dim=128, layer_idx=0, vocab_size=32768, rotary_dim=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            qk_norm=True,
            rotary_dim=rotary_dim,
        )
        self.gated_value_emb = GatedValueEmbedding(d_model)
        self.use_gated_value = (layer_idx % 2 == 0)
        self.norm2 = RMSNorm(d_model)
        self.mlp = ReLU2MLP(d_model, d_ffn)

    def forward(self, x, token_emb=None):
        normed = self.norm1(x)
        attn_out = self.attn(
            normed,
            token_emb=token_emb,
            gated_value_emb=self.gated_value_emb,
            apply_gated_value=self.use_gated_value,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
