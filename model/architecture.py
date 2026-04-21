import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TransformerBlockA, TransformerBlockB, RMSNorm
from .injection import InjectionBlock, PreludeNorm

MODEL_CONFIGS = {
    '100M': {
        'params': 114242560,
        'L_P': 1,
        'L_R': 1,
        'L_C': 1,
        'd_model': 1024,
        'd_ffn': 3520,
        'mu_rec': 16,
        'config_type': 'A',
    },
    '350M': {
        'params': 378558464,
        'L_P': 1,
        'L_R': 2,
        'L_C': 1,
        'd_model': 2048,
        'd_ffn': 7040,
        'mu_rec': 8,
        'config_type': 'A',
    },
    '140M': {
        'params': 144323136,
        'L_P': 2,
        'L_R': 2,
        'L_C': 2,
        'd_model': 768,
        'd_ffn': 3072,
        'mu_rec': 8,
        'n_heads': 6,
        'head_dim': 128,
        'config_type': 'B',
    },
    '370M': {
        'params': 388003328,
        'L_P': 4,
        'L_R': 4,
        'L_C': 4,
        'd_model': 1024,
        'd_ffn': 4096,
        'mu_rec': 8,
        'n_heads': 8,
        'head_dim': 128,
        'config_type': 'B',
    },
    '770M': {
        'params': 776655680,
        'L_P': 6,
        'L_R': 6,
        'L_C': 6,
        'd_model': 1280,
        'd_ffn': 5120,
        'mu_rec': 8,
        'n_heads': 10,
        'head_dim': 128,
        'config_type': 'B',
    },
    '1.3B': {
        'params': 1338591744,
        'L_P': 8,
        'L_R': 8,
        'L_C': 8,
        'd_model': 1536,
        'd_ffn': 6144,
        'mu_rec': 8,
        'n_heads': 12,
        'head_dim': 128,
        'config_type': 'B',
    },
}

CONFIG_A_VOCAB = 65536
CONFIG_B_VOCAB = 32768
CONFIG_A_DEFAULT_HEADS = 16


def _make_block(config_type, d_model, n_heads, d_ffn, max_seq_len, head_dim, layer_idx, vocab_size, rotary_dim=None):
    if config_type == 'A':
        return TransformerBlockA(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            layer_idx=layer_idx,
            vocab_size=vocab_size,
            rotary_dim=rotary_dim,
        )
    else:
        return TransformerBlockB(
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            layer_idx=layer_idx,
            vocab_size=vocab_size,
            rotary_dim=rotary_dim,
        )


class LoopedTransformer(nn.Module):

    def __init__(self, config_name, max_seq_len=2048, mu_rec=None, mu_bwd=None, config_type=None):
        super().__init__()
        if config_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config_name '{config_name}'. Available: {list(MODEL_CONFIGS.keys())}")
        cfg = MODEL_CONFIGS[config_name]
        config_type = config_type if config_type is not None else cfg['config_type']
        self.config_type = config_type
        self.config_name = config_name

        self.d_model = cfg['d_model']
        self.d_ffn = cfg['d_ffn']
        self.L_P = cfg['L_P']
        self.L_R = cfg['L_R']
        self.L_C = cfg['L_C']
        self.max_seq_len = max_seq_len
        self.mu_rec = mu_rec if mu_rec is not None else cfg['mu_rec']
        self.mu_bwd = mu_bwd if mu_bwd is not None else math.ceil(self.mu_rec / 2)

        vocab_size = CONFIG_A_VOCAB if config_type == 'A' else CONFIG_B_VOCAB
        self.vocab_size = vocab_size

        n_heads = cfg.get('n_heads', CONFIG_A_DEFAULT_HEADS)
        head_dim = cfg.get('head_dim', self.d_model // n_heads)

        self.embedding = nn.Embedding(vocab_size, self.d_model)

        layer_offset = 0
        self.prelude_blocks = nn.ModuleList([
            _make_block(
                config_type=config_type,
                d_model=self.d_model,
                n_heads=n_heads,
                d_ffn=self.d_ffn,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                layer_idx=layer_offset + i,
                vocab_size=vocab_size,
            )
            for i in range(self.L_P)
        ])
        layer_offset += self.L_P

        self.prelude_norm = PreludeNorm(self.d_model)

        self.injection_block = InjectionBlock(d_h=self.d_model, d_c=self.d_model)

        self.recurrent_blocks = nn.ModuleList([
            _make_block(
                config_type=config_type,
                d_model=self.d_model,
                n_heads=n_heads,
                d_ffn=self.d_ffn,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                layer_idx=layer_offset + i,
                vocab_size=vocab_size,
            )
            for i in range(self.L_R)
        ])
        layer_offset += self.L_R

        self.coda_blocks = nn.ModuleList([
            _make_block(
                config_type=config_type,
                d_model=self.d_model,
                n_heads=n_heads,
                d_ffn=self.d_ffn,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                layer_idx=layer_offset + i,
                vocab_size=vocab_size,
            )
            for i in range(self.L_C)
        ])

    def forward(self, token_ids):
        B, n = token_ids.shape
        token_emb = self.embedding(token_ids)

        x = token_emb
        for block in self.prelude_blocks:
            x = block(x, token_emb=token_emb)
        e = self.prelude_norm(x)

        if self.config_type == 'A':
            sigma = math.sqrt(2.0 / (5.0 * self.d_model))
        else:
            sigma = (2.0 / (5.0 * self.d_model)) ** 0.25

        h = torch.randn(B, n, self.d_model, device=token_ids.device, dtype=token_ids.dtype) * sigma

        A_bar, B_bar = self.injection_block()

        for _t in range(self.mu_rec):
            recurrent_out = h
            for block in self.recurrent_blocks:
                recurrent_out = block(recurrent_out, token_emb=token_emb)
            R_bar = recurrent_out - h
            h = A_bar * h + torch.matmul(e, B_bar.t()) + R_bar

        x = self.injection_block.c_proj(h)
        for block in self.coda_blocks:
            x = block(x, token_emb=token_emb)

        logits = F.linear(x, self.embedding.weight)
        return logits


class FixedDepthTransformer(nn.Module):

    def __init__(self, config_name, max_seq_len=2048, config_type=None):
        super().__init__()
        if config_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config_name '{config_name}'. Available: {list(MODEL_CONFIGS.keys())}")
        cfg = MODEL_CONFIGS[config_name]
        config_type = config_type if config_type is not None else cfg['config_type']
        self.config_type = config_type
        self.config_name = config_name

        self.d_model = cfg['d_model']
        self.d_ffn = cfg['d_ffn']
        self.L_R = cfg['L_R']
        self.max_seq_len = max_seq_len
        self.total_layers = 3 * self.L_R

        vocab_size = CONFIG_A_VOCAB if config_type == 'A' else CONFIG_B_VOCAB
        self.vocab_size = vocab_size

        n_heads = cfg.get('n_heads', CONFIG_A_DEFAULT_HEADS)
        head_dim = cfg.get('head_dim', self.d_model // n_heads)

        self.embedding = nn.Embedding(vocab_size, self.d_model)

        self.blocks = nn.ModuleList([
            _make_block(
                config_type=config_type,
                d_model=self.d_model,
                n_heads=n_heads,
                d_ffn=self.d_ffn,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                layer_idx=i,
                vocab_size=vocab_size,
            )
            for i in range(self.total_layers)
        ])

    def forward(self, token_ids):
        B, n = token_ids.shape
        token_emb = self.embedding(token_ids)

        x = token_emb
        for block in self.blocks:
            x = block(x, token_emb=token_emb)

        logits = F.linear(x, self.embedding.weight)
        return logits
