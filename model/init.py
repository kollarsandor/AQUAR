import math

import torch.nn as nn

from .blocks import RMSNorm


def scaled_init(weight, d_model):
    nn.init.normal_(weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * d_model)))


def scaled_zero_init(weight, d_model, is_output_proj=False):
    if is_output_proj:
        nn.init.zeros_(weight)
    else:
        nn.init.normal_(weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * d_model)))


_OUTPUT_PROJ_NAMES = frozenset({'o_proj', 'proj', 'down_proj', 'c_proj'})


def init_looped_model(model, config_type='B'):
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            d_model = module.in_features
            last_part = name.rsplit('.', 1)[-1]
            if config_type == 'A':
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * d_model)))
            else:
                if last_part in _OUTPUT_PROJ_NAMES:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * d_model)))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
