# Looped Transformer Language Model

A production-ready implementation of looped (recurrent) transformer language models with constrained spectral injection, configurable recurrence depth sampling, and comprehensive experiment suites for stability analysis, quality benchmarks, and scaling law validation.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Tokenizer Training](#tokenizer-training)
- [Model Training](#model-training)
- [Distributed Training](#distributed-training)
- [Evaluation](#evaluation)
- [Running Experiments](#running-experiments)
- [Scaling Law Fitting](#scaling-law-fitting)

---

## Installation

```bash
pip install torch transformers datasets tokenizers wandb scipy matplotlib numpy pyyaml regex
```

For CUDA support, install the appropriate PyTorch version for your GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
looped-lm/
├── model/
│   ├── architecture.py      # LoopedTransformer, FixedDepthTransformer, MODEL_CONFIGS
│   ├── blocks.py            # RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
│   ├── injection.py         # InjectionBlock, PreludeNorm, spectral_radius_A_bar
│   └── init.py              # scaled_init, scaled_zero_init, init_looped_model
├── training/
│   └── sampler.py           # RecurrenceDepthSampler, TruncatedBaselineSampler
├── scaling/
│   └── flops.py             # effective_params, training_flops, attention_flops_per_token
├── tokenizer/
│   ├── __init__.py
│   └── train.py             # BPE tokenizer training, FineWeb-Edu preparation
├── configs/
│   ├── 100M_looped.yaml     # Config A: 100M looped model
│   ├── 350M_looped.yaml     # Config A: 350M looped model
│   ├── 140M_looped.yaml     # Config B: 140M looped model
│   ├── 370M_looped.yaml     # Config B: 370M looped model
│   ├── 770M_looped.yaml     # Config B: 770M looped model
│   ├── 1.3B_looped.yaml     # Config B: 1.3B looped model
│   ├── 140M_baseline.yaml   # Config B: 140M fixed-depth baseline (6 layers)
│   ├── 370M_baseline.yaml   # Config B: 370M fixed-depth baseline (12 layers)
│   ├── 770M_baseline.yaml   # Config B: 770M fixed-depth baseline (18 layers)
│   └── 1.3B_baseline.yaml   # Config B: 1.3B fixed-depth baseline (24 layers)
├── experiments/
│   ├── __init__.py
│   ├── stability_sweep.py           # Exp 7.1: Stability Sweep
│   ├── quality_vs_baseline_loop.py  # Exp 7.2: Quality vs Looped Baseline
│   ├── ablation_stability_components.py  # Exp 7.3: Stability Ablation Table
│   ├── quality_vs_fixed_transformer.py   # Exp 7.4: Quality vs Fixed-Depth
│   ├── isoflop_sweep.py            # Exp 7.5: IsoFLOP Training Scaling
│   ├── test_time_saturation.py     # Exp 7.6: Test-Time Saturation
│   ├── unified_law.py              # Exp 7.7: Unified Scaling Law
│   ├── ablation_mu_rec.py          # Exp 7.8a: mu_rec sweep
│   ├── ablation_mu_bwd.py          # Exp 7.8b: mu_bwd sweep
│   ├── ablation_sampling_algo.py   # Exp 7.9: Sampling Algorithm Ablation
│   ├── ablation_per_sequence.py    # Exp 7.10: Per-Sequence Sampling
│   └── ablation_prelude_norm.py    # Exp 7.11: Prelude-Norm Ablation
├── tests/
│   ├── __init__.py
│   ├── test_injection.py
│   ├── test_flops.py
│   ├── test_sampler.py
│   └── test_model.py
└── README.md
```

## Configuration Reference

### Model Configurations

| Config | Type | d_model | n_heads | head_dim | d_ffn | L_P | L_R | L_C | mu_rec | mu_bwd | Vocab |
|--------|------|---------|---------|----------|-------|-----|-----|-----|--------|--------|-------|
| 100M   | A    | 768     | 12      | 64       | 3072  | 2   | 6   | 2   | 16     | 8      | 65536 |
| 350M   | A    | 1024    | 16      | 64       | 4096  | 2   | 10  | 2   | 16     | 8      | 65536 |
| 140M   | B    | 768     | 6       | 128      | 3072  | 2   | 6   | 2   | 8      | 4      | 32768 |
| 370M   | B    | 1024    | 8       | 128      | 4070  | 2   | 8   | 2   | 8      | 4      | 32768 |
| 770M   | B    | 1536    | 12      | 128      | 4070  | 2   | 8   | 2   | 8      | 4      | 32768 |
| 1.3B   | B    | 2048    | 16      | 128      | 8192  | 2   | 8   | 2   | 8      | 4      | 32768 |

### Config A vs Config B

- **Config A**: Uses ReLU^2 MLP, AdamW optimizer, standard packing, vocab_size=65536
- **Config B**: Uses SwiGLU MLP, Muon+AdamW optimizer, bestfit_crop packing, vocab_size=32768

### Key YAML Fields

```yaml
model:
  config_name: "140M"        # Model size identifier
  config_type: "B"           # A or B
  max_seq_len: 2048          # Maximum sequence length
  vocab_size: 32768          # Vocabulary size
  mu_rec: 8                  # Mean recurrence depth
  mu_bwd: 4                  # Mean backward (gradient) depth
  head_dim: 128              # Attention head dimension
  prelude_norm: true         # Enable PreludeNorm

training:
  adamw_lr: 8.0e-3           # AdamW learning rate (for Config B)
  muon_lr: 8.0e-3            # Muon learning rate
  adamw_beta1: 0.8           # AdamW beta1
  adamw_beta2: 0.95          # AdamW beta2
  adamw_weight_decay: 0.0    # AdamW weight decay
  muon_momentum: 0.95        # Muon momentum
  muon_weight_decay_start: 0.2
  muon_weight_decay_end: 0.0
  grad_clip: 1.0             # Gradient clipping norm
  cooldown_start_frac: 0.5   # Fraction of training at which cooldown starts
  total_tokens: 11_200_000_000
  global_batch_size: 256
  micro_batch_size: 32
  seq_len: 2048
  bf16: true                 # Use BF16 mixed precision

data:
  tokenizer_path: "tokenizer_config_b.json"
  train_path: null           # Set to your training data path
  val_path: null             # Set to your validation data path
  packing: "bestfit_crop"    # "standard" or "bestfit_crop"

evaluation:
  val_every: 2000            # Validate every N steps
  eval_tasks: [...]          # Evaluation tasks
  core_tasks: true           # Compute Core benchmark average
  core_ext_tasks: true       # Compute Core-Extended benchmark average
  test_time_sweep: false     # Evaluate at multiple T values
```

## Tokenizer Training

### Train Config B Tokenizer (FineWeb-Edu)

```bash
python -m tokenizer.train \
  --mode config_b \
  --input-path /path/to/fineteweb-edu.jsonl \
  --output-path tokenizer_config_b.json \
  --vocab-size 32768 \
  --max-chars-per-doc 10000 \
  --max-total-chars 2000000000
```

### Train Config A Tokenizer

```bash
python -m tokenizer.train \
  --mode config_a \
  --input-path /path/to/corpus.txt \
  --output-path tokenizer_config_a.json \
  --vocab-size 65536
```

### Create Simple Fallback Tokenizer

```bash
python -m tokenizer.train \
  --mode simple \
  --output-path tokenizer_simple.json \
  --vocab-size 65536
```

### Programmatic Usage

```python
from tokenizer import train_bpe_tokenizer, load_tokenizer

tokenizer = train_bpe_tokenizer(
    input_paths=["data/train.txt"],
    vocab_size=32768,
    output_path="tokenizer.json",
    min_frequency=2,
)

tokenizer = load_tokenizer("tokenizer.json")
encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(encoded.ids)
```

## Model Training

### Single GPU Training

```bash
python train.py \
  --config configs/140M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin
```

### Training with Override Parameters

```bash
python train.py \
  --config configs/370M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --training.micro-batch-size 16 \
  --training.global-batch-size 128
```

### Training by Model Size

| Model | Config File | GPUs | VRAM/GPU | Est. Time |
|-------|------------|------|----------|-----------|
| 100M  | 100M_looped.yaml | 1x A100 40GB | ~8 GB | ~12 hrs |
| 350M  | 350M_looped.yaml | 1x A100 40GB | ~16 GB | ~24 hrs |
| 140M  | 140M_looped.yaml | 1x A100 40GB | ~10 GB | ~18 hrs |
| 370M  | 370M_looped.yaml | 2x A100 40GB | ~20 GB | ~3 days |
| 770M  | 770M_looped.yaml | 4x A100 40GB | ~32 GB | ~8 days |
| 1.3B  | 1.3B_looped.yaml | 8x A100 80GB | ~50 GB | ~14 days |

### Baseline Models

```bash
python train.py --config configs/140M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/370M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/770M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/1.3B_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
```

## Distributed Training

### DDP (Distributed Data Parallel)

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/370M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=ddp
```

### FSDP (Fully Sharded Data Parallel)

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/1.3B_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=fsdp \
  --fsdp-sharding-strategy full \
  --fsdp-wrapping-policy transformer
```

### Multi-Node DDP

```bash
torchrun --nnodes=4 --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train.py \
  --config configs/770M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=ddp
```

## Evaluation

### Run All Evaluation Tasks

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --tasks hellaswag_0shot arc_c arc_e piqa boolq copa winogrande lambada
```

### Test-Time Recurrence Sweep

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --test-time-sweep \
  --sweep-range 1 24
```

### Compute Core and Core-Extended Benchmarks

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/370M_looped.yaml \
  --core \
  --core-extended
```

### WikiText Perplexity

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --wikitext-path /path/to/wikitext-103.txt
```

## Running Experiments

### Experiment 7.1: Stability Sweep

Sweeps peak learning rate across three model variants to verify stability.

```bash
python experiments/stability_sweep.py \
  --config configs/140M_looped.yaml \
  --output-dir results/stability_sweep \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~4 hours per variant, 3 variants x 5 LRs = 15 runs

**Output**: `results/stability_sweep/stability_sweep_results.json`

### Experiment 7.2: Quality vs Looped Baseline

Compares constrained-A looped model against unconstrained looped baseline with residual normalization.

```bash
python experiments/quality_vs_baseline_loop.py \
  --config configs/100M_looped.yaml \
  --output-dir results/quality_vs_baseline \
  --devices cuda:0

python experiments/quality_vs_baseline_loop.py \
  --config configs/350M_looped.yaml \
  --output-dir results/quality_vs_baseline \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~24 hours per scale, 2 scales (100M, 350M)

**Target**: >=6% PPL reduction over baseline

### Experiment 7.3: Stability Ablation Table

Retrofits transformer backbone with injection, progressively adding stability components.

```bash
python experiments/ablation_stability_components.py \
  --config configs/140M_looped.yaml \
  --output-dir results/stability_ablation \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~4 hours (4 configurations)

### Experiment 7.4: Quality vs Fixed-Depth Transformer

Full comparison across all model sizes.

```bash
python experiments/quality_vs_fixed_transformer.py \
  --config-dir configs \
  --output-dir results/quality_vs_fixed \
  --devices cuda:0
```

**Resources**: 1x A100 40GB per model size, ~18 hrs (140M), ~3 days (370M), ~8 days (770M), ~14 days (1.3B)

**Note**: For larger models, run each size independently on separate GPUs.

### Experiment 7.5: IsoFLOP Training Scaling Sweep

Sweeps mu_rec under fixed FLOP budgets to fit scaling laws.

```bash
python experiments/isoflop_sweep.py \
  --config configs/140M_looped.yaml \
  --output-dir results/isoflop_140M \
  --devices cuda:0

python experiments/isoflop_sweep.py \
  --config configs/370M_looped.yaml \
  --output-dir results/isoflop_370M \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~8 hours per config, 6 mu_rec values x 6 budgets (140M) or 3 budgets (370M)

**Output**: IsoLoss contour plots in PNG format, JSON results with parabolic fits

### Experiment 7.6: Test-Time Saturation Sweep

Evaluates exponential decay of PPL as test-time recurrence T increases.

```bash
python experiments/test_time_saturation.py \
  --config configs/140M_looped.yaml \
  --output-dir results/test_time_sat_140M \
  --devices cuda:0

python experiments/test_time_saturation.py \
  --config configs/370M_looped.yaml \
  --output-dir results/test_time_sat_370M \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~6 hours per config, 6 mu_rec values x T=1..24 evaluation

**Verification**: L_inf within 0.6% of L(mu_rec)

### Experiment 7.7: Unified Scaling Law Validation

Fits and validates the unified scaling law using IsoFLOP results.

```bash
python experiments/unified_law.py \
  --isoflop-results-dir results/isoflop_140M \
  --heldout-results-dir results/quality_vs_fixed \
  --output-dir results/unified_law \
  --devices cuda:0
```

**Resources**: CPU-only (no training), ~5 minutes

**Verification**: <=1.31% MAPE with predicted floor, <=0.17% with oracle floor

### Experiment 7.8a: mu_rec Ablation

```bash
python experiments/ablation_mu_rec.py \
  --config configs/140M_looped.yaml \
  --output-dir results/ablation_mu_rec \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~8 hours, 6 mu_rec values {4,8,14,20,26,32}

### Experiment 7.8b: mu_bwd Ablation

```bash
python experiments/ablation_mu_bwd.py \
  --config configs/140M_looped.yaml \
  --output-dir results/ablation_mu_bwd \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~6 hours, 5 mu_bwd values {4,6,8,10,12}

**Verification**: mu_bwd = ceil(mu_rec/2) is recommended

### Experiment 7.9: Sampling Algorithm Ablation

```bash
python experiments/ablation_sampling_algo.py \
  --config configs/100M_looped.yaml \
  --output-dir results/ablation_sampling \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~12 hours, 3 algorithms (full backprop, truncated, corrected)

**Verification**: Corrected algorithm matches full backprop, outperforms truncated

### Experiment 7.10: Per-Sequence vs Per-Microbatch Sampling

```bash
python experiments/ablation_per_sequence.py \
  --config configs/100M_looped.yaml \
  --output-dir results/ablation_per_seq \
  --devices cuda:0

python experiments/ablation_per_sequence.py \
  --config configs/350M_looped.yaml \
  --output-dir results/ablation_per_seq \
  --devices cuda:0
```

**Resources**: 1x A100 40GB, ~24 hours per scale, 2 modes x 2 scales

**Verification**: Per-sequence eliminates late-training spikes, <=2% overhead

### Experiment 7.11: Prelude-Norm Ablation at 1.3B

```bash
python experiments/ablation_prelude_norm.py \
  --config configs/1.3B_looped.yaml \
  --output-dir results/ablation_prelude_norm \
  --devices cuda:0
```

**Resources**: 2x A100 80GB, ~28 days total (2 runs of 104B tokens)

**Verification**: Without prelude-norm shows residual explosion; with prelude-norm remains stable

## Scaling Law Fitting

### Fit IsoFLOP Parabolic Law

```python
from experiments.isoflop_sweep import fit_parabolic_law

mu_values = [2, 4, 6, 8, 10, 12]
val_losses = [4.82, 4.15, 3.87, 3.72, 3.65, 3.61]

fit = fit_parabolic_law(mu_values, val_losses)
print(f"R-squared: {fit['r_squared']:.4f}")
print(f"Parameters: a={fit['params']['a']:.4f}, b={fit['params']['b']:.4f}, c={fit['params']['c']:.4f}")
```

### Fit Exponential Saturation Law

```python
from experiments.test_time_saturation import fit_exponential_decay, exponential_decay

T_values = list(range(1, 25))
ppl_values = [5.12, 4.55, 4.18, 3.92, 3.74, 3.61, 3.52, 3.46, 3.42, 3.39,
              3.37, 3.36, 3.35, 3.34, 3.34, 3.33, 3.33, 3.33, 3.33, 3.33,
              3.33, 3.33, 3.33, 3.33]

fit = fit_exponential_decay(T_values, ppl_values)
print(f"L_inf = {fit['L_inf']:.4f}")
print(f"Decay rate = {fit['rate']:.4f}")
print(f"R-squared = {fit['r_squared']:.4f}")
```

### Fit Unified Scaling Law

```python
from experiments.unified_law import fit_unified_law

T_values = [1, 2, 4, 8, 12]
loss_values = [5.12, 4.55, 4.02, 3.60, 3.45]

fit = fit_unified_law(T_values, loss_values)
print(f"L_inf = {fit['L_inf']:.4f}")
print(f"alpha = {fit['alpha']:.4f}")
print(f"beta = {fit['beta']:.4f}")
```

## Running Tests

```bash
cd /home/z/my-project/download/looped-lm

pytest tests/test_injection.py -v
pytest tests/test_flops.py -v
pytest tests/test_sampler.py -v
pytest tests/test_model.py -v

pytest tests/ -v
```

## Key Design Decisions

1. **Constrained Spectral Injection**: The InjectionBlock uses a parameterization `A_bar = exp(delta * (-exp(A)))` where `delta > 0` and `A` is unconstrained, guaranteeing `|A_bar_i| < 1` elementwise and thus `rho(A_bar) < 1`.

2. **Per-Sequence Depth Sampling**: Each sequence in a microbatch samples its own recurrence depth `T ~ Lambda(mu_rec)`, with gradient computed for `k = min(T, mu_bwd)` steps. This eliminates late-training loss spikes from microbatch-level sampling.

3. **Prelude Normalization**: A learned RMSNorm applied once before the first recurrence pass prevents residual magnitude explosion across many recurrent steps.

4. **Truncated BPTT with Correction**: The corrected sampling algorithm (`T ~ Lambda(mu_rec)`, `k = min(T, mu_bwd)`) matches full backpropagation while reducing memory proportional to `mu_bwd / mu_rec`.

5. **Weight Tying**: The LM head shares weights with the input embedding, reducing parameter count.
