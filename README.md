Hurkolt Transzformer Nyelvi Modell

Egy gyártásra kész implementáció hurkolt (rekurrens) transzformer nyelvi modellekhez korlátozott spektrális injektálással, konfigurálható rekurrencia mélység mintavételezéssel, valamint átfogó kísérleti suite-okkal a stabilitás elemzéséhez, minőségi benchmarkokhoz és skálázási törvény validációhoz.

 Tartalomjegyzék

- [Telepítés](telepítés)
- [Projekt Struktúra](projekt-struktúra)
- [Konfigurációs Referencia](konfigurációs-referencia)
- [Tokenizáló Tanítás](tokenizáló-tanítás)
- [Modell Tanítás](modell-tanítás)
- [Elosztott Tanítás](elosztott-tanítás)
- [Értékelés](értékelés)
- [Kísérletek Futtatása](kísérletek-futtatása)
- [Skálázási Törvény Illesztés](skálázási-törvény-illesztés)

---

 Telepítés

bash
pip install torch transformers datasets tokenizers wandb scipy matplotlib numpy pyyaml regex


CUDA támogatás esetén telepítse a megfelelő PyTorch verziót a GPU-jához:

bash
pip install torch --index-url https://download.pytorch.org/whl/cu121


 Projekt Struktúra


looped-lm/
├── model/
│   ├── architecture.py       LoopedTransformer, FixedDepthTransformer, MODEL_CONFIGS
│   ├── blocks.py             RMSNorm, CausalAttention, SwiGLUMLP, ReLU2MLP
│   ├── injection.py          InjectionBlock, PreludeNorm, spectral_radius_A_bar
│   └── init.py               scaled_init, scaled_zero_init, init_looped_model
├── training/
│   └── sampler.py            RecurrenceDepthSampler, TruncatedBaselineSampler
├── scaling/
│   └── flops.py              effective_params, training_flops, attention_flops_per_token
├── tokenizer/
│   ├── __init__.py
│   └── train.py              BPE tokenizer training, FineWeb-Edu preparation
├── configs/
│   ├── 100M_looped.yaml      Config A: 100M looped model
│   ├── 350M_looped.yaml      Config A: 350M looped model
│   ├── 140M_looped.yaml      Config B: 140M looped model
│   ├── 370M_looped.yaml      Config B: 370M looped model
│   ├── 770M_looped.yaml      Config B: 770M looped model
│   ├── 1.3B_looped.yaml      Config B: 1.3B looped model
│   ├── 140M_baseline.yaml    Config B: 140M fixed-depth baseline (6 layers)
│   ├── 370M_baseline.yaml    Config B: 370M fixed-depth baseline (12 layers)
│   ├── 770M_baseline.yaml    Config B: 770M fixed-depth baseline (18 layers)
│   └── 1.3B_baseline.yaml    Config B: 1.3B fixed-depth baseline (24 layers)
├── experiments/
│   ├── __init__.py
│   ├── stability_sweep.py            Exp 7.1: Stability Sweep
│   ├── quality_vs_baseline_loop.py   Exp 7.2: Quality vs Looped Baseline
│   ├── ablation_stability_components.py   Exp 7.3: Stability Ablation Table
│   ├── quality_vs_fixed_transformer.py    Exp 7.4: Quality vs Fixed-Depth
│   ├── isoflop_sweep.py             Exp 7.5: IsoFLOP Training Scaling
│   ├── test_time_saturation.py      Exp 7.6: Test-Time Saturation
│   ├── unified_law.py               Exp 7.7: Unified Scaling Law
│   ├── ablation_mu_rec.py           Exp 7.8a: mu_rec sweep
│   ├── ablation_mu_bwd.py           Exp 7.8b: mu_bwd sweep
│   ├── ablation_sampling_algo.py    Exp 7.9: Sampling Algorithm Ablation
│   ├── ablation_per_sequence.py     Exp 7.10: Per-Sequence Sampling
│   └── ablation_prelude_norm.py     Exp 7.11: Prelude-Norm Ablation
├── tests/
│   ├── __init__.py
│   ├── test_injection.py
│   ├── test_flops.py
│   ├── test_sampler.py
│   └── test_model.py
└── README.md


 Konfigurációs Referencia

 Modell Konfigurációk

| Konfig | Típus | d_model | n_heads | head_dim | d_ffn | L_P | L_R | L_C | mu_rec | mu_bwd | Vocab |
|--------|-------|---------|---------|----------|-------|-----|-----|-----|--------|--------|-------|
| 100M   | A     | 768     | 12      | 64       | 3072  | 2   | 6   | 2   | 16     | 8      | 65536 |
| 350M   | A     | 1024    | 16      | 64       | 4096  | 2   | 10  | 2   | 16     | 8      | 65536 |
| 140M   | B     | 768     | 6       | 128      | 3072  | 2   | 6   | 2   | 8      | 4      | 32768 |
| 370M   | B     | 1024    | 8       | 128      | 4070  | 2   | 8   | 2   | 8      | 4      | 32768 |
| 770M   | B     | 1536    | 12      | 128      | 4070  | 2   | 8   | 2   | 8      | 4      | 32768 |
| 1.3B   | B     | 2048    | 16      | 128      | 8192  | 2   | 8   | 2   | 8      | 4      | 32768 |

 Konfig A vs Konfig B

- Konfig A: ReLU^2 MLP-t használ, AdamW optimalizálót, standard csomagolást, vocab_size=65536
- Konfig B: SwiGLU MLP-t használ, Muon+AdamW optimalizálót, bestfit_crop csomagolást, vocab_size=32768

 Főbb YAML Mezők

yaml
model:
  config_name: "140M"         Modell méret azonosító
  config_type: "B"            A vagy B
  max_seq_len: 2048           Maximális sorozat hossz
  vocab_size: 32768           Szókincs méret
  mu_rec: 8                   Átlagos rekurrencia mélység
  mu_bwd: 4                   Átlagos backward (gradiens) mélység
  head_dim: 128               Figyelem fej dimenzió
  prelude_norm: true          PreludeNorm engedélyezése

training:
  adamw_lr: 8.0e-3            AdamW tanulási ráta (Konfig B esetén)
  muon_lr: 8.0e-3             Muon tanulási ráta
  adamw_beta1: 0.8            AdamW beta1
  adamw_beta2: 0.95           AdamW beta2
  adamw_weight_decay: 0.0     AdamW súlycsökkenés
  muon_momentum: 0.95         Muon momentum
  muon_weight_decay_start: 0.2
  muon_weight_decay_end: 0.0
  grad_clip: 1.0              Gradiens vágás norma
  cooldown_start_frac: 0.5    A tanítás hányadában kezdődik a cooldown
  total_tokens: 11_200_000_000
  global_batch_size: 256
  micro_batch_size: 32
  seq_len: 2048
  bf16: true                  BF16 vegyes pontosság használata

data:
  tokenizer_path: "tokenizer_config_b.json"
  train_path: null            Állítsa be a saját tanító adat elérési útját
  val_path: null              Állítsa be a saját validációs adat elérési útját
  packing: "bestfit_crop"     "standard" vagy "bestfit_crop"

evaluation:
  val_every: 2000             Értékelés minden N lépésben
  eval_tasks: [...]           Értékelési feladatok
  core_tasks: true            Core benchmark átlag számítása
  core_ext_tasks: true        Core-Extended benchmark átlag számítása
  test_time_sweep: false      Értékelés több T értéken


 Tokenizáló Tanítás

 Konfig B Tokenizáló Tanítása (FineWeb-Edu)

bash
python -m tokenizer.train \
  --mode config_b \
  --input-path /path/to/fineteweb-edu.jsonl \
  --output-path tokenizer_config_b.json \
  --vocab-size 32768 \
  --max-chars-per-doc 10000 \
  --max-total-chars 2000000000


 Konfig A Tokenizáló Tanítása

bash
python -m tokenizer.train \
  --mode config_a \
  --input-path /path/to/corpus.txt \
  --output-path tokenizer_config_a.json \
  --vocab-size 65536


 Egyszerű Tartalék Tokenizáló Létrehozása

bash
python -m tokenizer.train \
  --mode simple \
  --output-path tokenizer_simple.json \
  --vocab-size 65536


 Programozott Használat

python
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


 Modell Tanítás

 Egyetlen GPU-s Tanítás

bash
python train.py \
  --config configs/140M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin


 Tanítás Felülíró Paraméterekkel

bash
python train.py \
  --config configs/370M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --training.micro-batch-size 16 \
  --training.global-batch-size 128


 Tanítás Modell Méret Szerint

| Modell | Konfig Fájl          | GPU-k | VRAM/GPU | Becsült Idő |
|--------|----------------------|-------|----------|-------------|
| 100M   | 100M_looped.yaml     | 1x A100 40GB | ~8 GB    | ~12 óra     |
| 350M   | 350M_looped.yaml     | 1x A100 40GB | ~16 GB   | ~24 óra     |
| 140M   | 140M_looped.yaml     | 1x A100 40GB | ~10 GB   | ~18 óra     |
| 370M   | 370M_looped.yaml     | 2x A100 40GB | ~20 GB   | ~3 nap      |
| 770M   | 770M_looped.yaml     | 4x A100 40GB | ~32 GB   | ~8 nap      |
| 1.3B   | 1.3B_looped.yaml     | 8x A100 80GB | ~50 GB   | ~14 nap     |

 Baseline Modellek

bash
python train.py --config configs/140M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/370M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/770M_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin
python train.py --config configs/1.3B_baseline.yaml --data.train-path /path/to/train.bin --data.val-path /path/to/val.bin


 Elosztott Tanítás

 DDP (Distributed Data Parallel)

bash
torchrun --nproc_per_node=4 train.py \
  --config configs/370M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=ddp


 FSDP (Fully Sharded Data Parallel)

bash
torchrun --nproc_per_node=8 train.py \
  --config configs/1.3B_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=fsdp \
  --fsdp-sharding-strategy full \
  --fsdp-wrapping-policy transformer


 Több Csomópontos DDP

bash
torchrun --nnodes=4 --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train.py \
  --config configs/770M_looped.yaml \
  --data.train-path /path/to/train.bin \
  --data.val-path /path/to/val.bin \
  --distributed backend=ddp


 Értékelés

 Összes Értékelési Feladat Futtatása

bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --tasks hellaswag_0shot arc_c arc_e piqa boolq copa winogrande lambada


 Tesztidejű Rekurrencia Sweep

bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --test-time-sweep \
  --sweep-range 1 24


 Core és Core-Extended Benchmark Számítása

bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/370M_looped.yaml \
  --core \
  --core-extended


 WikiText Perplexity

bash
python evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/140M_looped.yaml \
  --wikitext-path /path/to/wikitext-103.txt


 Kísérletek Futtatása

 Kísérlet 7.1: Stabilitás Sweep

Három modellváltozatban sweepeli a csúcs tanulási rátát a stabilitás ellenőrzésére.

bash
python experiments/stability_sweep.py \
  --config configs/140M_looped.yaml \
  --output-dir results/stability_sweep \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~4 óra változatanként, 3 változat x 5 LR = 15 futtatás

Kimenet: results/stability_sweep/stability_sweep_results.json

 Kísérlet 7.2: Minőség a Hurkolt Baseline-hez képest

Összehasonlítja a korlátozott-A hurkolt modellt a korlátlan hurkolt baseline-nel maradék normalizációval.

bash
python experiments/quality_vs_baseline_loop.py \
  --config configs/100M_looped.yaml \
  --output-dir results/quality_vs_baseline \
  --devices cuda:0

python experiments/quality_vs_baseline_loop.py \
  --config configs/350M_looped.yaml \
  --output-dir results/quality_vs_baseline \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~24 óra skálánként, 2 skála (100M, 350M)

Cél: >=6% PPL csökkenés a baseline-hez képest

 Kísérlet 7.3: Stabilitás Abláció Táblázat

Transzformer gerincet injektálással lát el, fokozatosan hozzáadva a stabilitási komponenseket.

bash
python experiments/ablation_stability_components.py \
  --config configs/140M_looped.yaml \
  --output-dir results/stability_ablation \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~4 óra (4 konfiguráció)

 Kísérlet 7.4: Minőség Fix-Mélységű Transzformerhez képest

Teljes összehasonlítás az összes modellméretben.

bash
python experiments/quality_vs_fixed_transformer.py \
  --config-dir configs \
  --output-dir results/quality_vs_fixed \
  --devices cuda:0


Erőforrások: 1x A100 40GB modellméreténként, ~18 óra (140M), ~3 nap (370M), ~8 nap (770M), ~14 nap (1.3B)

Megjegyzés: Nagyobb modelleknél futtassa az egyes méreteket külön GPU-kon függetlenül.

 Kísérlet 7.5: IsoFLOP Tanítási Skálázási Sweep

Fix FLOP költségvetések mellett sweepeli a mu_rec értéket a skálázási törvények illesztéséhez.

bash
python experiments/isoflop_sweep.py \
  --config configs/140M_looped.yaml \
  --output-dir results/isoflop_140M \
  --devices cuda:0

python experiments/isoflop_sweep.py \
  --config configs/370M_looped.yaml \
  --output-dir results/isoflop_370M \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~8 óra konfigurációnként, 6 mu_rec érték x 6 költségvetés (140M) vagy 3 költségvetés (370M)

Kimenet: IsoLoss kontúr ábrák PNG formátumban, JSON eredmények parabolikus illesztésekkel

 Kísérlet 7.6: Tesztidejű Telítődés Sweep

Értékeli a PPL exponenciális csökkenését a tesztidejű rekurrencia T növekedésével.

bash
python experiments/test_time_saturation.py \
  --config configs/140M_looped.yaml \
  --output-dir results/test_time_sat_140M \
  --devices cuda:0

python experiments/test_time_saturation.py \
  --config configs/370M_looped.yaml \
  --output-dir results/test_time_sat_370M \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~6 óra konfigurációnként, 6 mu_rec érték x T=1..24 értékelés

Ellenőrzés: L_inf 0.6%-on belül L(mu_rec)-től

 Kísérlet 7.7: Egységes Skálázási Törvény Validáció

Illeszti és validálja az egységes skálázási törvényt az IsoFLOP eredmények felhasználásával.

bash
python experiments/unified_law.py \
  --isoflop-results-dir results/isoflop_140M \
  --heldout-results-dir results/quality_vs_fixed \
  --output-dir results/unified_law \
  --devices cuda:0


Erőforrások: Csak CPU (nincs tanítás), ~5 perc

Ellenőrzés: <=1.31% MAPE a prediktált padlóval, <=0.17% az oracle padlóval

 Kísérlet 7.8a: mu_rec Abláció

bash
python experiments/ablation_mu_rec.py \
  --config configs/140M_looped.yaml \
  --output-dir results/ablation_mu_rec \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~8 óra, 6 mu_rec érték {4,8,14,20,26,32}

 Kísérlet 7.8b: mu_bwd Abláció

bash
python experiments/ablation_mu_bwd.py \
  --config configs/140M_looped.yaml \
  --output-dir results/ablation_mu_bwd \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~6 óra, 5 mu_bwd érték {4,6,8,10,12}

Ellenőrzés: mu_bwd = ceil(mu_rec/2) ajánlott

 Kísérlet 7.9: Mintavételi Algoritmus Abláció

bash
python experiments/ablation_sampling_algo.py \
  --config configs/100M_looped.yaml \
  --output-dir results/ablation_sampling \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~12 óra, 3 algoritmus (teljes backprop, truncated, corrected)

Ellenőrzés: A corrected algoritmus megegyezik a teljes backpropagation-nel, felülmúlja a truncated-et

 Kísérlet 7.10: Sorozatonkénti vs Mikrokötegenkénti Mintavételezés

bash
python experiments/ablation_per_sequence.py \
  --config configs/100M_looped.yaml \
  --output-dir results/ablation_per_seq \
  --devices cuda:0

python experiments/ablation_per_sequence.py \
  --config configs/350M_looped.yaml \
  --output-dir results/ablation_per_seq \
  --devices cuda:0


Erőforrások: 1x A100 40GB, ~24 óra skálánként, 2 mód x 2 skála

Ellenőrzés: A per-sequence kiküszöböli a késői tanítási tüskéket, <=2% overhead

 Kísérlet 7.11: Prelude-Norm Abláció 1.3B-n

bash
python experiments/ablation_prelude_norm.py \
  --config configs/1.3B_looped.yaml \
  --output-dir results/ablation_prelude_norm \
  --devices cuda:0


Erőforrások: 2x A100 80GB, ~28 nap összesen (2 futás 104B tokennel)

Ellenőrzés: Prelude-norm nélkül maradék robbanást mutat; prelude-norm-mal stabil marad

 Skálázási Törvény Illesztés

 IsoFLOP Parabolikus Törvény Illesztése

python
from experiments.isoflop_sweep import fit_parabolic_law

mu_values = [2, 4, 6, 8, 10, 12]
val_losses = [4.82, 4.15, 3.87, 3.72, 3.65, 3.61]

fit = fit_parabolic_law(mu_values, val_losses)
print(f"R-squared: {fit['r_squared']:.4f}")
print(f"Parameters: a={fit['params']['a']:.4f}, b={fit['params']['b']:.4f}, c={fit['params']['c']:.4f}")


 Exponenciális Telítődési Törvény Illesztése

python
from experiments.test_time_saturation import fit_exponential_decay, exponential_decay

T_values = list(range(1, 25))
ppl_values = [5.12, 4.55, 4.18, 3.92, 3.74, 3.61, 3.52, 3.46, 3.42, 3.39,
              3.37, 3.36, 3.35, 3.34, 3.34, 3.33, 3.33, 3.33, 3.33, 3.33,
              3.33, 3.33, 3.33, 3.33]

fit = fit_exponential_decay(T_values, ppl_values)
print(f"L_inf = {fit['L_inf']:.4f}")
print(f"Decay rate = {fit['rate']:.4f}")
print(f"R-squared = {fit['r_squared']:.4f}")


 Egységes Skálázási Törvény Illesztése

python
from experiments.unified_law import fit_unified_law

T_values = [1, 2, 4, 8, 12]
loss_values = [5.12, 4.55, 4.02, 3.60, 3.45]

fit = fit_unified_law(T_values, loss_values)
print(f"L_inf = {fit['L_inf']:.4f}")
print(f"alpha = {fit['alpha']:.4f}")
print(f"beta = {fit['beta']:.4f}")


 Teszt Futtatása

bash
cd /home/z/my-project/download/looped-lm

pytest tests/test_injection.py -v
pytest tests/test_flops.py -v
pytest tests/test_sampler.py -v
pytest tests/test_model.py -v

pytest tests/ -v


 Főbb Dizájn Döntések

1. Korlátozott Spektrális Injektálás: Az InjectionBlock a A_bar = exp(delta  (-exp(A))) paraméterezést használja ahol delta > 0 és A korlátlan, garantálva hogy |A_bar_i| < 1 elemenként és így rho(A_bar) < 1.

2. Sorozatonkénti Mélység Mintavételezés: Minden sorozat egy mikrokötegben saját rekurrencia mélységet mintáz T ~ Lambda(mu_rec), a gradienst k = min(T, mu_bwd) lépésre számolja. Ez kiküszöböli a késői tanítási veszteség tüskéket a mikroköteg-szintű mintavételezésből.

3. Prelude Normalizáció: Egy tanult RMSNorm alkalmazva egyszer a első rekurrencia pass előtt megakadályozza a maradék magnitúdó robbanását sok rekurrens lépés során.

4. Truncated BPTT Javítással: A javított mintavételi algoritmus (T ~ Lambda(mu_rec), k = min(T, mu_bwd)) megegyezik a teljes backpropagation-nel miközben csökkenti a memóriát arányosan mu_bwd / mu_rec.

5. Súlymegosztás: Az LM fej megosztja a súlyait a bemeneti embeddinggel, csökkentve a paraméterszámot.
