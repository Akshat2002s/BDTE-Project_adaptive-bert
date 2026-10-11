# Adaptive BERT for financial intent triage at the edge

Big Data at the Edge project. One BERT-base model that is elastic in
width (nested, importance-ordered heads and FFN neurons) and depth (exit heads after
layers 4/8/12), distilled from the full model, with a queue-aware MDP controller that
picks the operating point at runtime. Task: Banking77 intent classification (77
classes) on an on-premise CPU gateway.

## Setup

Local (Windows, conda). The env, pip temp files and the HuggingFace model cache all
live on D: next to the repo (`..\.envs`, `..\.cache`), because C: has no room for the
CUDA build:

```bash
conda env create -p "D:\UvA_VU\Period 7\Big Data at the Edge\.envs\adaptive-bert" -f environment.yml
conda activate "D:\UvA_VU\Period 7\Big Data at the Edge\.envs\adaptive-bert"
pip install --force-reinstall --no-deps torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126   # GPU build
conda env config vars set HF_HOME="D:\UvA_VU\Period 7\Big Data at the Edge\.cache\huggingface"
```

The local GPU (GTX 1650, 4 GB) is enough for the baselines and single elastic runs.
DAS-5 is for sweeps and ablations, not a requirement.

DAS-5 (venv, same `requirements.txt`):

```bash
python -m venv ~/venvs/adaptive-bert && source ~/venvs/adaptive-bert/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt -e .
```

Check everything works: `pytest`.

## Workflow

```bash
python scripts/prepare_data.py                 # once: frozen split -> data/splits/ (commit it)
python scripts/train_baseline.py bert-base     # fp32; add --amp only on tensor-core GPUs (see configs/base.yaml)
python scripts/measure.py checkpoints/bert-base --name bert-base
python scripts/train_elastic.py                # -> checkpoints/elastic (teacher: checkpoints/bert-base)
python scripts/measure_elastic.py              # every (width, exit) and (width, threshold) point
python scripts/simulate.py                     # bursty traffic: fixed points vs greedy rule vs MDP
python scripts/plot_pareto.py                  # results/pareto_test.png
```

Ablations: `train_elastic.py --no-kd --out checkpoints/elastic-nokd` and
`--random-order --out checkpoints/elastic-random`, each followed by `measure_elastic.py <dir>`.

## Frozen decisions

Changing any of these invalidates every number already in `results/points.jsonl`.

| Decision | Value | Where |
|---|---|---|
| Validation split | 10% of official train, stratified, seed 0, indices committed | `data/splits/banking77_split.json` |
| Tuning vs reporting | thresholds and MDP tuned on **val**; Pareto reported on **test** | everywhere |
| Sequence length | 64 tokens, fixed, for training, eval and FLOPs | `configs/base.yaml` |
| FLOPs tool | `torch.utils.flop_counter.FlopCounterMode`, batch 1, 2 x MACs | `src/abert/flops.py` |
| Early-exit cost | expected FLOPs = sum of (exit fraction x FLOPs to that exit) | `flops.expected_flops` |
| Attention kernel | `attn_implementation="eager"` for every model | `src/abert/models.py` |
| Latency | CPU, batch 1, 4 threads, median + p95 over 100 runs, same machine for all points | `src/abert/bench.py` |
| Results | append-only `results/points.jsonl`, one line per (point, split) | `src/abert/registry.py` |

Why FlopCounterMode rather than fvcore/thop/ptflops: it counts the ops that actually
run, so an early-exit forward is counted correctly, and it needs no extra
dependency. `tests/test_flops.py` checks it against a hand calculation.

## Layout

```
configs/base.yaml        frozen settings, baseline / elastic / simulation hyperparameters
src/abert/data.py        Banking77 download, frozen split, tokenisation
src/abert/flops.py       the one FLOPs methodology
src/abert/bench.py       latency, size, params
src/abert/registry.py    results/points.jsonl read/write + Pareto mask
src/abert/models.py      static baseline builders
src/abert/train.py       fine-tuning loop (best epoch on val)
src/abert/elastic.py     ElasticBert: importance ordering, width slicing, exit heads
src/abert/controller.py  queue-aware MDP (value iteration) + greedy SLA baseline
src/abert/sim.py         discrete-event simulation of the gateway under bursty traffic
scripts/                 prepare_data, train_baseline, measure, train_elastic,
                         measure_elastic, simulate, plot_pareto, bench_precision
slurm/                   DAS-5 job scripts
tests/                   FLOPs, registry, split, elastic model and controller checks
```

## Plan

| Phase | Model and training | Benchmarking, controller, simulation |
|---|---|---|
| 1 | Static baselines trained and frozen | FLOPs/latency harness, baseline points registered |
| 2 | ElasticBert: importance ordering, slicing, exits, distillation (local GPU) | Exit calibration on val, per-config latency table |
| 3 | Fixes, INT8 dynamic quantisation, training sweeps on DAS-5 | MDP (value iteration), bursty traffic simulation, Pareto sweeps |
| 4 | Ablations | Slides |

## Run history

Nothing is overwritten; superseded runs are renamed. `results/points.jsonl` is
append-only, so the old lines are still there: entries timestamped before
2026-10-11T18:00 that point at `checkpoints/bert-base` or `checkpoints/elastic`
refer to the models now in the `-lr3e-5` / `-3ep` folders below.

| Run | Checkpoint | Logs / results | Notes |
|---|---|---|---|
| BERT-base, lr 3e-5 | `bert-base-lr3e-5` | `logs/*bert-base-lr3e-5.log` | 90.61% val / 91.20% test; beaten by DistilBERT, so retrained |
| Elastic, 3 epochs | `elastic-3ep` | `logs/*elastic-3ep.log`, `results/elastic-3ep_*`, `results/pareto_test_3ep.png` | Reference for the two ablations below |
| Ablation: no KD, 3 epochs | `elastic-nokd` | `logs/train_elastic-nokd.log` | Mean val 83.03% vs 83.10% with KD: no measurable effect |
| Ablation: random order, 3 epochs | `elastic-random` | `logs/train_elastic-random.log` | Mean val 79.85% vs 83.10%: importance ordering helps, most at narrow widths |
| BERT-base, lr 5e-5, failed | `bert-base-lr5e-5_failed-oom` | `logs/train_bert-base-lr5e-5_failed-oom.log` | Crashed saving epoch 2 (laptop out of RAM + page file); folder is inconsistent, see `FAILED.txt` |
| BERT-base, lr 5e-5 (final) | `bert-base` | `logs/*bert-base.log` | Final static baseline and distillation teacher |
| Elastic, 10 epochs (final) | `elastic` | `logs/*elastic.log`, `results/elastic_*` | Final adaptive model |

## Decided

- Exit heads: one per depth, shared across widths (the hidden size stays 768).
- Training rule for the extreme widths: sandwich rule (largest + smallest + one random
  middle width per step), KD from the frozen bert-base on every exit.

## Open questions

- Statically head-pruned baselines: which ratios (can reuse `importance.pt` from the
  elastic run).
