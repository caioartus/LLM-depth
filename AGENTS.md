# Guide for agents

This repo is a course project: a study of Transformer width vs depth at fixed compute, under SP and CompleteP. Start with [README.md](README.md) (overview, folder layout, reproduction steps). The report is [report/main.tex](report/main.tex) (built PDF: `report/build/main.pdf`).

No GPU is needed to check the results. Every run's logs are committed in `code/results/`, and the analysis notebooks read only that folder.

## Where each claim in the report comes from

| Report item | Evidence | Produced by |
|---|---|---|
| Table 1, Figure 1 (tuned LRs, LR sweeps) | `code/results/best_lr.csv`, `best_lr.md`, `results/runs/tune-*` | `code/analysis/collect_results.py`, `code/analysis/tuning.ipynb` |
| Table 2, Figure 2 left (final losses) | `code/results/runs.csv`, rows with `kind=final` (column `condition`: a, b, c, d, or a+b / c+d for a shared run) | `code/analysis/results.ipynb` |
| Extra SP 256×64 run at 2.4e-3 | `runs.csv`, row `check-sp-256x64-lr2.4e-3` | `results.ipynb` |
| Seed noise (Section 2.4) | `runs.csv`, rows with `kind=seed` plus the seed-0 512×16 final runs | `results.ipynb` |
| Warmup check (Figure 2 right) | `runs.csv`, rows `warmup-sp-256x64-*` vs `tune-sp-256x64-*` vs the 256×64 SP full runs | `code/analysis/analyse_warmup.ipynb`, `results.ipynb` |
| Figure 3 (MFU, training time) | `runs.csv`, columns `mfu`, `train_time_h` | `results.ipynb` → `code/results/plots/throughput.png` |
| Table 3 (FLOPs, tokens, steps) | `_flops_per_token` and `max_iters` in `code/config/depth/common.py`; `max_iters` in each `results/runs/*/config.json` | |
| Figures 4–5 (coordinate check) | `code/results/coord_check/*.csv` | `code/analysis/plot_coord_check.py` |

## Where the method is implemented

- Model, RoPE, CompleteP (residual scaling `m_L^-1` in `Block.forward`, init and per-group LR / Adam eps in `GPT.__init__` and `configure_optimizers`), FLOP count for MFU: `code/model.py`.
- Training loop changes (sequential data, fixed validation windows, final eval, logging, diagnostics): `code/train.py`, inside `### Begin depth-study code ###` blocks.
- Hyperparameters: `code/config/depth/common.py` (shared), `completep_common.py` (CompleteP multipliers), `tune.py` (10% tuning runs), `sp/` and `completep/` (one file per shape).
- Cluster scripts: `code/jobs/` (SLURM).
- Upstream code: see the README. `git diff 88458c3 HEAD` shows all of our changes.

## Run names (folders in `code/results/runs/`)

`tune-<param>-<shape>-lr<lr>` (tuning), `sp-512x16` (SP, conditions a and b), `sp-transfer-<shape>` (a), `sp-tuned-<shape>` (b), `completep-<shape>` (c, and d where the tuned LR equals the transferred one), `completep-tuned-<shape>` (d), `seed<n>-<param>-512x16` (extra seeds), `check-sp-256x64-lr2.4e-3` (extra full-length LR), `warmup-sp-256x64-lr<lr>` (tuning runs with 50% warmup). Each folder has `config.json` (the full resolved config, including the GPU), `eval.csv` (validation loss), `train.csv`, `diag.csv` (per-layer gradient and update statistics) and `run_info.txt` (git commit and command).
