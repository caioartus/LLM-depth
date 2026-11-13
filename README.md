# LLM depth: width vs depth at fixed compute, under SP and CompleteP

We train five GPT-2-style shapes with about 50M non-embedding parameters each (256×64, 384×28, 512×16, 640×10, 1024×4) at the same compute budget (≈ 8.48e17 FLOPs). Each shape is trained under four conditions:

| | LR transferred from 512×16 | LR tuned per shape |
|---|---|---|
| **SP** | (a) | (b) |
| **CompleteP** | (c) | (d) |

The write-up is in [report/main.tex](report/main.tex).

## Where this code comes from

The training code is a fork of [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) (branch `completep`), the reference implementation of CompleteP. nanoGPT-mup is itself a fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The last upstream commit is `88458c3`, and every commit after it is ours: `git diff 88458c3 HEAD` shows everything we changed.

## Folder layout

```
README.md            this file
report/              the report (LaTeX)                                    [ours]
code/                all code, configs and results
├── model.py         GPT model with muP / CompleteP                        [upstream, modified]
├── train.py         training loop                                         [upstream, modified]
├── configurator.py, csv_logging.py, bench.py, sample.py                   [upstream]
├── scaling_laws.ipynb, transformer_sizing.ipynb                           [upstream]
├── mup_examples/, completep_examples/, assets/                            [upstream] their coord checks and LR-transfer demos
├── README.md, LICENSE                                                     [upstream]
├── config/
│   ├── depth/       every config of this study (shapes, SP/CompleteP, tuning, coord check)   [ours]
│   ├── cluster_test.py                                                    [ours] quick cluster sanity run
│   └── the rest     (train_gpt2.py, eval_gpt2*.py, *shakespeare*.py)      [upstream]
├── data/
│   ├── slimpajama/prepare.py                                              [ours]
│   └── openwebtext/prepare.py                                             [upstream, modified] not used for the results
├── jobs/            SLURM scripts: data prep, training, resume-on-crash   [ours]
│   └── depth/       one script per experiment (tune, final, extra, coord check)
├── analysis/        collect_results.py, plot_coord_check.py, notebooks    [ours]
├── results/         logs of every run (CSV/JSON), tuned LRs, plots        [ours]
├── readings/        PDFs of the main papers                               [ours]
└── pyproject.toml, uv.lock, .python-version                               [ours]
```

**Our changes to upstream files:**
- `model.py`:
  - RoPE in place of learned position embeddings (blocks marked `### Begin RoPE code ###`)
  - FLOP count and MFU using our $C_\text{tok}$, measured against each GPU's own peak (blocks marked `### Begin depth-study code ###`)
- `train.py`: sequential (no-repeat) data reading, the same validation windows for every eval, a final eval at the last step, CSV logging with timing and MFU, per-layer gradient and Adam-update diagnostics (`diag.csv`), crash-safe checkpoints. All changes are inside `### Begin depth-study code ###` blocks.
- `data/openwebtext/prepare.py`: added a `--num_docs` option for taking a subset.

We removed upstream's `data/shakespeare*`. Everything else listed as upstream is unchanged.

## Reproducing the results

You can skip any step: every run's logs are already in `code/results/`, so the notebooks and the report work without a GPU.

**Requirements:**
- A SLURM cluster. The scripts use partition `gpu-long` and GPU names `a100-40` / `h100-96`, which are specific to our cluster; change them in `jobs/*.sbatch` and `jobs/depth/final.sh`.
- [uv](https://docs.astral.sh/uv/) with Python 3.14.

Run every command from `code/`.

```bash
cd code
uv sync --frozen                 # install the exact package versions from uv.lock
mkdir -p logs
```

**1. Data:** about 2.2B tokens from SlimPajama-6B, a ~1% sample of SlimPajama-627B. This writes a ~4.4 GB `data/slimpajama/train.bin` plus the Hugging Face cache (set `HF_HOME` to move the cache).
```bash
sbatch jobs/prepare_data.sh slimpajama --num_docs 2000000
```

**2. (Optional) Smoke test:** two 60-step runs, to check memory and resume.
```bash
jobs/depth/smoke_test.sh
```

**3. Coordinate check** (report appendix, figures `L_coord_check.png` and `width_coordinate_check.png`): one A100 job of 10-step runs, 3 seeds.
```bash
sbatch jobs/depth/coord_check.sbatch
uv run --frozen python analysis/collect_results.py
uv run --frozen python analysis/plot_coord_check.py     # -> results/coord_check/plots/{depth,width}.png
```

**4. LR tuning** (Table 2 and figure `tuning_transfer.png`): 5 shapes × 5 LRs per parameterisation, each on 10% of the shape's tokens, on A100-40GB.
```bash
jobs/depth/tune.sh sp
jobs/depth/tune.sh completep
uv run --frozen python analysis/collect_results.py      # picks the optimum per shape -> results/best_lr.csv
```
The figure is the "Final val loss vs LR" plot in `analysis/tuning.ipynb`.

**5. Final runs** (main results, conditions (a)–(d)): full compute budget, on H100-96GB, 2–6 h each. The script reads its learning rates from `results/best_lr.csv`.
```bash
DRY_RUN=1 jobs/depth/final.sh    # optional: print what would be submitted
jobs/depth/final.sh
uv run --frozen python analysis/collect_results.py
```

**6. Extra runs:**
- seeds 1–2 of both 512×16 runs, for noise estimates
- SP 256×64 at LR 2.4e-3 at full length
- the warmup check: SP 256×64 tuning runs with 50% warmup

```bash
jobs/depth/extra.sh
uv run --frozen python analysis/collect_results.py
```

**7. Analysis notebooks.** These read only `results/`:

| Notebook | Produces |
|---|---|
| `analysis/results.ipynb` | final losses, gaps, seed noise, loss curves, throughput, diagnostics; writes `results/plots/main_figure.png` (report Figure 1) |
| `analysis/tuning.ipynb` | LR sweeps, optima per shape (`tuning_transfer.png`) |
| `analysis/analyse_warmup.ipynb` | the warmup check (10% vs 50% warmup) |

**8. Report:** copy the figures into `report/figures/`, then build:
```bash
cd ../report && latexmk -pdf -outdir=build main.tex
```

**Notes:**
- Re-running any `jobs/depth/*.sh` script only submits runs that are missing: finished runs are skipped and crashed runs resume from their checkpoint.
- Runs write to `code/out/<name>/`, which is not in git because of the checkpoints. `collect_results.py` copies the small log files into `results/`.
- The seed is 0 for every run except the extra seeds, and every run reads the same tokens in the same order.
