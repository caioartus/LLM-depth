# Experiment spec: depth/width allocation at fixed compute (50M scale)

## Research question
At fixed compute, what is the best depth/width allocation, and does it depend on the parameterization (SP or CompleteP, each with the LR transferred from one shape or tuned per shape)? Testing Depth Delusion's claim (loss rises past a critical depth) is one prediction within this.

## Fixed setup (all runs)
- Decoder-only transformer, pre-LN, tied embeddings (unembedding $= W_E^\top$), no dropout
- Tied embeddings with CompleteP: shared matrix, logits $= m_N^{-1}\, h\, W_E^\top$ (only the forward multiplier differs; init, LR and AdamW $\epsilon$ are identical for emb/unemb in CompleteP Table 1). SP: logits $= h\, W_E^\top$. Deviation from the CompleteP paper (untied), matches their nanoGPT-mup code.
- RoPE positional encoding
- $d_{\text{head}} = 64$ (heads $= d/64$), FFN width $4d$
- Sequence length $T = 2048$
- Dataset: SlimPajama. GPT-2 tokenizer, vocab padded to $V = 50{,}304$
- LR schedule: linear warmup over 10% of the run's steps (CompleteP rule $\min(10\%, 375\text{M tokens})$; cap never binds at this scale), then cosine decay to 0 at the run's last step. Every run, including tuning runs, is stretched to its own length.
- Attention scaling: $Q^\top K/\sqrt{d_{\text{head}}}$ (SP), $Q^\top K/d_{\text{head}}$ (CompleteP)
- Non-embedding params: $P_{\text{ne}} = 12\,d^2 L$ (held ≈ 50M across shapes)
- Total params (tied): $P_{\text{total}} = P_{\text{ne}} + dV$

## Optimizer and batch (all conditions, taken from CompleteP)
Principle: identical across all three conditions; only LR and parameterization differ.
- AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$
- $\epsilon_{\text{base}} = 10^{-16}$, optimizer states in fp32. CompleteP (c) applies its Table 1 $\epsilon$ scaling on top of this base (hidden: $\epsilon\, m_N^{-1} m_L^{-\alpha}$; emb/unemb: $\epsilon\, m_N^{-1}$); SP uses $10^{-16}$ unscaled.
- Weight decay $\lambda_{\text{base}} = 0$ (AdamW reduces to Adam; CompleteP's $\lambda\, m_N$ rule is moot)
- Gradient clipping: global norm 1.0 (same as Depth Delusion; check value in CompleteP's nanoGPT-mup code)
- Batch: 128 sequences × 2048 = 262,144 tokens per step, fixed across shapes and conditions (same token count per step as Depth Delusion: 256 × 1024). Gradient accumulation as needed; micro-batch chosen per shape for memory (only the effective batch matters). 256×64 has ≈ $2\times$ the activation memory of 512×16 (scales with $Ld$).
- Precision: bf16 autocast, fp32 master weights and optimizer states

Why $\epsilon$ matters here: Adam's update is $\approx g/(\sqrt{v}+\epsilon)$. For $|g| \gg \epsilon$ gradient scale cancels, so "gradient starvation" barely reduces per-layer update size; if deep-layer gradients approach $\epsilon$, the update shrinks to $\approx g/\epsilon$ and starvation becomes real. Main runs use CompleteP's $10^{-16}$ for all conditions. $\epsilon$ is not copied from Depth Delusion (their reported $10^{-8}$, Table 5, is not taken as reliable); sensitivity to $\epsilon$ is tested separately as an ablation.

## Compute accounting
Per-token training cost (forward + backward):
$$C_{\text{tok}} = 6P_{\text{ne}} + 6dV + 6LTd$$
- $6P_{\text{ne}}$: transformer blocks
- $6dV$: unembedding matmul (runs whether tied or not; embedding lookup ≈ 0 FLOPs)
- $6LTd$: attention scores + weighted sum (causal)

Run cost: $C = C_{\text{tok}} \times D$, with $D$ = training tokens.
MFU $= \text{tokens/s} \times C_{\text{tok}} / \text{peak FLOP/s}$

## Budget
- Reference model: $d = 512$, $L = 16$ ($P_{\text{ne}} = 50.3$M, $P_{\text{total}} = 76.1$M)
- Reference tokens: 20 per total parameter (Chinchilla convention) → $D_{\text{ref}} = 1.52$B
- Fixed per-run budget: $C = D_{\text{ref}} \times C_{\text{tok}}(\text{ref}) = 1.52\text{B} \times 5.57\times10^{8} \approx 8.48\times10^{17}$ FLOPs
- Every final run gets the same $C$; tokens per shape: $D = C / C_{\text{tok}}(d, L)$
- Budget declared in FLOPs; wall-clock reported separately (≈2.2 h on A100 at 35% MFU for the reference; deep-narrow shapes slower; to be measured)

## Shapes
| d    | L  | heads | d/L | P_ne  | P_total | C_tok  | Tokens D | Tokens/P_total | Steps | Warmup steps |
|------|----|-------|-----|-------|---------|--------|----------|----------------|-------|--------------|
| 256  | 64 | 4     | 4   | 50.3M | 63.2M   | 5.81e8 | 1.46B    | 23.1           | 5,570 | 557          |
| 384  | 28 | 6     | 14  | 49.5M | 68.9M   | 5.45e8 | 1.55B    | 22.6           | 5,913 | 591          |
| 512  | 16 | 8     | 32  | 50.3M | 76.1M   | 5.57e8 | 1.52B    | 20.0           | 5,798 | 580          |
| 640  | 10 | 10    | 64  | 49.2M | 81.3M   | 5.67e8 | 1.50B    | 18.4           | 5,722 | 572          |
| 1024 | 4  | 16    | 256 | 50.3M | 101.8M  | 6.61e8 | 1.28B    | 12.6           | 4,883 | 488          |

Steps $= D / 262{,}144$. Tuning runs (10% of tokens) ≈ 490–590 steps with ≈ 49–59 warmup steps.
Fallback if over budget: drop 640×10.

## Conditions
- (a) SP-transfer: SP, LR tuned on 512×16 only, transferred unchanged to all other shapes. Mimics Depth Delusion's procedure: one LR ($3\times10^{-4}$) for every shape, and their only reported LR sweep is at depth 16 (App. A.1; width unstated, run at a different batch size than their main runs). Same protocol as (c), so (a) vs (c) isolates the parameterization.
- (b) SP-tuned: SP, LR tuned per shape (upper bound for SP)
- (c) CompleteP: $\alpha = 1$, LR tuned on 512×16 only, transferred unchanged to all other shapes
- (d) CompleteP-tuned: CompleteP, $\eta_{\text{base}}$ tuned per shape (upper bound for CompleteP). Added after the first CompleteP sweeps (2026-10-27) found the 1024×4 optimum 2 grid steps below 512×16's ($2^{-9}$ vs $2^{-7}$, ≈0.18 val loss at the transferred LR)

The design is 2×2: {SP, CompleteP} × {transferred LR, per-shape tuned LR}. Reading the gaps: (a) → (b) = cost of SP's LR not transferring across shapes; (c) → (d) = the same for CompleteP; (b) vs (d) = the parameterizations compared with both properly tuned, so the depth conclusion does not depend on transfer being perfect. A U-curve in (a)/(c) that disappears in (b)/(d) points to LR detuning rather than depth itself.

Why transfer across shapes at fixed size is in CompleteP's scope: its claim is that the optimal $\eta_{\text{base}}$ is invariant to width $N$ (the μP part) and depth $L$ (the $\alpha = 1$ part) separately; a reshape at fixed $P_{\text{ne}}$ is both at once (512×16 → 1024×4 is width ×2, depth ÷4), and total size never enters Table 1. The paper relies on this in its own $N{:}L$ study (Section 5 / Fig. 4a; App. G.1: one $\eta_{\text{base}} = 2^{-8}$ from $L_{\text{base}} = 2$ for every shape, 50M at 20 TPP, the same design as ours). Caveats: the guarantee is asymptotic ($N, L \to \infty$) and derived at a fixed token count (their Section 7), while our shapes get 1.28–1.55B tokens and are tuned at a 10% horizon; 1024×4 ($m_L = 2$) is furthest from the large-depth limit; and our tied embeddings (≈50% of 1024×4's parameters, LR $\eta_{\text{base}}$ unscaled in width) are a deviation from the paper.

## Tuning protocol (excluded from per-run budget)
- Each tuning run: 10% of the shape's full tokens, schedule compressed to that length, same batch size as final runs
- Optimum = lowest validation loss at the end of the tuning run
- Before sweeps: coordinate checks for CompleteP implementation

### LR grids (5 values, ×2 spacing)
SP (conditions a, b), global LR applied to all parameters:
$$\eta \in \{1.5\times10^{-4},\ 3\times10^{-4},\ 6\times10^{-4},\ 1.2\times10^{-3},\ 2.4\times10^{-3}\}$$
Centered on $6\times10^{-4}$ (nanoGPT default at 124M); contains Depth Delusion's $3\times10^{-4}$, so we can report where our optimum lands relative to theirs.

CompleteP (condition c), base LR $\eta_{\text{base}}$ defined at CompleteP's base shape ($N_{\text{base}} = 256$, $L_{\text{base}} = 2$); per-parameter LRs follow Table 1:
$$\eta_{\text{base}} \in \{2^{-10},\ 2^{-9},\ 2^{-8},\ 2^{-7},\ 2^{-6}\} \approx \{9.8\times10^{-4},\ 1.95\times10^{-3},\ 3.9\times10^{-3},\ 7.8\times10^{-3},\ 1.56\times10^{-2}\}$$
Centered on CompleteP's tuned value $\eta_{\text{base}} = 2^{-8} \approx 0.0039$ (App. G.1; tuned at batch 128, $\lambda = 0$, 300M tokens, Fig. 2). Effective LRs at 512×16 ($m_N = 2$, $m_L = 8$): hidden matrices $\eta_{\text{base}}\, m_N^{-1} = \eta_{\text{base}}/2$; emb/unemb and LN $\eta_{\text{base}}$.

Edge rule: if the optimum falls on a grid edge, add one more ×2 point beyond that edge (one extra 10% run) and re-pick.

### Sweeps
- SP sweeps: all 5 shapes (25 short runs). 512×16 optimum → LR for (a); per-shape optima → LRs for (b)
- CompleteP sweeps: all 5 shapes (25 short runs). 512×16 optimum → LR for (c); per-shape optima → LRs for (d). (First 512×16, 256×64 and 1024×4; 384×28 and 640×10 added for (d), so both parameterizations have a 5-point optimal-LR-vs-shape curve.)
- Transfer check: if the CompleteP optimum at any shape differs from 512×16 by more than one grid step, flag it (implementation bug or failed transfer) and report. The SP sweeps give the matching picture for SP (expected to drift). Result so far: 256×64 +0 steps, 1024×4 −2 steps (flagged; loss curves smooth, no instability).

## Final runs (14 + up to 4 for (d), each at $C \approx 8.48\times10^{17}$, seed 0)
(a) and (b) share the 512×16 run, and so do (c) and (d) (same LR by construction).
1.  SP 512×16 (serves as both SP-transfer and SP-tuned)
2.  SP-transfer 256×64
3.  SP-transfer 384×28
4.  SP-transfer 640×10
5.  SP-transfer 1024×4
6.  SP-tuned   256×64
7.  SP-tuned   384×28
8.  SP-tuned   640×10
9.  SP-tuned   1024×4
10. CompleteP  256×64
11. CompleteP  384×28
12. CompleteP  512×16
13. CompleteP  640×10
14. CompleteP  1024×4
15. –18. CompleteP-tuned 256×64, 384×28, 640×10, 1024×4 (added 2026-10-27)

If a tuned optimum (SP or CompleteP) equals the 512×16 LR for some shape, the transfer and tuned runs coincide; skip the duplicate. From the sweeps so far: SP-tuned 640×10 and 1024×4 are skipped (1.2e-3 everywhere except 256×64 and 384×28 at 6e-4), CompleteP-tuned 256×64 is skipped, CompleteP-tuned 1024×4 runs at $2^{-9}$; 384×28 and 640×10 depend on their sweeps.

Optional if time remains:
- Seeds 1–2 for 384×28 and 512×16 (all 3 conditions)
- CompleteP 512×4 diagnostic sweep (out of scope for now, 5 short runs of a 12.6M model, not a study shape, tuned with 512×16's token count): 512×16 → 512×4 changes only depth, 512×4 → 1024×4 only width. If the optimum shifts at 512×4, the 1024×4 drift is a shallow-depth (finite-$L$) effect; if not, it comes from width or the tied embedding.
- $\epsilon$ ablation: SP-transfer 256×64 with $\epsilon = 10^{-8}$ (PyTorch default; also the value Depth Delusion reports). Tests whether a larger $\epsilon$ degrades deep models, i.e. whether $\epsilon$ alone can produce a depth penalty. Everything else unchanged.

## Diagnostics (logged in every run)
- Per-layer gradient RMS over training, compared against $\epsilon$ (fraction of entries with $|g| < \epsilon$ per layer); feeds the gradient-starvation analysis and the $\epsilon$ discussion
- Per-layer update RMS (Adam update magnitude), to see whether gradient decay translates into update decay

## Cost estimate
- Final runs: 14 full-run equivalents, + up to 4 for (d)
- Tuning: 50 short runs × 10% = 5 full-run equivalents (+0.1 per edge extension)
- Total ≈ 19–23 full-run equivalents (originally planned: 18) ≈ 39 GPU-hours on A100 (at 35% MFU; more if deep shapes run at lower MFU)
- Optional $\epsilon$ ablation: +1 full-run equivalent

## Known limitations (to report)
- LR tuned at a 10% horizon; optimal LR shifts with training length, but the bias is shared across shapes and conditions
- CompleteP's transfer guarantee is asymptotic and derived at a fixed token count; our shapes differ in tokens (iso-FLOP) and the shallowest (L = 4) is far from the large-depth limit (see "Conditions")
- Single seed unless optional seed runs are done
- Tied embeddings (CompleteP paper used untied)
- Optimizer HPs taken from CompleteP, not tuned. Differences from Depth Delusion's reported setup (their Table 5; not independently verifiable, code unreleased): $\epsilon$ $10^{-16}$ vs $10^{-8}$ (tested by ablation); weight decay 0 vs 0.1; schedule 10% warmup + cosine to 0 vs 2,000-step warmup + cosine to $3\times10^{-5}$; seq length 2048 vs 1024 (tokens per step identical, 262,144); iso-FLOP tokens vs their fixed 6.4B tokens; condition (a) uses our own tuned LR, not their $3\times10^{-4}$.

## Context
- Depth Delusion predicts $D_{\text{crit}} \approx 2.43 \ln W$ ≈ 15 at $d=512$, ≈ 13.5 at $d=256$ → $L \in \{28, 64\}$ past it, $L \in \{4, 10, 16\}$ before it
- CompleteP's best 50M shape: 512×16; their tuned $\eta_{\text{base}} = 2^{-8}$, reused untuned for all parameterizations and shapes in their shape study



---

# Implementation notes (for agents working on this repo)

The spec above is the research plan. This section says how it is implemented and what is still missing.
Code changes for the study are marked `### Begin depth-study code ###` (train.py) and `### Begin RoPE code ###` (model.py).

## Code map
- **model.py**: RoPE replaces the learned position embedding `wpe` ("rotate half" layout, `rope_base = 10000`, applied to q/k in every attention layer, nothing added to the residual stream). `from_pretrained` is disabled (GPT-2 weights need `wpe`). CompleteP itself (residual scaling `m_L^-alpha`, Table 1 LR/eps/init groups in `configure_optimizers`, attention 1/d_head when `mup_enabled`, logits / m_N) was already in the EleutherAI nanoGPT-mup code.
- **train.py** additions:
  - `learning_rate` must be passed on the command line. The configs set it to -1 and train.py asserts it is > 0.
  - `sequential_train_data`: reads train.bin once, front to back. Step s always gets tokens [s·262144, (s+1)·262144), whatever the micro-batch. Training resumes at the right position.
  - Evals use a fixed-seed generator, so the same 1,600 val windows are scored at every eval, in every run.
  - There is always an eval after the last step, and the run stops right after it, so exactly `max_iters` steps are trained.
  - `run_log`: `out_dir/train.csv` every 10 steps (loss = mean over the step's micro-batches, tokens, lr, time_ms / tok_per_s / mfu averaged over the steps since the previous row, cumulative train_time_s, peak_mem_gb). `out_dir/eval.csv` at every eval (train/val loss, train_time_s, wall_time_s, plus coordinate-check columns when enabled).
  - Timing: `train_time_s` counts training steps only (evals, checkpointing and the first step of each process, i.e. compile, are left out); `wall_time_s` counts everything. Both are saved in the checkpoint and continue on resume. MFU (`model.estimate_mfu`) uses this spec's C_tok (causal attention term 6LTd; it was nanoGPT's 6N + 12LTd) and the bf16 dense peak of the GPU it runs on (`model.gpu_peak_flops`, from the GPU name; MIG slices get their share). `config.json` records `gpu_name` and `gpu_peak_flops`. Both are flushed row by row. On start, rows at or after the start iter are dropped, so a resumed run never has duplicate rows.
  - Checkpoints are saved atomically (tmp file + rename).
  - Diagnostics (`diag_interval = 100` in common.py, off for the coordinate check): `out_dir/diag.csv`, long format, one row per parameter tensor every 100 steps and at the last step. Columns: iter, layer (block index, empty for wte / ln_f), kind (`attn.c_attn`, `mlp.c_proj`, `ln_1`, ...), the tensor's own eps and lr (CompleteP scales both per group), param_rms, grad_rms and frac_g_below_eps (the gradient Adam gets: mean over micro-batches, after clipping), frac_sqrtv_below_eps (fraction where Adam's denominator sqrt(v_hat) < eps), update_rms (actual parameter change of the step), update_over_lr. `train.csv` also has grad_norm (global norm before clipping). Verified: losses are identical with diagnostics on and off.
- **config/depth/**:
  - `common.py` holds the shared settings and derives `n_head`, `max_iters` (from the FLOP budget formula), warmup (10%) and `lr_decay_iters` from `n_embd`/`n_layer`. A shape file sets `n_embd`, `n_layer` and `batch_size` (the micro-batch: 8 for 256x64 and 384x28, 16 otherwise, sized for an A100-40GB), then execs common.py. common.py doubles the micro-batch on GPUs with >= 75GB (A100-80, H100-96); the coordinate-check configs opt out with `_keep_micro_batch`. Only the effective batch (128) matters, and the eval windows are the same for any micro-batch.
  - `sp/<d>x<L>.py` and `completep/<d>x<L>.py` are the 5 shapes each. `completep_common.py` sets m_N = d/256, m_L = L/2, alpha = 1. `completep/512x4.py` is the optional diagnostic shape (tuning only, not run for now): its run length is 512x16's, not the FLOP-budget one.
  - `tune.py` is appended after a shape config. It shrinks the run to `tune_fraction = 0.1` of the tokens, with the schedule compressed to match and no checkpoints.
  - `coord_check_common.py` and `{sp,completep}/coord_check.py`: 10 steps at a constant LR of 2^-8, with coordinate-check logging and compile off. The shape is passed as `--n_embd= --n_layer= --batch_size=` *before* the config file.
  - SP-transfer and SP-tuned share configs and differ only in `--learning_rate`; so do CompleteP and CompleteP-tuned.
- **jobs/**:
  - `submit.sh NAME config [args]` submits a run. Protecting finished work comes first: a name that is queued, running or finished is skipped (exit 0, nothing touched), so every submit script can be rerun and only submits what is missing. A different command under an existing name (e.g. a changed LR) is refused, also when that run finished. A run with logs but no `ckpt.pt` (crashed/diverged tuning run) is refused, since rerunning it starts at step 0 and overwrites its logs; `RESTART=1` allows it. The state comes from `run_state.sh` (finished = last `eval.csv` iter equals `max_iters` in `config.json`).
  - `train.sbatch` writes to `out/<name>/`. Resubmitting the same name and command resumes from `ckpt.pt`; a different command under an existing name is refused. It repeats the finished / partial check when the job starts (a run can finish while another copy waits in the queue). `run_info.txt` has one block per attempt.
  - `jobs/depth/`:
    - `smoke_test.sh`: 60-step runs of 256x64 and 1024x4.
    - `coord_check.sbatch`: SP and CompleteP, L = 2..64 at d = 256 and d = 256..1024 at L = 2, 3 seeds, output in `out/coord_check/`. Finished seeds are skipped.
    - `tune.sh sp|completep [shape] [lrs]`: the CLAUDE.md grids (5 shapes each), plus the edge-rule extra points. `tune.sh completep 512x4` runs the optional diagnostic sweep (not part of the plan for now).
    - `final.sh`: reads `results/best_lr.csv`, and refuses (before submitting anything) if an LR is missing or on the grid edge. Submits (a)–(d); a tuned run whose LR equals the 512x16 one is skipped as a duplicate. Submits all final runs on one GPU type, the 96GB H100 by default (`GPU=gpu:a100-80:1 jobs/depth/final.sh` for another), so wall-clock times are comparable. Tuning and the coordinate check run on A100-40s (train.sbatch / coord_check.sbatch).
    - `status.py`: sacct state, progress, losses and log age for every run; problems marked `!!`.
    - `DRY_RUN=1` prints a script's commands and each run's state (finished / resume / partial / new) without submitting.
- **analysis/**:
  - `collect_results.py` copies the small logs from `out/` (gitignored) to `results/` (committed) and writes `results/runs.csv` (one row per run, including training/wall time, tokens/s and MFU), `best_lr.csv` and `best_lr.md`. `best_lr.md` holds the LRs tried per shape, the edge-rule flags, unfinished runs and the CompleteP transfer check (every shape vs 512x16). `runs.csv` has a `condition` column for final runs (`a`, `b`, `c`, `d`, or `a+b` / `c+d` for a run that serves both), derived from the run's LR: the 512x16 optimum = transfer, the shape's own optimum = tuned.
  - `plot_coord_check.py` writes `results/coord_check/plots/{depth,width}.png`.
  - `tuning.ipynb` reads only `results/` and plots the LR sweeps: final val loss vs LR per shape, optimal LR vs shape (transfer), loss curves per run, and the LR schedule.

## Run names
`tune-<sp|completep>-<d>x<L>-lr<lr>`, `sp-512x16` (shared by conditions a and b), `sp-transfer-<shape>`, `sp-tuned-<shape>`, `completep-<shape>` (c; also d where the tuned LR equals 512x16's), `completep-tuned-<shape>` (d), `smoke-*` (ignored by collect). `collect_results.py` classifies runs by these prefixes.

## Workflow on the cluster
1. `sbatch jobs/prepare_data.sh slimpajama --num_docs 2000000`. That gives about 2.2B tokens; the longest run (384x28) needs 1.56B, and sequential reading asserts if train.bin is too short. Val is SlimPajama's own validation split.
2. `jobs/depth/smoke_test.sh`. Check peak memory and that both runs finish.
3. `sbatch jobs/depth/coord_check.sbatch`, then collect and `plot_coord_check.py`.
4. `jobs/depth/tune.sh sp` and `jobs/depth/tune.sh completep`, then collect, apply the edge rule if flagged, and commit `results/`.
5. `jobs/depth/final.sh`, then collect and commit.
6. Added 2026-10-27, after (a)–(c) finals had run: `jobs/depth/tune.sh completep` (submits only the 10 missing 384x28 / 640x10 runs); collect; then `jobs/depth/final.sh` again, which skips every finished final run and submits only the (d) runs. Check with `DRY_RUN=1` first.

## Status (as of 2026-10-26)
- Everything above was tested locally on CPU with fake data in a scratch copy: logs, exact step count, bit-identical resume, collect, final.sh, plot, the train.sbatch resume guard (with stub uv and nvidia-smi). Nothing has been run on the cluster yet. Not committed.
- The "Diagnostics" section is implemented (2026-10-27, `diag_interval`, see the code map). Tuning runs started before that have no `diag.csv`. Not implemented: the optional SP + 1/d_head attention ablation.
- Condition (d), the extra CompleteP sweeps, the optional 512x4 diagnostic config and the skip / refuse guards in `submit.sh` / `train.sbatch` (2026-10-27): tested locally in a scratch copy with stub squeue / sbatch / uv against copies of the real `results/runs/` logs (finished runs skipped, partial refused, checkpointed resumed, changed LR refused, only missing runs submitted). `tuning.ipynb` plots only the 5 study shapes (not 512x4).

## Decisions and open discrepancies (for the user to confirm)
- **Tuning fraction:** 10% as this spec says. The user once said "20% I think"; to change it, edit `tune_fraction` in `config/depth/tune.py` and the text above.
- **Step counts:** computed exactly from the budget formula. They differ from the Shapes table by ≤0.3% (e.g. 512x16: 5,805 vs 5,798; 384x28: 5,932 vs 5,913), which looks like rounding in the table.
- **LR decay:** the CompleteP paper uses linear decay to 0 after warmup. This spec (and the code) uses cosine.
- **SP attention scale:** the CompleteP paper uses 1/d_head for all parameterizations, SP included. This spec keeps SP at 1/sqrt(d_head). With d_head fixed at 64 this is only a constant factor of 8.

## Working conventions
- The user's Mac has little disk space. Never download, install or cache anything without saying what, how big and where. Heavy data prep runs on the cluster.
- Test in the scratchpad, never inside the repo; a test once deleted a real `train.bin`.
- The shell is zsh, which does not word-split `$VAR`. Run multi-word argument tests under `bash`.
