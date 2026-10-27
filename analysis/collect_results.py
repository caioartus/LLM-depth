"""
Collects the depth-study runs from out/ (not in git: checkpoints are large) into results/
(small files, meant to be committed) and picks the best learning rates from the tuning runs.
Run from the repo root on the cluster whenever runs have finished, then commit results/:

    uv run --frozen python analysis/collect_results.py
    git add results && git commit -m "results: ..."

It writes
    results/runs/<name>/        train.csv, eval.csv, diag.csv, config.json, run_info.txt of every run
    results/coord_check/<name>.csv   eval.csv of every coordinate-check run
    results/runs.csv            one row per run: kind, parameterization, shape, lr, status, final val loss,
                                training / wall time (h), tokens/s and MFU over the whole run
    results/best_lr.csv         per (parameterization, shape): the tuned LR. jobs/depth/final.sh reads this
    results/best_lr.md          the same as a readable table, with every LR tried and the CLAUDE.md checks

Run kinds come from the names the scripts in jobs/depth/ give: tune-..., smoke-... (skipped), others are final runs.
"""

import csv
import json
import math
import shutil
from pathlib import Path

OUT, RESULTS = Path("out"), Path("results")
RESULTS.mkdir(exist_ok=True)
REFERENCE_SHAPE = "512x16"  # the LR of conditions (a) and (c) is tuned here (CLAUDE.md)
VOCAB, A100_PEAK = 50304, 312e12


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")  # plain \n, so shell tools can read it
        writer.writeheader()
        writer.writerows(rows)


# ---- copy the logs of every run ----
runs = []
for out_dir in sorted(OUT.glob("*/")):
    if out_dir.name.startswith("smoke-") or not (out_dir / "config.json").exists():
        continue
    config = json.loads((out_dir / "config.json").read_text())
    if not config.get("run_log"):
        continue  # not a depth-study run
    dest = RESULTS / "runs" / out_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "eval.csv", "diag.csv", "config.json", "run_info.txt"]:
        if (out_dir / name).exists():
            shutil.copy2(out_dir / name, dest / name)

    evals = read_csv(out_dir / "eval.csv") if (out_dir / "eval.csv").exists() else []
    finished = bool(evals) and int(evals[-1]["iter"]) == config["max_iters"]
    # wall-clock and throughput over the whole run (train.py sums them over crash + resume).
    # MFU uses C_tok from CLAUDE.md "Compute accounting" and the run's GPU peak (train.py writes it to
    # config.json; runs from before that were all on A100-40s)
    timing = {}
    peak = config.get("gpu_peak_flops", A100_PEAK)
    if finished and evals[-1].get("train_time_s"):
        d, L, T = config["n_embd"], config["n_layer"], config["block_size"]
        train_s = float(evals[-1]["train_time_s"])
        tok_per_s = config["max_iters"] * config["batch_size"] * config["gradient_accumulation_steps"] * T / train_s
        timing = {"train_time_h": f"{train_s / 3600:.3f}", "wall_time_h": f"{float(evals[-1]['wall_time_s']) / 3600:.3f}",
                  "tok_per_s": f"{tok_per_s:.0f}",
                  "mfu": f"{100 * tok_per_s * (6 * 12 * d**2 * L + 6 * d * VOCAB + 6 * L * T * d) / peak:.2f}" if peak else ""}
    runs.append({
        "name": out_dir.name,
        "kind": "tune" if out_dir.name.startswith("tune-") else "final",
        "parameterization": "completep" if config["depth_alpha_enabled"] else "sp",
        "shape": f"{config['n_embd']}x{config['n_layer']}",
        "d": config["n_embd"],
        "L": config["n_layer"],
        "lr": config["learning_rate"],
        "max_iters": config["max_iters"],
        "last_eval_iter": evals[-1]["iter"] if evals else "",
        # not finished = still running, crashed, or diverged (train.py stops on a NaN loss)
        "status": "finished" if finished else "not finished",
        "final_val_loss": evals[-1]["val_loss"] if finished else "",
        **{k: timing.get(k, "") for k in ["train_time_h", "wall_time_h", "tok_per_s", "mfu"]},
    })
write_csv(RESULTS / "runs.csv", runs, list(runs[0]) if runs else ["name"])

for cc_dir in sorted((OUT / "coord_check").glob("*/")):
    if (cc_dir / "eval.csv").exists():
        (RESULTS / "coord_check").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cc_dir / "eval.csv", RESULTS / "coord_check" / f"{cc_dir.name}.csv")

# ---- best LR per (parameterization, shape) from the tuning runs ----
groups = {}
for run in runs:
    if run["kind"] == "tune":
        groups.setdefault((run["parameterization"], run["shape"]), []).append(run)

best_rows, md = [], ["# Tuned learning rates", "",
                     "Written by `analysis/collect_results.py` from the tuning runs (10% of each shape's tokens).",
                     "Optimum = lowest val loss after the last step (CLAUDE.md \"Tuning protocol\").", ""]
best_lr = {}
for (param, shape), group in sorted(groups.items(), key=lambda kv: (kv[0][0], -int(kv[0][1].split("x")[1]))):
    group.sort(key=lambda r: r["lr"])
    done = [r for r in group if r["status"] == "finished"]
    md += [f"## {param} {shape}", "", "| LR | val loss | |", "|---|---|---|"]
    best = min(done, key=lambda r: float(r["final_val_loss"])) if done else None
    for r in group:
        mark = "**best**" if r is best else ("not finished (running, crashed or diverged)" if r["status"] != "finished" else "")
        loss = f"{float(r['final_val_loss']):.4f}" if r["final_val_loss"] else "-"
        md.append(f"| {r['lr']:.4g} | {loss} | {mark} |")
    md.append("")
    if best is None:
        md += ["No finished runs yet.", ""]
        continue
    # edge rule: the optimum is the smallest or largest LR tried -> add one more x2 point beyond it
    on_edge = best["lr"] in (group[0]["lr"], group[-1]["lr"])
    if on_edge:
        md += [f"⚠ **Optimum on the grid edge.** Edge rule: add one more point at "
               f"{best['lr'] * (2 if best['lr'] == group[-1]['lr'] else 0.5):.4g} "
               f"(`jobs/depth/tune.sh {param} {shape} <lr>`) and collect again.", ""]
    unfinished = len(group) - len(done)
    if unfinished:
        md += [f"⚠ {unfinished} run(s) not finished, check `jobs/depth/status.py` before trusting this optimum.", ""]
    best_lr[(param, shape)] = best["lr"]
    best_rows.append({"parameterization": param, "shape": shape, "best_lr": best["lr"],
                      "val_loss": best["final_val_loss"], "lrs_tried": ";".join(f"{r['lr']:g}" for r in group),
                      "unfinished": unfinished, "on_edge": on_edge})

# CompleteP transfer check: optimum at the extremes within one grid step (x2) of the reference shape
if ("completep", REFERENCE_SHAPE) in best_lr:
    md += ["## CompleteP transfer check", "",
           f"Optimum at the depth extremes vs {REFERENCE_SHAPE} (more than one grid step = flag, CLAUDE.md \"Sweeps\").", ""]
    ref = best_lr[("completep", REFERENCE_SHAPE)]
    for shape in ["256x64", "1024x4"]:
        if ("completep", shape) in best_lr:
            steps = math.log2(best_lr[("completep", shape)] / ref)
            md.append(f"- {shape}: {steps:+.0f} grid steps " + ("⚠ **FLAG: transfer failed or implementation bug**" if abs(steps) > 1.01 else "(ok)"))
    md.append("")

write_csv(RESULTS / "best_lr.csv", best_rows, ["parameterization", "shape", "best_lr", "val_loss", "lrs_tried", "unfinished", "on_edge"])
(RESULTS / "best_lr.md").write_text("\n".join(md))
print(f"{len(runs)} runs -> {RESULTS}/runs/, {len(best_rows)} tuned shapes -> {RESULTS}/best_lr.csv and best_lr.md")
