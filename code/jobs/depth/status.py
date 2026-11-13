"""
Overview of all runs: SLURM state, progress, latest losses and time since the last log line.
Run from the repo root on the cluster:

    uv run --frozen python jobs/depth/status.py                    # once
    watch -n 60 uv run --frozen python jobs/depth/status.py        # refresh every minute

Rows marked !! need attention: the job failed, was cancelled, timed out or ran out of memory,
or it is RUNNING but has not written a log line for 15 minutes (hung).
To see why a job died: tail -50 logs/<name>-<job id>.out
"""

import csv
import json
import subprocess
import time
from pathlib import Path

OK_STATES = {"RUNNING", "PENDING", "COMPLETED"}
HUNG_AFTER_MIN = 15


def slurm_jobs():
    """name -> (job id, state, elapsed) of the latest job with that name in the last 14 days."""
    try:
        out = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-S", "now-14days", "--format=JobID,JobName,State,Elapsed"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}  # not on the cluster
    jobs = {}
    for line in out.splitlines():  # sacct lists jobs oldest first, so later ones win
        job_id, name, state, elapsed = line.split("|")
        jobs[name] = (job_id, state.split()[0], elapsed)  # "CANCELLED by 123" -> "CANCELLED"
    return jobs


def last_row(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


jobs = slurm_jobs()
names = {d.name for d in Path("out").glob("*/") if (d / "run_info.txt").exists()} | set(jobs)

table = []
for name in sorted(names):
    out_dir = Path("out") / name
    job_id, state, elapsed = jobs.get(name, ("-", "-", "-"))
    config = json.loads((out_dir / "config.json").read_text()) if (out_dir / "config.json").exists() else {}
    train, evals = last_row(out_dir / "train.csv"), last_row(out_dir / "eval.csv")

    progress = f"{int(train['iter']) + 1}/{config['max_iters']}" if train and "max_iters" in config else "-"
    if evals and "max_iters" in config and int(evals["iter"]) == config["max_iters"]:
        progress = f"{config['max_iters']}/{config['max_iters']}"  # finished (train.csv only logs every 10 steps)
    train_loss = f"{float(train['loss']):.3f}" if train else "-"
    val_loss = f"{float(evals['val_loss']):.3f}@{evals['iter']}" if evals else "-"
    age_min = (time.time() - (out_dir / "train.csv").stat().st_mtime) / 60 if train else None
    last_log = f"{age_min:.0f} min ago" if age_min is not None else "-"

    problem = state not in OK_STATES | {"-"} or (state == "RUNNING" and age_min is not None and age_min > HUNG_AFTER_MIN)
    table.append(("!!" if problem else "", name, job_id, state, elapsed, progress, train_loss, val_loss, last_log))

header = ("", "run", "job", "state", "elapsed", "step", "train loss", "val loss@step", "last log")
table.sort(key=lambda r: (r[0] == "", r[1]))  # problems first
widths = [max(len(str(r[i])) for r in [header] + table) for i in range(len(header))]
for r in [header] + table:
    print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
n_problems = sum(r[0] == "!!" for r in table)
print(f"\n{len(table)} runs, {n_problems} need attention" + (" (marked !!)" if n_problems else ""))
