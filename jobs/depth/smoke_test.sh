#!/bin/bash
# Quick end-to-end check of the cluster setup: two 60-step runs with the real configs, the two
# shapes that use the most GPU memory (256x64: deepest, 1024x4: widest).
# Each takes a few minutes and checkpoints every 20 steps.
#
# Usage (from the repo root, after data/slimpajama is prepared):
#   jobs/depth/smoke_test.sh
#
# Then check:
#   1. both jobs finish:              uv run --frozen python jobs/depth/status.py
#   2. memory fits:                   grep "peak mem" logs/smoke-*.out | tail
#      (the micro-batch is fine if peak mem stays well below 40GB)
#   3. logs are written:              cat out/smoke-sp-256x64/eval.csv
#   4. optional, crash recovery: while a job is training, after its first "saving checkpoint"
#      line in logs/smoke-*.out, cancel it (scancel <job id>), then run this script again.
#      The cancelled run should print "resuming from ..." and continue from that step
#      (the other one, if already finished, is skipped by jobs/submit.sh).
#
# Afterwards delete them: rm -r out/smoke-* logs/smoke-*

set -euo pipefail

SMOKE_ARGS=(--learning_rate=6e-4 --max_iters=60 --warmup_iters=6 --lr_decay_iters=60 --eval_interval=20 --eval_iters=10)

jobs/submit.sh smoke-sp-256x64 config/depth/sp/256x64.py "${SMOKE_ARGS[@]}"
jobs/submit.sh smoke-completep-1024x4 config/depth/completep/1024x4.py "${SMOKE_ARGS[@]}"
