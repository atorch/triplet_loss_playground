#!/usr/bin/env bash
# Runs every condition for each held-out digit, then summarizes. ~10 min per run on 4 CPU cores.
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=2
EPOCHS=${EPOCHS:-12}
for digit in ${HOLDOUTS:-7 4 9}; do
  for flags in "" "--augment" "--outlier-exposure" "--augment --outlier-exposure"; do
    echo "=== holdout=$digit flags=[$flags] $(date -u +%H:%M:%S)"
    uv run src/experiment.py --holdout-digit "$digit" --epochs "$EPOCHS" $flags > /dev/null 2>&1 \
      && echo "done" || echo "FAILED"
  done
done
uv run src/compare.py
