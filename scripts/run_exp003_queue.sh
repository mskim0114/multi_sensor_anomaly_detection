#!/usr/bin/env bash
# One queue lane of EXP-20260907-003 (2^3 factorial ablation, 8 conditions x 3 seeds).
#
#   usage: scripts/run_exp003_queue.sh LANE NLANES GPU
#   e.g.   for L in 0 1 2 3; do nohup scripts/run_exp003_queue.sh $L 4 $((L/2)) > lane$L.log 2>&1 & done
#
# The 24 jobs are ordered seed-major (all 8 conditions of seed 42, then 123, then 456)
# and dealt round-robin to the lanes, so each seed completes as early as possible.
# A job whose results.json already exists is skipped; the trainer itself refuses to
# overwrite a finished run, so re-launching a lane after an interruption is safe.
#
# Environment: SERVER-TRAINING profile (docs/SERVER_ENVIRONMENT.md). Override the venv
# or evidence directory with FACTORY_VENV / EXP003_EVIDENCE.
set -u
LANE=$1; NLANES=$2; GPU=$3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVID="${EXP003_EVIDENCE:-$HOME/review_runs/20260918_exp003}"
VENV="${FACTORY_VENV:-$HOME/venvs/factory_training}"
mkdir -p "$EVID/logs"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"
i=0
# Override the default full design with EXP003_SEEDS="7 2026" and EXP003_CONDS="1 0 1;1 1 1"
# (each condition is "ms se sc"); used for the SE follow-up seeds (연구노트 #20).
SEEDS="${EXP003_SEEDS:-42 123 456}"
CONDS="${EXP003_CONDS:-0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1}"
IFS=';' read -r -a COND_LIST <<< "$CONDS"
for seed in $SEEDS; do
  for triple in "${COND_LIST[@]}"; do
    read -r ms se sc <<< "$triple"
    if (( i % NLANES == LANE )); then
      cond="ms${ms}_se${se}_sc${sc}_seed${seed}"
      if [[ -f "results/factorial_ablation/${cond}/results.json" ]]; then
        echo "$(date +%T) lane$LANE skip  $cond (done)"
      else
        echo "$(date +%T) lane$LANE gpu$GPU start $cond"
        python -m src.train_factorial_ablation --multiscale "$ms" --se "$se" --supcon "$sc" \
               --seed "$seed" --gpu "$GPU" > "$EVID/logs/${cond}.log" 2>&1
        echo "$(date +%T) lane$LANE gpu$GPU end   $cond exit=$?"
      fi
    fi
    i=$((i+1))
  done
done
echo "$(date +%T) lane$LANE finished"
