#!/bin/bash
#SBATCH --job-name=exactbo-e4-other
#SBATCH --output=experiments/exactbo/experiment4/logs/other-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/other-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00

# Step 5 of the Vision stress test, one H200, uncapped (no max_target_boxes):
#   A. problem2d/problem5d, two initial-design sizes, seeds 0-2: AutoBound vs
#      interval bounds, run to convergence (epsilon_X = 1e-5, <= 100 partitions)
#   B. problem10d with seeds 1 and 2
#   C. box_sampling = "center" (problem10d, problem5d)
#   D. autobound_degree = 3 vs 2 on problem2d/problem5d
# Each run is capped at 15 minutes so one runaway configuration cannot use the job.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_other_settings.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E4="${REPO_ROOT}/experiments/exactbo/experiment4"
OUT="${E4}/results/other_partition_counts.csv"
SUMMARY="${E4}/results/other_runs.jsonl"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${E4}"

run() {  # run <label> <args...>
  local label="$1"; shift
  echo "##### ${label} #####"
  timeout 900 python stress_partitioning.py --label "${label}" --out "${OUT}" --summary "${SUMMARY}" "$@" \
    || echo "RUN ${label}: FAILED or timed out (exit $?)"
}

# A. lower dimensions, to convergence
for spec in "problem2d 4" "problem2d 32" "problem5d 8" "problem5d 32"; do
  read -r problem n0 <<< "${spec}"
  for seed in 0 1 2; do
    for bm in autobound interval; do
      run "A/${problem}/n0=${n0}/seed=${seed}/logei/${bm}" \
        --problem "${problem}" --n0 "${n0}" --seed "${seed}" --bound-method "${bm}" --max-partitions 100
    done
  done
done

# B. problem10d, other seeds (AutoBound + EI to partition 8, as in step 1)
for seed in 1 2; do
  run "B/problem10d/n0=32/seed=${seed}/ei/autobound" --seed "${seed}" --acquisition ei --max-partitions 9
done

# C. center sampling
run "C/problem10d/n0=32/seed=0/ei/autobound/center" --acquisition ei --max-partitions 9 --box-sampling center
run "C/problem5d/n0=8/seed=0/logei/autobound/center" --problem problem5d --n0 8 --max-partitions 100 --box-sampling center

# D. AutoBound degree 3 on small d
run "D/problem2d/n0=4/seed=0/logei/autobound/deg3" --problem problem2d --n0 4 --max-partitions 100 --autobound-degree 3
run "D/problem5d/n0=8/seed=0/logei/autobound/deg3" --problem problem5d --n0 8 --max-partitions 100 --autobound-degree 3
