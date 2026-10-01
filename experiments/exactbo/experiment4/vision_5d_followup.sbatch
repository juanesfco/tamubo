#!/bin/bash
#SBATCH --job-name=exactbo-e4-5d
#SBATCH --output=experiments/exactbo/experiment4/logs/5d-%j.out
#SBATCH --error=experiments/exactbo/experiment4/logs/5d-%j.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00

# Follow-up to step 5: problem5d with 32 initial points ran out of memory even
# with AutoBound, while 8 initial points converged in ~15 s. Compare logEI
# (relative epsilon_ei) with EI (absolute) and record the target growth per split.
# Submit from the repo root:
#   sbatch experiments/exactbo/experiment4/vision_5d_followup.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
E4="${REPO_ROOT}/experiments/exactbo/experiment4"
export PYTHONUNBUFFERED=1

module load CUDA/12.9.1
source "${REPO_ROOT}/envs/tamubo_exactbo/bin/activate"
cd "${E4}"

for acq in logei ei; do
  echo "##### problem5d n0=32 seed=0 ${acq}/autobound #####"
  timeout 1200 python stress_partitioning.py --problem problem5d --n0 32 --seed 0 --acquisition "${acq}" \
    --max-partitions 100 --progress --label "F/problem5d/n0=32/seed=0/${acq}/autobound" \
    --out results/followup_partition_counts.csv --summary results/followup_runs.jsonl \
    || echo "RUN ${acq}: FAILED or timed out (exit $?)"
done
