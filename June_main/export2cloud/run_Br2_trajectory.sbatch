#!/bin/bash
#SBATCH --job-name=br2_traj
#SBATCH --partition=256G56c
#SBATCH --qos=qos7d
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --array=0-1
#SBATCH --output=br2_traj_%A_%a.log

# Trajectory noisy-sim + parallel CDR. One 28-core socket handles 28 of the
# default 30 independent CDR circuits at once. CMX stays disabled in main_Br2.py.
set -euo pipefail

BONDS=(2.2 3.8)
BOND=${BONDS[$SLURM_ARRAY_TASK_ID]}

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: Br2 trajectory bond=${BOND}"
echo "Node=${SLURMD_NODENAME:-unknown}, CPUs=${SLURM_CPUS_PER_TASK}"

source activate cross_chips
cd "${SLURM_SUBMIT_DIR}"

export BR2_BOND_LENGTH="${BOND}"
export GLOBAL_NUM_SHOTS=8192
export PARALLEL_CDR_WORKERS="${SLURM_CPUS_PER_TASK}"
export PARALLEL_CDR_START_METHOD="fork"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -u main_Br2_trajectory.py
