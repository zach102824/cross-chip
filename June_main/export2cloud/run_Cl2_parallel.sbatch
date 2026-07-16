#!/bin/bash
#SBATCH --job-name=cl2_parallel
#SBATCH --partition=256G56c
#SBATCH --qos=qos7d
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --array=0-1
#SBATCH --output=cl2_parallel_%A_%a.log

# One 28-core socket handles 28 of the default 30 independent CDR circuits at once.
set -euo pipefail

BONDS=(1.2 2.2)
BOND=${BONDS[$SLURM_ARRAY_TASK_ID]}

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: Cl2 bond=${BOND}"
echo "Node=${SLURMD_NODENAME:-unknown}, CPUs=${SLURM_CPUS_PER_TASK}"

source activate cross_chips
cd "${SLURM_SUBMIT_DIR}"

export CL2_BOND_LENGTH="${BOND}"
# VQE: one CDR call at a time, using almost all cores for its 30 training circuits.
export PARALLEL_CDR_WORKERS="${SLURM_CPUS_PER_TASK}"
export PARALLEL_CDR_START_METHOD="fork"
# CMX: seven concurrent moment calls, each with four spawn-safe CDR workers.
export PARALLEL_CMX_WORKERS=7
export PARALLEL_CMX_CDR_WORKERS=4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -u main_Cl2_parallel.py
