#!/bin/bash
#SBATCH --job-name=cl2_py
#SBATCH --partition=256G56c
#SBATCH --qos=qos7d
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=cl2_%A_%a.log

# Array tasks map to bond lengths (Å):
#   task 0 -> 1.2
#   task 1 -> 2.2
BONDS=(1.2 2.2)
BOND=${BONDS[$SLURM_ARRAY_TASK_ID]}

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: Cl2 bond=${BOND}"

# Activate conda environment
source activate cross_chips

# Run from the directory where this sbatch was submitted (export2cloud)
cd "${SLURM_SUBMIT_DIR}"

export CL2_BOND_LENGTH="${BOND}"
python main_Cl2.py
