#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=hrf_opt
#SBATCH --ntasks=1
#SBATCH --array=1-12
#SBATCH --time=04:30:00
#SBATCH --cpus-per-task=192
#SBATCH --output=slurm_output_%A.out

srun python all_brain_fits_fixed.py $SLURM_ARRAY_TASK_ID