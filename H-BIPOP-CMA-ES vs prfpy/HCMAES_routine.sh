#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=hrf_opt
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=192
#SBATCH --output=slurm_output_%A.out

srun python H_CMA_ES.py $SLURM_ARRAY_TASK_ID