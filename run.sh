#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=PPCI_test        # Job name
#SBATCH --output=logs/PPCI_%j.log   # Output file
#SBATCH --time=2:00:00             # Maximum walltime
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --mem=10G                    # Memory allocation
#SBATCH --partition=gpu100             # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                  # Ensure you load your bash profile
module load conda
conda activate crl                  # Activate your conda environment


epochs=40
seeds=10
estimator="CF"


srun python ./src/run_gen.py --epochs ${epochs} --seeds ${seeds} --estimator ${estimator}

