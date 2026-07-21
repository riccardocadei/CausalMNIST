#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=cfl_mnist             # Job name
#SBATCH --output=logs/cfl_%j.log         # Output file
#SBATCH --time=2:00:00                   # Maximum walltime
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --mem=10G                        # Memory allocation
#SBATCH --partition=gpu              # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                     # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                      # Ensure you load your bash profile
module load conda
conda activate crl                      # Activate your conda environment

train_ratio=0.1024
pU=0.5
pW=0.5
n_seeds=10



# Run experiments
srun python src/run_cfl.py --train_ratio ${train_ratio} --pU ${pU} --pW ${pW} --exp RCT --n_seeds ${n_seeds} 
srun python src/run_cfl.py --train_ratio ${train_ratio} --pU ${pU} --pW ${pW} --exp OS --n_seeds ${n_seeds} 