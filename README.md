# CausalMNIST
Synthetic Benchmark for *Prediction-Powered Treatment Effect Estimation* based on MNIST dataset manipulation.

## Problem
Estimating the treatment effect of the pen color to the magnitude of handwritten digits without digit annotation.

## Data Generating Process
See Appendix C of the corresponding paper for a detailed description of the data generating process.

## Run
Run:

`srun python ./src/run_gen.py --pW 0.5 --pU 0.02 --exp OS --epochs 40 --seeds 50`

to reproduce the experiments in Table 1. 
