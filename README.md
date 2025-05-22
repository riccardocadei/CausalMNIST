# CausalMNIST
Synthetic Benchmark for *Prediction-Powered Treatment Effect Estimation* based on MNIST dataset manipulation.

## Problem
Estimating the treatment effect of the pen color to the magnitude of handwritten digits without digit annotation.

## Data Generating Process
See Appendix C of the corresponding paper for a detailed description of the data generating process.

## Run
Run:

```bash
python ./src/run_gen.py --pW 0.5 --pU 0.02 --exp OS --epochs 40 --seeds 50
```

to reproduce the experiments in Table 1, and load them in `results.ipynb` for visualization.

## Requirements
Install the required packages using:

```bash
pip install -r requirements.txt
```
