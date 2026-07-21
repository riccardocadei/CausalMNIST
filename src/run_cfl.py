
import numpy as np
import pandas as pd
import argparse
import os
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from dataset import CausalMNIST
import matplotlib.pyplot as plt

from models import compute_effect, ConvNet
from train import training

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description="CFL - semisupervised.")
    parser.add_argument('--e', type=int, default=1, help='Environment')
    parser.add_argument('--pW', type=float, default=0.5, help='pW')
    parser.add_argument('--pU', type=float, default=0.3, help='pU')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples')
    parser.add_argument('--exp', type=str, default='RCT', help='Version of the experiment')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Train ratio')
    parser.add_argument('--n_seeds', type=int, default=50, help='Seed')
    return parser


def main(args):
    start = time.time()
    cfl_weights = [0] + list(np.logspace(-2, 8, num=11)) 
    seeds = list(range(args.n_seeds))
    i = 0
    all_metrics = pd.DataFrame(columns=["cfl_weight", "seed", "best_epoch", "ATE", "PPATE", "acc", "bacc"])
    for seed in seeds:
        print(f"Seed: {seed}", flush=True)
        dataset = CausalMNIST(root='./data',
                            N=args.N,
                            e=args.e,
                            pW=args.pW,
                            pU=args.pU,
                            exp=args.exp,
                            verbose=False,
                            seed=seed,
                            force_generation=True)
        ATE = compute_effect(dataset, method="AIPW", pred=False, total=False, econml=False, train_ratio=1)
        for cfl_weight in cfl_weights:
            if i==0:
                print(f"CFL weight: {cfl_weight} (run {i+1}/{len(cfl_weights)*len(seeds)})", flush=True)
            else:  
                partial = time.time() - start
                total = partial * (len(cfl_weights)*len(seeds)) / i
                print(f"CFL weight: {cfl_weight} (run {i+1}/{len(cfl_weights)*len(seeds)}, ETA: {int(partial/60)}m{int(partial%60)}s/ {int(total/60)}m{int(total%60)}s)", flush=True)
            # train model
            model = ConvNet()
            model = training(model,
                            dataset, 
                            train_ratio=args.train_ratio,
                            epochs=40,
                            lr=0.001,
                            batch_size=64,
                            k_inv=0,
                            method='ERM',
                            verbose=False,
                            log_dir='runs',
                            eval=True,
                            gpu=True,
                            cfl=cfl_weight)
            dataset.Y_hat = model.cond_exp(dataset.X.to(model.device)).detach().cpu().numpy()
            PPATE = compute_effect(dataset, method="AIPW", pred=True, total=False, econml=False, train_ratio=1)
            acc = accuracy_score(dataset.Y_hat.round(), dataset.Y.numpy())
            bacc = balanced_accuracy_score(dataset.Y_hat.round(), dataset.Y.numpy())
            all_metrics_i = {
                "cfl_weight": cfl_weight,
                "seed": seed,
                "best_epoch": model.best_epoch,
                "ATE": ATE,
                "PPATE": PPATE,
                "acc": acc,
                "bacc": bacc,
                #"TERB": abs(ATE-PPATE)/ATE*100
            }
            all_metrics.loc[i] = all_metrics_i
            i += 1

    results_dir = f'results/cfl/{args.train_ratio}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    all_metrics.to_csv(f'{results_dir}/pW_{args.pW}_pU_{args.pU}_e_{args.e}_exp_{args.exp}.csv', index=False)

    all_metrics["TERB"] = (all_metrics["PPATE"]-0.3)/0.3*100
    plt.figure()
    plt.xlabel(r"$\lambda_{CFL}$")
    plt.ylabel("TERB (%)", color='tab:blue')
    plt.xscale('log')
    all_metrics['cfl_weight'] = all_metrics['cfl_weight'].replace(0, 0.0001)
    plt.errorbar(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['TERB'].mean(), yerr=all_metrics.groupby('cfl_weight')['TERB'].std(), fmt='o', color='tab:blue')
    plt.plot(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['TERB'].mean(), '--', color='tab:blue')
    plt.fill_between(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['TERB'].mean()-all_metrics.groupby('cfl_weight')['TERB'].std(), all_metrics.groupby('cfl_weight')['TERB'].mean()+all_metrics.groupby('cfl_weight')['TERB'].std(), alpha=0.2, color='skyblue')

    #plt.xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], [f"0\n(supervised)",  0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]);

    plt.twinx()
    plt.ylabel("Balanced Accuracy", color='tab:red')
    plt.errorbar(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['bacc'].mean(), yerr=all_metrics.groupby('cfl_weight')['bacc'].std(), fmt='o', color='tab:red', label="Accuracy");
    plt.plot(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['bacc'].mean(), '--', color='tab:red', label="Balanced Accuracy");
    plt.fill_between(all_metrics['cfl_weight'].unique(), all_metrics.groupby('cfl_weight')['bacc'].mean()-all_metrics.groupby('cfl_weight')['bacc'].std(), all_metrics.groupby('cfl_weight')['bacc'].mean()+all_metrics.groupby('cfl_weight')['bacc'].std(), alpha=0.2, color='pink')
    plt.ylim(0.45, 1)
    plt.axvline(x=0.00031622776601683794, color='black', linestyle='--', label=r"separation", alpha=0.2)

    # save the plot
    plt.savefig(f'{results_dir}/pW_{args.pW}_pU_{args.pU}_e_{args.e}_exp_{args.exp}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)