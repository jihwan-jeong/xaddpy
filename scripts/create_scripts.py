import itertools
from os import path
import os
import argparse

# Job file snippet
job_str = """#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --account=def-ssanner
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

module load gurobi/9.5.0
source ~/xadd/bin/activate
cd ~/projects/def-ssanner/${USER}/xadd-bilinear

"""

run_str = """#!/bin/bash

"""
# Each combination of hyperparameters are experimented 5 times with different random seeds
hyperparams = dict(
    intopt=dict(                                # 4 * 4 * 3 * 5 = 240 experiments
        damping=[1e-6, 1e-4, 1e-3, 1e-2],
        thr=[1e-1, 1e-2, 1e-3, 1e-4],
        lr=[1e-3, 1e-2, 1e-1],
        timeout_iter=[500],
    ),
    qptl=dict(
        tau=[10, 1e2, 1e4, 1e5],                # 4 * 3 * 5 = 60 experiments
        lr=[1e-3, 1e-2, 1e-1],
        timeout_iter=[500],
    ),
    spoplus=dict(
        lr=[1e-3, 1e-2, 1e-1],                  # 3 * 5 = 15 experiments
    ),
    twostage=dict(
        lr=[1e-3, 1e-2, 1e-1],                  # 3 * 5 = 15 experiments
    )
)

common_args = dict(
    # shortest_path=dict(
    #     num_runs=[1],
    #     size=[50, 100, 200],
    #     nonlinearity=[1, 2, 5, 10],
    #     eps_bar=[0, 0.25],
    #     seed=list(range(5)),
    #     epochs=[300],
    #     num_edges=[24],
    # ),
    # energy_scheduling=dict(
    #     num_runs=[1],
    #     size=[20, 50, 100],
    #     prob_config=[1, 3],
    #     dataset=[1, 3],
    #     seed=list(range(5)),
    #     epochs=[300],
    # ),
    classification=dict(
        epochs=[300],
        seed=list(range(5)),
        num_runs=[1],
        nonlinearity=[1, 2, 5, 10],
        size=[50, 100, 200],
        eps_bar=[0, 0.25],
        class_dim=[5],
    )
)

# Number of run commands per domain
num_files = dict(
    # shortest_path=dict(
    #     intopt=500,
    #     qptl=250,
    #     spoplus=360,
    #     twostage=360,
    # ),
    # energy_scheduling=dict(
    #     intopt=300,
    #     qptl=100,
    #     spoplus=15,
    #     twostage=15,
    # ),
    classification=dict(
        intopt=20,
        qptl=20,
        spoplus=80,
        twostage=150,
    ),
)

domains = ('classification', ) #('shortest_path', 'classification', 'energy_scheduling')
baselines = list(hyperparams.keys())
dir_path = path.join(path.curdir, 'scripts')

def create_scripts(args):
    for alg in baselines:
        alg_path = path.join(dir_path, alg)
        algo_configs = hyperparams[alg]

        for dom in domains:
            alg_dom_path = path.join(alg_path, dom)
            os.makedirs(alg_dom_path, exist_ok=True)

            cnt = 0
            f_cnt = 0

            fname = f"job_{alg}_{dom}_{f_cnt}.sh" if args.job else f"run_{alg}_{dom}_{f_cnt}.sh"
            run_cmds = ""
            all_configs = algo_configs.copy()
            all_configs.update(common_args[dom])
            keys = tuple(all_configs.keys())
            for cfg in itertools.product(*list(all_configs.values())):
                run_cmd = f"python xaddpy/experiments/predopt/{dom}/run_experiments.py --algo {alg}"
                for k, v in zip(keys, cfg):
                    run_cmd += f" --{k} {v}"
                run_cmds += f"{run_cmd}\n"
                cnt += 1
                if cnt == num_files[dom][alg]:
                    # Write a job file here
                    cnt = 0
                    run_cmds = job_str + run_cmds if args.job else run_str + run_cmds

                    with open(path.join(alg_dom_path, fname), 'w') as f:
                        f.write(run_cmds)

                    run_cmds = ""
                    f_cnt += 1
                    fname = f"job_{alg}_{dom}_{f_cnt}.sh" if args.job else f"run_{alg}_{dom}_{f_cnt}.sh"

            # Remaining run commands
            run_cmds = job_str + run_cmds if args.job else run_str + run_cmds

            with open(path.join(alg_dom_path, fname), 'w') as f:
                f.write(run_cmds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--job', dest='job', default=False, action='store_true',
                        help='If True, create job scripts; otherwise, create a script with python commands')
    args = parser.parse_args()

    create_scripts(args)
