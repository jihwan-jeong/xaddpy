import itertools
from os import path
import os
import argparse

# Job file snippet
job_str = """#!/bin/bash

#SBATCH --time=28:00:00
#SBATCH --account=def-ssanner
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

module load gurobi/9.5.0
source ~/xaddpy-parth/bin/activate
cd ~/projects/def-ssanner/${USER}/parth/xadd-bilinear

"""
job_str2 = """#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --account=def-ssanner
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

module load gurobi/9.5.0
source ~/xaddpy-parth/bin/activate
cd ~/projects/def-ssanner/${USER}/parth/xadd-bilinear

"""

run_str = """#!/bin/bash

"""
# Each combination of hyperparameters are experimented 5 times with different random seeds
hyperparams = dict(
    emspo=dict(),
    # intopt=dict(                                # 4 * 4 * 3 * 5 = 240 experiments
    #     damping=[1e-6, 1e-4, 1e-3, 1e-2],
    #     thr=[1e-1, 1e-2, 1e-3, 1e-4],
    #     lr=[1e-3, 1e-2, 1e-1],
    #     timeout_iter=[500],
    # ),
    # qptl=dict(
    #     tau=[10, 1e2, 1e4, 1e5],                # 4 * 3 * 5 = 60 experiments
    #     lr=[1e-3, 1e-2, 1e-1],
    #     timeout_iter=[500],
    # ),
    # spoplus=dict(
    #     lr=[1e-3, 1e-2, 1e-1],                  # 3 * 5 = 15 experiments
    # ),
    # twostage=dict(
    #     lr=[1e-3, 1e-2, 1e-1],                  # 3 * 5 = 15 experiments
    # )
)

common_args = dict(
    shortest_path=dict(
        epsilon=[0.001],
        timeout=[86400],
        theta_constr=[-1, 1],
        num_runs=[1],
        size=[50, 100, 200],
        nonlinearity=[1, 2, 5, 10],
        eps_bar=[0, 0.25],
        seed=list(range(5)),
        num_edges=[24],
        verbose=[''],
    ),
    energy_scheduling=dict(  # real data
        epsilon=[0.00001],
        leaf_minmax_no_prune=[''],
        # timeout=[86400],
        timeout=[27000],  # 7.5hrs
        theta_method=[1],
        num_runs=[1],
        size=[20, 50, 100],     # TODO: this can change...
        prob_config=[1],        # TODO: double-check whether we are going to use this config file
        # prob_config=[1],
        dataset=[3],
        seed=list(range(5)),
        verbose=[''],
    ),
    # energy_scheduling2=dict(  # generated data
    #     epsilon=[0.001],
    #     leaf_minmax_no_prune=[''],
    #     timeout=[86400],
    #     theta_constr=[-1, 1],
    #     num_runs=[1],
    #     size=[20, 50, 100],     # TODO: this can change...
    #     prob_config=[3],        # TODO: double-check whether we are going to use this config file
    #     seed=list(range(5)),
    #     use_gen_data=[''],
    #     nonlinearity=[1, 2, 5, 10],
    #     eps_bar=[0, 0.25],
    #     verbose=[''],
    # ),
    classification=dict(
        epsilon=[0.001],
        # timeout=[86400],
        timeout=[27000],  # 7.5hrs
        # theta_constr=[-1, 1],
        theta_method=[1],
        class_dim=[5],
        seed=list(range(5)),
        num_runs=[1],
        nonlinearity=[1, 2, 5, 10],
        size=[50, 100, 200],
        eps_bar=[0, 0.25],
        verbose=[''],
    )
)

# Number of run commands per domain
num_files = dict(
    shortest_path=dict(
        intopt=500,
        qptl=250,
        spoplus=360,
        twostage=360,
        emspo=1  # 240
    ),
    energy_scheduling=dict(
        intopt=12,
        qptl=12,
        spoplus=15,
        twostage=15,
        emspo=1  # 30
    ),
    classification=dict(
        intopt=500,
        qptl=250,
        spoplus=360,
        twostage=360,
        emspo=1  # 240
    ),
)

domains = ('shortest_path', 'classification', 'energy_scheduling')
baselines = list(hyperparams.keys())
dir_path = path.join(path.curdir, 'scripts')

def create_scripts(args):
    run_cmds = ""
    for alg in baselines:
        alg_path = path.join(dir_path, alg)
        algo_configs = hyperparams[alg]

        for dom in domains:
            alg_dom_path = path.join(alg_path, dom)
            os.makedirs(alg_dom_path, exist_ok=True)

            cnt = 0
            f_cnt = 0

            fname = f"job_{alg}_{dom}_{f_cnt}.sh" if args.job else f"run_{alg}_{dom}_{f_cnt}.sh"
            all_configs = algo_configs.copy()
            all_configs.update(common_args[dom])
            keys = tuple(all_configs.keys())
            for cfg in itertools.product(*list(all_configs.values())):
                run_cmd = f"python xaddpy/experiments/predopt/{dom}/run_experiments.py --algo {alg}"
                for k, v in zip(keys, cfg):
                    if v is not None:
                        run_cmd += f" --{k} {v}"
                run_cmds += f"{run_cmd}\n"
                cnt += 1
                if cnt == num_files[dom][alg]:
                    # Write a job file here
                    cnt = 0
                    js = job_str2 if dom in ['energy_scheduling', 'classification'] else job_str
                    run_cmds = js + run_cmds if args.job else run_str + run_cmds

                    with open(path.join(alg_dom_path, fname), 'w') as f:
                        f.write(run_cmds)

                    run_cmds = ""
                    f_cnt += 1
                    fname = f"job_{alg}_{dom}_{f_cnt}.sh" if args.job else f"run_{alg}_{dom}_{f_cnt}.sh"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--job', dest='job', default=False, action='store_true',
                        help='If True, create job scripts; otherwise, create a script with python commands')
    args = parser.parse_args()

    create_scripts(args)
