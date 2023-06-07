import argparse
import os.path as path
import sys
from typing import Union

import numpy as np
from scipy import stats

import examples.dblp.util as util
import examples.dblp.xor.gurobi_xor as gurobi_xor
from examples.dblp.xor.util import plot_runtimes, solve_dblp_with_xor
from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.utils.logger import logger, set_log_file_path
from xaddpy.utils.lp_util import return_model_info

sys.setrecursionlimit(10**6)


def run_experiments(args):
    method = args.method
    if method == 'sve':
        run_instance(args)
    elif method == 'gurobi':
        run_baseline(args)
    else:
        raise ValueError("Method should be specified as either 'sve' or 'gurobi'")


def run_baseline(args: argparse.Namespace):
    min_n = args.min_n
    max_n = max(args.max_n, min_n)
    
    t_lst = []
    timedout_lst = []
    ny = args.ny

    for n in range(min_n, max_n + 1):
        obj_val_lst = []
        t_gurobi_lst = []
        n_timeout = 0
        for s in args.seed:
            model_name = f'dblp_xor_n_{n}_seed_{s}_gurobi'
            args.model_name = model_name
            args.results_dir = util.get_results_dir(args)
            args.date_time = util.get_date_time()
            args.log_dir = set_log_file_path(args)

            m, is_timeout, info = \
                gurobi_xor.create_and_optimize_dblp_with_xor_constraints(
                    n, ny, s, model_name, args
                )
            t, obj_val = info['lapse'], info['obj_val']
            n_timeout += 1 if is_timeout else 0
            t_gurobi_lst.append(t)
            obj_val_lst.append(obj_val)

        num_cvars, num_bvars, num_constrs = return_model_info(m)

        # Write results on the file (one line per one `n' and multiple seeds)
        results_dir = args.results_dir
        results_fname = path.join(results_dir, 'results.txt')
        create = True if not path.exists(results_fname) else False

        with open(results_fname, 'a+') as txtfile:
            # Write the header
            if create:
                txtfile.write(f'n,ny,objective,timeout,t_gurobi,t_gurobi_se,# cont vars,# bin vars,# constrs\n')
            txtfile.write(f'{n},{ny},{np.mean(obj_val_lst)},{n_timeout},{np.mean(t_gurobi_lst):.4f},'
                          f'{get_se(t_gurobi_lst):.4f},{num_cvars},{num_bvars},{num_constrs}\n'
                          )
        t_lst.append(np.mean(t_gurobi_lst))
        timedout_lst.append(n_timeout)

    if args.plot:
        fig, ax = plot_runtimes(t_lst, tick_interval=3)


def run_instance(args):
    min_n = args.min_n
    max_n = max(args.max_n, min_n)
    t_lst = []
    ny = args.ny

    for n in range(min_n, max_n + 1):
        obj_val_lst = []
        t_symb_lst, t_milp_lst, t_model_lst, n_int_nodes_lst,\
            n_term_nodes_lst, n_cvar_milp_lst, n_bvar_milp_lst, n_constr_lst = \
                [], [], [], [], [], [], [], []

        for s in args.seed:
            # Handle logging and result recording
            model_name = f'dblp_xor_n_{n}_seed_{s}'
            args.model_name = model_name
            args.results_dir = util.get_results_dir(args)
            args.date_time = util.get_date_time()
            args.log_dir = set_log_file_path(args)
            args.name = model_name
            
            # Run experiments
            context = XADD(vars(args))
            g_model, info_dict = solve_dblp_with_xor(context=context,
                                                     n=n,
                                                     ny=ny,
                                                     seed=s,
                                                     model_name=model_name,
                                                     args=args)

            num_cvars_milp, num_bvars_milp, num_constrs_milp = return_model_info(g_model)
            
            t_symb_lst.append(info_dict['time_symb'])
            t_milp_lst.append(info_dict['time_milp'])
            t_model_lst.append(info_dict['time_modeling'])
            n_cvar_milp_lst.append(num_cvars_milp)
            n_bvar_milp_lst.append(num_bvars_milp)
            n_constr_lst.append(num_constrs_milp)
            obj_val_lst.append(info_dict['obj_value'])
            
            if args.verbose:
                logger.info(f"")

        # Write results on the file (one line per one `n' and multiple seeds)
        results_dir = args.results_dir
        results_fname = path.join(results_dir, 'results.txt')
        create = True if not path.exists(results_fname) else False
        t_total = np.array(t_symb_lst) + np.array(t_milp_lst)
        with open(results_fname, 'a+') as txtfile:
            # Write the header
            if create:
                txtfile.write(f'n,ny,objective,t_total,t_total_se,'
                              f't_sym,t_sym_se,t_model,t_model_se,t_gurobi,t_gurobi_se,'
                              f'# cont vars,# cont vars se,# bin vars,# bin vars se,# constrs,'
                              f'# constrs se\n')
            n_total_nodes = np.array(n_int_nodes_lst) + np.array(n_term_nodes_lst)
            txtfile.write(f'{n},{ny},{np.mean(obj_val_lst)},{np.mean(t_total):.4f},{get_se(t_total):.4f},'
                          f'{np.mean(t_symb_lst):.4f},{get_se(t_symb_lst):.4f},'
                          f'{np.mean(t_model_lst):.4f},{get_se(t_model_lst):.4f},{np.mean(t_milp_lst):.4f},'
                          f'{get_se(t_milp_lst):.4f},{np.mean(n_cvar_milp_lst):.2f},{get_se(n_cvar_milp_lst):.2f},'
                          f'{np.mean(n_bvar_milp_lst):.2f},{get_se(n_bvar_milp_lst):.2f},{np.mean(n_constr_lst):.2f},'
                          f'{get_se(n_constr_lst):.2f}\n'
                          )
        t_total_mean = np.mean(t_total)
        t_lst.append(t_total_mean)

    if args.plot:
        plot_runtimes(t_lst, tick_interval=3)


def get_se(arr: Union[np.ndarray, list]):
    return 0 if len(arr) == 1 else stats.sem(arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', dest='domain', default='xor', type=str)
    parser.add_argument('--method', dest='method', default='sve', type=str)     # 'sve' or 'gurobi'
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
    parser.add_argument('--seed', dest='seed', nargs='+', type=int)
    parser.add_argument('--solver_type', dest='solver_type', type=int, default=0,
                        help='Type 0: defer the casemins till the very end; '
                             'type 1: min out each variable sequentially')
    parser.add_argument('--leaf_minmax_no_prune', default=False, action='store_true',
                        help='prune_equality turned off during addition of independent decisions.')

    parser.add_argument('--ny', dest='ny', type=int, default=None)
    parser.add_argument('--n_constrs_y', dest='n_constrs_y', type=int, default=15,
                        help='The number of linear constraints over y')
    parser.add_argument('--n_ri_per_z', dest='n_ri_per_z', type=int, default=-1,
                        help='The number of buckets linked by a single z variable')
    parser.add_argument('--max_n', dest='max_n', type=int, default=1,
                        help="The maximum number value for n upto which we iterate")
    parser.add_argument('--min_n', dest='min_n', type=int, default=1,
                        help="The minimum number value for n")
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=1e-3,
                        help='A small value to handle strict inequality')
    parser.add_argument('--timeout', dest='timeout', type=float, default=1800,
                        help='Timeout for Gurobi optimization')
    parser.add_argument('--time_interval', dest='time_interval', type=float, default=1500,
                        help='Periodically output some experimental results')
    parser.add_argument('--use_q', dest='use_q', action='store_true')

    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Whether to plot runtimes vs. number of decision variables')


    args = parser.parse_args()

    run_experiments(args)
