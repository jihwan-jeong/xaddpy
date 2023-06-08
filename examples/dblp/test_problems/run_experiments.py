import argparse
import json
import os
import os.path as path
import sys
import time

import numpy as np
from gurobipy import GRB
from scipy import stats

import examples.dblp.solve as dblp_solve
from examples.dblp import util as dblp_util
from examples.dblp.gurobi import construct_dblp_gurobi_model
from examples.dblp.util import build_xadd_from_json
from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.milp.solve import solve_milp
from xaddpy.utils.logger import logger, set_log_file_path
from xaddpy.utils.lp_util import return_model_info

sys.setrecursionlimit(10**6)


def create_prob_inst(args, seed):
    check = False
    # Vincente et al. (1992)... problem configuration needs to be defined
    if args.cfg is not None:
        assert len(args.cfg) == 5
        k11, k12, k13, k1, k2 = args.cfg
        if args.skip_generation:
            fname_json = f'./examples/dblp/test_problems/prob_instances/dblp_inst_{k11}_{k12}_{k13}_{k1}_{k2}_opt_{args.opt_var}_seed{seed}.json'
            check = path.exists(fname_json)
            if not check:
                logger.info(f"File {fname_json} does not exist! Create...")
        if (args.skip_generation and not check) or not args.skip_generation:
            fname_json = dblp_util.create_prob_json(seed,
                                                    dir_path='examples/dblp/test_problems/prob_instances',
                                                    cfg=[k11, k12, k13, k1, k2],
                                                    opt_var_set_id=args.opt_var)
    # Randomized test instances
    else:
        if args.nx is not None and args.ny is not None:
            fname_json = f'./experiments/dblp/test_problems/prob_instances/' \
                         f'dblp_inst_{args.nx}_{args.ny}_density_{args.density}_opt_{args.opt_var}_seed{seed}.json'
        elif args.nx is None and args.ny is None:
            fname_json = f'./experiments/dblp/test_problems/prob_instances/' \
                         f'dblp_inst_density_{args.density}_opt_{args.opt_var}_seed{seed}.json'
        else:
            raise ValueError

        if args.skip_generation:
            check = path.exists(fname_json)
            if not check:
                logger.info(f"File {fname_json} does not exist! Create...")
        if (args.skip_generation and not check) or not args.skip_generation:
            fname_json = dblp_util.create_prob_json(seed,
                                                    dir_path='examples/dblp/test_problems/prob_instances',
                                                    density=args.density,
                                                    nx=args.nx,
                                                    ny=args.ny)
    return fname_json


def run_baseline(args):
    args.results_dir = dblp_util.get_results_dir(args)
    args.date_time = dblp_util.get_date_time()

    for s in args.seed:
        fname_json = create_prob_inst(args, s)
        args.json_file = fname_json
        model_name = fname_json.split('/')[-1].split('.')[0]
        args.model_name = model_name + '_gurobi'
        args.log_dir = set_log_file_path(args)

        # Construct the model
        m, info_dict = construct_dblp_gurobi_model(args)
        
        # Optimize the model
        stime = time.time()
        m.optimize()
        etime = time.time()
        lapse = etime - stime

        status = m.status
        num_dec_vars_0 = info_dict['num_dec_vars_0']
        num_dec_vars_1 = info_dict['num_dec_vars_1']
        num_constraints = info_dict['num_constraints']

        if status != GRB.OPTIMAL:
            logger.debug(f"Optimization was stopped with status {status}")
            continue
        obj_value = m.objVal
        logger.info(f"Optimal objective: {obj_value}")
        logger.info(f"Number of decision variables 0: {num_dec_vars_0}")
        logger.info(f"Number of decision variables 1: {num_dec_vars_1}")
        logger.info(f"Number of constraints: {num_constraints}")
        logger.info(f"Seed: {s}")
        logger.info(f"Model name: {model_name}")

    # Write the results to a file
    results_dir = args.results_dir
    output_file = path.join(results_dir, args.output_file)
    create = True if not path.exists(output_file) else False
    with open(output_file, 'a+') as f:
        if args.cfg is not None:
            # Write the header
            if create:
                f.write('nx,ny,n_constraints,objective,time,config\n')
            # Write the results
            f.write(f'{args.nx},{args.ny},{num_constraints},{obj_value},{lapse},'
                    f'{"_".join(map(lambda x: str(x), args.cfg))}\n')       
    

def run_instance(args, seed=None) -> dict:
    """Runs a single experiment"""    
    # Prepare model name and xadd file name from json file name
    fname_json = args.json_file
    model_name = fname_json.split('/')[-1].split('.')[0]
    fname_xadd = fname_json.replace('.json', '.xadd')
    try:
        with open(fname_json, "r") as json_file:
            prob_instance = json.load(json_file)
    except:
        print("Failed to open file!!")
        raise FileNotFoundError
    
    # Gurobi model
    m = dblp_util.setup_gurobi_model(model_name, epsilon=args.epsilon)

    args.model_name = model_name
    args.results_dir = dblp_util.get_results_dir(args)
    args.date_time = dblp_util.get_date_time()
    args.log_dir = set_log_file_path(args)
    
    # Perform symbolic minimization over one set of decision variables
    error, obj_value = False, None
    if args.check_instance:
        context = XADD(vars(args))
        context.link_json_file(fname_json)
        _, dblp_xadd, _ = build_xadd_from_json(context, fname_json, model_name)
        if dblp_xadd == 2:
            logger.info(f'seed: {seed}\tShould be skipped')
        else:
            logger.info(f'seed: {seed}\tShould be included')
        return
    try:
        context, eq_constr_dict, time_sve = \
                            dblp_solve.solve(
                                solver_type=args.solver_type,
                                save_xadd=True,             # Should save the resulting XADD to a file
                                verbose=True,
                                fname_xadd=fname_xadd,
                                args=args,
                            )

        # Import the resulting MILP XADD
        context = XADD(vars(args))
        variables, eq_constr_dict = context.import_lp_xadd(fname_xadd)
        obj_node_id = context.get_objective()
        num_int_nodes, num_term_nodes = dblp_util.get_num_nodes(context, obj_node_id)
        logger.info(f'Number of decision/terminal nodes: {num_int_nodes} / {num_term_nodes}')

        # Compile the MILP from XADD and solve it
        timeout = 3600
        time_interval = 1800
        info_dict = solve_milp(
            context,
            m,
            variables,
            eq_constr_dict,
            verbose=True,
            timeout=timeout,
            time_interval=time_interval,
            args=args,
        )
        time_modeling, time_milp, obj_value = info_dict['time_modeling'],\
                                              info_dict['time_milp'],\
                                              info_dict['obj_value']
    except Exception as e:
        error = True
        logger.debug(e)

    logger.info(f"Optimal objective: {obj_value}")

    # Some additional info to record
    num_dec_vars_0 = len(prob_instance['cvariables0'])     # Variables that are treated as free variables during symbolic min
    num_dec_vars_1 = len(prob_instance['cvariables1'])     # Variables that are symbolically optimized
    opt_var_set = prob_instance.get('min-var-set-id')
    num_constraints = len(prob_instance['ineq-constr']) + len(prob_instance['eq-constr'])
    num_local_opts = prob_instance.get('num-local-opts', None)
    num_global_opts = prob_instance.get('num-global-opts', None)
    if error:
        obj_value, infeasible, time_sve, time_modeling, time_milp, num_int_nodes, num_term_nodes\
            = None, None, None, None, None, None, None
        num_cvars_milp, num_bvars_milp, num_constrs_milp = 0, 0, 0
    else:
        infeasible = True if obj_value is None else False
        num_cvars_milp, num_bvars_milp, num_constrs_milp = return_model_info(m)
    return dict(
        num_dec_vars_0=num_dec_vars_0,
        num_dec_vars_1=num_dec_vars_1,
        opt_var_set=opt_var_set,
        num_constraints=num_constraints,
        num_local_opts=num_local_opts,
        num_global_opts=num_global_opts,
        obj_value=obj_value,
        infeasible=infeasible,
        time_sve=time_sve,
        time_modeling=time_modeling,
        time_milp=time_milp,
        num_int_nodes=num_int_nodes,
        num_term_nodes=num_term_nodes,
        num_cvars_milp=num_cvars_milp,
        num_bvars_milp=num_bvars_milp,
        num_constrs_milp=num_constrs_milp,
    )
    

def run_experiments(args: argparse.Namespace):
    t_symb_lst, t_milp_lst, t_model_lst, n_int_nodes_lst, n_term_nodes_lst,\
        n_cvar_milp_lst, n_bvar_milp_lst, n_constr_lst = [], [], [], [], [], [], [], []
    
    num_dec_vars_0, num_dec_vars_1, opt_var_set, density, num_constraints,\
        num_local_opts, num_global_opts, obj_value, infeasible, time_sve,\
            time_modeling, time_milp, num_int_nodes, num_term_nodes, \
                num_cvars_milp, num_bvars_milp, num_constrs_milp, fname_json = [None] * 18
    
    for s in args.seed:
        fname_json = create_prob_inst(args, seed=s)
        args.json_file = fname_json

        if args.check_instance:
            run_instance(args, s)
            continue

        # Run experiments
        res_dict = run_instance(args)
        num_dec_vars_0 = res_dict['num_dec_vars_0']
        num_dec_vars_1 = res_dict['num_dec_vars_1']
        opt_var_set = res_dict['opt_var_set']
        num_constraints = res_dict['num_constraints']
        num_local_opts = res_dict['num_local_opts']
        num_global_opts = res_dict['num_global_opts']
        obj_value = res_dict['obj_value']
        infeasible = res_dict['infeasible']
        
        t_symb_lst.append(res_dict['time_sve'])
        t_milp_lst.append(res_dict['time_milp'])
        t_model_lst.append(res_dict['time_modeling'])
        n_int_nodes_lst.append(res_dict['num_int_nodes'])
        n_term_nodes_lst.append(res_dict['num_term_nodes'])
        n_cvar_milp_lst.append(res_dict['num_cvars_milp'])
        n_bvar_milp_lst.append(res_dict['num_bvars_milp'])
        n_constr_lst.append(res_dict['num_constrs_milp'])

    if args.check_instance:
        return
    # Write on the file
    results_dir = args.results_dir
    output_file = path.join(results_dir, args.output_file)
    create = True if not path.exists(output_file) else False
    with open(output_file, 'a+') as txtfile:

        if args.cfg is not None:
            # Write the header
            if create:

                txtfile.write(f'nx,ny,opt_var,n_constraints,# local,# global,objective,infeasible,'
                              f't_sym,t_sym_se,t_model,t_model_se,t_milp,t_milp_se,# int nodes,# leaf nodes,'
                              f'# milp cont vars,# milp bin vars,# milp constrs,config\n')
            txtfile.write(f'{num_dec_vars_0},{num_dec_vars_1},{opt_var_set},{num_constraints},{num_local_opts},'
                          f'{num_global_opts},{obj_value},{infeasible},{np.mean(t_symb_lst):.4f},'
                          f'{stats.sem(t_symb_lst):.4f},{np.mean(t_model_lst):.4f},{stats.sem(t_model_lst):.4f},'
                          f'{np.mean(t_milp_lst):.4f},{stats.sem(t_milp_lst):.4f},{np.mean(n_int_nodes_lst):.4f},'
                          f'{np.mean(n_term_nodes_lst):.4f},{np.mean(n_cvar_milp_lst):.4f},'
                          f'{np.mean(n_bvar_milp_lst):.4f},{np.mean(n_constr_lst):.4f},'
                          f'{"_".join(map(lambda x: str(x), args.cfg))}\n')
        else:
            # Write the header
            if create:
                txtfile.write(f'nx,ny,opt_var,density,n_constraints,objective,infeasible,'
                              f't_sym,t_sym_se,t_model,t_model_se,t_milp,t_milp_se,# nodes,# nodes se,'
                              f'# milp cont vars,# milp cont vars se,# milp bin vars,# milp bin vars se,# milp constrs,'
                              f'# milp constrs se\n')
            n_total_nodes = np.array(n_int_nodes_lst) + np.array(n_term_nodes_lst)
            txtfile.write(f'{num_dec_vars_0},{num_dec_vars_1},{opt_var_set},{args.density},{num_constraints},'
                          f'{obj_value},{infeasible},{np.mean(t_symb_lst):.4f},'
                          f'{stats.sem(t_symb_lst):.4f},{np.mean(t_model_lst):.4f},{stats.sem(t_model_lst):.4f},'
                          f'{np.mean(t_milp_lst):.4f},{stats.sem(t_milp_lst):.4f},'
                          f'{np.mean(n_total_nodes):.2f},{stats.sem(n_total_nodes)},{np.mean(n_cvar_milp_lst):.2f},'
                          f'{stats.sem(n_cvar_milp_lst):.2f},{np.mean(n_bvar_milp_lst):.2f},'
                          f'{stats.sem(n_bvar_milp_lst):.2f},{np.mean(n_constr_lst):.2f},{stats.sem(n_constr_lst):.2f}\n'
                          )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', dest='domain', default='vincente', type=str)    # 'vincente' or 'random'
    parser.add_argument('--method', dest='method', default='sve', type=str)     # 'sve' or 'gurobi'
    parser.add_argument('--cfg', dest='cfg', nargs='+', type=int,
                        help='Problem specific configurations')
    parser.add_argument('--density', dest='density', type=float, default=1.0)
    parser.add_argument('--skip_generation', dest='skip_generation', action='store_true')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)

    # for instances due to Vicente et al., './experiments/dblp/results1.txt'
    # for random instances, './experiments/dblp/results2.txt'
    parser.add_argument('--output_file', dest='output_file', type=str)
    parser.add_argument('--seed', dest='seed', nargs='+', type=int)
    parser.add_argument('--solver_type', dest='solver_type', type=int, default=1,
                        help='Type 0: defer the casemins till the very end; '
                             'type 1: min out each variable sequentially')
    parser.add_argument('--leaf_minmax_no_prune', default=False, action='store_true',
                        help='prune_equality turned off during addition of independent decisions.')

    parser.add_argument('--nx', dest='nx', type=int, default=None)
    parser.add_argument('--ny', dest='ny', type=int, default=None)
    parser.add_argument('--opt_var', dest='opt_var', type=int, default=1,
                        help='0: minimize out x variables;'
                             '1: minimize out y variables')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0)
    parser.add_argument('--check_instance', dest='check_instance', action='store_true')
    args = parser.parse_args()

    if args.method == 'gurobi':
        run_baseline(args)
    else:
        run_experiments(args)
