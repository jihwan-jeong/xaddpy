from os import path

import numpy as np
from sklearn import preprocessing
import torch

from xaddpy.models.base import Oracle
from xaddpy.models.emspo import EMSPO
from xaddpy.models.spo_plus import SPOPlusShortest
from xaddpy.models.qptl import QPTLShortest
from xaddpy.models.intopt import IntOptShortest
from xaddpy.models.two_stage import TwoStageShortest

from xaddpy.experiments.predopt.helper import \
    run_emspo, test_algorithm, get_feature_dim, get_param_dim, get_results_dir, get_config_dir, get_model_name
from xaddpy.utils.util import get_date_time
from xaddpy.experiments.predopt.shortest_path.util import generate_dataset, make_json
from xaddpy.utils.logger import logger, set_log_file_path


def var_name_rule(x):
    """Returns values to be used for sorting variables"""
    x = str(x)
    x_split = x.split('_')
    if len(x_split) == 1:
        return float(str(x)[1:])
    elif len(x_split) == 2:
        return (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]))
    elif len(x_split) == 3:
        return (float(str(x).split("_")[0][1:]), float(str(x).split("_")[1]), float(str(x).split("_")[2]))


def run_baselines(alg: str, *args, **kwargs):
    cls = globals()[alg]
    bs = cls(*args, **kwargs)
    init_cost, opt_cost, init_regret = bs.test_model(bs.X_train, bs.y_train, dom='shortest')
    logger.info('Test initial regret loss with random model parameters')
    logger.info(f'Cost: {init_cost:.10f}'.ljust(25) + f'Optimal: {opt_cost:.10f}'.ljust(25) + \
                f'Regret: {init_regret:.10f}'.ljust(25))

    # Skip run if parameter already exists
    if not path.exists(kwargs['fname_param']):
        bs.fit(**kwargs)

        # Load the best parameters
        bs.load_params(kwargs['fname_param'])
    else:
        logger.info(f"Parameter file {kwargs['fname_param']} already exists.... exiting")
        exit(0)

    # Load the best parameters
    bs.load_params(kwargs['fname_param'])
    return bs


def run(args):
    # Method argument --- 'emspo': our approach / 'baselines': all baselines ('spoplus', 'qptl', 'intopt', 'twostage')
    algo = args.algo
    num_runs = args.num_runs
    size = args.size
    num_edges = args.num_edges
    use_validation = args.use_validation
    standardize = args.standardize
    epochs = args.epochs
    embed_dim = args.embed_dim
    linear = len(embed_dim) == 0
    l2_lamb = args.l2_lamb
    domain = 'shortest_path'
    args.timeout = args.timeout if args.timeout != -1 else float('inf')
    args.domain = domain
    args.date_time = get_date_time()
    args.config_dir = get_config_dir(args)
    args.results_dir = get_results_dir(args)

    args.model_name = model_name = get_model_name(args)
    args.log_dir = set_log_file_path(args)

    logger.info(f'Configuration: {args}')
    dir_path = path.join(path.curdir, 'xaddpy/experiments/predopt/shortest_path')
    node_arc_incidence_fname = f'node_edge_incidence_{num_edges}edges.npy'

    if num_runs > 1:
        args.seed = list(range(num_runs))
    else:
        args.seed = [args.seed if args.seed is not None else 0]

    # Generate dataset according to configuration
    fname_json = make_json(num_edges, args)        # Create the .json file
    X_train, c_train, X_test, c_test = generate_dataset(args)

    # Loop through runs
    for r in range(num_runs):
        logger.info(f'----------------------- Run {r + 1} / {num_runs} -----------------------')
        X_train_r, c_train_r, X_test_r, c_test_r = X_train[r], c_train[r], X_test[r], c_test[r]
        X_val, c_val = None, None

        if use_validation:
            val_size = int(size * 0.2)
            X_val, c_val = X_train_r[-val_size:], c_train_r[-val_size:]
            X_train, c_train = X_train_r[:-val_size], c_train_r[:-val_size]

        scaler = None
        if standardize:
            scaler = preprocessing.StandardScaler().fit(X_train_r)
            X_train_r = scaler.transform(X_train_r)
            X_test_r = scaler.transform(X_test_r)
            if use_validation:
                X_val = scaler.transform(X_val)

        dataset = dict(
            train=[X_train_r, c_train_r],
            test=[X_test_r, c_test_r],
            val=[X_val, c_val],
        )
        logger.info("Dataset prepared")

        target_dim = c_train_r.shape[1]  # Number of edges = number of decision variables
        input_dim = X_train_r.shape[1]  # Number of features
        feature_dim = get_feature_dim(input_dim, embed_dim)
        param_dim = get_param_dim(args, target_dim, feature_dim)
        fname_param = path.join(args.results_dir, f'{model_name}__seed_{args.seed[r]}.npy')

        # Firstly, create the oracle and presolve for the true decisions
        oracle = Oracle(domain, fname=fname_json, var_name_rule=var_name_rule, )
        oracle.presolve(c_train_r)
        logger.info('Oracle presolve done')

        logger.info('Start running experiments')
        torch.manual_seed(args.seed[r])
        res = {}

        if algo == 'emspo':
            # model_name = f'shortest_path_edges_{num_edges}_param_deg_{args.nonlinearity}_eps_{args.eps_bar}_size_{args.size}' \
            #              f'_seed_{args.seed}_{args.domain}_{args.date_time}'

            emspo = run_emspo(target_dim, input_dim, dataset, embed_dim, feature_dim, param_dim, linear, scaler, oracle,
                      l2_lamb, use_validation, model_name, fname_param, fname_json, args, var_name_rule=var_name_rule)

            # Load the EMSPO model with the optimized (and saved) parameters
            emspo.load_params(fname_param)

            # Test the EMSPO model
            emspo_res = {'emspo': test_algorithm(emspo, dataset, 'train', 'shortest')}
            res.update(emspo_res)

        else:
            baselines = ['twostage', 'spoplus', 'qptl', 'intopt'] if algo == 'baselines' else [algo]
            bs_dict = {'spoplus': 'SPOPlusShortest', 'twostage': 'TwoStageShortest',
                       'qptl': 'QPTLShortest', 'intopt': 'IntOptShortest'}
            A = np.load(path.join(dir_path, 'data', node_arc_incidence_fname))

            kwargs = dict(
                model_name=model_name,
                target_dim=target_dim,
                input_dim=input_dim,
                dataset=dataset,
                embed_dim=embed_dim,
                linear=linear,
                oracle=oracle,
                scaler=scaler,
                batch_size=10,
                lr=args.lr,
                use_validation=use_validation,
                l2_lamb=l2_lamb,
                epochs=epochs,
                A=A,
                fname_param=fname_param,  # .npy file to be saved

                # QPTL parameters
                tau=args.tau,
                timeout_iter=args.timeout_iter,

                # IntOpt parameters
                method=1,
                max_iter=100,
                damping=args.damping,
                smoothing=False,
                thr=args.thr,
                mu0=None,
            )

            # Run the baseline algorithms
            alg_inst_dict = {k: run_baselines(bs_dict[k], **kwargs) for k in baselines}

            # Check train loss
            bs_res = {k: test_algorithm(alg_inst_dict[k], dataset=dataset, config='train', domain='shortest')
                      for k in bs_dict if k in baselines}
            res.update(bs_res)

        logger.info(f"\nTrain result:\n" +
                    '\n'.join([f'{k}'.ljust(20) + f'Cost: {res[k][0]:.10f}'.ljust(25) +
                               f'Optimal: {res[k][1]:.10f}'.ljust(25) +
                               f'Regret: {res[k][2]:.10f}'.ljust(25)
                               for k in res.keys()]))

        res_file = path.join(f'{args.results_dir}', 'results.csv')
        if path.exists(res_file):
            with open(res_file, 'a') as f:
                f.write(f"{model_name},{args.seed[r]},{','.join(list(map(str, res[args.algo])))}\n")
        else:
            with open(res_file, 'w') as f:
                f.write('Model Name,seed,Actual Cost,Optimal Cost,Regret\n')
                f.write(f"{model_name},{args.seed[r]},{','.join(list(map(str, res[args.algo])))}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_edges', dest='num_edges', type=int, default=12,
                        help='Number of edges')
    parser.add_argument('--size', dest='size', type=int, default=200,
                        help='Number of data instances')
    parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=5,
                        help='The feature dimensionality')
    parser.add_argument('--nonlinearity', dest='nonlinearity', type=int, default=1,
                        help='Degree of nonlinearity in data generation (1: linear)')
    parser.add_argument('--eps_bar', dest='eps_bar', type=float, default=0,
                        help='Multiplicative noise is sampled from Uniform(1-eps_bar, 1+eps_bar)')
    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help='Random seed to use')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If set to True, you can check Gurobi optimization log file')

    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--use_validation', dest='use_validation', default=False, action='store_true',
                        help='Whether to use a validation set')
    parser.add_argument('--standardize', dest='standardize', default=True, action='store_false',
                        help='Whether to standardize the train set')
    parser.add_argument('--algo', dest='algo', type=str, default='emspo',
                        help='Which algorithm to use for experiment')

    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float,
                        help='The learning rate of optimizer')
    parser.add_argument('--embed_dim', dest='embed_dim', default=[], type=int, nargs='+',
                        help='Hidden layer dimensions (if exist) of a neural network')
    parser.add_argument('--l2_lamb', dest='l2_lamb', default=0, type=float,
                        help='The hyperparameter controlling L2 regularization of parameters')

    # Parameters related to EMSPO
    parser.add_argument('--epsilon', type=float, default=0.0001,
                        help='Epsilon to be added to inequalities when constructing a MILP')
    parser.add_argument('--timeout', type=int, default=200,
                        help='Timout for gurobi model optimization (in sec)')
    parser.add_argument('--time_interval', type=int, default=3600,
                        help='Time interval to cache incumbent parameter solution in SPO MILP')
    parser.add_argument('--leaf_minmax_no_prune', default=False, action='store_true',
                        help='prune_equality turned off during addition of independent decisions.')
    parser.add_argument('--scheme', default='bruteforce', type=str,
                        help='bruteforce: use indicator constraints; benders: use combinatorial Benders decomposition')
    parser.add_argument('--theta_method', type=int, default=1,
                        help='How to handle theta=0 pathological solutions (values: 0, 1)')
    parser.add_argument('--theta_constr', default=False, type=float,
                        help='Whether to fix the absolute value of the sum of parameters at a specific value')
    parser.add_argument('--force', default=None, type=int,
                        help='Whether to recompute the optimal parameters (1: recompute; otherwise: exit)')
    parser.add_argument('--nodefilestart', default=30, type=int,
                        help='Write nodes to disk if memory used exceeds parameter')

    # Parameters related to baselines
    parser.add_argument('--timeout_iter', type=float, default=500,
                        help='Timout per iteration for QPTL and IntOpt (in sec)')
    parser.add_argument('--thr', default=0.1, type=float,
                        help='The threshold hyperparameter used in IntOpt')
    parser.add_argument('--damping', default=1e-6, type=float,
                        help='The damping factor used in IntOpt')
    parser.add_argument('--tau', default=20000, type=float,
                        help='The quadratic regularizer used in QPTL')
    args = parser.parse_args()

    run(args)
