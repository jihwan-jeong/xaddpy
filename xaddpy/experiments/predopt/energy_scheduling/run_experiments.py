from sklearn import preprocessing
from os import path
from torch import optim
import torch
import sys
sys.setrecursionlimit(10**6)

from xaddpy.experiments.predopt.energy_scheduling.util import generate_dataset, make_json
from xaddpy.experiments.predopt.energy_scheduling import gurobi_model

from xaddpy.experiments.predopt.helper import \
    run_emspo, test_algorithm, get_feature_dim, get_param_dim, get_results_dir, get_config_dir, get_model_name
from xaddpy.utils.util import get_date_time
from xaddpy.utils.logger import logger, set_log_file_path

from xaddpy.models.spo_plus import SPOPlusEnergy
from xaddpy.models.qptl import QPTLEnergy
from xaddpy.models.intopt import IntOptEnergy
from xaddpy.models.two_stage import TwoStageEnergy


def run_baselines(alg: str, *args, **kwargs):
    cls = globals()[alg]
    bs = cls(*args, **kwargs)
    init_cost, opt_cost, init_regret = bs.test_model(bs.X_train, bs.y_train, dom='energy')
    logger.info('Test initial regret loss with random model parameters')
    logger.info(f'Cost: {init_cost:.5f}'.ljust(25) + f'Optimal: {opt_cost:.5f}'.ljust(25) + \
                f'Regret: {init_regret:.5f}'.ljust(25))

    # Skip run if parameter already exists
    if not path.exists(kwargs['fname_param']):
        bs.fit(**kwargs)

        # Load the best parameters
        bs.load_params(kwargs['fname_param'])
    else:
        logger.info(f"Parameter file {kwargs['fname_param']} already exists.... exiting")
        exit(0)

    return bs


def run(args):
    num_runs = args.num_runs
    use_validation = args.use_validation
    standardize = args.standardize
    algo = args.algo    # algo in ('spoplus', 'qptl', 'intopt', 'emspo')
    size = args.size
    epochs = args.epochs
    embed_dim = args.embed_dim
    linear = True if len(embed_dim) == 0 else False
    l2_lamb = args.l2_lamb
    domain = 'energy_scheduling'
    args.timeout = args.timeout if args.timeout != -1 else float('inf')
    args.domain = domain
    args.date_time = get_date_time()
    args.config_dir = get_config_dir(args)
    args.results_dir = get_results_dir(args)

    args.model_name = model_name = get_model_name(args) #f'{args.domain}_{args.date_time}'
    args.log_dir = set_log_file_path(args)

    logger.info('Configuration: %(config)s', {'config': args})

    if num_runs > 1:
        args.seed = list(range(num_runs))
    else:
        args.seed = [args.seed if args.seed is not None else 0]

    if args.prob_config == 0:
        prob_config_fname = 'day01.txt'  # (3m, 10j)
    elif args.prob_config == 1:
        prob_config_fname = 'day01_modified.txt'  # (1m, 2j)
    elif args.prob_config == 2:
        prob_config_fname = 'day01_modified2.txt'  # (2m, 7j)
    elif args.prob_config == 3:
        prob_config_fname = 'day01_modified3.txt'  # (1m, 3j)
    else:
        raise ValueError('Unknown prob_config specified.')

    # Import parameters of the optimization problem
    dataset_dir = path.join(path.curdir, 'xaddpy/experiments/predopt/energy_scheduling/data')
    prob_config_path = path.join(dataset_dir, prob_config_fname)
    prob_configs = gurobi_model.read_config(prob_config_path, args)

    fname_json = make_json(args, prob_config_path)      # TODO: Check the directory and what info is put in

    # Generate dataset according to configuration
    X_train_multi, c_train_multi, X_test, c_test = generate_dataset(args)

    # Loop through runs
    for r in range(num_runs):
        logger.info(f'----------------------- Run {r + 1} / {num_runs} -----------------------')
        X_train, c_train = X_train_multi[r], c_train_multi[r]
        X_val, c_val = None, None
        num_data = X_train.shape[0]

        if use_validation:
            val_size = int(num_data * 0.2)
            X_val, c_val = X_train[-val_size:, :], c_train[-val_size:, :]
            X_train, c_train = X_train[:-val_size, :], c_train[:-val_size, :]

        scaler = None
        if standardize:
            dim2 = X_train.shape[1]
            scaler = preprocessing.StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
            X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(-1, dim2, X_train.shape[-1])
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(-1, dim2, X_test.shape[-1])
            if use_validation:
                X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(-1, dim2, X_val.shape[-1])

        dataset = {'train': [X_train, c_train],
                   'test': [X_test, c_test],
                   'val': [X_val, c_val]}
        logger.info('Dataset prepared')

        # Set necessary dimensions
        target_dim = 1  # single value of r_t to be predicted

        input_dim = X_train.shape[2]
        feature_dim = get_feature_dim(input_dim, embed_dim)
        param_dim = get_param_dim(args, target_dim, feature_dim)
        fname_param = path.join(args.results_dir, f'{model_name}__seed_{args.seed[r]}.npy')

        # Firstly, create the oracle, build and presolve the model
        oracle = gurobi_model.GurobiICON(**prob_configs)
        oracle.make_model()
        oracle.compute_sols(c_train, presolve=True)
        logger.info('Presolve done')

        logger.info('Start running experiments')
        torch.manual_seed(args.seed[r])
        res = {}

        if algo == "emspo":
            emspo = run_emspo(target_dim, input_dim, dataset, embed_dim, feature_dim, param_dim, linear, scaler, oracle,
                              l2_lamb, use_validation, model_name, fname_param, fname_json, args, prob_configs=prob_configs)

            # Load the EMSPO model with the optimized (and saved) parameters
            emspo.load_params(fname_param)

            # Test the EMSPO model
            emspo_res = {'emspo': test_algorithm(emspo, dataset, 'train', 'energy')}
            res.update(emspo_res)

        else:
            baselines = ['twostage', 'spoplus', 'qptl', 'intopt'] if algo == 'baselines' else [algo]
            bs_dict = {'spoplus': 'SPOPlusEnergy', 'twostage': 'TwoStageEnergy',
                       'qptl': 'QPTLEnergy', 'intopt': 'IntOptEnergy'}

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
                lr=0.001,
                use_validation=use_validation,
                l2_lamb=l2_lamb,
                epochs=epochs,
                optimizer=optim.Adam,
                sample_per_day=args.sample_per_day,
                prob_configs=prob_configs,
                fname_param=fname_param,                # .npy file to be saved

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
            bs_res = {k: test_algorithm(alg_inst_dict[k], dataset=dataset, config='train', domain='energy')
                      for k in bs_dict if k in baselines}
            res.update(bs_res)

        logger.info(f"\nTrain result:\n" +
                    '\n'.join([f'{k}'.ljust(20) + f'Cost: {res[k][0]:.5f}'.ljust(25) +
                               f'Optimal: {res[k][1]:.5f}'.ljust(25) +
                               f'Regret: {res[k][2]:.5f}'.ljust(25)
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

    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--use_validation', dest='use_validation', default=False, action='store_true',
                        help='Whether to use a validation set')
    parser.add_argument('--train_test_ratio', dest='train_test_ratio', type=float, default=0.7,
                        help='The portion of the train set')
    parser.add_argument('--standardize', dest='standardize', default=True, action='store_false',
                        help='Whether to standardize the train set')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help='Random seed to use')

    parser.add_argument('--size', dest='size', default=100, type=int,
                        help='Number of days from the data to include in training')
    parser.add_argument('--sample_per_day', dest='sample_per_day', default=24, type=int,
                        help='Number of samples taken per day')
    parser.add_argument('--presolve', dest='presolve', default=True, action='store_false')
    parser.add_argument('--algo', dest='algo', type=str, default='emspo',
                        help='Which algorithm to use for experiment')
    parser.add_argument('--prob_config', dest='prob_config', default=1, type=int,
                        help='Which configuration file to use.... 0 (full), 1, 2, 3 available')
    parser.add_argument('--dataset', type=int, default=1, help='Dataset selection. Can be 1 (full), 2.')

    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float,
                        help='The learning rate of optimizer')
    parser.add_argument('--embed_dim', dest='embed_dim', default=[], type=int, nargs='+',
                        help='Hidden layer dimensions (if exist) of a neural network')
    parser.add_argument('--l2_lamb', dest='l2_lamb', default=0, type=float,
                        help='The hyperparameter controlling L2 regularization of parameters')


    # Parameters related to EMSPO
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Epsilon to be added to inequalities when constructing a MILP')
    parser.add_argument('--timeout', type=int, default=200,
                        help='Timout for gurobi model optimization (in sec)')
    parser.add_argument('--time_interval', type=int, default=3600,
                        help='Time interval to cache incumbent parameter solution in SPO MILP')
    parser.add_argument('--leaf_minmax_no_prune', default=False, action='store_true',
                        help='prune_equality turned off during addition of independent decisions.')
    parser.add_argument('--var_reorder', type=int, default=None,
                        help='Can reorder variables for elimination in increasing (+1) or decreasing (-1) order of appearence.')
    parser.add_argument('--scheme', default='bruteforce', type=str,
                        help='bruteforce: use indicator constraints; benders: use combinatorial Benders decomposition')
    parser.add_argument('--theta_method', type=int, default=0,
                        help='How to handle theta=0 pathological solutions (values: 0, 1)')
    parser.add_argument('--theta_constr', type=int, default=None,
                        help='Equality constraint for sum of all theta parameters')
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
