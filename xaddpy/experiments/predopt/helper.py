from xaddpy.models.base import Base
from xaddpy.xadd.xadd import XADD
from xaddpy.models.emspo import EMSPO

from xaddpy.xadd.symbolic_solver import solve_n_record_min_or_argmin
from xaddpy.xadd.solve import solve_predopt
from xaddpy.utils.gurobi_util import setup_gurobi_model

from xaddpy.utils.logger import logger

from torch import optim
import torch

import numpy as np
from os import path
import os
import argparse


class Domain:
    SHORTEST_PATH = 'shortest_path'
    SHORTEST_PATH2 = 'shortest_path2'
    ENERGY_SCHEDULING = 'energy_scheduling'
    CLASSIFICATION = 'classification'
    CLASSIFICATION2 = 'classification2'


def get_param_dim(args, target_dim, feature_dim):
    # if args.theta_method == 0 and args.algo:
    #     param_dim = (target_dim, feature_dim[0])        # (n_x, d)
    # else:
    param_dim = (target_dim, feature_dim[0] + 1)    # (n_x, d + 1) where `d` is the number of features
    return param_dim


def get_feature_dim(input_dim, embed_dim=[]):
    if len(embed_dim) != 0:
        feature_dim = (embed_dim[-1], )     # The output of the last hidden layer acts as features
    else:
        feature_dim = (input_dim, )
    return feature_dim


def test_algorithm(alg: Base, dataset: dict, config: str, domain: str, **kwargs):
    actual_cost, opt_cost, regret = alg.test_model(dataset[config][0], dataset[config][1], dom=domain)
    return actual_cost, opt_cost, regret


def run_emspo(
        target_dim, input_dim, dataset, embed_dim, feature_dim, param_dim, linear, scaler,
        oracle, l2_lamb, use_validation, model_name, fname_param, fname_json, args, **kwargs):
    prob_configs = kwargs.get('prob_configs', None)
    var_name_rule = kwargs.get('var_name_rule', None)
    fname_arg_xadd = fname_json.replace('.json', '_argmin.xadd')
    epsilon = args.epsilon
    time_interval = args.time_interval

    context: XADD = None

    # Exact MILP reduction of SPO
    emspo = EMSPO(
        domain=args.domain,
        target_dim=target_dim,
        input_dim=input_dim,
        dataset=dataset,
        embed_dim=embed_dim,
        feature_dim=feature_dim,
        param_dim=param_dim,
        theta_method=args.theta_method,
        linear=linear,
        scaler=scaler,
        oracle=oracle,
        l2_lamb=l2_lamb,
        learn_feature=not linear,
        optimizer=optim.Adam,
        lr=args.lr,
        use_validation=use_validation,
        batch_size=10,
        theta_constr=args.theta_constr,
        fname_param=fname_param,
        prob_configs=prob_configs,
    )

    # Compute the optimal parameters or load from the saved array
    param_exists = path.exists(fname_param)
    if param_exists:
        if args.force is None:
            logger.warning("Optimized parameter already exists... You need to pass `--force 1' to recompute")
            exit(1)

    if not param_exists or (param_exists and args.force == 1):
        logger.info(f'Parameter file "{fname_param.split("/")[-1]}" does not exist. Run the EMSPO model.')
        eq_constr_dict = None

        # Learn features first if using nonlinear neural network model
        if len(embed_dim) != 0:
            emspo.fit(args.epochs)
            train_feature = emspo.get_embedding(
                torch.tensor(dataset['train'][0], dtype=torch.float)).detach().numpy()
            test_feature = emspo.get_embedding(
                torch.tensor(dataset['test'][0], dtype=torch.float)).detach().numpy()
            dataset['train'][0] = train_feature
            dataset['test'][0] = test_feature

        # Check if the closed-form argmin solution has been computed
        try:
            logger.info(f'Import the saved argmin XADD from {fname_arg_xadd.split("/")[-1]}')
            context = XADD(vars(args))
            variables, eq_constr_dict = \
                context.import_arg_xadd(
                    fname_arg_xadd,
                    feature_dim=feature_dim,
                    param_dim=param_dim,
                    model_name=model_name,
                )
            logger.info(f'File {fname_arg_xadd.split("/")[-1]} successfully imported.')

        except AssertionError as e:
            logger.error(e)
            exit(1)

        except FileNotFoundError as e:
            logger.error(e)
            logger.info('Argmin solution has not been computed... Start symbolic variable elimination')
            context, eq_constr_dict, time_symbolic = \
                solve_n_record_min_or_argmin(
                    get_solution=True,
                    model_name=model_name,
                    solver_type=1,  # Compute casemin between partitions immediately (not deferring till the end)
                    save_xadd=True,
                    verbose=True,
                    fname_xadd=fname_arg_xadd,
                    args=vars(args)
                )
            logger.info('Argmin solution computed')

        except Exception as e:
            logger.error(e)
            exit(1)

        # Add argmin solution nodes to special nodes and flush cache to free up memory
        context.add_annotations_to_special_nodes()
        context.flush_caches()

        # Set up a Gurobi model and update g_model parameters
        g_model = setup_gurobi_model(
            model_name=model_name,
            epsilon=epsilon,
            var_to_bound=context._var_to_bound,
            scheme=args.scheme,
        )

        # psutil.virtual_memory().total / (1024.**3)
        nodefilestart = getattr(args, 'nodefilestart', None)
        if nodefilestart != None:
            g_model._model.Params.NodefileStart = nodefilestart

        emspo.update_gurobi_model(g_model)

        # Compile a MILP and solve it
        param = solve_predopt(
            context,
            g_model,
            emspo,
            dataset,
            eq_constr_dict,
            domain=args.domain,
            var_name_rule=var_name_rule,
            epsilon=epsilon,
            verbose=args.verbose,
            timeout=args.timeout,
            time_interval=time_interval,
            prob_config=prob_configs,
            args=args,
        )
        np.save(fname_param, param)
    return emspo


def get_config_dir(args: argparse.Namespace):
    domain = args.domain
    config_dir = path.join(path.curdir, f'xaddpy/experiments/predopt/{domain}/prob_instances/')
    if domain == Domain.ENERGY_SCHEDULING:
        config_dir = path.join(config_dir, f'pc_{args.prob_config}__n_{args.sample_per_day}')    # Presolve assumed
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_results_dir(args: argparse.Namespace):
    domain = args.domain
    folder = get_folder_name(args)
    results_dir = path.join(path.curdir, f'xaddpy/experiments/predopt/{domain}/results/{folder}/{args.algo}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_folder_name(args: argparse.Namespace):
    domain = args.domain
    folder_name = ''

    if domain == Domain.SHORTEST_PATH or domain == Domain.CLASSIFICATION:
        nl = args.nonlinearity
        size = args.size
        eps_bar = args.eps_bar
        folder_name += f'nl_{nl}__size_{size}__eps_{eps_bar}'
        folder_name += f'__ne_{args.num_edges}' if domain == Domain.SHORTEST_PATH else ''
        folder_name += f'__cd_{args.class_dim}' if domain == Domain.CLASSIFICATION else ''
    elif domain == Domain.ENERGY_SCHEDULING:
        # to_prune = 'np' if args.leaf_minmax_no_prune else 'p'
        size = args.size
        data_type = args.dataset
        folder_name += f'config_{args.prob_config}'
        folder_name = folder_name + f"__{'pre' if args.presolve else 'no_pre'}" + f'__size_{size}__type_{data_type}'
    else:
        raise ValueError('Unknown domain specified.')
    return folder_name


def get_model_name(args: argparse.Namespace):
    model_name = ''
    domain = args.domain

    if args.algo == 'emspo':
        # theta constr; epsilon; timelimit;
        model_name += f'emspo_th_{args.theta_constr}_' if args.theta_constr is not None else ''
        model_name += f'eps_{args.epsilon}__time_{args.timeout}'

    elif args.algo == 'spoplus':
        model_name += f'spo_epoch_{args.epochs}__lr_{args.lr}'
    elif args.algo == 'intopt':
        model_name += f'intopt_epoch_{args.epochs}__lr_{args.lr}_dp__{args.damping}__thr_{args.thr}__time_{args.timeout_iter}'
    elif args.algo == 'qptl':
        model_name += f'qptl_epoch_{args.epochs}__lr_{args.lr}_tau__{args.tau}__time_{args.timeout_iter}'
    elif args.algo == 'twostage':
        model_name += f'mse_epoch_{args.epochs}__lr_{args.lr}'
    else:
        raise ValueError
    return model_name
