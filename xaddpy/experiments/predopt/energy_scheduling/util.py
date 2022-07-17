import json
from os import path
import os

import torch

from xaddpy.experiments.predopt.energy_scheduling import gurobi_model
from xaddpy.utils.logger import logger

import sympy as sp
import numpy as np
import pandas as pd


def generate_dataset(args):
    train_test_ratio = args.train_test_ratio
    sample_per_day = args.sample_per_day
    dataset_dir = path.join(path.curdir, 'xaddpy/experiments/predopt/energy_scheduling/data')

    def shuffle_and_collect(X_train, y_train):
        np.random.seed(0)
        idxs = list(range(len(X_train)))
        np.random.shuffle(idxs)
        
        X_train = X_train[idxs]
        y_train = y_train[idxs]

        size_per_run = args.size
        X_ret, y_ret = [], []
        for s in args.seed:
            X_train_i = X_train[s * size_per_run: (s + 1) * size_per_run]
            y_train_i = y_train[s * size_per_run: (s + 1) * size_per_run]
            X_ret.append(X_train_i)
            y_ret.append(y_train_i)
        return X_ret, y_ret

    if args.dataset == 1:
        fname = 'prices2013.dat'
    elif args.dataset == 2:
        fname = 'prices2013_modified.dat'
    elif args.dataset == 3:
        fname = 'gen_data__eb0.25__nl1.npy'
        fpath = path.join(dataset_dir, fname)
        X_train, y_train, X_test, y_test = np.load(fpath, allow_pickle=True)
        X_ret, y_ret = shuffle_and_collect(X_train, y_train)
        return X_ret, y_ret, X_test, y_test
    else: 
        raise ValueError('Unknown dataset specified.')

    data_path = path.join(dataset_dir, fname)

    X_train, y_train, X_test, y_test = get_energy_data(data_path, train_test_ratio, sample_per_day)
    X_train, X_test = X_train[:, :, 1:], X_test[:, :, 1:]     # Remove the group ID representing each day

    if args.size == -1:     # When size is not specified, use the entire dataset
        args.size = X_train.shape[0]
    X_ret, y_ret = shuffle_and_collect(X_train, y_train)

    return X_ret, y_ret, X_test, y_test


def get_energy_pandas(fname):
    df = pd.read_csv(fname, delim_whitespace=True, quotechar='"')

    # Remove unnecessary columns
    df.drop(['#DateTime', 'Holiday', 'ActualWindProduction', 'SystemLoadEP2'], axis=1, inplace=True)

    # Remove columns with missing values
    df.drop(['ORKTemperature', 'ORKWindspeed'], axis=1, inplace=True)

    # Missing value treatment
    # Impute missing CO2 intensities linearly
    df.loc[df.loc[:, 'CO2Intensity'] == 0, 'CO2Intensity'] = np.nan
    df.loc[:, 'CO2Intensity'].interpolate(inplace=True)

    # Remove remaining 3 days with missing values
    grouplength = 48
    for i in range(0, len(df), grouplength):
        day_has_nan = pd.isnull(df.loc[i: i+ (grouplength - 1)]).any(axis=1).any()
        if day_has_nan:
            df.drop(range(i, i + grouplength), inplace=True)

    # Data is sorted by year, month, day, periodofday; these should be independent of learning
    df.drop(['Day', 'Year', 'PeriodOfDay'], axis=1, inplace=True)

    # Insert group identifier at the beginning
    grouplength = 48
    length = int(len(df) / 48)  # 789
    gids = [gid for gid in range(length) for i in range(grouplength)]
    df.insert(0, 'groupID', gids)

    return df


def get_energy_data(fname, train_test_ratio=0.7, sample_per_day=24, batch_dim=True):
    df = get_energy_pandas(fname)
    length = df['groupID'].nunique()
    grouplength = 48
    assert grouplength % sample_per_day == 0

    # Numpy arrays, X contains groupID as the first column
    # SMPEP2: the actual price of this time period. SMPEA: the price forecast for this period
    X1g = df.loc[:, df.columns != 'SMPEP2'].values
    y = df.loc[:, 'SMPEP2'].values

    # No negative values allowed
    for i in range(len(y)):
        y[i] = max(y[i], 0)

    # Ordered split per complete group
    train_len = int(train_test_ratio * length)

    # Skip some data to reduce the problem size
    skip = grouplength // sample_per_day
    X1g = X1g[::skip, :]
    y = y[::skip]

    # Split
    X_1gtrain = X1g[:sample_per_day * train_len]
    y_train = y[:sample_per_day * train_len]
    X_1gtest = X1g[sample_per_day * train_len:]
    y_test = y[sample_per_day * train_len:]

    if batch_dim:
        X_1gtrain = X_1gtrain.reshape((-1, sample_per_day, X_1gtrain.shape[1]))
        y_train = y_train.reshape((-1, sample_per_day))
        X_1gtest = X_1gtest.reshape((-1, sample_per_day, X_1gtest.shape[1]))
        y_test = y_test.reshape((-1, sample_per_day))
    return X_1gtrain, y_train, X_1gtest, y_test


def make_json(args, prob_config_path):
    dir_path = args.config_dir

    # Read problem parameters
    prob_configs = gurobi_model.read_config(prob_config_path, args)

    num_machines = prob_configs['num_machines']
    num_tasks = prob_configs['num_tasks']
    num_resources = prob_configs['num_resources']
    MAX_CAPACITY = prob_configs['MAX_CAPACITY']
    RESOURCE_USE = prob_configs['RESOURCE_USE']  # Resource use of task j for resource r
    DURATION = prob_configs['DURATION']  # Duration of task j
    E_START = prob_configs['E_START']  # Earliest start of task j
    L_END = prob_configs['L_END']  # Latest end of task j
    POWER_USE = prob_configs['POWER_USE']
    POWER_USE_FRAC = prob_configs['POWER_USE_FRAC']
    IDLE = prob_configs['IDLE']
    UP = prob_configs['UP']
    DOWN = prob_configs['DOWN']
    q = prob_configs['q']
    N = 1440 // q
    TASKS = range(num_tasks)
    MACHINES = range(num_machines)

    ICON = gurobi_model.GurobiICON(
        num_machines,
        num_tasks,
        num_resources,
        MAX_CAPACITY,
        RESOURCE_USE,
        DURATION,
        E_START,
        L_END,
        POWER_USE,
        IDLE,
        UP,
        DOWN,
        q,
        reset=True,
        presolve=args.presolve,
        relax=False,
        verbose=False,
        warmstart=False,
        method=-1
    )
    ICON.make_model()
    model = ICON.model
    os.makedirs(path.join(dir_path, 'data'), exist_ok=True)
    model.write(path.join(dir_path, 'data', 'energy.mps'))
    del model

    logger.info('Gurobi model built and model mps file has been written')

    constr_type = {}
    ns = {}
    section = None
    num_var = num_machines * num_tasks * N
    prev_var_str = None
    constr_to_var = {}
    x = {}

    if args.presolve:
        num_var = len(ICON.model.getVars())
        model_constrs = ICON.model.getConstrs()
        reorder_constr = {int(c.ConstrName[1:]): idx for idx, c in enumerate(model_constrs)}

    logger.info('Read .mps file to generate a json file')
    with open(path.join(dir_path, 'data', 'energy.mps'), 'r') as lp_file:
        for line in lp_file.readlines():
            if line.strip() == 'ROWS':
                section = 'rows'
                continue
            elif line.strip() == 'COLUMNS':
                section = 'columns'

                num_constr = len(constr_type)
                A = np.zeros((num_constr, num_var), dtype=int)
                b = np.zeros((num_constr, 1), dtype=int)
                var_vector = sp.Matrix([])
                continue
            elif line.strip() == 'RHS':
                section = 'rhs'
                continue
            elif line.strip() == 'BOUNDS':
                break

            if section == 'rows':
                ty, constr = line.strip().split()
                if ty == 'N' and constr == 'OBJ':
                    continue
                constr_type[constr] = ty
            elif section == 'columns':
                var_str, row, coef = line.strip().split()
                if var_str == 'MARKER':
                    continue
                j, mc, t = tuple(map(int, var_str.strip('x').strip('[').strip(']').split(',')))
                var_str = 'x{}_{}_{}'.format(j, mc, t)

                if var_str != prev_var_str:
                    curr_num_var = len(var_vector)
                    var = sp.symbols(var_str)
                    ns[var_str] = var
                    var_vector = sp.Matrix([var_vector, var])
                    prev_var_str = var_str
                    x[(j, mc, t)] = var
                if row not in constr_to_var:
                    constr_to_var[row] = ns[var_str]
                row = int(row[1:])
                row = reorder_constr[row] if args.presolve else row
                col = curr_num_var
                coef = int(coef)

                A[row, col] = coef
            elif section == 'rhs':
                _, row, rhs = line.strip().split()
                row = int(row[1:])
                row = reorder_constr[row] if args.presolve else row
                rhs = int(rhs)
                b[row, 0] = rhs

    logger.info('Setting up (in)equality constraints in str type')
    eq_constr_lst = []
    ineq_constr_lst = []
    for constr, ty in constr_type.items():
        row = int(constr[1:])
        row = reorder_constr[row] if args.presolve else row
        lhs_i = (A[row, :][None, :] * var_vector)[0] - b[row, 0]
        # lhs_i = lhs[row]
        if ty == 'E':
            var = constr_to_var[constr]
            rhs = sp.solveset(lhs_i, var).args[0]
            eq_constr_lst.append("{} = {}".format(var, rhs))
        elif ty == 'L':
            ineq_constr_lst.append('{} <= 0'.format(lhs_i))
        else:
            ineq_constr_lst.append('{} >= 0'.format(lhs_i))

    min_vals, max_vals = [0] * num_var, [1] * num_var
    variables = list(var_vector)
    feature_dim = (8,)
    xadd = ''
    is_positive = 0
    is_minimize = 1

    # Objective function: note that price
    logger.info('Computing the objective expression')
    price = sp.Matrix(sp.symbols(f'c1:{args.sample_per_day + 1}'))
    obj = sum([x[(j, mc, t)] * sum(price[t: t + DURATION[j]]) * POWER_USE_FRAC[j] * q / 60  # Can reduce variable count by using L_END start limit for t.
               for j in TASKS for t in range(N - DURATION[j] + 1) for mc in MACHINES if (j, mc, t) in x])                

    fname = 'energy_scheduling.json'

    res_json = {}
    res_json['prob-type'] = 'predict-then-optimize'
    res_json['cvariables0'] = [str(v) for v in variables]
    res_json['cvariables1'] = []
    res_json['min-values'] = min_vals
    res_json['max-values'] = max_vals
    res_json['bvariables'] = []
    res_json['feature-dim'] = feature_dim
    res_json['ineq-constr'] = ineq_constr_lst
    res_json['eq-constr'] = eq_constr_lst
    res_json['xadd'] = ""
    res_json['is-positive'] = 0
    res_json['is-minimize'] = 1
    res_json['min-var-set-id'] = 0
    res_json['objective'] = str(obj)

    logger.info('Writing to .json file')
    # dir_json = path.join(dir_path, 'prob_instance')
    # os.makedirs(dir_json, exist_ok=True)
    # fname = path.join(dir_json, fname)
    fname = path.join(dir_path, fname)

    with open(fname, 'w') as f_json:
        json.dump(res_json, f_json, indent=4)
    
    return fname


""" Matrix operation related utility functions (used by QPTL and IntOpt) """
""" The code is copied from https://github.com/JayMan91/NeurIPSIntopt """

def set_up_milp_matrix(prob_configs: dict, algo):

    num_machines = prob_configs['num_machines']     # nbMachines: number of machine
    num_tasks = prob_configs['num_tasks']           # nbTasks: number of task
    num_resources = prob_configs['num_resources']   # nb resources: number of resources
    MAX_CAPACITY = prob_configs['MAX_CAPACITY']     # MC[m][r] resource capacity of machine m for resource r
    RESOURCE_USE = prob_configs['RESOURCE_USE']     # U[f][r] resource use of task f for resource r
    DURATION = prob_configs['DURATION']             # D[f] duration of tasks f
    E_START = prob_configs['E_START']               # E[f] earliest start of task f
    L_END = prob_configs['L_END']                   # L[f] latest end of task f
    POWER_USE = prob_configs['POWER_USE']           # P[f] power use of tasks f
    POWER_USE_FRAC = prob_configs['POWER_USE_FRAC']
    IDLE = prob_configs['IDLE']                     # idle[m] idle cost of server m
    UP = prob_configs['UP']                         # up[m] startup cost of server m
    DOWN = prob_configs['DOWN']                     # down[m] shut-down cost of server m
    q = prob_configs['q']                           # time resolution
    presolve = prob_configs['presolve']

    Machines = range(num_machines)
    Tasks = range(num_tasks)
    Resources = range(num_resources)
    N = 1440 // q

    ### G and h
    G = torch.zeros((num_machines * N, num_tasks * num_machines * N))
    h = torch.zeros(num_machines * N)
    F = torch.zeros((N, num_tasks * num_machines * N))
    for m in Machines:
        for t in range(N):
            h[m * N + t] = MAX_CAPACITY[m][0]
            for f in Tasks:
                c_index = (f * num_machines + m) * N
                G[t + m * N, (c_index + max(0, t - DURATION[f] + 1)):(c_index + (t + 1))] = 1
                F[t, (c_index + max(0, t - DURATION[f] + 1)):(c_index + (t + 1))] = POWER_USE[f]

    if algo == 'qptl':
        G2 = torch.eye((num_tasks * num_machines * N))
        G3 = -1 * torch.eye((num_tasks * num_machines * N))
        h2 = torch.ones(num_tasks * num_machines * N)
        h3 = torch.zeros(num_tasks * num_machines * N)
        G = torch.cat((G, G2, G3))
        h = torch.cat((h, h2, h3))

    ### A and b
    A1 = torch.zeros((num_tasks, num_tasks * num_machines * N))
    A2 = torch.zeros((num_tasks, num_tasks * num_machines * N))
    A3 = torch.zeros((num_tasks, num_tasks * num_machines * N))

    for f in Tasks:
        A1[f, (f * N * num_machines):((f + 1) * N * num_machines)] = 1
        A2[f, (f * N * num_machines):(f * N * num_machines + E_START[f])] = 1
        A3[f, (f * N * num_machines + L_END[f] - DURATION[f] + 1):((f + 1) * N * num_machines)] = 1
    b = torch.cat((torch.ones(num_tasks), torch.zeros(2 * num_tasks)))
    A = torch.cat((A1, A2, A3))
    return A, b, G, h, torch.transpose(F, 0, 1)
