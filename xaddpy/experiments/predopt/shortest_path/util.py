import argparse
import json
import os
import os.path as path
import numpy as np


def generate_dataset(args):
    """Generates the noisy shortest path dataset according to the provided configurations."""
    X_train, c_train, X_test, c_test = [], [], [], []
    for s in args.seed:
        size = args.size                    # 200 used in SPOTree
        test_size = 1000                    # as in SPOTree
        deg = args.nonlinearity
        eps_bar = args.eps_bar
        feature_dim = args.feature_dim
        num_edges = args.num_edges          # 24 edges in the 4x4 grid

        np.random.seed(s)
        X_train_i = np.random.uniform(0, 1, size=(size, feature_dim))
        param_matrix = np.random.binomial(1, 0.5, size=(num_edges, feature_dim))

        eps = np.random.uniform(1-eps_bar, 1+eps_bar, size=(size, num_edges))
        linear_cost = np.einsum('ij, kj -> ki', param_matrix, X_train_i)  # size=(size, num_edges)
        c_train_i = np.multiply(np.power(linear_cost / np.sqrt(feature_dim) + 1, deg), eps)

        X_test_i = np.random.uniform(0, 1, size=(test_size, feature_dim))
        linear_cost_test = np.einsum('ij, kj -> ki', param_matrix, X_test_i)   # size=(test_size, num_edges)
        eps_test = np.random.uniform(1 - eps_bar, 1 + eps_bar, size=(test_size, num_edges))
        c_test_i = np.multiply(np.power(linear_cost_test / np.sqrt(feature_dim) + 1, deg), eps_test)

        X_train.append(X_train_i)
        c_train.append(c_train_i)
        X_test.append(X_test_i)
        c_test.append(c_test_i)
    return X_train, c_train, X_test, c_test


def make_json(num_edges: int, args: argparse.Namespace):
    """
    Create a .json file of a shortest path problem to be read by `xadd_parse_utils.py` and `models/base.py` files.
    We place a decision variable per each edge, and there are either 12 (3 x 3) or 24 (4 x 4) edges in total.
    """
    assert num_edges in [12, 24, 40]

    # Create dim * dim grid, where dim = number of vertices
    if num_edges == 12:
        dim = 3
    elif num_edges == 24:
        dim = 4
    elif num_edges == 40:
        dim = 5
    else:
        raise ValueError
    edge_list = [(i, i+1) for i in range(1, dim**2 + 1) if i % dim != 0]
    edge_list += [(i, i+dim) for i in range(1, dim**2 + 1) if i <= dim ** 2 - dim]
    edge_dict = {}

    cvariables = []
    eq_constr_dict = {}
    ineq_constr_dict = {}
    start_with = [0] * dim ** 2

    count = 0
    eq_constr = ''

    for i in range(1, dim**2 + 1):
        for j in range(1, dim**2 + 1):
            if (i, j) not in edge_list:
                continue
            v_edge = f"x{i}_{j}"            # a decision variable associated with the edge (i, j)
            cvariables.append(v_edge)

            edge_dict[(i, j)] = count       # count for validation
            count += 1

            # Encode equality constraints as the form of `var = RHS`
            if i not in eq_constr_dict:
                eq_constr_dict[i] = f"{v_edge} = -("
                start_with[i-1] = -1
            elif eq_constr_dict[i][-1] != '+' and eq_constr_dict[i][-1] != '-' and eq_constr_dict[i][-1] != '[':
                eq_constr_dict[i] += f'+{v_edge}'
            elif eq_constr_dict[i][-1] == '+' or eq_constr_dict[i][-1] == '(' or eq_constr_dict[i][-1] == '[':
                eq_constr_dict[i] += v_edge
            else:
                raise ValueError

            if j not in eq_constr_dict:
                eq_constr_dict[j] = f"{v_edge} = "
                start_with[j-1] = 1
            else:
                eq_constr_dict[j] += f"-{v_edge}"

    # Complete the equality constraints format
    for i in eq_constr_dict:
        if i == 1:
            eq_constr_dict[i] += ")+ 1"
        elif i == dim ** 2:
            eq_constr_dict[i] += "+ 1"
        else:
            if start_with[i] == -1:
                eq_constr_dict[i] += ")"
    eq_constr = list(eq_constr_dict.values())

    ineq_constr = ""
    objective = ""

    # dir_path = path.join(path.curdir, 'xaddpy/experiments/predopt/shortest_path/prob_instances')
    # os.makedirs(dir_path, exist_ok=True)
    dir_path = args.config_dir
    fname = path.join(dir_path, f'shortest_path_{num_edges}edges.json')

    res_json = dict()
    res_json['prob-type'] = 'predict-then-optimize'
    res_json['cvariables0'] = cvariables
    res_json['cvariables1'] = []
    res_json['min-values'] = [0] * len(cvariables)
    res_json['max-values'] = [1] * len(cvariables)
    res_json['bvariables'] = []
    res_json['ineq-constr'] = ineq_constr  # ""
    res_json['eq-constr'] = eq_constr
    res_json['xadd'] = ""
    res_json['is-minimize'] = 1
    res_json['min-var-set-id'] = 0
    res_json['objective'] = objective

    with open(fname, 'w') as fjson:
        json.dump(res_json, fjson, indent=4)

    return fname
