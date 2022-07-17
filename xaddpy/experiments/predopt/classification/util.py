import argparse
import json
import os
import os.path as path
import numpy as np


def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))


def generate_dataset(args):
    """
    Generates the noisy multiclass classification dataset according to the provided configurations.

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    X_train, c_train, X_test, c_test = [], [], [], []
    for s in args.seed:
        size = args.size
        test_size = 1000
        deg = args.nonlinearity
        eps_bar = args.eps_bar
        class_dim = args.class_dim
        feature_dim = args.feature_dim
        np.random.seed(s)
        param = np.random.binomial(1, 0.5, size=feature_dim)

        def generate_feature_costs(size: int):
            X = np.random.randn(size, feature_dim)
            eps = np.random.uniform(1-eps_bar, 1+eps_bar, size=size)

            dotprod = np.dot(X, param)
            score = sigmoid(np.power(dotprod, deg) * np.sign(dotprod) * eps)
            label = np.ceil(10 * score)
            cost = np.tile(np.arange(1, class_dim + 1), (size,1))
            cost = np.absolute(cost - label.reshape((size, 1)))
            return X, cost

        X_train_s, c_train_s = generate_feature_costs(size)
        X_test_s, c_test_s = generate_feature_costs(test_size)
        X_train.append(X_train_s)
        c_train.append(c_train_s)
        X_test.append(X_test_s)
        c_test.append(c_test_s)
    return X_train, c_train, X_test, c_test


def make_json(class_dim: int, feature_dim: int, args: argparse.Namespace):
    """
    Create a .json file of a multiclass classification problem to be read by `xadd_parse_utils.py` and `models/base.py` files.

    Args:
        class_dim (int): [description]
        feature_dim (int): [description]

    Returns:
        [type]: [description]
    """
    assert feature_dim == 5

    cvariables = [f'x{i}' for i in range(1, class_dim + 1)]
    eq_constr = [f'{cvariables[0]} = -' + '-'.join([v for v in cvariables[1:]]) + '+ 1']
    ineq_constr = ''
    objective = ''

    # dir_path = path.join(path.curdir, 'xaddpy/experiments/predopt/classification/prob_instances')
    # os.makedirs(dir_path, exist_ok=True)
    dir_path = args.config_dir
    fname = path.join(dir_path, f'classification_{class_dim}classes.json')

    res_json = dict()
    res_json['prob-type'] = 'predict-then-optimize'
    res_json['cvariables0'] = cvariables
    res_json['cvariables1'] = []
    res_json['min-values'] = [0] * len(cvariables)
    res_json['max-values'] = [1] * len(cvariables)
    res_json['bvariables'] = []
    res_json['ineq-constr'] = ineq_constr
    res_json['eq-constr'] = eq_constr
    res_json['xadd'] = ""
    res_json['is-minimize'] = 1
    res_json['min-var-set-id'] = 0
    res_json['objective'] = objective

    with open(fname, 'w') as fjson:
        json.dump(res_json, fjson, indent=4)

    return fname
