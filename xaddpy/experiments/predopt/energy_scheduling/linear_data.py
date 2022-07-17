import numpy as np
from os import path
from sklearn.linear_model import LinearRegression
from xaddpy.experiments.predopt.energy_scheduling.util import generate_dataset


def get_file_name(args):
    fname = f'gen_data__eb{args.eps_bar}__nl{args.nonlinearity}'
    return fname


def generate_linear_data(args):
    deg = args.nonlinearity
    eps_bar = args.eps_bar
    np.random.seed(args.seed)
    dataset_dir = path.join(path.curdir, 'xaddpy/experiments/predopt/energy_scheduling/data')

    if isinstance(args.seed, int):
        args.seed = [args.seed]

    # retrieve original dataset
    X_train, c_train, X_test, c_test = generate_dataset(args)
    X_train, c_train = X_train[0], c_train[0]
    X_train = X_train.reshape(-1, X_train.shape[-1])
    c_train = c_train.reshape(-1)
    X_test = X_test.reshape(-1, X_test.shape[-1])
    c_test = c_test.reshape(-1)

    def generate_data(X, c):
        size = X.shape[0]
        reg = LinearRegression().fit(X, c)
        eps = np.random.uniform(1 - eps_bar, 1 + eps_bar, size=size)
        dotprod = np.dot(X, reg.coef_) + reg.intercept_
        c_gen = np.power(dotprod, deg) * eps        
        return c_gen
    
    c_gen_train = generate_data(X_train, c_train)
    c_gen_test = generate_data(X_test, c_test)

    X_train = X_train.reshape(-1, args.sample_per_day, X_train.shape[-1])
    c_gen_train = c_gen_train.reshape(-1, args.sample_per_day)
    X_test = X_test.reshape(-1, args.sample_per_day, X_test.shape[-1])
    c_gen_test = c_gen_test.reshape(-1, args.sample_per_day)

    fpath = path.join(dataset_dir, get_file_name(args))
    np.save(fpath, [X_train, c_gen_train, X_test, c_gen_test])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--eps-bar', type=float, default=0.25, help='Multiplicative noise half-width')
    parser.add_argument('--nonlinearity', type=int, default=1, help='Data nonlinearity (1: linear)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--train_test_ratio', type=float, default=0.7, help='The portion of the train set')
    parser.add_argument('--sample_per_day', default=24, type=int, help='Number of samples taken per day')
    parser.add_argument('--dataset', type=int, default=1, help='Dataset. Can be 1 (entire data) or 2 (single day)')
    parser.add_argument('--size', type=int, default=-1, help='The size of the dataset to be generated')

    args = parser.parse_args()
    generate_linear_data(args)