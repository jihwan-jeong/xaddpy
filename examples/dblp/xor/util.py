import argparse
import time
from typing import Tuple

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sympy as sp
from gurobipy import GRB, quicksum
from scipy.sparse import random

from examples.dblp.util import setup_gurobi_model
from examples.dblp.xor.bucket_elim import (
    run_symbolic_bucket_elimination, run_symbolic_bucket_elimination_multi)
from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.milp.solve import solve_milp
from xaddpy.utils.logger import logger
from xaddpy.utils.lp_util import Model


def solve_dblp_with_xor(
        context: XADD,
        n: int,
        ny: int,
        seed: int,
        model_name: str,
        args: argparse.Namespace,
):
    # Perform the MILP reduction of a DBLP via Symbolic Bucket Elimination
    n_constrs_y = args.n_constrs_y
    n_ri_per_z = args.n_ri_per_z

    sbe_res_dict, dblp, time_symb = create_xor_dblp_and_eliminate_xs(
        context,
        n,
        ny,
        seed,
        n_constrs_y,
        use_q=args.use_q,
        n_ri_per_z=n_ri_per_z
    )

    # Retrieve some XADD information
    # num_int_nodes, num_term_nodes = get_num_nodes(context, milp_xadd)

    # Incorporate y constraints and the cost d.T y into the objective   (skip cause already added)
    m = setup_gurobi_model(
        model_name,
        epsilon=args.epsilon,
        var_to_bound=context._var_to_bound,
    )
    logger.info("Insert y constraints and the objective term into each of the resulting XADDs")
    add_y_constrs_and_costs(context, dblp, sbe_res_dict)
    context.set_objective(sbe_res_dict)

    # Build the MILP model and solve (time the process)
    logger.info("Build the MILP model and optimze it")
    m, info_dict = build_milp_from_xadd_and_solve(
        context,
        m,
        args,
        model_name,
        dblp
    )
    info_dict['time_symb'] = time_symb
    logger.info(f'Time taken for Gurobi to solve the MILP: '
                f'{info_dict["time_milp"]:.5f}\tObjective: {info_dict["obj_value"]:.5f}')
    
    return m, info_dict


def create_xor_dblp_and_eliminate_xs(
        context: XADD,
        n: int,
        ny: int,
        seed: int,
        n_constrs_y: int,
        n_ri_per_z: int,
        use_q: bool = False,
) -> Tuple[int, dict, float]:
    if n_ri_per_z == -1 or n_ri_per_z > n:        # Single z variable
        n_ri_per_z = n
    logger.info("Create sympy decision variables and link them to XADD object")
    xs, ys, zs = setup_context_and_decision_vars(context, 
                                                 n=n,
                                                 ny=ny,
                                                 n_ri_per_z=n_ri_per_z)
    ys = sp.Matrix(ys)

    # Create coeffcients of a DBLP
    logger.info("Create the coefficients of DBLP (skipping XOR constraints)")
    dblp = create_dblp_skip_xor_constrs(n1=n, n2=ny,
                                        seed=seed,
                                        n_constrs_y=n_constrs_y,
                                        nz=len(zs))
    B, b, Q, c, d, cz = tuple(map(lambda x: dblp[x], ['B', 'b', 'Q', 'c', 'd', 'cz']))
    dblp['y'] = ys
    dblp['zs'] = zs
    dblp['dec_vars'] = list(ys) + zs

    # Create a Gurobi model used for computing bounds over q
    def configure_gurobi_model(m: gp.Model, ny: int, B: sp.Matrix, b: sp.Matrix):
        ys = m.addVars(range(ny), lb=-10, ub=10, vtype=GRB.CONTINUOUS, name='y')

        for i in range(B.shape[0]):
            m.addConstr(quicksum(B[i, j] * ys[j] for j in range(ny)) <= b[i], name=f'{i}th_constr')
        m.update()

    def compute_bounds_over_q(c, Qi, incl_constr=False):
        if not incl_constr:
            m.remove(m.getConstrs())
        ys = m.getVars()
        obj = c + quicksum(Qi[j] * ys[j] for j in range(ny))
        m.setObjective(obj, sense=GRB.MINIMIZE)
        m.optimize()
        lb = sp.nsimplify(obj.getValue())

        m.setObjective(obj, sense=GRB.MAXIMIZE)
        m.optimize()
        ub = sp.nsimplify(obj.getValue())
        return lb, ub

    m = gp.Model('ComputeBounds')
    configure_gurobi_model(m, ny, B, b)

    # Encode the XOR constraints as XADD and prepare buckets of x variables
    logger.info("Prepare the buckets of variables and their associated symbolic functions")
    buckets, r_dict, q_dict = prepare_buckets(context, xs, zs, n_ri_per_z, dblp)

    # Need to do substitution `q_i <- c_i + \sum_j Q_{ij} y_j'
    # Also, update the bounds on q variables
    logger.info("Create the dictionary mapping 'q_i' var -> 'c_i + \\sum_j Q_ij y_j'")
    subst_dict = {}
    bound_dict = {}
    for i, q_i in q_dict.items():
        subst_in = sp.expand(c[i - 1] + (Q[i - 1, :] * ys)[0])
        subst_out = q_i
        subst_dict[subst_out] = subst_in

        # compute the bounds over q
        if use_q:
            lb, ub = compute_bounds_over_q(c[i - 1], Q[i - 1, :], incl_constr=True)
            bound_dict[q_i] = (lb, ub)
            context.update_bounds(bound_dict)

    if not use_q:
        logger.info('Substitute y expressions into q_i')
        for i, (_, f_lst) in buckets.items():
            fi = f_lst[0]
            fi = context.substitute(fi, subst_dict=subst_dict)
            f_lst[0] = fi

    # Start SBE: the outcome is saved as an XADD node which has all x variables eliminated (q vars left)
    logger.info('Run symbolic bucket elimination')
    stime = time.time()
    sbe_res_dict = run_symbolic_bucket_elimination_multi(context,
                                                         buckets,
                                                         solver_type=0)
    etime = time.time()

    time_taken = etime - stime
    logger.info(f"n: {n}, nx :{3 * n}, ny: {ny}\tTime taken for SVE: {time_taken}")

    # Replace q_i variables with original y expressions
    logger.info("Substitute y expressions into q_i")
    keys = sorted(sbe_res_dict.keys())
    for i in keys:
        res_i = sbe_res_dict[i]
        res_i = context.substitute(res_i, subst_dict)

        # Optionally.... Test if providing the y constraints helps (seems to be helping a lot!)
        # res_i = add_y_constrs_and_costs(context, dblp, res_i)
        sbe_res_dict[i] = res_i

    return sbe_res_dict, dblp, time_taken


def setup_context_and_decision_vars(
        context: XADD, n: int, ny: int, n_ri_per_z: int
) -> Tuple[tuple, tuple, list]:
    nx = 3 * n
    nz = n - n_ri_per_z + 1

    # Create x and y symbols
    xs = sp.symbols(f'x1:{nx + 1}')
    ys = sp.symbols(f'y1:{ny + 1}')
    zs = sp.symbols(f'z1:{nz + 1}')

    ns = {str(v): v for v in xs + ys + zs}
    dec_vars = xs + ys + zs

    bound_dict = {}
    min_vals = [-10] * (nx + ny) + [-50] * nz
    max_vals = [10] * (nx + ny) + [50] * nz
    for i, (lb, ub) in enumerate(zip(min_vals, max_vals)):
        lb, ub = sp.S(lb), sp.S(ub)
        bound_dict[ns[str(dec_vars[i])]] = (lb, ub)

    context.update_decision_vars(min_var_set=set(xs), free_var_set=set(ys))
    context.update_bounds(bound_dict)
    context.update_name_space(ns)
    xs = {i: xs[i - 1] for i in range(1, len(xs) + 1)}
    return xs, ys, list(zs)


def create_dblp_skip_xor_constrs(n1, n2, seed, n_constrs_y=15, nz=1):
    """
    Creates a DBLP defined over decision variables x and y whose cardinalities are nx and ny, respectively.
    XOR constraints are pre-defined and fixed according to the formulation given in our paper.
    The polyhedron defining the feasible region of y is generated randomly, and we follow the same procedure used for
    the other experiments.

    Args:
        n1 (int)
        n2 (int)
        seed (int)
        n_constrs_y (int)

    Returns:

    """
    # Set up some parameters
    lb, ub = -10, 10
    low, high = -10, 10     # integer coefficients are sampled from uniform(low, high)
    rng = np.random.RandomState(seed)
    rv_obj = stats.randint(low, high)
    rv_obj.random_state = rng
    rv = rv_obj.rvs
    density = 0.3

    # Generate Q matrix
    while True:
        Q = random(n1, n2, density=density, data_rvs=rv, dtype=np.int, random_state=rng).A
        if np.count_nonzero(Q) != 0:
            break

    # Generate c, d coefficient vectors
    c, d, cz = tuple(map(lambda n: rv(n), [n1, n2, nz]))

    # For checking feasibility over y
    ## Slack test ensures that not all constraints hold at equality at optimum
    def test_slack(B, b):
        m.remove(m.getConstrs())
        for i in range(B.shape[0]):
            constr = quicksum(B[i, j] * ys[j] for j in range(n2)) - b[i] + S <= 0
            m.addConstr(constr)
        m.setObjective(S, sense=GRB.MAXIMIZE)
        m.optimize()
        status = m.status
        if status == GRB.INF_OR_UNBD:
            m.setParam('DualReductions', 0)
            m.optimize()
            status = m.status
            m.setParam('DualReductions', 1)
        obj_val = m.objVal if status == GRB.OPTIMAL else 1e100
        if status != GRB.UNBOUNDED and obj_val < 1e-4:
            return False
        else:
            return True

    m = gp.Model('FeasibilityChecker')
    S = m.addVar(name='S')              # For testing slack value
    m.Params.OutputFlag = 0
    ys = m.addVars(range(n2), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name='y')

    while True:
        # Generate constraints over y: By <= b
        b = rv(n_constrs_y)
        while True:
            B = random(n_constrs_y * int(1/density), n2, density=density, data_rvs=rv, dtype=np.int, random_state=rng).A
            nonzero_cnts = (B != 0).sum(axis=1)
            nonzero_rows = nonzero_cnts.nonzero()[0]
            if len(nonzero_rows) >= n_constrs_y:
                B = B[nonzero_rows[:n_constrs_y], :]
                break

        try:
            m.remove(m.getConstrs())
        except AttributeError:
            pass

        m.addConstrs(quicksum(B[i, j] * ys[j] for j in range(n2)) <= b[i] for i in range(n_constrs_y))
        m.setObjective(1)
        m.update()
        m.optimize()
        status = m.status
        if status == GRB.OPTIMAL:
            pass_slack_test = test_slack(B, b)
            if pass_slack_test:
                break

    Q, B, c, d, cz, b = tuple(map(
        lambda arr: sp.Matrix(arr), [Q, B, c, d, cz, b]
    ))

    return dict(
        B=B,
        b=b,
        Q=Q,
        c=c,
        d=d,
        cz=cz,
    )


def prepare_buckets(
        context: XADD, xs: dict, zs: list, n_ri_per_z: int, dblp: dict,
) -> Tuple[dict, dict, dict]:
    buckets = {}
    nx = len(xs)
    n = nx // 3
    zs = sp.Matrix(zs)
    cz = dblp['cz']

    r_dict = {}
    q_dict = {}
    for i in range(1, n + 1):
        buckets[i] = ([xs[j] for j in range(3 * i - 2, 3 * i + 1)], [])
        if i not in r_dict:
            # Get the node id corresponding to r_i
            r_dict[i] = return_ri_func(context, xs, zs, i, n_ri_per_z)

            # Create a leaf node whose expression is q_i
            q_i = sp.symbols(f'q{i}')
            q_i_id = context.get_leaf_node(q_i)

            # The function goes into the bucket is `q_i * r_i' which is bilinear
            q_i_r_i = context.apply(r_dict[i], q_i_id, 'prod')

            # Add linear objective over z
            z_obj_i = context.get_leaf_node((sp.Matrix(zs).T * cz)[0] / sp.S(n))
            obj_i = context.apply(q_i_r_i, z_obj_i, 'add')

            buckets[i][1].append(obj_i)

            # Later, we should substitute `q_i <- c_i + \sum_k Q_{ik} y_k'
            # For now, we can just store q_i in a dictionary
            q_dict[i] = q_i

    return buckets, r_dict, q_dict


def build_milp_from_xadd_and_solve(
        context: XADD, m: Model, args: argparse.Namespace, model_name: str, dblp: dict
) -> Tuple[Model, dict]:
    timeout = args.timeout
    time_interval = args.time_interval
    dec_vars = dblp['dec_vars']

    # g_model.setParam('FeasibilityTol', 1e-7)
    m.setParam('OptimalityTol', 1e-7)
    info_dict = solve_milp(
        context,
        m,
        dec_vars,
        eq_constr_dict=dict(),
        verbose=False,
        timeout=timeout,
        time_interval=time_interval,
        args=args,
    )
    return m, info_dict


def add_y_constrs_and_costs(context: XADD, dblp: dict, res_dict: dict):
    B, b, d, ys = tuple(map(lambda x: dblp[x], ['B', 'b', 'd', 'y']))

    # Retrieve constraint expressions over y from B and b
    y_constr_lst = []
    for i in range(B.shape[0]):
        lhs = (B[i, :] * ys)[0]
        rhs = b[i]
        dec_expr = lhs <= rhs
        y_constr_lst.append((dec_expr, True))

    # Add decision nodes of the constraints to the objective XADD
    context._prune_equality = False
    for dec_expr, val in y_constr_lst:
        high = context.NEG_oo
        low = context.oo
        dec_id, is_reversed = context.get_dec_expr_index(dec_expr, create=True)
        constr_node_id = context.get_internal_node(
            dec_id=dec_id,
            low=low if not is_reversed else high,
            high=high if not is_reversed else low,
        )
        # constr = context.get_dec_node(dec, low_val, high_val)

        # Instantiate binary Gurobi variable in advance and fix their value such that y constraints are
        # always enforced...
        # assert f'ind_{dec_id}' not in g_model._var_cache
        # indicator = g_model.addVar(vtype=GRB.BINARY, name=f'ind_{dec_id}')
        # ind_val = 1 if not is_reversed else 0
        # g_model.addConstr(indicator == ind_val, name=f'Orig_constr:dec{dec_id}_eq_{ind_val}')

        for i in sorted(res_dict.keys()):
            obj_i = res_dict[i]
            obj_i = context.apply(constr_node_id, obj_i, 'max')
            obj_i = context.reduce_lp(obj_i)
            res_dict[i] = obj_i

    context._prune_equality = True

    # Add d.T y to all the leaf values (only once)
    y_cost = (d.T * ys)[0]
    y_cost_id = context.get_leaf_node(y_cost)
    i = list(res_dict.keys())[0]
    obj_i = res_dict[i]
    obj_i = context.apply(obj_i, y_cost_id, 'add')
    res_dict[i] = obj_i
    return res_dict


def return_ri_func(
        context: XADD, xs: dict, zs: list, i: int, n_ri_per_z: int = 1,
) -> int:
    neg_one = context.get_leaf_node(sp.S(-1))
    x_i = context.get_leaf_node(xs[3 * i  - 2])
    x_i_1 = context.get_leaf_node(xs[3 * i - 1])
    x_i_2 = context.get_leaf_node(xs[3 * i])

    dec_1_id, _ = context.get_dec_expr_index(xs[3 * i - 2] - xs[3 * i - 1] <= 0, create=True)
    dec_2_id, _ = context.get_dec_expr_index(xs[3 * i - 1] - xs[3 * i] <= 0, create=True)

    xor_false_res = context.apply(context.apply(x_i, x_i_1, 'min'), neg_one, 'prod')
    xor_false_res = context.apply(context.apply(x_i, x_i_1, 'max'), xor_false_res, 'add')
    xor_true_res = context.apply(context.apply(x_i_1, x_i_2, 'max'), neg_one, 'prod')
    xor_true_res = context.apply(context.apply(x_i_1, x_i_2, 'min'), xor_true_res, 'add')

    inode_low = context.get_inode_canon(
        dec_2_id,
        low=xor_false_res,
        high=xor_true_res,
    )
    inode_high = context.get_inode_canon(
        dec_2_id,
        low=xor_true_res,
        high=xor_false_res,
    )
    r_i = context.get_inode_canon(
        dec_1_id,
        low=inode_low,
        high=inode_high,
    )
    r_i = context.reduce_lp(r_i)

    for j in range(max(1, i - n_ri_per_z + 1), min(len(zs), i) + 1):
        z_dec, _ = context.get_dec_expr_index(zs[j - 1] <= 0, create=True)
        z_node = context.get_inode_canon(
            z_dec,
            low=neg_one,
            high=context.ONE,
        )

        r_i = context.apply(r_i, z_node, 'prod')
    return r_i

    
def plot_runtimes(time_lst: list, tick_interval=25):
    fig, ax = plt.subplots(figsize=(10, 12))

    max_n = len(time_lst)
    ax.set_xlabel('n')
    ax.set_ylabel('Runtime (ms)')

    x_axis = range(1, max_n + 1)
    xticks = range(0, max_n + 1, tick_interval)
    ax.plot(x_axis, time_lst, 'k.')
    ax.set_xticks(xticks)
    plt.show()
    return fig, ax
