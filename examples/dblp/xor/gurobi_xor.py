import argparse
import time
from typing import Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum

import examples.dblp.xor.util as util
from xaddpy.utils.logger import logger


def create_and_optimize_dblp_with_xor_constraints(
        n: int,
        ny: int,
        seed: int,
        model_name: str,
        args: argparse.Namespace
) -> Tuple[gp.Model, bool, dict]:
    n_constrs_y = args.n_constrs_y

    logger.info("Create the problem instance")
    dblp = util.create_dblp_skip_xor_constrs(n, ny, seed, n_constrs_y)
    B, b, Q, c, d, cz = tuple(map(lambda x: dblp[x], ['B', 'b', 'Q', 'c', 'd', 'cz']))

    nx = 3 * n
    epsilon = args.epsilon

    # Create gurobi model and add variables
    m = gp.Model(model_name)

    # Create decision variables
    rs = m.addVars(range(1, n + 1), lb=-20, ub=20, name='r')
    xs = m.addVars(range(1, nx + 1), lb=-10, ub=10, name='x')
    ys = m.addVars(range(1, ny + 1), lb=-10, ub=10, name='y')
    cs = m.addVars(range(1, n + 1), lb=-float('inf'), name='c')
    z = m.addVar(lb=-50, ub=50, name='z')
    m.update()

    # Add y constraints
    logger.info(f"Add {B.shape[0]} constraints over y ...")
    for i in range(B.shape[0]):
        lhs = quicksum(B[i, j - 1] * ys[j] for j in range(1, ny + 1))
        rhs = b[i]
        constr = lhs <= rhs
        m.addConstr(constr, name=f'{i}th_y_constr')

    # Add XOR constraints
    logger.info("Define helper variables for XOR constraints")
    ## Define helper variables
    fs = m.addVars(range(1, n + 1), lb=-20, ub=20, name='f')
    gs = m.addVars(range(1, n + 1), lb=-20, ub=20, name='g')
    ps = m.addVars(range(1, n + 1), lb=-20, ub=20, name='p')
    qs = m.addVars(range(1, n + 1), lb=-20, ub=20, name='q')
    vs = m.addVars(range(1, n + 1), lb=-20, ub=20, name='v')
    ws = m.addVars(range(1, n + 1), lb=-20, ub=20, name='w')

    b1 = m.addVars(range(1, n + 1), vtype=GRB.BINARY, name='b1')
    b2 = m.addVars(range(1, n + 1), vtype=GRB.BINARY, name='b2')
    b3 = m.addVars(range(1, n + 1), vtype=GRB.BINARY, name='b3')
    bz = m.addVar(vtype=GRB.BINARY, name='bz')
    m.update()

    obj = quicksum(d[j - 1] * ys[j] for j in range(1, ny + 1)) + cz[0] * z

    logger.info("Add XOR constraints via logical modeling")
    # z constraints
    m.addGenConstrIndicator(bz, True, z >= 0, name=f'GC_bz_True_impl_z_geq_0')
    m.addGenConstrIndicator(bz, False, z <= 0, name=f'GC_bz_False_impl_z_leq_0')

    # Add XOR constraints per each r_i
    for i in range(1, n + 1):
        xi_0, xi_1, xi_2 = xs[3 * i - 2], xs[3 * i - 1], xs[3 * i]

        # Associate b1 with x_i >= x_{i+1}; b2 with x_{i+1} >= x_{i+2}
        m.addGenConstrIndicator(b1[i], True, xi_0 >= xi_1 + epsilon,
                                name=f'GC_b1{i}_True_impl_x{3 * i - 2}_geq_x{3 * i - 1}')
        m.addGenConstrIndicator(b1[i], False, xi_0 <= xi_1 - epsilon,
                                name=f'GC_b1{i}_False_impl_x{3 * i - 2}_leq_x{3 * i - 1}')
        m.addGenConstrIndicator(b2[i], True, xi_1 >= xi_2 + epsilon,
                                name=f'GC_b2{i}_True_impl_x{3 * i - 1}_geq_x{3 * i}')
        m.addGenConstrIndicator(b2[i], False, xi_1 <= xi_2 - epsilon,
                                name=f'GC_b2{i}_False_impl_x{3 * i - 1}_leq_x{3 * i}')

        # XOR constraints
        m.addConstr(b3[i] <= b1[i] + b2[i], name=f'b3_{i}_leq_add(b1,b2)')
        m.addConstr(b3[i] >= b1[i] - b2[i], name=f'b3_{i}_geq_sub(b1,b2)')
        m.addConstr(b3[i] >= - b1[i] + b2[i], name=f'b3_{i}_geq_sub(b2,b1)')
        m.addConstr(b3[i] <= 2 - b1[i] - b2[i], name=f'b3_{i}_leq_sub(2,add(b1,b2))')

        # Objective value constraints
        m.addGenConstrIndicator(b3[i], True, rs[i] == fs[i], name=f'GC_b3{i}_True_impl_r{i}_eq_f{i}')
        m.addGenConstrIndicator(b3[i], False, rs[i] == gs[i], name=f'GC_b3{i}_False_impl_r{i}_eq_g{i}')
        m.addGenConstrIndicator(bz, True, fs[i] == ps[i], name=f'GC_bz_True_impl_f{i}_eq_p{i}')
        m.addGenConstrIndicator(bz, False, fs[i] == qs[i], name=f'GC_bz_False_impl_f{i}_eq_q{i}')
        m.addGenConstrIndicator(bz, True, gs[i] == vs[i], name=f'GC_bz_True_impl_g{i}_eq_v{i}')
        m.addGenConstrIndicator(bz, False, gs[i] == ws[i], name=f'GC_bz_False_impl_g{i}_eq_w{i}')

        m.addGenConstrIndicator(b2[i], True, ps[i] == xi_1 - xi_2,
                                name=f'GC_b2{i}_True_impl_p{i}_eq_sub(x{3 * i - 1},x{3 * i})')
        m.addGenConstrIndicator(b2[i], False, ps[i] == xi_2 - xi_1,
                                name=f'GC_b2{i}_False_impl_p{i}_eq_sub(x{3 * i},x{3 * i - 1})')
        m.addConstr(ps[i] + qs[i] == 0, name=f'p{i}_eq_q{i}')

        m.addGenConstrIndicator(b1[i], True, vs[i] == xi_1 - xi_0,
                                name=f'GC_b1{i}_True_impl_v{i}_eq_sub(x{3 * i - 1},x{3 * i - 2})')
        m.addGenConstrIndicator(b1[i], False, vs[i] == xi_0 - xi_1,
                                name=f'GC_b1{i}_False_impl_v{i}_eq_sub(x{3 * i - 2},x{3 * i - 1})')
        m.addConstr(vs[i] + ws[i] == 0, name=f'v{i}_eq_w{i}')


        # Add objective
        m.addConstr(quicksum(Q[i - 1, j - 1] * ys[j] for j in range(1, ny + 1)) + c[i - 1] == cs[i], name=f'coeff_{i}')
        obj += rs[i] * cs[i]

    m.setObjective(obj, sense=GRB.MINIMIZE)
    m.setParam('NonConvex', 2)
    m.setParam('OutputFlag', 1 if args.verbose else 0)

    logger.info(f"Start optimization... Timeout={args.timeout}s")
    m.setParam('TimeLimit', args.timeout)
    stime = time.time()
    m.optimize()
    etime = time.time()

    time_taken = etime - stime
    try:
        obj_val = m.objVal
        is_timeout = m.status == GRB.TIME_LIMIT
        logger.info(f'n: {n}\tObjective: {obj_val}\tTime: {time_taken}\t{"Timed out!" if is_timeout else ""}')
    except AttributeError:
        obj_val = None
        is_timeout = False
        logger.info(f"Instance {model_name} is infeasible.. cannot retrieve the objective value")
    info = dict(
        lapse=time_taken,
        obj_val=obj_val
    )
    return m, is_timeout, info
