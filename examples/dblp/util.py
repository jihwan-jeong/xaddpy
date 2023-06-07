import argparse
import json
import os
from os import path
from typing import Callable, List, Optional, Tuple

import gurobipy as gp
import numpy as np
import pulp as pl
import sympy as sp
from gurobipy import GRB, quicksum
from scipy import stats
from scipy.linalg import block_diag
from scipy.sparse import random

from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.utils.logger import logger
from xaddpy.utils.lp_util import GurobiModel
from xaddpy.utils.util import (compute_rref_filter_eq_constr,  
                               get_date_time, get_num_nodes)    # DO NOT REMOVE
from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar

Ak = sp.Matrix([[0, 1], [-2, -1], [2, -1]])
ak = sp.Matrix([[2], [-2], [2]])
ck = sp.Matrix([[-1], [-1]])
dk1 = ck.copy()
dk2 = - 2
Qk1 = sp.eye(2)
Qk2 = sp.Matrix([[1], [1]])
Bk2 = sp.Matrix([[-1], [1]])
bk2 = sp.Matrix([[0], [2]])


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def get_delta_and_rho(
        class_id,
        n,
):
    rho = sp.zeros(n, 1)
    if class_id == 0:
        # delta = np.random.uniform(1, 3, size=(n, 1))
        delta = sp.Matrix(np.random.randint(1, 3, size=(n, 1)))
    elif class_id == 1:
        delta = sp.Matrix([sp.S(3) for _ in range(n)]) + sp.zeros(n, 1)
    elif class_id == 2:
        # delta = np.abs(np.random.normal(0, 100, size=(n, 1))) + 3
        delta = sp.Matrix(np.random.randint(4, 6, size=(n, 1)))
    else:
        delta = sp.Matrix([sp.S(5) / 2 for _ in range(n)]) + sp.zeros(n, 1)
        rho = sp.Matrix([sp.S(3) / 2 for _ in range(n)]) + sp.zeros(n, 1)
    return delta, rho


def generate_B_matrix(delta, rho):
    if isinstance(delta, sp.Matrix):
        delta = delta[0]
    if isinstance(rho, sp.Matrix):
        rho = rho[0]
    return sp.Matrix([[-delta, 1], [delta - rho, 1], [rho, -2]])


def generate_b_vector(delta, rho):
    if isinstance(delta, sp.Matrix):
        delta = delta[0]
    if isinstance(rho, sp.Matrix):
        rho = rho[0]
    return sp.Matrix([[0], [2 * delta - rho], [0]])


def return_household_matrix(v: sp.Matrix):
    dim = len(v)
    H = sp.eye(dim) - 2 / (v.T * v)[0] * (v * v.T)
    return H


def generate_dblp_instance(
        k11,
        k12,
        k13,
        k1,
        k2,
        transform=True,
        seed=0,
) -> Tuple[tuple, dict]:
    """
    This function follows the test problem generation steps given in
        "Generation of Disjointly Constrained Bilinear Programming Test Problems", Vincente et al. (1992)


    """
    np.random.seed(seed)
    nx = 2 * k2
    ny = k1 + k2

    # Instantiate a Gurobi model used for slack test
    m = gp.Model('slack-test')
    S = m.addVar(name='S')
    m.Params.OutputFlag = 0
    xs = m.addVars(range(nx), lb=float('-inf'), ub=float('inf'), name='x')
    ys = m.addVars(range(ny), lb=float('-inf'), ub=float('inf'), name='y')

    def test_slack(A, a, B, b):
        m.remove(m.getConstrs())
        m.addConstrs(quicksum(A[i, j] * xs[j] for j in range(nx)) - a[i] + S <= 0 for i in range(A.shape[0]))
        m.addConstrs(quicksum(B[i, j] * ys[j] for j in range(ny)) - b[i] + S <= 0 for i in range(B.shape[0]))
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


    # Coefficient matrix A
    A = sp.Matrix(block_diag(*[Ak] * k2))

    # Constant vector a
    a = sp.Matrix.vstack(*[ak] * k2)

    # Coefficient matrix B and vector b
    Bk1_lst = []
    b1_lst = []
    for i, n in enumerate([k11, k12, k13, k1 - (k11+k12+k13)]):
        if n == 0:
            continue
        delta, rho = get_delta_and_rho(i, n)
        Bk1 = sp.Matrix(block_diag(*map(lambda x: generate_B_matrix(x[0], x[1]), zip(delta, rho))))
        b1 = sp.Matrix.vstack(*list(map(lambda x: generate_b_vector(x[0], x[1]), zip(delta, rho))))
        Bk1_lst.append(Bk1)
        b1_lst.append(b1)
    B1 = sp.Matrix(block_diag(*Bk1_lst))
    b1 = sp.Matrix.vstack(*b1_lst)
    B2 = sp.Matrix.diag([Bk2] * (k2 - k1))
    b2 = sp.Matrix([[bk2] for _ in range(k2 - k1)])
    if k1 == 0:
        B = B2
        b = b2
    elif k2 - k1 > 0:
        B = sp.Matrix(block_diag(B1, B2))
        b = sp.Matrix.vstack(b1, b2)
    else:
        B, b = B1, b1

    # Coefficient matrix Q
    Q = sp.Matrix(block_diag(*([Qk1]*k1 + [Qk2] * (k2 - k1))))

    # Coefficient vector c
    c = sp.Matrix.vstack(*[ck] * k2)

    # Coefficient vector d
    d = sp.Matrix.vstack(*[sp.Matrix.vstack(*[dk1] * k1)])
    if k2 - k1 > 0:
        d = sp.Matrix.vstack(*[d, sp.Matrix([dk2 for _ in range(k2 - k1)])])


    if transform:
        while True:
            # Random Householder matrices (for sparsity, only a few entries are nonzero)
            vx = sp.Matrix(np.random.randint(1, 4, size=(nx, 1)))
            idxs = np.random.choice(range(nx), size=max(nx-1, 0), replace=False)
            vx = sp.Matrix([vx[i] if i not in idxs else 0 for i in range(len(vx))])
            vy = sp.Matrix(np.random.randint(1, 4, size=(ny, 1)))
            idxs = np.random.choice(range(ny), size=max(ny-1, 0), replace=False)
            vy = sp.Matrix([vy[i] if i not in idxs else 0 for i in range(len(vy))])
            Hx = return_household_matrix(vx)  # np.eye(nx) - 2 / (vx.T @ vx) * vx @ vx.T
            Hy = return_household_matrix(vy)  # np.eye(ny) - 2 / (vy.T @ vy) * vy @ vy.T

            # Positive definite diagonal matrices Dx and Dy
            Dx = sp.Matrix(np.diag(np.random.randint(1, 4, size=(nx))))
            Dy = sp.Matrix(np.diag(np.random.randint(1, 4, size=(ny))))

            # The transformation matrices
            Mx = Dx * Hx
            My = Dy * Hy

            # Transform coefficients
            c_bar = Mx.T * c
            d_bar = My.T * d
            Q_bar = Mx.T * Q * My
            A_bar = A * Mx
            B_bar = B * My

            pass_slack_test = test_slack(A_bar, a, B_bar, b)
            if pass_slack_test:
                break
    else:
        c_bar, d_bar, Q_bar, A_bar, B_bar = c, d, Q, A, B

    # Number of local and global minima
    num_local_opts = 4 ** (k11 + k12 + k13) * 2 ** (k1 - (k11 + k12 + k13))
    num_global_opts = 2 ** k11 * 3 ** k12
    prob_info = dict(
        num_local_opts=num_local_opts,
        num_global_opts=num_global_opts,
    )

    return (c_bar, d_bar, Q_bar, A_bar, a, B_bar, b), prob_info


def create_prob_json(
        seed,
        dir_path,
        cfg=None,
        transform=True,
        opt_var_set_id=1,
        density=None,
        nx=None,
        ny=None,
        fname=None,
):
    """
    Create a .json file of a DBLP problem instance.
    """
    ineq_constr = []
    # Vincente et al. (1992)
    if cfg is not None:
        k11, k12, k13, k1, k2 = cfg
        nx, ny = 2 * k2, k1 + k2
        fname = f"dblp_inst_{k11}_{k12}_{k13}_{k1}_{k2}_opt_{opt_var_set_id}_seed{seed}.json" if fname is None \
            else fname

        (c, d, Q, A, a, B, b), prob_info = generate_dblp_instance(
            k11=k11,
            k12=k12,
            k13=k13,
            k1=k1,
            k2=k2,
            seed=seed,
            transform=transform,
        )
    # Randomized problems with varying density
    else:
        assert density is not None
        if fname is None:
            if nx is None and ny is None:
                fname = f"dblp_inst_density_{density}_opt_{opt_var_set_id}_seed{seed}.json"
            else:
                fname = f"dblp_inst_{nx}_{ny}_density_{density}_opt_{opt_var_set_id}_seed{seed}.json"
        (c, d, Q, A, a, B, b), prob_info = generate_rand_dblp_instances(density, seed, nx=nx, ny=ny)
        nx, ny = len(c), len(d)

    x = sp.symbols(f'x0:{nx}')
    y = sp.symbols(f'y0:{ny}')

    x_vec, y_vec = sp.Matrix(x), sp.Matrix(y)
    obj = c.T @ x_vec + x_vec.T @ Q @ y_vec + d.T @ y_vec

    x_constr_lhs = A @ x_vec
    y_constr_lhs = B @ y_vec

    for i, lhs in enumerate(x_constr_lhs):
        constr = lhs <= a[i]
        ineq_constr.append(f"{constr}")

    for i, lhs in enumerate(y_constr_lhs):
        constr = lhs <= b[i]
        ineq_constr.append(f"{constr}")

    os.makedirs(dir_path, exist_ok=True)
    fname = path.join(dir_path, fname)

    config = {}
    config['cvariables0'] = list(map(str, x))
    config['cvariables1'] = list(map(str, y))
    config['bvariables'] = []
    config['ineq-constr'] = ineq_constr
    config['eq-constr'] = []
    config['xadd'] = ''
    config['is-minimize'] = 1
    config['min-var-set-id'] = opt_var_set_id
    config['objective'] = f"{obj[0]}"

    if cfg is not None:
        config['num-local-opts'] = prob_info.get('num_local_opts', [])
        config['num-global-opts'] = prob_info['num_global_opts']
        config['min-values'] = ['-oo'] * (nx + ny)
        config['max-values'] = ['oo'] * (nx + ny)

    else:
        config['min-values'] = [0] * (nx + ny)
        config['max-values'] = [prob_info.get('ub', 20)] * (nx + ny)
    
    # Add additional info
    config['optimal-objective'] = prob_info.get('obj_val', None)
    
    with open(fname, 'w') as fjson:
        json.dump(config, fjson, indent=4)

    return fname


def generate_rand_dblp_instances(
        density, seed, nx=8, ny=4, nA=15, nB=15
) -> Tuple[tuple, dict]:
    # A DBLP can be specified by c, d, Q, A, B, a, b. 
    # Additionally, to make the problem bounded always, place lower (0) and upper bounds (50) on variables.
    # `density` can change from 0 to 1
    # With the generated instance, check whether the problem is feasible using Gurobi.

    if nx is None and ny is None:
        nx, ny = 8, 4
    elif nx is None or ny is None:
        raise ValueError

    # Number of x and y variables set to 8
    # nA, nB = 15, 15  # in total there are 30 constraints
    ub = 50
    low, high = -10, 10  # integer coefficients are sampled from uniform(low, high)
    rng = np.random.RandomState(seed)
    rv_obj = stats.randint(low, high)
    rv_obj.random_state = rng
    rv = rv_obj.rvs


    logger.info("Generating randomized DBLP instances and check for feasibility")
    logger.info(f"Density: {density}, nx: {nx}, ny: {ny}, nA: {nA}, nB: {nB}, seed: {seed}")
    
    # Objective coefficients
    c, d = tuple(map(lambda n: rv(n), [nx, ny]))
    # Generate Q: ensure at least one of Q is non-zero
    while True:
        Q = random(nx, ny, density=density, data_rvs=rv, dtype=np.int, random_state=rng).A
        if np.count_nonzero(Q) != 0:
            break

    # For checking feasibility
    def test_slack(A, a, B, b):
        model.remove(model.getConstrs())

        model.addConstrs(quicksum(A[i, j] * xs[j] for j in range(nx)) - a[i] + S <= 0 for i in range(A.shape[0]))
        model.addConstrs(quicksum(B[i, j] * ys[j] for j in range(ny)) - b[i] + S <= 0 for i in range(B.shape[0]))
        model.setObjective(S, sense=GRB.MAXIMIZE)

        model.optimize()
        status = model.status
        if status == GRB.INF_OR_UNBD:
            model.setParam('DualReductions', 0)
            model.optimize()
            status = model.status
            model.setParam('DualReductions', 1)
        obj_val = model.objVal if status == GRB.OPTIMAL else 1e100
        if status != GRB.UNBOUNDED and obj_val < 1e-4:
            return False
        else:
            return True

    model = gp.Model('FeasibilityChecker')
    S = model.addVar(name='S')  # For testing slack value
    model.Params.OutputFlag = 0
    status = False
    xs = model.addVars(range(nx), lb=0, ub=ub, vtype=GRB.CONTINUOUS, name='x')
    ys = model.addVars(range(ny), lb=0, ub=ub, vtype=GRB.CONTINUOUS, name='y')
    
    while True:
        # Generate A
        a = rv(nA)
        while True:
            A = random(nA * int(1/density), nx, density=density, data_rvs=rv, dtype=np.int, random_state=rng).A
            nonzero_cnts = (A != 0).sum(axis=1)
            nonzero_rows = nonzero_cnts.nonzero()[0]
            if len(nonzero_rows) >= nA:
                A = A[nonzero_rows[:nA], :]
                break

        # Generate B
        b = rv(nB)
        while True:
            B = random(nB * int(1/density), ny, density=density, data_rvs=rv, dtype=np.int, random_state=rng).A
            nonzero_cnts = (B != 0).sum(axis=1)
            nonzero_rows = nonzero_cnts.nonzero()[0]
            if len(nonzero_rows) >= nB:
                B = B[nonzero_rows[:nB], :]
                break

        # Check feasibility
        try:
            model.remove(model.getConstrs())
        except AttributeError:
            pass

        model.addConstrs(quicksum(A[i, j] * xs[j] for j in range(nx)) <= a[i] for i in range(nA))
        model.addConstrs(quicksum(B[i, j] * ys[j] for j in range(ny)) <= b[i] for i in range(nB))
        model.setObjective(1)
        model.update()
        model.optimize()
        status = model.status
        if status == GRB.OPTIMAL:
            obj = quicksum(c[i] * xs[i] for i in range(len(xs)))
            obj += quicksum(Q[i, j] * xs[i] * ys[j] for i in range(len(xs)) for j in range(len(ys)))
            obj += quicksum(d[i] * ys[i] for i in range(len(ys)))
            model.setObjective(obj)
            model.setParam('NonConvex', 2)
            model.update()
            model.optimize()
            assert model.status == GRB.OPTIMAL
            obj_val = model.objVal
            
            # Test for slack value
            pass_slack_test = test_slack(A, a, B, b)
            if pass_slack_test:
                logger.info(f"Feasible instance generated: optimal objective: {obj_val}")
                break

    # Make all elements Sympy objects
    Q, A, B, c, d, a, b = tuple(map(
        lambda arr: sp.Matrix(arr), [Q, A, B, c, d, a, b]
    ))

    prob_info = {'ub': ub, 'obj_val': obj_val}

    return (c, d, Q, A, a, B, b), prob_info


def build_xadd_from_json(
        context: XADD, fname: str, var_name_rule: Optional[Callable] = None
) -> Tuple[dict, int, dict]:
    try:
        with open(fname, 'r') as f:
            prob_instance = json.load(f)
    except:
        raise FileNotFoundError(f'File {fname} not found')
    
    # Check the loaded file
    check_json(prob_instance)
    context.link_json_file(fname)
    
    # Which set of variables to minimize? 0 or 1
    min_var_set_id = prob_instance['min-var-set-id']

    # Namespace to be used to define sympy symbols
    ns = {}

    # is_minimize?
    is_min = True if prob_instance['is-minimize'] else False

    # Create Sympy symbols for cvariables and bvariables
    if len(prob_instance['cvariables0']) == 1 and isinstance(prob_instance['cvariables0'][0], int):
        cvariables0 = sp.symbols('x1:%s' % (prob_instance['cvariables0'][0]+1))
    else:
        cvariables0 = []
        for v in prob_instance['cvariables0']:
            cvariables0.append(sp.symbols(v))
    if len(prob_instance['cvariables1']) == 1 and isinstance(prob_instance['cvariables1'][0], int):
        cvariables1 = sp.symbols('y1:%s' % (prob_instance['cvariables1'][0]+1))
    else:
        cvariables1 = []
        for v in prob_instance['cvariables1']:
            cvariables1.append(sp.symbols(v))
    cvariables = cvariables0 + cvariables1

    if len(prob_instance['bvariables']) == 1 and isinstance(prob_instance['bvariables'][0], int):
        bvariables = sp.symbols(f"b1:{prob_instance['bvariables'][0] + 1}", integer=True)
    else:
        bvariables = []
        for v in prob_instance['bvariables']:
            bvariables.append(sp.symbols(v, integer=True))

    # Retrieve dimensions of problem instance
    cvar_dim = len(cvariables)                      # Continuous variable dimension
    bvar_dim = len(prob_instance['bvariables'])     # Binary variable dimension
    var_dim = cvar_dim + bvar_dim                   # Total dimensionality of variables
    assert bvar_dim + cvar_dim > 0, "No decision variables provided"
    # assert (bvar_dim == 0 and cvar_dim != 0) or (bvar_dim != 0 and cvar_dim == 0) # Previously, we accepted either continuous or binary variables, not both

    variables = list(cvariables) + list(bvariables)
    ns.update({str(v): v for v in variables})

    # retrieve lower and upper bounds over decision variables
    min_vals = prob_instance['min-values']
    max_vals = prob_instance['max-values']

    if len(min_vals) == 1 and len(min_vals) == len(max_vals):       # When a single number is used for all cvariables
        min_vals = min_vals * cvar_dim
        max_vals = max_vals * cvar_dim

    assert len(min_vals) == len(max_vals) and len(min_vals) == cvar_dim, \
        "Bound information mismatch!\n cvariables: {}\tmin-values: {}\tmax-values: {}".format(
            prob_instance['cvariables'],
            prob_instance['min-values'],
            prob_instance['max-values'])

    bound_dict = {}
    for i, (lb, ub) in enumerate(zip(min_vals, max_vals)):
        lb, ub = sp.S(lb), sp.S(ub)
        bound_dict[ns[str(cvariables[i])]] = (lb, ub)
    
    # Update XADD attributes
    variables = [ns[str(v)] for v in variables]
    if var_name_rule is not None:
        XADD.set_variable_ordering_func(var_name_rule)

    bvariables = [ns[str(bvar)] for bvar in bvariables]
    cvariables0 = [ns[str(cvar)] for cvar in cvariables0]
    cvariables1 = [ns[str(cvar)] for cvar in cvariables1]
    context.update_bounds(bound_dict)
    context.update_name_space(ns)

    # Read constraints and link with the created Sympy symbols
    # If an initial xadd is directly provided in str type, need also return it
    ineq_constrs = []
    eq_constr_dict = {}
    for const in prob_instance['ineq-constr']:
        ineq_constrs.append(sp.sympify(const, locals=ns))
    
    if prob_instance['xadd']:
        init_xadd = parse_xadd_grammar(prob_instance['xadd'], ns)[1][0]
    else:
        init_xadd = None
        # Handle equality constraints separately.
        # Equality constraints are firstly converted to a system of linear equations. Then, the reduced row echelon form
        # of the coefficient matrix tells us linearly independent equality constraints. Only put these constraints
        # when building the initial LP XADD.
        eq_constr_dict, variables = compute_rref_filter_eq_constr(prob_instance['eq-constr'],
                                                                  variables,
                                                                  locals=ns)
        
    assert (len(ineq_constrs) + len(eq_constr_dict) == 0 and init_xadd is not None) or \
           (len(ineq_constrs) + len(eq_constr_dict) != 0 and init_xadd is None), \
           "When xadd formulation is provided, make sure no other constraints passed (vice versa)"
    
    # Read in objective function if provided
    obj = prob_instance['objective']
    if obj:
        obj = sp.expand(sp.sympify(prob_instance['objective'], locals=ns))
    
    # Build XADD from constraints and the objective
    if (obj and init_xadd is None):
        constrs_and_objective = ineq_constrs + [obj]
        dblp_xadd = build_initial_xadd_lp(context, constrs_and_objective, is_min=is_min)

    # Build XADD from given case statement
    elif init_xadd is not None:
        dblp_xadd = context.build_initial_xadd(init_xadd)
    else:
        raise ValueError("No objective function or initial XADD provided")
    
    context._prune_equality = False
    # Substitute in equality constraints to `dblp_xadd`
    # It is guaranteed that we don't need to substitute one equality constraint into another;
    # this is due to the fact that we have reduced the equality constraints 
    # into the reduced row echelon form.
    for v_i, rhs in eq_constr_dict.items():
        ## TODO: how to handle binary variables?
        # The equality constraint over v_i can be seen as its annotation
        v_i_anno = context.get_leaf_node(rhs)
        context.update_anno(v_i, v_i_anno)

        # Substitute into the problem formulation
        dblp_xadd = context.reduce_lp(context.substitute(dblp_xadd, {v_i: rhs}))

        # Keep track of the order.. later retrieve argmin(max) using this
        context.add_eliminated_var(v_i)
        variables.remove(v_i)

        # Add bound constraints of v_i if exists
        lb, ub = bound_dict[v_i]
        bound_constraints = []

        if lb != -sp.oo:
            comp = (rhs >= lb)
            bound_constraints.append((comp, True))
        if ub != sp.oo:
            comp = (rhs <= ub)
            bound_constraints.append((comp, True))
        for d, b in bound_constraints:
            high_val = sp.oo if (b and not is_min) or (not b and is_min) else -sp.oo
            low_val = -sp.oo if (b and not is_min) or (not b and is_min) else sp.oo
            bound_constraint = context.get_dec_node(d, low_val, high_val)
            dblp_xadd = context.apply(bound_constraint, dblp_xadd, 'min' if not is_min else 'max')
            dblp_xadd = context.reduce_lp(dblp_xadd)
    context._prune_equality = True
    dblp_xadd = context.reduce_lp(dblp_xadd)
    
    # Final variables set
    min_vars = cvariables0 if min_var_set_id == 0 else cvariables1
    free_vars = cvariables0 if min_var_set_id == 1 else cvariables1
    variables = dict(
        min_var_list=[ns[str(cvar)] for cvar in min_vars if ns[str(cvar)] in variables],
        free_var_list=[ns[str(cvar)] for cvar in free_vars if ns[str(cvar)] in variables],
    )
    context._free_var_set.update(free_vars)
    context._min_var_set.update(min_vars)
    return variables, dblp_xadd, eq_constr_dict


def build_initial_xadd_lp(context: XADD, constrs_and_objective: List, is_min: bool = True):
    """
    Given LP constraints, recursively build initial XADD and return the id of the root node.
    """
    if len(constrs_and_objective) == 1:
        return context.get_leaf_node(constrs_and_objective[0])
    
    dec_expr = constrs_and_objective[0]
    dec, is_reversed = context.get_dec_expr_index(dec_expr, create=True)
    low = context.oo if is_min else context.NEG_oo
    high = build_initial_xadd_lp(context, constrs_and_objective[1:], is_min)

    # Swap low and high if the decision is reversed
    if is_reversed:
        low, high = high, low
    return context.get_inode_canon(dec, low, high)


def check_json(json_loaded):
    assert "cvariables0" in json_loaded and "cvariables1" in json_loaded, \
        "list of cvariables not in .json file.. exiting.."
    assert "bvariables" in json_loaded, "bvariables not in .json file.. exiting.."
    assert "min-values" in json_loaded, "min-values not in .json file.. exiting.."
    assert "max-values" in json_loaded, "max-values not in .json file.. exiting.."
    assert "min-var-set-id" in json_loaded, "min-var-set not in .json file.. exiting.."
    assert "ineq-constr" in json_loaded, "ineq-constr not in .json file.. exiting.."
    assert "eq-constr" in json_loaded, "eq-constr not in .json file.. exiting.."
    assert "is-minimize" in json_loaded, "is-minimize not in .json file.. exiting.."
    assert (len(json_loaded['cvariables0']) + len(json_loaded['cvariables1'])) * len(json_loaded['bvariables']) == 0, \
        "Currently, either only cvariables or bvariables can be defined, not both."


def get_config_dir(args: argparse.Namespace):
    domain = args.domain
    config_dir = path.join(path.curdir, f'xaddpy/experiments/dblp/{domain}/prob_instances/')
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_results_dir(args: argparse.Namespace):
    results_dir = path.join(path.curdir, f'results/dblp/{args.domain}')
    try:
        method = args.method
        results_dir = path.join(results_dir, f'{method}')
    except:
        pass
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


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


def setup_gurobi_model(model_name, m: GurobiModel = None, **kwargs):
    try:
        assert m is None or isinstance(m._solver, pl.GUROBI)
        if m is None:
            m = GurobiModel(model_name, backend='gurobi_custom', **kwargs)
        else:
            m.set_model_name(model_name)
        m.setParam('OutputFlag', 0)
        m.setAttr('_best_obj', float('inf'))
        m.setAttr('_time_log', 0)
        return m
    except gp.GurobiError as e:
        logger.error(f'Error code {e.errno}: {e}')
        raise e
    except AttributeError:
        logger.error('Encountered an attribute error')
        raise e
