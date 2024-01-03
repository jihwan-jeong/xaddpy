import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp
import symengine.lib.symengine_wrapper as core
from symengine.lib.symengine_wrapper import Matrix

from sympy.solvers import solveset

try:
    from gurobipy import GRB
    relConverter = {core.GreaterThan: GRB.GREATER_EQUAL, core.StrictGreaterThan: GRB.GREATER_EQUAL,
                core.LessThan: GRB.LESS_EQUAL, core.StrictLessThan: GRB.LESS_EQUAL,
                core.Eq: GRB.EQUAL}
except:
    relConverter = None
    pass

from xaddpy.utils.global_vars import REL_REVERSED, REL_TYPE, REL_TYPE_SYM

typeConverter = {
    core.Integer: int, core.Float: float, core.Zero: int, core.NegativeOne: int,
    core.One: int, core.Rational: float, core.Half: float, core.RealDouble: float,
    core.Infinity: lambda x: float(sp.S(x)),
    core.NegativeInfinity: lambda x: float(sp.S(x)),
}


def reverse_expr(expr: core.Rel):
    """
    Given a symengine relational expression, reverse the expression.

    Args:
        expr (core.Rel): A symengine relational expression.
    :return:
    """
    assert isinstance(expr, core.Rel), (
        f'{expr} is not an instance of core.Rel, instead it is of type {type(expr)}.'
    )
    lhs, rhs = expr.args
    rel = REL_TYPE[type(expr)]
    rel = REL_REVERSED[rel]
    return REL_TYPE_SYM[rel](lhs, rhs)
    


def export_argmin_solution(fname, context, g_model, ):
    res_dict = {}

    # Retrieve the name space of SymEngine variables and save
    ns = context._name_space
    res_dict['namespace'] = ns

    # Save decision variables and their argmin solutions
    var_set = context._decisionVars
    var_to_anno = context.get_all_anno()
    res_dict['argmin'] = {}
    for v, anno in var_to_anno.items():
        node = context.get_exist_node(anno)
        res_dict['argmin'][str(v)] = str(node)
    res_dict['decision_vars'] = var_set

    # Save Gurobi model
    res_dict['gurobi_model'] = g_model

    # Output a file
    with open(fname, 'wb') as f_pickle:
        pickle.dump(res_dict, f_pickle)


def compute_rref_filter_eq_constr(eq_constr_str, variables):
    """
    eq_constr_str is a list of strings corresponding to initially given equality constraints in .json file.
    This function returns two things:
        1) dictionary
        2) list
    Args:
        eq_constr_str:   (list) List of equality constraints in str type.
        variables:       (list) List of symengine variables
    Returns:
        (tuple) (eq_constr_dict, variables)
    """
    if len(eq_constr_str) == 0:
        return {}, variables

    eq_constr_dict = {}
    eq_constr_lst = []
    eq_constr = []
    variables = sp.Matrix(variables)

    # Each equality constraint transformed to lhs - rhs = 0
    for const in eq_constr_str:
        lhs, rhs = const.split('=')
        lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
        expr = lhs - rhs
        if expr.free_symbols.intersection(variables.free_symbols):
            eq_constr.append(expr)

    if len(eq_constr) == 0:
        return {}, [core.Symbol(str(v)) for v in variables]

    # Build the coefficient matrix (constants are also appended at the rightmost column)
    A = sp.Matrix([[row.coeff(v) for v in variables] for row in eq_constr])
    C = sp.Matrix([row.as_coefficients_dict()[1] for row in eq_constr])
    A_ = sp.Matrix.hstack(A, C)

    # Compute the reduced row echelon form of the A_ matrix
    R, pivots = A_.rref()               # R is the reduced matrix; pivots is a tuple of pivot column indices
    num_lin_indep_constr = len(pivots)  # Number of pivots indicates the # of linearly independent equality constraints
    R = R[:num_lin_indep_constr, :]     # Remove rows of zeros

    # Append 1 at the end of variables column and perform matrix multiplication of R and the variable vector.
    E = R * sp.Matrix.vstack(variables, sp.Matrix([1]))
    for p, eq in zip(pivots, E):
        pvar = variables[p]
        rhs = solveset(eq, pvar).args[0]
        eq_constr_dict[core.Symbol(str(pvar))] = core.S(rhs)

    return eq_constr_dict, [core.Symbol(str(v)) for v in variables]


def center_features(feature):
    mean = np.mean(feature, axis=0)
    feature = feature - mean
    return feature, mean


def decenter_features(feature, mean):
    return feature + mean


def get_coefficients(expr, var_lst: List[core.Symbol]) -> Tuple[float, ...]:
    """
    Given a symengine expression and a list of variables,
        returns the coefficients of the variables in the expression.
    """
    coeffs = expr.as_coefficients_dict()
    return tuple([float(coeffs.get(var, 0.0)) for var in var_lst])


def get_multiplied_expr(expr: core.Basic, var: core.Symbol) -> core.Basic:
    """
    Given a symengine expression and a symbol,
        return the expression that is multiplied to the variable.
    """
    return expr.as_coefficients_dict()[var]


def is_bilinear(expr):
    """
    Given a symengine expression, check whether it is bilinear.
    """
    return any(map(lambda t: len(t.free_symbols) >= 2, expr.args))


def get_depth(context, node_id, depth, lst):
    node = context.get_exist_node(node_id)

    if node._is_leaf:
        lst.append(depth)
        return

    low = node._low
    high = node._high

    get_depth(context, low, depth + 1, lst)
    get_depth(context, high, depth + 1, lst)


def get_num_nodes(context, node_id):
    """Returns the total number of decision nodes and terminal nodes given the root node id.
    Args:
        context (xadd.XADD)
        node_id (int) The root node ID
    """
    num_nodes = [0, 0]
    counted = set()

    def _get_num_nodes(context, node_id):
        node = context.get_exist_node(node_id)
        new_node = False
        if node_id not in counted:
            new_node = True
        counted.add(node_id)

        if node._is_leaf:
            if new_node:
                num_nodes[1] += 1
            return

        if new_node:
            num_nodes[0] += 1

        # Recurse
        low = node._low
        high = node._high
        _get_num_nodes(context, low)
        _get_num_nodes(context, high)

    # Call the function
    _get_num_nodes(context, node_id)

    return num_nodes[0], num_nodes[1]   # Number of internal, terminal nodes


def get_bound(var: core.Symbol, expr: core.Rel) -> Tuple[core.Basic, bool]:
    """
    Return either lower bound or upper bound of 'var' from `expr`.

    Args:
        var:     (core.Symbol) target variable
        expr:    (core.Rel) An inequality over 'var'

    Returns:
        (core.Basic, bool) a symengine expression along with 
            the boolean value indicating whether an upper or lower bound.
            True for upper bound, False for lower bound.
    """
    assert isinstance(expr, core.Rel), (
        f'{expr} is not an instance of core.Rel, instead it is of type {type(expr)}.'
    )
    lt = isinstance(expr, core.LessThan) or isinstance(expr, core.StrictLessThan)
    lhs, rhs = expr.args
    lhs = (lhs - rhs).expand()

    # Solve to get the bound over the variable.
    sol_set = core.solve(lhs, var)
    sols = sol_set.args
    assert len(sols) == 1, f"More than one solution found: {sol_set}"
    sol_expr = sols[0].expand()

    # check whether upper bound or lower bound over 'var'
    # if ub: 'var' <= upper bound, else: 'var' >= lower bound
    # If the coefficient of `var` was negative, should swap the direction of the inequality.
    c = lhs.as_coefficients_dict()[var]
    ub = (lt and c > 0) or (not lt and c < 0)
    return sol_expr, ub


def get_date_time():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")


def check_sym_boolean(expr: core.Basic) -> bool:
    return isinstance(expr, core.BooleanAtom) or expr.is_Boolean or str(expr).startswith('Bernoulli')


def sample_rvs(
        rvs: List[core.Symbol], 
        rv_type: List[str],
        params: List[Tuple],
        rng: np.random.Generator,
        expectation: bool = False
) -> Dict[core.Symbol, float]:
    assert len(rvs) == len(rv_type), "Length mismatch"
    res = {}
    for rv, type, param in zip(rvs, rv_type, params):
        if type == 'UNIFORM':
            assert len(param) == 2 and param[0] <= param[1]
            low, high = param
            res[rv] = rng.uniform(low, high) if not expectation else (low + high) / 2
        elif type == 'NORMAL':
            assert len(param) == 2 and param[1] >= 0
            mu, sigma = param
            res[rv] = rng.normal(mu, sigma) if not expectation else mu
        else:
            raise NotImplementedError
    return res


def check_expr_linear(expr: core.Basic) -> bool:
    """Checks whether the given symengine expression is linear or not.
    
    If there exists a term that is nonlinear in any variables (including bilinear),
        the function will return False. Otherwise, this will return True.
    
    Args:
        expr (core.Basic): The expression to check linearity.

    Returns:
        bool: True if linear; otherwise False.
    """
    if isinstance(expr, core.Rel):
        expr = expr.args[0]
    coeffs_dict = expr.as_coefficients_dict()
    for term in coeffs_dict:
        # Nonlinear terms.
        if not isinstance(term, core.Number) and not isinstance(term, core.Symbol):
            return False
    return True
