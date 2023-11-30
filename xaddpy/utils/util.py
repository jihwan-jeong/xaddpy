import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import symengine
import symengine.lib.symengine_wrapper as core
from sympy import sympify
from sympy.matrices import Matrix
from sympy.solvers import solveset

try:
    from gurobipy import GRB
    relConverter = {core.GreaterThan: GRB.GREATER_EQUAL, core.StrictGreaterThan: GRB.GREATER_EQUAL,
                core.LessThan: GRB.LESS_EQUAL, core.StrictLessThan: GRB.LESS_EQUAL,
                core.Eq: GRB.EQUAL}
except:
    relConverter = None
    pass

from xaddpy.utils.global_vars import REL_REVERSED, REL_TYPE
from xaddpy.utils.symengine import BooleanVar

typeConverter = {core.Integer: int, core.Float: float, core.Zero: int, core.NegativeOne: int,
                 core.One: int, core.Rational: float, core.Half: float,
                 core.Infinity: float, core.NegativeInfinity: float}


def reverse_expr(expr):
    """
    Given a symengine inequality expression, reverse the expression.
    :param expr:
    :return:
    """

    lhs, rhs, rel = expr.lhs, expr.rhs, REL_TYPE[type(expr)]
    rel = REL_REVERSED[rel]
    raise RuntimeError("TODO: Implement this!")
    return core.Rel(lhs, rhs, rel)


def export_argmin_solution(fname, context, g_model, ):
    res_dict = {}

    # Retrieve the name space of sympy variables and save
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


def compute_rref_filter_eq_constr(eq_constr_str, variables, locals):
    """
    eq_constr_str is a list of strings corresponding to initially given equality constraints in .json file.
    This function returns two things:
        1) dictionary
        2) list
    :param eq_constr_str:   (list) List of equality constraints in str type.
    :param variables:       (list) List of sympy variables
    :param locals:          (dict) Namespace of Sympy variables
    :return:
    """
    if len(eq_constr_str) == 0:
        return {}, variables

    eq_constr_dict = {}
    eq_constr_lst = []
    eq_constr = []
    variables = Matrix(variables)

    # Each equality constraint transformed to lhs - rhs = 0
    for const in eq_constr_str:
        lhs, rhs = const.split('=')
        lhs, rhs = sympify(lhs, locals=locals), sympify(rhs, locals=locals)
        expr = lhs - rhs
        if expr.free_symbols.intersection(variables.free_symbols):
            eq_constr.append(expr)

    if len(eq_constr) == 0:
        return {}, list(variables)

    # Build the coefficient matrix (constants are also appended at the rightmost column)
    A = Matrix([[row.coeff(v) for v in variables] for row in eq_constr])
    C = Matrix([row.as_coefficients_dict()[1] for row in eq_constr])
    A_ = Matrix.hstack(A, C)

    # Compute the reduced row echelon form of the A_ matrix
    R, pivots = A_.rref()               # R is the reduced matrix; pivots is a tuple of pivot column indices
    num_lin_indep_constr = len(pivots)  # Number of pivots indicates the # of linearly independent equality constraints
    R = R[:num_lin_indep_constr, :]     # Remove rows of zeros

    # Append 1 at the end of variables column and perform matrix multiplication of R and the variable vector.
    E = R * Matrix.vstack(variables, Matrix([1]))
    for p, eq in zip(pivots, E):
        pvar = variables[p]
        rhs = solveset(eq, pvar).args[0]
        eq_constr_dict[pvar] = rhs

    return eq_constr_dict, list(variables)


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
    return expr.coeff(var)


def is_bilinear(expr):
    """
    Given a symengine expression, check whether it is bilinear.
    """
    return any(map(lambda t: len(t.free_symbols) >= 2, terms))


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


def get_bound(var, expr):
    """
    Return either lower bound or upper bound of 'var' from expr.
    :param var:     (core.Symbol) target variable
    :param expr:    (core.Rel) An inequality over 'var'
    :return:        (core.Basic, bool) a sympy expression along with the boolean value indicating whether an upper
                    or lower bound. True for upper bound, False for lower bound.
    """
    comp = core.solve(expr, var)

    if isinstance(comp, core.And):
        assert len(comp.args) == 2, "No more than 3 terms should be generated as a result of solve(ineq)!"
        args1rhs = comp.args[1].canonical.rhs             # when 'var' is the only variable in the
        i = 0 if (args1rhs == core.oo) or (args1rhs == -core.oo) else 1
        comp = comp.args[i]
    else:
        raise RuntimeError("TODO: Handle this case!")
        comp = sympy.simplify(comp)

    comp = comp.canonical
    # check whether upper bound or lower bound over 'var'
    # if ub: 'var' <= upper bound, else: 'var' >= lower bound
    ub = isinstance(comp, core.LessThan) or isinstance(comp, core.StrictLessThan)
    expr = comp.rhs
    return expr, ub


def get_date_time():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f")


def check_sym_boolean(expr: core.Basic):
    return isinstance(expr, core.BooleanAtom) or isinstance(expr, BooleanVar)


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
