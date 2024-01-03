import sympy
import symengine.lib.symengine_wrapper as core
from sympy.codegen.rewriting import optimize, optims_c99


def log1p(x: core.Basic):
    log1p = core.log(x + 1)
    log1p_sp = log1p._sympy_()
    log1p = optimize(log1p_sp, optims_c99)
    return core.sympify(log1p)


def _to_float(x, *args):
    if isinstance(x, core.Rational):
        return x
    elif isinstance(x, core.BooleanAtom):
        return float(bool(x))
    else:
        return float(x)


REL_TYPE = {core.LessThan: '<=', core.StrictLessThan: '<',
            core.GreaterThan: '>=', core.StrictGreaterThan: '>',
            core.Equality: '==', core.Unequality: '!='}
RELATIONAL_OPERATOR = {val: key for key, val in REL_TYPE.items()}
REL_REVERSED = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
REL_NEGATED = {'>': '<=', '<': '>=', '>=': '<', '<=': '>'}
REL_TYPE_SYM = {
    '<': core.StrictLessThan, '<=': core.LessThan,
    '>': core.StrictGreaterThan, '>=': core.GreaterThan,
    '==': core.Eq, '!=': core.Unequality,
}
OP_TYPE = {core.Mul: 'prod', core.Add: 'sum'}
UNARY_OP = {
    'sin': core.sin, 'cos': core.cos, 'tan': core.tan, 'exp': core.exp, 'abs': abs,
    'log': core.log, 'log2': lambda x: core.log(x, 2), 'log10': lambda x: core.log(x, 10),
    'tanh': core.tanh, 'cosh': core.cosh, 'sinh': core.sinh,
    'sqrt': core.sqrt, 'pow': lambda x, n: core.Pow(x, n),
    'floor': core.floor, 'ceil': core.ceiling,
    'log1p': log1p, '-': lambda x: -x, '+': lambda x: x,
    'int': lambda x, *args: 1 if x else 0,
    'float': _to_float,
}
EPSILON = 1e-1
TIMEOUT = 200
TIME_INTERVAL = 10
MIPGap = 5e-3
ACCEPTED_RV_TYPES = {'UNIFORM', 'NORMAL', 'BERNOULLI'}

try:
    import gurobipy
    from gurobipy import GRB
    LP_BACKEND = 'gurobi'
    REL_REVERSED_GUROBI = {GRB.GREATER_EQUAL: GRB.LESS_EQUAL, GRB.LESS_EQUAL: GRB.GREATER_EQUAL}
except:
    LP_BACKEND = 'pulp'
    REL_REVERSED_GUROBI = {}
