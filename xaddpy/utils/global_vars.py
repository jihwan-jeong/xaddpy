import sympy
import sympy.core.relational as relational
from sympy.codegen.rewriting import optimize, optims_c99


def log1p(x: sympy.Basic):
    log1p = sympy.log(x + 1)
    log1p = optimize(log1p, optims_c99)
    return log1p


REL_TYPE = {relational.LessThan: '<=', relational.StrictLessThan: '<',
            relational.GreaterThan: '>=', relational.StrictGreaterThan: '>',
            relational.Equality: '==', relational.Unequality: '!='}
RELATIONAL_OPERATOR = {val: key for key, val in REL_TYPE.items()}
REL_REVERSED = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
REL_NEGATED = {'>': '<=', '<': '>=', '>=': '<', '<=': '>'}
OP_TYPE = {sympy.core.Mul: 'prod', sympy.core.Add: 'sum'}
UNARY_OP = {
    'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 'exp': sympy.exp, 'abs': abs,
    'log': sympy.log, 'log2': lambda x: sympy.log(x, 2), 'log10': lambda x: sympy.log(x, 10),
    'tanh': sympy.tanh, 'cosh': sympy.cosh, 'sinh': sympy.sinh,
    'sqrt': sympy.sqrt, 'pow': lambda x, n: sympy.Pow(x, n), 
    'floor': sympy.floor, 'ceil': sympy.ceiling,
    'log1p': log1p, '-': lambda x: -x, '+': lambda x: x,
    'int': lambda x, *args: 1 if x else 0,
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
