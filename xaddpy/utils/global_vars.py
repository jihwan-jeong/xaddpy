import sympy.core.relational as relational
import sympy
from sympy.codegen.rewriting import optims_c99, optimize


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
    'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 'exp': sympy.exp,
    'log': sympy.log, 'log2': lambda x: sympy.log(x, 2), 'log10': lambda x: sympy.log(x, 10),
    'tanh': sympy.tanh, 'cosh': sympy.cosh, 'sinh': sympy.sinh,
    'sqrt': sympy.sqrt, 'pow': lambda x, n: sympy.Pow(x, n),
    'log1p': log1p, '-': lambda x: -x
}
EPSILON = 1e-1
TIMEOUT = 200
TIME_INTERVAL = 10
MIPGap = 5e-3
ACCEPTED_RV_TYPES = {'UNIFORM', 'NORMAL'}
