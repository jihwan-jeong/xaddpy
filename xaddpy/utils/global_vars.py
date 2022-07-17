import sympy.core.relational as relational
import sympy
from gurobipy import *

REL_TYPE = {relational.LessThan: '<=', relational.StrictLessThan: '<',
            relational.GreaterThan: '>=', relational.StrictGreaterThan: '>'}
REL_REVERSED = {'>': '<', '<': '>', '>=': '<=', '<=': '>='}
REL_NEGATED = {'>': '<=', '<': '>=', '>=': '<', '<=': '>'}
REL_REVERSED_GUROBI = {GRB.GREATER_EQUAL: GRB.LESS_EQUAL, GRB.LESS_EQUAL: GRB.GREATER_EQUAL}
OP_TYPE = {sympy.core.Mul: 'prod', sympy.core.Add: 'sum'}
EPSILON = 1e-1
TIMEOUT = 200
TIME_INTERVAL = 10
MIPGap = 5e-3
