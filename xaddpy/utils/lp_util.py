from abc import ABC
from abc import abstractmethod
from typing import Union, Optional, Dict, List, cast
import sympy as sp
import pulp as pl
from pulp import const

from xaddpy.utils.util import typeConverter
from xaddpy.utils.global_vars import EPSILON

FeasibilityTol = 1e-6       # Gurobi default feasibility tolerance
IntFeasTol = 1e-5           # Gurobi default integer feasibility tolerance
OptimalityEpsilon = 1e-3

try:
    from gurobipy import GRB
    GRB_SOL_STATUS_TO_PULP_STATUS = {
        GRB.INF_OR_UNBD: pl.LpStatusUndefined,
        GRB.UNBOUNDED: pl.LpStatusUnbounded,
        GRB.OPTIMAL: pl.LpStatusOptimal,
        GRB.INFEASIBLE: pl.LpStatusInfeasible
    }
except:
    GRB_SOL_STATUS_TO_PULP_STATUS = {}


class BasePULPModel(ABC):
    def __init__(self, **kwargs):
        self._model = None
        self._solver: pl.LpSolver = kwargs.get('backend', pl.LpSolverDefault)
        self._status = None
        self.sympy_to_pulp = {}
        self.epsilon = kwargs.get('epsilon', EPSILON)
        self._var_to_bound = kwargs.get('var_to_bound', {})  # sympy variable to its bound
        self._name_to_var = {}
        self.use_solver_directly = False

    @property
    def use_solver_directly(self) -> bool:
        return self._use_solver_directly
    
    @use_solver_directly.setter
    def use_solver_directly(self, use: bool):
        self._use_solver_directly = use
    
    def toggle_direct_solver_on(self):
        self.use_solver_directly = True
    
    def toggle_direct_solver_off(self):
        self.use_solver_directly = False
    
    @abstractmethod
    def solve(self, callback: Optional[callable] = None):
        pass

    @abstractmethod
    def getVars(self):
        pass

    @abstractmethod
    def getVarByName(self, name, *args, **kwargs):
        pass

    @abstractmethod
    def setParam(self, paramname: str, newval: Union[int, str, float]):
        pass

    @abstractmethod
    def setAttr(self, attrname: str, newval):
        pass

    @property
    @abstractmethod
    def ModelName(self):
        pass

    @abstractmethod
    def setObjective(self, expr, sense=None):
        pass

    @property
    @abstractmethod
    def status(self):
        pass

    @abstractmethod
    def addVar(self, *args, **kwargs):
        pass

    @abstractmethod
    def addConstr(self, *args, **kwargs):
        pass


class Model(BasePULPModel):
    def __init__(
            self, 
            name: str,
            backend: str,
            sense: int,
            msg: bool = False,
            **kwargs
    ):
        if backend.lower() == 'gurobi':
            backend = pl.GUROBI(msg=msg, **kwargs)
        elif backend.lower() == 'pulp':
            # backend = pl.COIN_CMD(msg=msg, **kwargs)
            backend = pl.GLPK_CMD(msg=msg, **kwargs)
            backend.msg = msg
        else:
            raise NotImplementedError
        super().__init__(backend=backend, **kwargs)
        self._model = pl.LpProblem(name, sense=sense)

    def set_sympy_to_pulp_dict(self, sp_to_pulp_dict: dict):
        self.sympy_to_pulp = sp_to_pulp_dict

    def setAttr(self, attrname: str, newval):
        if attrname[0] != '_':
            attrname = '_' + attrname
        setattr(self, attrname, newval)

    def setObjective(self, obj):
        if self._model.objective is None:
            self._model += obj
        else:
            # Otherwise, need to rebuild the model
            model = pl.LpProblem(self._model.name, sense=self._model.getSense())
            model += obj, "objective"
            constraints = self._model.constraints()
            for name, constr in constraints.items():
                model.addConstraint(constr, name=name)
            self._model = model
    
    def reset(self):
        self._model = pl.LpProblem(self._model.name, sense=self._model.getSense())
        self.toggle_direct_solver_off()

    def get_constraint_by_name(self, name):
        return self._model.constraints.get(name)

    def getVars(self):
        return self._model.variables()

    def addVar(
            self, lb=0, ub=float('inf'), name='', vtype=const.LpContinuous, *args, **kwargs
    ) -> None:
        assert name != '', 'Variable name should be specified'
        if vtype == const.LpContinuous:
            v = pl.LpVariable(name=name, lowBound=lb, upBound=ub, cat=vtype)
        elif vtype == const.LpBinary:
            v = pl.LpVariable(name=name, cat=vtype)
        else:
            raise ValueError(f"Variable type {vtype} is not supported.")
        self._model.addVariable(v)
        self._name_to_var[name] = v
        return v

    def addConstr(self, constr: pl.LpConstraint, name='', *args, **kwargs):
        assert name != '', "A constraint should have a name"
        self._model.addConstraint(constr, name=name)

    def get_constraints(self):
        return self._model.constraints

    @property
    def status(self):
        return self._model.status

    @property
    def objVal(self):
        return self._model.objective.value()

    def setParam(self, paramname: str, newval: Union[int, str, float]):
        if isinstance(self._solver, pl.GUROBI):
            self._model.solverModel.setParam(paramname, newval)
            self.toggle_direct_solver_on()

    @property
    def ModelName(self):
        return self._model.Params.ModelName

    def getVarByName(self, name):
        return self._name_to_var.get(name)

    def getVars(self):
        return self._model.variables()

    def solve(self, *args, **kwargs) -> Optional[int]:
        if self._use_solver_directly:
            self._solver.callSolver(self._model)
            return GRB_SOL_STATUS_TO_PULP_STATUS[self._model.solverModel.status]
        else:
            return self._model.solve(solver=self._solver)


def is_val(x, val) -> bool:
    """Checks whether `x' equals to `val' up to the integrality tolerance"""
    return val - IntFeasTol <= x <= val + IntFeasTol


def convert_to_pulp_expr(
        expr: sp.Basic, model: Model, incl_bound: bool = True, binary: bool = False,
):
    """Given a sympy expression, convert it to a PULP expression. 
    An expression can be a simple linear expression or linear inequality.

    Args:
        expr (sympy.Basic): SymPy expression
        model (lp_util.PULPModel): PULP model class
        incl_bound (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    sympy2pulp = model.sympy_to_pulp

    # If cached, return immediately
    if expr in sympy2pulp:
        return sympy2pulp[expr]
    
    # Recursively convert Sympy expression to PULP one
    if isinstance(expr, sp.Number) and not isinstance(expr, sp.core.numbers.NaN):
        return typeConverter[type(expr)](expr)

    elif isinstance(expr, sp.core.numbers.NaN):
        return float('inf')

    elif isinstance(expr, sp.Symbol):
        if model is None:
            raise ValueError

        var_str = str(expr)
        var = model.getVarByName(var_str)
        
        if var is not None:
            return var

        if expr._assumptions.get('bool', False) and binary:
            var = model.addVar(name=var_str, vtype=pl.LpBinary)
        elif expr._assumptions.get('bool', False):
            var = model.addVar(lb=0, ub=1, name=var_str, vtype=pl.LpContinuous)
        elif incl_bound:
            bound = model._var_to_bound.get(expr, (float('-inf'), float('inf')))
            lb, ub = bound
            var = model.addVar(lb=lb, ub=ub, name=var_str, vtype=pl.LpContinuous)
        else:
            var = model.addVar(lb=float('-inf'), ub=float('inf'), name=var_str, vtype=pl.LpContinuous)
        return var

    res = [convert_to_pulp_expr(arg_i, model, incl_bound=incl_bound) for arg_i in expr.args]

    # Operation between args0 and args1 is either Add or Mul
    if isinstance(expr, sp.Add):
        ret = pl.lpSum(res)
    elif isinstance(expr, sp.Mul):
        ret = 1
        for t in res:
            ret *= t
    else:
        raise NotImplementedError("Operation not recognized!")

    # Store in cache
    sympy2pulp[expr] = ret
    return ret


def set_equality_constraint(var, rhs, pl_model, incl_bound=False):
    if isinstance(var, sp.Basic):
        pl_var = convert_to_pulp_expr(var, pl_model, incl_bound=incl_bound)
    else:
        pl_var = var

    if isinstance(rhs, sp.Basic):
        rhs = convert_to_pulp_expr(rhs, pl_model, incl_bound=incl_bound)

    pl_model.addConstr(pl_var == rhs, f"Equality_constraint_for_{str(pl_var)}")