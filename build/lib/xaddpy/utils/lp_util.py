import warnings
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import pulp as pl
import sympy as sp
from pulp import const

from xaddpy.utils.logger import logger

try:
    import gurobipy as gp
except:
    pass

from xaddpy.utils.global_vars import EPSILON, LP_BACKEND, REL_REVERSED_GUROBI
from xaddpy.utils.util import relConverter, typeConverter

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


class GUROBI(pl.GUROBI):
    def buildSolverModel(self, lp: pl.LpProblem):
        from pulp import PulpSolverError, constants, log

        ### Copy the original code from the parent class ###
        log.debug("set the sense of the problem")
        if lp.sense == constants.LpMaximize:
            lp.solverModel.setAttr("ModelSense", -1)
        if self.timeLimit:
            lp.solverModel.setParam("TimeLimit", self.timeLimit)
        gapRel = self.optionsDict.get("gapRel")
        logPath = self.optionsDict.get("logPath")
        if gapRel:
            lp.solverModel.setParam("MIPGap", gapRel)
        if logPath:
            lp.solverModel.setParam("LogFile", logPath)

        log.debug("add the variables to the problem")
        for var in lp.variables():
            lowBound = var.lowBound
            if lowBound is None:
                lowBound = -gp.GRB.INFINITY
            upBound = var.upBound
            if upBound is None:
                upBound = gp.GRB.INFINITY
            obj = lp.objective.get(var, 0.0)
            varType = gp.GRB.CONTINUOUS
            if var.cat == constants.LpInteger and self.mip:
                varType = gp.GRB.INTEGER
            var.solverVar = lp.solverModel.addVar(
                lowBound, upBound, vtype=varType, obj=obj, name=var.name
            )
        if self.optionsDict.get("warmStart", False):
            # Once lp.variables() has been used at least once in the building of the model.
            # we can use the lp._variables with the cache.
            for var in lp._variables:
                if var.varValue is not None:
                    var.solverVar.start = var.varValue

        lp.solverModel.update()
        log.debug("add the Constraints to the problem")
        for name, constraint in lp.constraints.items():
            # build the expression
            expr = gp.LinExpr(
                list(constraint.values()), [v.solverVar for v in constraint.keys()]
            )
            if constraint.sense == constants.LpConstraintLE:
                relation = gp.GRB.LESS_EQUAL
            elif constraint.sense == constants.LpConstraintGE:
                relation = gp.GRB.GREATER_EQUAL
            elif constraint.sense == constants.LpConstraintEQ:
                relation = gp.GRB.EQUAL
            else:
                raise PulpSolverError("Detected an invalid constraint type")
            constraint.solverConstraint = lp.solverModel.addConstr(
                expr, relation, -constraint.constant, name
            )
        lp.solverModel.update()
        #### End of the original code from the parent class ####
        
        # Add indicator constraints
        log.debug("add the indicator constraints to the problem")
        for i_constr in lp._i_constr_cache.values():
            binvar, binval, lhs, sense, rhs, name = i_constr
            binvar = binvar.solverVar
            gp_lhs = self.get_gurobi_expr(lhs)
            gp_rhs = self.get_gurobi_expr(rhs)
            if not isinstance(gp_rhs, (int, float)):
                gp_lhs, gp_rhs = gp_rhs - gp_lhs, 0
            lp.solverModel.addGenConstrIndicator(binvar, binval, gp_lhs, sense, gp_rhs, name)
        lp.solverModel.update()

    def get_gurobi_expr(
            self, expr: Union[sp.core.numbers.Number, int, float, pl.LpAffineExpression, pl.LpVariable]
    ):
        if isinstance(expr, pl.LpAffineExpression):
            return self.pulp_expr_to_gurobi(expr)
        elif isinstance(expr, pl.LpVariable):
            return expr.solverVar
        elif isinstance(expr, (int, float, sp.core.numbers.Number)):
            return int(expr) if int(expr) == expr else float(expr)

    def pulp_expr_to_gurobi(self, expr: pl.LpAffineExpression):
        """
        Convert a Pulp expression to a Gurobi expression
        """
        g_expr = gp.LinExpr(
            list(expr.values()), [v.solverVar for v in expr.keys()]
        )
        g_expr += expr.constant
        return g_expr


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
            backend: str = '',              # 'gurobi', 'pulp', 'gurobi_custom'
            sense: int = pl.LpMinimize,     # pl.LpMinimize: 1, pl.LpMaximize: -1
            msg: bool = False,
            **kwargs
    ):
        # Override the backend with the default solver
        if backend == '':
            backend = LP_BACKEND
            logger.info(f"PULP backend undefined... By default, {LP_BACKEND} is used")
        if backend.lower() == 'gurobi':
            backend = pl.GUROBI(msg=msg, **kwargs)
        elif backend.lower() == 'pulp':
            # backend = pl.COIN_CMD(msg=msg, **kwargs)
            backend = pl.GLPK_CMD(msg=msg, **kwargs)
            backend.msg = msg
        elif backend.lower() == 'gurobi_custom':
            backend = GUROBI(msg=msg, **kwargs)
        else:
            raise NotImplementedError
        super().__init__(backend=backend, **kwargs)
        self._model = pl.LpProblem(name, sense=sense)

    def set_model_name(self, name: str):
        self._model.name = name
    
    def set_sympy_to_pulp_dict(self, sp_to_pulp_dict: dict):
        self.sympy_to_pulp = sp_to_pulp_dict

    def setAttr(self, attrname: str, newval):
        if attrname[0] != '_':
            attrname = '_' + attrname
        setattr(self, attrname, newval)
        setattr(self._model, attrname, newval)

    def setObjective(self, obj):
        if self._model.objective is None:
            self._model += obj
        else:
            # Otherwise, need to rebuild the model
            model = pl.LpProblem(self._model.name, sense=self._model.getSense())
            model += obj, "objective"
            constraints = self._model.constraints
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
    
    def addVars(
            self, 
            indices: Iterable[int], 
            lb=0, ub=float('inf'), 
            name='', 
            vtype=const.LpContinuous, 
            *args, 
            **kwargs,
    ) -> List[pl.LpVariable]:
        var_lst = [None] * len(indices)
        for i, ind in enumerate(indices):
            v = self.addVar(lb, ub, f'{name}_({ind})', vtype, *args, **kwargs)
            var_lst[i] = v
        return var_lst

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
            if not self.use_solver_directly:
                warnings.warn(f"Setting a parameter {paramname} for Gurobi model yet `use_solver_directly` is set to False")
            self._model.solverModel.setParam(paramname, newval)

    @property
    def ModelName(self):
        return self._model.Params.ModelName

    def getVarByName(self, name):
        return self._name_to_var.get(name)

    def getVars(self):
        return self._model.variables()

    def solve(self, *args, **kwargs) -> Optional[int]:
        if self._use_solver_directly:
            self._solver.callSolver(self._model, **kwargs)
            return GRB_SOL_STATUS_TO_PULP_STATUS[self._model.solverModel.status]
        else:
            return self._model.solve(solver=self._solver, **kwargs)


class GurobiModel(Model):
    def __init__(
            self,
            name: str,
            backend: str = '',
            sense: int = pl.LpMinimize,
            msg: bool = False,
            **kwargs
    ):
        super().__init__(name, backend, sense, msg, **kwargs)
        assert isinstance(self._solver, pl.GUROBI)
        self._i_constr_cache = {}   # indicator constraints

    def addGenConstrIndicator(
            self,
            binvar: pl.LpVariable,
            binval: Union[bool, int],
            lhs,
            sense,
            rhs,
            name=''
    ):
        if not hasattr(self._model, '_i_constr_cache'):
            self._model._i_constr_cache = self._i_constr_cache
        i_constr = (binvar, binval, lhs, sense, rhs, name)
        self._i_constr_cache[name] = i_constr
    
    def addDecNodeIndicatorConstr(
            self,
            dec: int,
            expr: sp.core.relational.Relational,
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        """
        Given the decision ID and the corresponding inequality expression,
        create an indicator variable (if not exists) and add the logical constraints.

        Args:
              dec (int): The integer decision ID
              expr (Relational): The associated sympy expression in canonical form
              size (int): (if provided) The size of the dataset
        """
        rel = type(expr)
        if rel == sp.core.relational.Eq:
            raise NotImplementedError(
                "Equality constraints are not supported yet. It can always be reformulated."
            )
        else:
            self._addDecNodeIndicatorConstrIneq(dec, expr, size, *args, **kwargs)

    def _addDecNodeIndicatorConstrIneq(
            self,
            dec: int,
            expr: sp.core.relational.Relational,
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        """
        Given the decision ID and the corresponding inequality expression,
        create an indicator variable (if not exists) and add the logical constraints.

        Args:
              dec (int): The integer decision ID
              expr (Relational): The associated sympy expression in canonical form
              size (int): (if provided) The size of the dataset
        """
        lhs, rel, rhs = expr.lhs, type(expr), expr.rhs
        rel = relConverter[rel]
        rev_rel = REL_REVERSED_GUROBI[rel]

        ind_var_name = f'ind_{dec}'

        constr_name = f'GC_({ind_var_name})_ineq'
        check = constr_name in self._i_constr_cache
        if not check:
            if size is None:
                indicator = self.getVarByName(ind_var_name)
                lhs = convert_to_pulp_expr(lhs, self, incl_bound=True)
                self.addGenConstrIndicator(indicator,
                                           True,
                                           lhs,
                                           rel,
                                           rhs - self.epsilon if rel == '<'\
                                                else rhs + self.epsilon,
                                           name=f'{constr_name}_(True)')
                self.addGenConstrIndicator(indicator, 
                                           False, 
                                           lhs, 
                                           rev_rel, 
                                           rhs - self.epsilon if rev_rel == '<'\
                                            else rhs + self.epsilon,
                                           name=f'{constr_name}_(False)')
            else:
                for i in range(size):
                    indicator = self.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = convert_to_pulp_expr(lhs, self, i, incl_bound=True)
                    self.addGenConstrIndicator(
                        indicator, True, lhs_i, rel, rhs - self.epsilon if rel == '<' else rhs + self.epsilon,
                        name=f'{constr_name}_(True)__{i}'
                    )
                    self.addGenConstrIndicator(
                        indicator, False, lhs_i, rev_rel, rhs - self.epsilon if rev_rel == '<' else rhs + self.epsilon,
                        name=f'{constr_name}_(False)__{i}'
                    )
    
    def addIntNodeIndicatorConstr(
            self,
            dec: int,
            node_id: int,
            low: Union[int, sp.Basic],
            high: Union[int, sp.Basic],
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        skip_low, skip_high = False, False
        if low == sp.oo or low == -sp.oo:
            skip_low = True
        if high == sp.oo or high == -sp.oo:
            skip_high = True

        ind_var_name = f'ind_{dec}'
        constr_name = f'GC_(icvar_{node_id})_eq'
        check = constr_name in self._i_constr_cache

        if not check:
            if size is None:
                indicator = self.getVarByName(ind_var_name)
                par_val = self.getVarByName(f'icvar_{node_id}')
                if not skip_high:
                    high_val = convert_rhs(self, high, incl_bound=True)
                    self.addGenConstrIndicator(
                        indicator, True, par_val, '=', high_val,
                        name=f"GC_({ind_var_name})_(True)_({par_val})_eq_rhs"
                    )
                if not skip_low:
                    low_val = convert_rhs(self, low, incl_bound=True)
                    self.addGenConstrIndicator(
                        indicator, False, par_val, '=', low_val,
                        name=f"GC_({ind_var_name})_(False)_({par_val})_eq_rhs"
                    )
            else:
                for i in range(size):
                    indicator = self.getVarByName(f'{ind_var_name}__{i}')
                    par_val_i = self.getVarByName(f'icvar_{node_id}__{i}')
                    if not skip_high:
                        high_val_i = convert_rhs(self, high, data_idx=i, incl_bound=True)
                        self.addGenConstrIndicator(
                            indicator, True, par_val_i, '=', high_val_i,
                            name=f"GC_({ind_var_name})_(True)_({par_val_i})_eq_rhs"
                        )
                    if not skip_low:
                        low_val_i = convert_rhs(self, low, data_idx=i, incl_bound=False)
                        self.addGenConstrIndicator(
                            indicator, False, par_val_i, '=', low_val_i,
                            name=f"GC_({ind_var_name})_(False)_({par_val_i})_eq_rhs"
                        )

    def setParam(self, paramname: str, newval: Union[int, str, float]):
        if isinstance(self._solver, pl.GUROBI):
            if not self.use_solver_directly:
                warnings.warn(f"Setting a parameter {paramname} for Gurobi model yet `use_solver_directly` is set to False")
            if not hasattr(self._model, 'solverModel') or self._model.solverModel is None:
                self._model.solverModel = gp.Model(self._model.name)
            self._model.solverModel.setParam(paramname, newval)
    
    def setAttr(self, attrname: str, newval):
        super().setAttr(attrname, newval)
        if not hasattr(self._model, 'solverModel') or self._model.solverModel is None:
            self._model.solverModel = gp.Model(self._model.name)
        setattr(self._model.solverModel, attrname, newval)

def is_val(x, val) -> bool:
    """Checks whether `x' equals to `val' up to the integrality tolerance"""
    return val - IntFeasTol <= x <= val + IntFeasTol


def convert_to_pulp_expr(
        expr: sp.Basic,
        model: Model,
        incl_bound: bool = True,
        binary: bool = False,
        data_idx: Optional[int] = None,
):
    """Given a sympy expression, convert it to a PULP expression. 
    An expression can be a simple linear expression or linear inequality.

    Args:
        expr (sympy.Basic): SymPy expression
        model (lp_util.PULPModel): PULP model class
        incl_bound (bool, optional): _description_. Defaults to True.
        binary (bool, optional): Is binary variable? Defaults to False.
        data_idx (int, optional): The data index for EMSPO. Defaults to None.
    
    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    sympy2pulp = model.sympy_to_pulp

    # If cached, return immediately
    if data_idx is None and expr in sympy2pulp:
        return sympy2pulp[expr]
    elif data_idx is not None and (expr, data_idx) in sympy2pulp:
        return sympy2pulp[(expr, data_idx)]
    
    # Recursively convert Sympy expression to PULP one
    if isinstance(expr, sp.Number) and not isinstance(expr, sp.core.numbers.NaN):
        return typeConverter[type(expr)](expr)

    elif isinstance(expr, sp.core.numbers.NaN):
        return float('inf')

    elif isinstance(expr, sp.Symbol):
        if model is None:
            raise ValueError

        var_str = str(expr) if data_idx is None else f'{str(expr)}__{data_idx}'
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

    res = [convert_to_pulp_expr(arg_i,
                                model,
                                incl_bound=incl_bound,
                                binary=binary,
                                data_idx=data_idx) for arg_i in expr.args]

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
    if data_idx is None:
        sympy2pulp[expr] = ret
    else:
        sympy2pulp[(expr, data_idx)] = ret
    return ret


def convert_rhs(
        m: Model,
        expr_or_node_id: Union[sp.Basic, int],
        data_idx: Optional[int] = None,
        incl_bound: bool = True
):
    if isinstance(expr_or_node_id, sp.core.numbers.Number):
        return float(expr_or_node_id)
    elif isinstance(expr_or_node_id, int):
        return m.getVarByName(f'icvar_{expr_or_node_id}') if data_idx is None else \
                m.getVarByName(f'icvar_{expr_or_node_id}__{data_idx}')
    else:
        return convert_to_pulp_expr(expr_or_node_id,
                                    m,
                                    data_idx=data_idx,
                                    incl_bound=incl_bound)


def set_equality_constraint(var, rhs, pl_model, incl_bound=False):
    if isinstance(var, sp.Basic):
        pl_var = convert_to_pulp_expr(var, pl_model, incl_bound=incl_bound)
    else:
        pl_var = var

    if isinstance(rhs, sp.Basic):
        rhs = convert_to_pulp_expr(rhs, pl_model, incl_bound=incl_bound)

    pl_model.addConstr(pl_var == rhs, f"Equality_constraint_for_{str(pl_var)}")


def return_model_info(m: Model) -> Tuple[int, int, int]:
    vars_lst = m.getVars()
    num_cvar = 0
    num_bvar = 0
    
    try:
        m.update()
        for v in vars_lst:
            if v.vtype == GRB.BINARY:
                num_bvar += 1
            else:
                num_cvar += 1
        num_constrs = len(m.getConstrs())   # TODO: nonlinear constraints wouldn't be captured
    except:
        for v in vars_lst:
            if v.cat == const.LpBinary:
                num_bvar += 1
            else:
                num_cvar += 1
        num_constrs = len(m.get_constraints())
    return num_cvar, num_bvar, num_constrs
