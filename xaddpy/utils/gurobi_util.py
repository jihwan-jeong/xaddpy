import time
from abc import ABC
from abc import abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod
import os.path as path
import os
from scipy.sparse import hstack
import scipy.sparse as sparse
import psutil

import gurobipy as gp
import numpy as np
import sympy
from gurobipy import GRB
from typing import Union, Optional
from tqdm import tqdm

from xaddpy.utils.logger import logger
from xaddpy.utils.util import typeConverter, relConverter
from xaddpy.utils.global_vars import EPSILON, REL_REVERSED_GUROBI
from xaddpy.utils.milp_encoding import convert2GurobiExpr

FeasibilityTol = 1e-6       # Gurobi default feasibility tolerance
IntFeasTol = 1e-5           # Gurobi default integer feasibility tolerance
OptimalityEpsilon = 1e-3

class BaseGurobiModel(ABC):
    def __init__(self, **kwargs):
        self._model = None
        self._status = None
        self.sympy_to_gurobi = {}
        self.epsilon = kwargs.get('epsilon', EPSILON)
        self._var_to_bound = kwargs.get('var_to_bound', {})  # sympy variable to its bound

    @abstractmethod
    def optimize(self, callback: Optional[callable] = None):
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


class Model(BaseGurobiModel):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self._model = gp.Model(name)

    def set_sympy_to_gurobi_dict(self, sp2gb_dict):
        self.sympy_to_gurobi = sp2gb_dict

    def setAttr(self, attrname: str, newval, model=False):
        if attrname[0] != '_':
            attrname = '_' + attrname
        if not model:
            setattr(self, attrname, newval)
        else:
            setattr(self._model, attrname, newval)

    def setParam(self, paramname: str, newval: Union[int, str]):
        self._model.setParam(paramname, newval)

    def setObjective(self, expr, sense=None):
        self._model.setObjective(expr, sense=sense)

    def update(self):
        self._model.update()

    def getConstrByName(self, name):
        return self._model.getConstrByName(name)

    def getVars(self):
        return self._model.getVars()

    def addVar(self, lb=0, ub=float('inf'), name='', vtype=GRB.CONTINUOUS, *args, **kwargs):
        assert name != '', 'Variable name should be specified'
        if vtype == GRB.CONTINUOUS:
            v = self._model.addVar(lb=lb, ub=ub, name=name, vtype=vtype)
        elif vtype == GRB.BINARY:
            v = self._model.addVar(name=name, vtype=vtype)
        else:
            raise ValueError(f"Variable type {vtype} is not supported.")
        v._VarName = name
        return v

    def addConstr(self, constr: gp.TempConstr, name='', *args, **kwargs):
        assert name != '', "A constraint should have a name"
        c = self._model.addConstr(constr, name=name)
        c._ConstrName = name
        return c

    def remove(self, items):
        self._model.remove(items)

    def getConstrs(self):
        return self._model.getConstrs()

    @property
    def status(self):
        return self._model.status

    @property
    def objVal(self):
        return self._model.objVal

    def getQConstrs(self):
        return self._model.getQConstrs()

    @property
    def ModelName(self):
        return self._model.Params.ModelName

    def getVarByName(self, name, *args, **kwargs):
        return self._model.getVarByName(name)

    def getVars(self):
        return self._model.getVars()

    def optimize(self, *args, **kwargs):
        self._model.optimize()


class GurobiModel(BaseGurobiModel):
    """
    A helper class for handling variables and constraints of a Gurobi model.
    This is to be used for applications where frequently calling model.update() is bottleneck.
    """
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self._model = gp.Model(name)
        self._model._wrapper = self
        self._status = self._model.status
        self._var_cache = {}
        self._l_constr_cache = {}   # linear constraints
        self._q_constr_cache = {}   # quadratic (bilinear) constraints
        self._i_constr_cache = {}   # indicator constraints

    @property
    def num_constrs(self):
        n_lconstr = len(self._l_constr_cache)
        n_qconstr = len(self._q_constr_cache)
        n_iconstr = len(self._i_constr_cache)
        return n_lconstr + n_qconstr + n_iconstr

    def optimize(self, callback: Optional[callable] = None):
        if callback is not None:
            self._model.optimize(callback)
        else:
            self._model.optimize()
        self._status = self._model.status
        return self._status

    def remove(self, items):
        self._model.remove(items)

    @property
    def status(self):
        return self._status

    def getVars(self):
        return self._model.getVars() if len(self._model.getVars()) > 0 else list(self._var_cache)

    def addVars(self, *args, **kwargs):
        return self._model.addVars(*args, **kwargs)

    def addVar(self, lb=0, ub=float('inf'), name='', vtype=GRB.CONTINUOUS, *args, **kwargs):
        assert name != '', 'Variable name should be specified'
        if vtype == GRB.CONTINUOUS:
            v = self._model.addVar(lb=lb, ub=ub, name=name, vtype=vtype)
        elif vtype == GRB.BINARY:
            v = self._model.addVar(name=name, vtype=vtype)
            if 'ind' in name:       # node with shallow depth has higher priority
                ind_id = int(name.split('__')[0].split('_')[1])
                v.BranchPriority = ind_id
            # v.Start = 0             # Simply populate with all zero solution
        else:
            raise ValueError(f"Variable type {vtype} is not supported.")
        self._var_cache[name] = v
        v._VarName = name           # this attribute can be accessed even before calling model.update()
        return v

    def getVarByName(self, name, *args, **kwargs):
        return self._var_cache.get(name, None)

    def getConstrs(self):
        return self._model.getConstrs()

    def getConstrByName(self, name, *args, **kwargs):
        c_type = 'l'
        constr = self._l_constr_cache.get(name, None)
        if constr is None:
            constr = self._q_constr_cache.get(name, None)
            c_type = 'q'
        return constr, c_type

    def addConstr(self, constr: gp.TempConstr, name='', *args, **kwargs):
        assert name != '', "A constraint should have a name"
        c = self._model.addConstr(constr, name=name[:20])
        if isinstance(c, gp.Constr):
            self._l_constr_cache[name] = c
        elif isinstance(c, gp.QConstr):
            self._q_constr_cache[name] = c
        else:
            raise ValueError("Unrecognized constraint type provided to ``GurobiModel.addConstr``")
        c._ConstrName = name        # this attribute can be accessed even before calling model.update()
        return c

    def setParam(self, paramname: str, newval: Union[int, str]):
        self._model.setParam(paramname=paramname, newval=newval)

    def setAttr(self, attrname: str, newval, model=True):
        if attrname[0] != '_':
            attrname = '_' + attrname
        if not model:
            setattr(self, attrname, newval)
        else:
            setattr(self._model, attrname, newval)

    @property
    def ModelName(self):
        return self._model.ModelName

    @property
    def solCount(self):
        return self._model.solCount

    @property
    def objVal(self):
        return self._model.objVal

    @property
    def x(self):
        return self._model.x

    @property
    def _best_obj(self):
        return self._model._best_obj

    def setObjective(self, expr, sense=None):
        self._model.setObjective(expr, sense=sense)

    def update(self):
        self._model.update()

    def write(self, filename: str):
        self._model.write(filename=filename)

    def addGenConstrIndicator(self, binvar: gp.Var, binval: Union[bool, int], lhs, sense=None, rhs=None, name=''):
        self._model.addGenConstrIndicator(binvar, binval, lhs, sense, rhs, name)


class BFGurobiModel(GurobiModel):
    """Implements the `brute-force (BF)' MIP encoding of XADDs"""
    def __init__(self, name, **kwargs):
        super(BFGurobiModel, self).__init__(name, **kwargs)
        self._model._wrapper = self
        self.scheme = 'bruteforce'

    def addDecNodeIndicatorConstr(
            self,
            dec: int,
            expr: sympy.core.relational.Relational,
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
                lhs = convert2GurobiExpr(lhs, self, incl_bound=True)
                self.addGenConstrIndicator(
                    indicator, True, lhs, rel, rhs - self.epsilon if rel == '<' else rhs + self.epsilon,
                    name=f'{constr_name}_(True)'
                )
                self.addGenConstrIndicator(
                    indicator, False, lhs, rev_rel, rhs - self.epsilon if rev_rel == '<' else rhs + self.epsilon,
                    name=f'{constr_name}_(False)'
                )
            else:
                for i in range(size):
                    indicator = self.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = convert2GurobiExpr(lhs, self, i, incl_bound=True)
                    self.addGenConstrIndicator(
                        indicator, True, lhs_i, rel, rhs - self.epsilon if rel == '<' else rhs + self.epsilon,
                        name=f'{constr_name}_(True)__{i}'
                    )
                    self.addGenConstrIndicator(
                        indicator, False, lhs_i, rev_rel, rhs - self.epsilon if rev_rel == '<' else rhs + self.epsilon,
                        name=f'{constr_name}_(False)__{i}'
                    )
            self._i_constr_cache[constr_name] = True

    def addIntNodeIndicatorConstr(
            self,
            dec: int,
            node_id: int,
            low: Union[int, sympy.Basic],
            high: Union[int, sympy.Basic],
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        skip_low, skip_high = False, False
        if low == sympy.oo or low == -sympy.oo:
            skip_low = True
        if high == sympy.oo or high == -sympy.oo:
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
                        indicator, True, par_val == high_val,
                        name=f"GC_({ind_var_name})_(True)_({par_val})_equalto_({str(high_val)[:20].replace(' ', '')})"
                    )
                if not skip_low:
                    low_val = convert_rhs(self, low, incl_bound=True)
                    self.addGenConstrIndicator(
                        indicator, False, par_val == low_val,
                        name=f"GC_({ind_var_name})_(False)_({par_val})_equalto_({str(low_val)[:20].replace(' ', '')})"
                    )
            else:
                for i in range(size):
                    indicator = self.getVarByName(f'{ind_var_name}__{i}')
                    par_val_i = self.getVarByName(f'icvar_{node_id}__{i}')
                    if not skip_high:
                        high_val_i = convert_rhs(self, high, i, incl_bound=True)
                        self.addGenConstrIndicator(
                            indicator, True, par_val_i == high_val_i,
                            name=f"GC_({ind_var_name})_(True)_({par_val_i})_"
                                 f"equalto_({str(high_val_i)[:20].replace(' ', '')})"
                        )
                    if not skip_low:
                        low_val_i = convert_rhs(self, low, i, incl_bound=False)
                        self.addGenConstrIndicator(
                            indicator, False, par_val_i == low_val_i,
                            name=f"GC_({ind_var_name})_(False)_({par_val_i})_"
                                 f"equalto_({str(low_val_i)[:20].replace(' ', '')})"
                        )
            self._i_constr_cache[constr_name] = True

    @property
    def _param_array(self):
        return self._model._param_array

class CBGurobiModel(BaseGurobiModel):
    """Implements the combinatorial Benders' decomposition"""
    def __init__(self, name, num_mis=1, **kwargs):
        super(CBGurobiModel, self).__init__(**kwargs)

        self._masterproblem = MasterProblem(name, wrapper=self)
        self._subproblem = SubProblem(name, wrapper=self, num_mis=num_mis)
        self._model_name = name
        self.scheme = 'benders'
        self._constr_cache = set()
        self._initialized = False

    @property
    def status(self):
        try:
            return self._masterproblem._model.status
        except Exception as e:
            logger.error(e)

    @property
    def _best_obj(self):
        return self._masterproblem._model._best_obj

    @property
    def solCount(self):
        return self._subproblem._model.solCount

    @property
    def ModelName(self):
        return self._model_name

    @property
    def _param_array(self):
        return self._subproblem.param_array

    def initialize(self, callback):
        self._subproblem.initialize(callback)
        self._masterproblem.initialize()

        self._initialized = True

    def optimize(self, callback: Optional[callable] = None):
        if not self._initialized:
            self.initialize(callback)
        mp_status = self._masterproblem.optimize(callback)
        return mp_status

    def update(self):
        self._masterproblem.update()
        self._subproblem.update()

    def getVars(self) -> list:
        return self._masterproblem.getVars() + self._subproblem.getVars()

    def setObjective(self, expr, sense=None):
        """Note: The indicator variables in the master problem do not affect the objective in our problem"""
        self._subproblem.set_objective(expr)

    def setParam(self, paramname: str, newval: Union[int, str]):
        self._masterproblem.setParam(paramname=paramname, newval=newval)
        if paramname != 'OutputFlag':
            self._subproblem.setParam(paramname=paramname, newval=newval)

    def setAttr(self, attrname: str, newval, model=True):
        if attrname[0] != '_':
            attrname = '_' + attrname
        if attrname == "_param_array":
            self._subproblem.setAttr(attrname, newval, model)
            self._masterproblem.setAttr(attrname, newval, model)
        else:
            self._masterproblem.setAttr(attrname, newval, model)
            self._subproblem.setAttr(attrname, newval, model)

    def addVar(self, lb=0, ub=float('inf'), name='', vtype=None, *args, **kwargs):
        assert vtype is not None, "vtype should be provided"

        if vtype == GRB.CONTINUOUS:
            v = self._subproblem.addVar(lb=lb, ub=ub, name=name)
        elif vtype == GRB.BINARY:
            v = self._masterproblem.addVar(name=name)
        else:
            raise ValueError(f"Variable type {vtype} is not recognized")
        v._VarName = name
        return v

    def addConstr(self, constr: gp.TempConstr, name: str = '', size: int = None, *args, **kwargs):
        """
        Adds a typical linear constraint (not logical) to the model.
        While all binary variables are dealt with in the master problem,
        linear constraints are added to the subproblem.
        """
        self._subproblem.addConstr(constr, name)

    def addLogicalConstr(
            self, dec: int, indicator: int, constr: gp.TempConstr, data_idx: int = None, node_id: int = None,
    ):
        self._subproblem.addLogicalConstr(dec, indicator, constr, data_idx, node_id)

    def getVarByName(self, name, binary=False, *args, **kwargs):
        if binary:
            return self._masterproblem.getVarByName(name)
        else:
            return self._subproblem.getVarByName(name)

    def addDecNodeIndicatorConstr(
            self,
            dec: int,
            expr: sympy.Basic,
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        lhs, rel, rhs = expr.lhs, type(expr), expr.rhs
        rel = relConverter[rel]
        rev_rel = REL_REVERSED_GUROBI[rel]

        ind_var_name = f'ind_{dec}'
        constr_name = f'GC_({ind_var_name})_ineq'
        check = constr_name in self._constr_cache
        if not check:
            if size is None:
                lhs = convert2GurobiExpr(lhs, self, incl_bound=True)
                rhs = rhs - self.epsilon if rel == '<' else rhs + self.epsilon
                rev_rhs = rhs - self.epsilon if rev_rel == '<' else rhs + self.epsilon
                self.addLogicalConstr(dec, 1, (lhs <= rhs) if rel == '<' else (lhs >= rhs))
                self.addLogicalConstr(dec, 0, (lhs <= rev_rhs) if rev_rel == '<' else (lhs >= rev_rhs))
            else:
                for i in range(size):
                    lhs_i = convert2GurobiExpr(lhs, self, i, incl_bound=True)
                    rhs_i = rhs - self.epsilon if rel == '<' else rhs + self.epsilon
                    rev_rhs_i = rhs - self.epsilon if rev_rel == '<' else rhs + self.epsilon
                    self.addLogicalConstr(
                        dec, 1, (lhs_i <= rhs_i) if rel == '<' else (lhs_i >= rhs_i), data_idx=i,
                    )
                    self.addLogicalConstr(
                        dec, 0, (lhs_i <= rev_rhs_i) if rev_rel == '<' else (lhs_i >= rev_rhs_i), data_idx=i,
                    )
            self._constr_cache.add(constr_name)

    def addIntNodeIndicatorConstr(
            self,
            dec: int,
            node_id: int,
            low: Union[int, sympy.Basic],
            high: Union[int, sympy.Basic],
            size: Optional[int] = None,
            *args,
            **kwargs,
    ):
        skip_low, skip_high = False, False
        if low == sympy.oo or low == -sympy.oo:
            skip_low = True
        if high == sympy.oo or high == -sympy.oo:
            skip_high = True
        constr_name = f'GC_(icvar_{node_id})_eq'
        check = constr_name in self._constr_cache

        if not check:
            if size is None:
                par_val = self.getVarByName(f'icvar_{node_id}', binary=False)
                if not skip_high:
                    high_val = convert_rhs(self, high, incl_bound=True)
                    self.addLogicalConstr(dec, 1, (par_val == high_val), node_id=node_id)
                if not skip_low:
                    low_val = convert_rhs(self, low, incl_bound=True)
                    self.addLogicalConstr(dec, 0, (par_val == low_val), node_id=node_id)
            else:
                for i in range(size):
                    par_val_i = self.getVarByName(f'icvar_{node_id}__{i}', binary=False)
                    if not skip_high:
                        high_val_i = convert_rhs(self, high, i, True)
                        self.addLogicalConstr(dec, 1, (par_val_i == high_val_i), node_id=node_id, data_idx=i)
                    if not skip_low:
                        low_val_i = convert_rhs(self, low, i, True)
                        self.addLogicalConstr(dec, 0, (par_val_i == low_val_i), node_id=node_id, data_idx=i)
            self._constr_cache.add(constr_name)

    def write(self, filename: str):
        self._masterproblem._model.write(filename=filename.replace('.lp', '_master.lp'))
        self._subproblem._model.write(filename=filename.replace('.lp', '_sub.lp'))


class MasterProblem:
    """
    The master problem class in combinatorial Benders' decomposition.
    All decision variables of this class are binary. They correspond to indicator variables encoding logical
    relationships of XADD.

    As XADD
    """
    def __init__(self, name, wrapper):
        self._model = gp.Model(f'{name}_MP')
        self._model.Params.LazyConstraints = 1
        self._model.Params.PreCrush = 1
        self._model.Params.OutputFlag = 0
        self._model._wrapper = wrapper
        self._var_tupledict = gp.tupledict()
        self._model._vars = self._var_tupledict
        self._init_constr_tupledict = gp.tupledict()
        self._num_added_cuts = 0

    def initialize(self):
        pass

    def optimize(self, callback: Optional[callable] = None):
        status = None
        while status != GRB.INFEASIBLE and status != GRB.TIME_LIMIT:
            self._model.optimize(callback)
            status = self._model.status
        return status

    def setAttr(self, attrname: str, newval, model=True):
        if attrname[0] != '_':
            attrname = '_' + attrname
        if not model:
            setattr(self, attrname, newval)
        else:
            setattr(self._model, attrname, newval)

    def update(self):
        self._model.update()

    def setParam(self, paramname: str, newval: Union[int, str]):
        self._model.setParam(paramname=paramname, newval=newval)

    def getVars(self):
        return self._model.getVars()

    def addVar(self, name):
        v = self._model.addVar(vtype=GRB.BINARY, name=name)

        # When data index is appended to the name of variable
        name_lst = name.split('__')
        dec_id = int(name_lst[0].split('_')[1])
        if len(name_lst) > 1:
            data_idx = int(name_lst[1])
            self._var_tupledict[dec_id, data_idx] = v
        else:
            self._var_tupledict[dec_id, ] = v
        return v

    def getVarByName(self, name):
        name_lst = name.split('__')
        dec_id = int(name_lst[0].split('_')[1])
        if len(name_lst) > 1:
            data_idx = int(name_lst[1])
            return self._var_tupledict.get((dec_id, data_idx), None)
        else:
            return self._var_tupledict.get((dec_id, ), None)

    def addConstr(self, constr: gp.TempConstr, name='', *args, **kwargs):
        c = self._model.addConstr(constr, name=name)
        self._init_constr_tupledict.update({len(self._init_constr_tupledict): c})

    def getConstrs(self):
        return self._model.getConstrs()

    def compute_cb_cut(self, mis: gp.tuplelist, sol):
        """
        Given a tuple of indices denoting constraints that form an MIS,
        build a combinatorial Benders' cut and add to the model.

        Args:
            mis (tuplelist): A set of tuples of indices denoting a minimally infeasible subsystem
            sol (tupledict): A tupledict mapping indices to current integer solution values
        """
        lhs = 0
        for idx in mis:
            idx = idx[:-1]
            val = sol[idx]
            var = self._var_tupledict[idx]
            lhs += var if is_val(val, 0) else (1 - var)
        cut = lhs >= 1
        return cut


class SubProblem:
    def __init__(self, name, wrapper, num_mis: int = 10, num_init_mis: int = 100):
        self._model = gp.Model(f'{name}_SP')
        self._model.setParam('Method', 1)           # Use the dual simplex algorithm
        self._model.Params.FeasibilityTol = 1e-6
        self._model.Params.OutputFlag = 0
        # self._model.Params.IISMethod = 1
        # self._model.setParam('DualReductions', 0)
        self._model._wrapper = wrapper
        self._wrapper = wrapper
        # self._z = self._model.addVar(lb=float('-inf'), ub=float('inf'), name='z')

        self._ub_related_constr = None
        self._var_bounded_tupledict = gp.tupledict()
        self._var_free_tupledict = gp.tupledict()
        self._var_ub = None
        self._bs_basic = None
        self._num_vars = 0
        # self._idx_bounded_vars = np.empty(shape=(0, ), dtype=int)
        # self._idx_free_vars = np.empty(shape=(0,), dtype=int)

        self._ineq_tupledict = gp.tupledict()
        self._eq_tupledict = gp.tupledict()
        self._constr_tupledict = gp.tupledict()     # A cache of non-logical constraints

        self._optimality_constr = None
        self._ub = float('inf')
        self._sol = None
        self.obj = None

        self._full_model = False
        self._num_mis = num_mis
        self._incumbent_sol = None
        self._temp_cand_bin_sol = None
        self._initialized = False

        self._added_eq_constrs = gp.tupledict()
        self._added_ineq_constrs = gp.tupledict()

        self._dec_set = set()
        self._num_init_mis = num_init_mis
        self._num_added_logical_eq_constrs = 0
        self._num_added_logical_ineq_constrs = 0

        # self._idx_cache = np.array([None] * 1000)
        # self._logic_eq_idx = np.zeros(1000, dtype=int)
        # self._logic_ineq_idx = np.zeros(1000, dtype=int)
        # self._basic_eq_idx = np.zeros(1000, dtype=int)
        # self._basic_ineq_idx = np.zeros(1000, dtype=int)
        # self._logic_eq_num = 0
        # self._logic_ineq_num = 0
        # self._basic_eq_num = 0
        # self._basic_ineq_num = 0

        self._rng = np.random.RandomState(0)

    @property
    def num_constrs(self):
        return len(self._constr_tupledict) + len(self._added_eq_constrs) + len(self._added_ineq_constrs)

    @property
    def masterproblem(self):
        return self._wrapper._masterproblem

    def random_initialize_mp_vars(self, all=None):
        vals = gp.tupledict()
        for idx, v in self.masterproblem._var_tupledict.items():
            vals[idx] = all if all is not None else float(self._rng.randint(2))
        return vals

    def initialize(self, callback):
        mp = self.masterproblem
        stime = self._stime = time.time()
        logger.info(f"Initialization begins")
        logger.info(f"Variable types (MP): {len(mp.getVars())}")
        logger.info("Time".center(10) + "|" +
                    "UB".center(20) + "|" + "Cuts added".center(15))

        # Initial feasible solution to MP
        vals = self.random_initialize_mp_vars(all=1)

        # Initially, update the upper bound and get the incumbent (since SP is feasible)
        self.build(vals)
        _ = self.optimize_or_return_mis()

        # Rebuild the SP model and add initial constraints to MP
        for n in range(self._num_init_mis):
            if n != 0:
                mp._model.optimize(callback)
                if mp._model.status == GRB.OPTIMAL:
                    vals = mp._model.getAttr('x', mp._var_tupledict)
                elif mp._model.status == GRB.INFEASIBLE:
                    return
                # vals = self.random_initialize_mp_vars()

            self.build(vals)
            status, mis_lst = self.optimize_or_return_mis(self._num_mis)
            for mis in mis_lst:
                if len(mis) == 0:
                    continue
                init_cut = mp.compute_cb_cut(mis, vals)
                name = 'Init_CB_cut:' + \
                       ''.join([f'{i[0]}{f"{i[1]}" if len(i) > 1 else ""}{vals[i[:-1]]}' for i in mis])
                mp.addConstr(init_cut, name=name[:20])
                mp._num_added_cuts += 1
            mp.update()
        etime = time.time()
        logger.info(f"Initialized: {len(mp.getConstrs())} CB cuts added")
        logger.info(f"Initialization time: {etime - stime}s")

        logger.info("Time".center(10) + "|" +
                    "UB".center(20) + "|" + "Cuts added".center(15))


    @property
    def param_array(self):
        return self._model._param_array

    def setAttr(self, attrname: str, newval, model=True):
        if attrname[0] != '_':
            attrname = '_' + attrname
        if not model:
            setattr(self, attrname, newval)
        else:
            setattr(self._model, attrname, newval)

    @property
    def bs_basic(self):
        if self._bs_basic is None:
            self._bs_basic = self._model.getAttr('RHS', self._constr_tupledict)
            return self._bs_basic
        else:
            return self._bs_basic

    @property
    def model_size(self):
        return len(self._dec_set)

    def update(self):
        self._model.update()

    def setParam(self, paramname: str, newval: Union[int, str]):
        self._model.setParam(paramname=paramname, newval=newval)

    def getVars(self):
        return self._model.getVars()

    def set_objective(self, obj):
        self.obj = obj

    def addVar(self, name, lb=0, ub=float('inf')):
        assert (ub == float('inf') and lb == float('-inf')) or (ub != float('inf') and lb == 0),\
            "Only support bounds that are either (-oo, oo) or (0, ub)!"
        v = self._model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
        if ub == float('inf'):
            self._var_free_tupledict[name] = v
            # self._idx_free_vars = np.hstack([self._idx_free_vars, self._num_vars])
            self._num_vars += 1
        else:
            self._var_bounded_tupledict[name] = v
            # self._idx_bounded_vars = np.hstack([self._idx_bounded_vars, self._num_vars])
            self._num_vars += 1
        return v

    def getVarByName(self, name):
        v = self._var_bounded_tupledict.get(name, None)
        if v is None:
            v = self._var_free_tupledict.get(name, None)
        return v

    def clear(self):
        self._model.remove(self._added_eq_constrs)
        self._model.remove(self._added_ineq_constrs)
        self._added_ineq_constrs.clear()
        self._added_eq_constrs.clear()
        # self._logic_eq_idx.fill(0)
        # self._logic_ineq_idx.fill(0)
        # self._logic_eq_num = 0
        # self._logic_ineq_num = 0
        self._num_added_logical_ineq_constrs = 0
        self._num_added_logical_eq_constrs = 0
        # if self._ub_related_constr is not None:
        #     self._model.remove(self._ub_related_constr)
        #     self._ub_related_constr = None

    def addConstr(self, constr: gp.TempConstr, name='', *args, **kwargs):
        c = self._model.addConstr(constr, name=name)
        # if self._basic_eq_num >= len(self._basic_eq_idx) - 5:
        #     self._basic_eq_idx = np.concatenate((self._basic_eq_idx, np.zeros(1000, dtype=int)))
        # if self._basic_ineq_num >= len(self._basic_ineq_idx) - 5:
        #     self._basic_ineq_idx = np.concatenate((self._basic_ineq_idx, np.zeros(1000, dtype=int)))

        # if constr._sense == '=':
        #     self._basic_eq_idx[self._basic_eq_num] = self.num_constrs
        #     self._basic_eq_num += 1
        # else:
        #     self._basic_ineq_idx[self._basic_ineq_num] = self.num_constrs
        #     self._basic_ineq_num += 1
        self._constr_tupledict.update({len(self._constr_tupledict): c})

    def addLogicalConstr(
            self, dec: int, indicator: int, constr: gp.TempConstr, data_idx: int = None, node_id: int = None,
    ):
        """
        Adds a logical constraint.
        Note that all inequality constraints are added in 'Ax >= b' form.

        Args:

        """
        name = f'GC_(ind_{dec})_({"True" if indicator == 1 else "False"})_'
        if data_idx is None:
            self._dec_set.add(dec)
            constr_id = (dec, indicator)
        else:
            self._dec_set.add((dec, data_idx))
            constr_id = (dec, data_idx, indicator)
        sense = constr._sense
        if sense == '=':
            assert node_id is not None
            constr_id += (node_id, )

            name += f'eq_(icvar_{node_id})'
            name += f'__{data_idx}' if data_idx is not None else ''
            self._eq_tupledict[constr_id] = (constr, name)
        elif sense == '<' or sense == '>':
            name += f'ineq'
            name += f'__{data_idx}' if data_idx is not None else ''
            if sense == '<':
                constr = constr._lhs * -1 >= -1 * constr._rhs
            self._ineq_tupledict[constr_id] = (constr, name)

    def init(self):
        if not self._initialized:
            # for v in self.getVars():
            #     self._model.setAttr('IISConstrForce', v, 0)
            self._model.setAttr('IISConstrForce', self._constr_tupledict.values(), 0)
            self._initialized = True

    def update_ub_and_optimality_constr(self, new_ub):
        """Add the optimality constraint in `a.T x >= b' form"""

        # When added for the first time, appended to the tupledict at the end
        if self._ub == float('inf'):
            self._ub = new_ub
            c_ = - self.obj >= - self._ub + OptimalityEpsilon      # obj <= UB - epsilon, or -obj >= - UB + epsilon
            c = self._model.addConstr(c_, name='optimality')
            # self._basic_ineq_idx[self._basic_ineq_num] = len(self._constr_tupledict)
            # self._basic_ineq_num += 1
            self._model.update()
            self._constr_tupledict.update({len(self._constr_tupledict): c})
            self.bs_basic.update({len(self.bs_basic): c.RHS})
            self._optimality_constr = c
            self._model.setAttr('IISConstrForce', c, 0)
        else:
            # diff = self._ub - new_ub
            c = self._optimality_constr #self._constr_tupledict[len(self._constr_tupledict)]
            orig_rhs = self._model.getAttr('RHS', [c])[0]
            # new_rhs = orig_rhs + diff
            self._model.setAttr('RHS', c, - new_ub + OptimalityEpsilon)
            self._model.update()
            self.bs_basic.update({len(self.bs_basic) - 1: c.RHS})
            self._ub = new_ub

    def build(self, vals):
        """
        Given a tupledict of binary variables to their values (only integers),
        select the corresponding linear constraints and add them to the subproblem model.
        Additionally, the optimality constraint is also added.

        Args:
             vals (gp.tupledict): keys are (dec, ) or (dec, data_idx, )... and
                                    values are the values of the corresponding binary variables (relaxed)
        """
        # Firstly, remove all existing logical constraints and the upper bound related constraint (if exists)
        self.clear()
        self.init()

        if len(vals) == self.model_size:
            self._full_model = True
            self._temp_cand_bin_sol = vals
        else:
            self._full_model = False

        # Then, revert back the constraints independent of the binary variables to the original constraints
        # self._model.setAttr('RHS', self._constr_tupledict, self.bs_basic)
        # for i, ci in self._constr_tupledict.items():
        #     self._model.chgCoeff(ci, self._z, 0)

        # Now, add the activated constraints specified by the integer solutions in `vals'
        for idx, val in vals.items():
            # while self._logic_ineq_num >= len(self._logic_ineq_idx) - 5000:
            #     self._logic_ineq_idx = np.concatenate((self._logic_ineq_idx, np.zeros(5000, dtype=int)))
            # while self._logic_eq_num >= len(self._logic_eq_idx) - 5000:
            #     self._logic_eq_idx = np.concatenate((self._logic_eq_idx, np.zeros(5000, dtype=int)))
            # while self.num_constrs >= len(self._idx_cache) - 5000:
            #     self._idx_cache = np.concatenate((self._idx_cache, np.array([None] * 5000)))

            if val > 0.5:
                idx += (1, )
                # Multiple equality constraints can exist per decision
                j = 0
                for ci, namei in self._eq_tupledict.select(*idx, '*'):
                    constr = self._model.addConstr(ci, name=namei)
                    # self._logic_eq_idx[self._logic_eq_num] = self.num_constrs   #idx + (j,)
                    # self._idx_cache[self.num_constrs] = idx + (j, )
                    self._added_eq_constrs.update({idx + (j, ): constr})
                    self._num_added_logical_eq_constrs += 1
                    # self._logic_eq_num += 1
                    j += 1

                # Add a single inequality constraint
                constr = self._model.addConstr(self._ineq_tupledict[idx][0], name=self._ineq_tupledict[idx][1])
                # self._logic_ineq_idx[self._logic_ineq_num] = self.num_constrs  # idx + (j,)
                # self._idx_cache[self.num_constrs] = idx
                self._added_ineq_constrs.update({idx: constr})
                self._num_added_logical_ineq_constrs += 1
                # self._logic_ineq_num += 1
            else:
                idx += (0, )
                # Multiple equality constraints can exist per decision
                j = 0
                for ci, namei in self._eq_tupledict.select(*idx, '*'):
                    constr = self._model.addConstr(ci, name=namei)
                    # self._logic_eq_idx[self._logic_eq_num] = self.num_constrs  # idx + (j,)
                    # self._idx_cache[self.num_constrs] = idx + (j,)
                    self._added_eq_constrs.update({idx + (j, ): constr})
                    self._num_added_logical_eq_constrs += 1
                    # self._logic_eq_num += 1
                    j += 1

                # Add a single inequality constraint
                constr = self._model.addConstr(self._ineq_tupledict[idx][0], name=self._ineq_tupledict[idx][1])
                # self._logic_ineq_idx[self._logic_ineq_num] = self.num_constrs  # idx + (j,)
                # self._idx_cache[self.num_constrs] = idx
                self._added_ineq_constrs.update({idx: constr})
                self._num_added_logical_ineq_constrs += 1
                # self._logic_ineq_num += 1

        self._model.setObjective(self.obj)

    def optimize_or_return_mis(self, num_mis=None):
        self._model.optimize()
        status = self._model.status
        mis_lst = []

        # When an optimal solution is found for the subproblem, update the incumbent
        if status == GRB.OPTIMAL and self._full_model:
            obj = self._model.getAttr('objVal')
            mp = self._wrapper._masterproblem
            time_lapsed = time.time() - self._stime

            assert obj <= self._ub + FeasibilityTol
            self._incumbent_sol = (self._temp_cand_bin_sol,
                                   self._model.getAttr('x', self._var_bounded_tupledict),
                                   self._model.getAttr('x', self._var_free_tupledict))
            self.update_ub_and_optimality_constr(obj)
            logger.info(f"{time_lapsed:.7f}".center(10) + "|" + f"{self._ub:.15f}".center(20) +
                        "|" + f"{mp._num_added_cuts}".center(15))

        # When the subsystem is infeasible, compute the MIS
        elif status == GRB.INFEASIBLE or GRB.INF_OR_UNBD:
            mis_lst = self.compute_mis(num=self._num_mis if num_mis is None else num_mis, dual=True)

        # Since we know the primal is not unbounded...
        # elif status == GRB.INF_OR_UNBD:
            # The primal should be infeasible not unbounded...
            # self._model.setParam('DualReductions', 0)
            # self._model.optimize()
            # mis_lst = self.compute_mis(num=self._num_mis if num_mis is None else num_mis)  # , dual=True)
            # # Directly use the dual to compute MIS
            # if self._model.status == GRB.INFEASIBLE:
            #
            # else:
            #     raise ValueError
            # self._model.setParam('DualReductions', 1)
        return status, mis_lst

    # def compute_mis_via_dual(self, weight, num):
    #     dual = gp.Model('Dual')
    #     dual.Params.OutputFlag = 0
    #     dual.Params.Method = 1
    #     dual_sol_eq, dual_sol_ineq = None, None
    #     coeff_matrix = self._model.getA()
    #     rhs_vector = np.array(self._model.getAttr('RHS', self._model.getConstrs()), dtype=np.float64)
    #     logic_eq_idx = self._logic_eq_idx[:self._logic_eq_num]
    #     logic_ineq_idx = self._logic_ineq_idx[:self._logic_ineq_num]
    #     basic_eq_idx = self._basic_eq_idx[:self._basic_eq_num]
    #     basic_ineq_idx = self._basic_ineq_idx[:self._basic_ineq_num]
    #
    #     A = coeff_matrix[logic_ineq_idx][:, self._idx_bounded_vars]
    #     A_ = coeff_matrix[basic_ineq_idx][:, self._idx_bounded_vars]
    #     B = coeff_matrix[logic_ineq_idx][:, self._idx_free_vars]
    #     B_ = coeff_matrix[basic_ineq_idx][:, self._idx_free_vars]
    #     C = coeff_matrix[logic_eq_idx][:, self._idx_bounded_vars]
    #     C_ = coeff_matrix[basic_eq_idx][:, self._idx_bounded_vars]
    #     D = coeff_matrix[logic_eq_idx][:, self._idx_free_vars]
    #     D_ = coeff_matrix[basic_eq_idx][:, self._idx_free_vars]
    #
    #     #
    #     # A, C, A_, C_ = tuple(
    #     #     map(lambda x: coeff_matrix[x], (logic_ineq_idx, logic_eq_idx, basic_ineq_idx, basic_eq_idx))
    #     # )
    #     b, d, b_, d_ = tuple(
    #         map(lambda x: rhs_vector[x], (logic_ineq_idx, logic_eq_idx, basic_ineq_idx, basic_eq_idx))
    #     )
    #     if self._var_ub is None:
    #         self._var_ub = np.array(self._model.getAttr('UB', self._var_bounded_tupledict).values()).reshape(-1, 1)
    #     u = self._var_ub
    #     U = sparse.identity(n=self._var_ub.size)
    #
    #     p = dual.addMVar(shape=A.shape[0], vtype=GRB.CONTINUOUS, name='p')
    #     p_ = dual.addMVar(shape=A_.shape[0], vtype=GRB.CONTINUOUS, name='p_')
    #     q = dual.addMVar(shape=C.shape[0], vtype=GRB.CONTINUOUS, lb=float('-inf'), name='q')
    #     q_ = dual.addMVar(shape=C_.shape[0], vtype=GRB.CONTINUOUS, lb=float('-inf'), name='q_')
    #     r_ = dual.addMVar(shape=self._var_ub.size, vtype=GRB.CONTINUOUS, name='r_')
    #
    #     num_vars = sum(map(lambda x: x.shape[0], [p, p_, q, q_, r_]))
    #
    #     # p.T A + q.T C - r.T U <= 0
    #     pqr = gp.MVar(gp.tuplelist(p) + gp.tuplelist(p_) + gp.tuplelist(q) + gp.tuplelist(q_) + gp.tuplelist(r_))
    #     pq = pqr[:-r_.shape[0]]
    #     bd_u = np.concatenate([np.array(gp.tuplelist(b) + gp.tuplelist(b_) + gp.tuplelist(d) + gp.tuplelist(d_)),
    #                            -u.flatten()]).reshape(-1, 1)
    #     ACU = hstack(list(map(lambda x: x.transpose(), [A, A_, C, C_, U])))
    #     BD = hstack(list(map(lambda x: x.transpose(), [B, B_, D, D_])))
    #     dual.addMConstrs(ACU, pqr, '<', np.zeros(ACU.shape[0]), name='bounded_var_related_constr')
    #     dual.addMConstrs(BD, pq, '=', np.zeros(BD.shape[0]), name='free_var_related_constr')
    #     dual.addConstr(bd_u.T @ pqr == num_vars * 10, name='bounded_dual')
    #     # dual.addConstr(vars_vec @ bd__ == num_vars, name='bounded')     # As in Parker and Ryan (1994)
    #
    #     mis_set = set()
    #     for i in range(num):
    #         w1 = weight[i, :p.shape[0]]
    #         w2 = weight[i, p.shape[0]:]
    #         obj = p @ w1 + q @ w2
    #         dual.setObjective(obj, sense=GRB.MAXIMIZE)
    #
    #         dual.optimize()
    #         status = dual.status
    #
    #         if status == GRB.UNBOUNDED or status == GRB.INF_OR_UNBD:
    #             dual.setObjective(0)
    #             dual.optimize()
    #             # break
    #         ineq_val = check_nonzero(p.x).keys()
    #         eq_val = check_nonzero(q.x).keys()
    #         dual_support_eq = self._idx_cache[logic_eq_idx[eq_val]]
    #         dual_support_ineq = self._idx_cache[logic_ineq_idx[ineq_val]]
    #         mis_eq = {k[:-1] for k in dual_support_eq}
    #         mis_ineq = set(dual_support_ineq)
    #         mis_eq.update(mis_ineq)
    #         mis_set.add(frozenset(mis_eq))
    #     return list(mis_set), status

    def compute_mis(self, num: int = 10, dual=False):
        """
        Given the indices of activated linear constraints, compute the minimally infeasible subsystem by solving
        the dual of the system.  Return `num' number of such systems.

        For example, consider the following infeasible linear system:
        S = {x:     Ax >= b },  then the alternative polyhedron is  P = { y, u, v:  y.T A + u.T C = 0
                    Cx = d                                                          y.T b + u.T d = 1
                                                                                    y >= 0, u free    }
        Due to Gleeson and Ryan (1990) and its generalization in Parker and Ryan (1994),
        we know that the supports (nonzero elements) of the vertices of P have the 1-to-1 correspondence with
        the MIS of S.  Furthermore, an weighted version is possible by considering the following problem:
            max     w1.T y + w2.T u
            s.t.    y.T A + u.T C = 0
                    y.T b + u.T d = 1
                    y >= 0
                    w free
        where w1 and w2 are the weights given to each and every constraint from A and C, respectively.
        In our case, we give 0 weights to the constraints that are independent of the indicator variables from the MP.

        Instead of solving this dual formulation explicitly, we can modify the original problem specified in self._model
        such that it corresponds to the dual of the above problem.

        That is, we solve
            min     z
            s.t.    [A | b] x_ >= w1
                    [C | d] x_ = w2
                    z free
            where x_ = (x.T, z).T with z \in \mathbb{R}.

        Args:
            num (int)
        """
        w = self._rng.binomial(1, 0.5, num * (self.num_constrs - len(self._constr_tupledict))).reshape(num, -1)

        """Set up the dual to compute MIS"""
        mis_lst = []
        # if dual:
        #     mis_lst, status = self.compute_mis_via_dual(w, num)

        """Use Gurobi internal function to compute MIS (or IIS)"""
        mis_set = set()
        try:
            self._model.computeIIS()
            mis_eq = check_nonzero(self._model.getAttr('IISConstr', self._added_eq_constrs))
            mis_ineq = check_nonzero(self._model.getAttr('IISConstr', self._added_ineq_constrs))
            mis_ineq = set(mis_ineq.keys())
            mis_ineq.update({k[:-1] for k in mis_eq.keys()})
            mis_set.add(frozenset(mis_ineq))
            mis_lst = list(mis_set)
            return mis_lst
        except gp.GurobiError as e:
            logger.error(e)
            return mis_lst

        #
        # mis_eq = check_nonzero(dual_sol_eq)  # index: (dec, data_idx, 0 or 1, counter)
        # mis_ineq = check_nonzero(dual_sol_ineq)  # index: (dec, data_idx, 0 or 1)
        #
        # # The support (nonzero elements) of the dual corresponds to an MIS
        # mis_ineq = set(mis_ineq.keys())
        # mis_ineq.update({k[:-1] for k in mis_eq.keys()})
        # mis_set.add(frozenset(mis_ineq))

        # if len(mis_lst) == 0:
        #     mis_lst, status = self.compute_mis_via_primal(w, num)
        # if len(mis_lst) >= 1:
        #     return gp.tuplelist(mis_lst)
        # else:
        #     return []

    # def compute_mis_via_primal(self, w, num):
    #     mis_set = set()
    #     for i in range(num):
    #         """Modify the primal instead of solving the dual directly"""
    #         # Get the RHS of the original logical constraints
    #         # Note: RHS of non-linking constraints can be accessed via self.bs_basic
    #         bs_logic_eq = self._model.getAttr('RHS', self._added_eq_constrs)
    #         bs_logic_ineq = self._model.getAttr('RHS', self._added_ineq_constrs)
    #
    #         # Make the RHS of the logical constraints to be the weight values
    #         # 0 for non-linking constraints
    #         self._model.setAttr('RHS', self._added_eq_constrs.values(), w[i, :len(bs_logic_eq)])
    #         self._model.setAttr('RHS', self._added_ineq_constrs.values(), w[i, len(bs_logic_eq):])
    #         self._model.setAttr('RHS', self._constr_tupledict, 0)
    #
    #         # Add the constraint related to upper bounds on some variables
    #         self.add_ub_related_constr()
    #
    #         # Modify the constraints to [A | b] [x ... z] >= w form
    #         # Note the order of bs values and constraints should be matched exactly
    #         for bi, ci in zip(bs_logic_eq.values(), self._added_eq_constrs.values()):
    #             self._model.chgCoeff(ci, self._z, bi)
    #         for bi, ci in zip(bs_logic_ineq.values(), self._added_ineq_constrs.values()):
    #             self._model.chgCoeff(ci, self._z, bi)
    #         for bi, ci in zip(self.bs_basic.values(), self._constr_tupledict.values()):
    #             self._model.chgCoeff(ci, self._z, bi)
    #
    #         # The objective becomes `z' (Note: the scale of objective may matter due to numerical reasons)
    #         self._model.setObjective(self._z * self.num_constrs)
    #
    #         # Optimize the primal
    #         self._model.optimize()
    #         status = self._model.status
    #         """Retrieve the dual solutions and compute MIS"""
    #         if status == GRB.Status.INF_OR_UNBD:    # Can't get dual solutions in this case
    #             continue
    #         dual_sol_eq = self._model.getAttr('Pi', self._added_eq_constrs)
    #         dual_sol_ineq = self._model.getAttr('Pi', self._added_ineq_constrs)
    #         mis_eq = check_nonzero(dual_sol_eq)  # index: (dec, data_idx, 0 or 1, counter)
    #         mis_ineq = check_nonzero(dual_sol_ineq)  # index: (dec, data_idx, 0 or 1)
    #
    #         # The support (nonzero elements) of the dual corresponds to an MIS
    #         mis_ineq = set(mis_ineq.keys())
    #         mis_ineq.update({k[:-1] for k in mis_eq.keys()})
    #         mis_set.add(frozenset(mis_ineq))
    #     mis = gp.tuplelist(mis_set)
    #     return mis, status
    #
    # def add_ub_related_constr(self):
    #     # Save below since frequently called
    #     if self._var_ub is None:
    #         self._var_ub = np.array(self._model.getAttr('UB', self._var_bounded_tupledict).values()).reshape(-1, 1)
    #         self._bounded_vars = gp.MVar(self._var_bounded_tupledict.values())
    #
    #     # Sparse identity matrix
    #     I = sparse.identity(n=sel£f._bounded_vars.shape[0])
    #
    #     # Add the UB related constraint and save its pointer (note that this one is directly added to the model)
    #     self._ub_related_constr = self._model.addConstr(I @ self._bounded_vars + self._var_ub @ gp.MVar(self._z) <= 0)


def setup_gurobi_model(model_name, g_model: GurobiModel = None, scheme: str = "bruteforce", **kwargs):
    assert scheme == 'bruteforce' or scheme == 'benders'
    try:
        if g_model is None:
            g_model = BFGurobiModel(model_name, **kwargs) if scheme == 'bruteforce' \
                else CBGurobiModel(model_name, **kwargs)
        else:
            g_model.set_model_name(model_name)
            g_model.update()
        g_model.setParam('OutputFlag', 0)
        g_model.setAttr('_best_obj', float('inf'))
        g_model.setAttr('_time_log', 0)
        return g_model
    except gp.GurobiError as e:
        logger.error('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        logger.error('Encountered an attribute error')


def convert_rhs(model, expr_or_node_id, data_idx=None, incl_bound=True):
    if isinstance(expr_or_node_id, sympy.core.numbers.Number):
        return float(expr_or_node_id)
    elif isinstance(expr_or_node_id, int):
        return model.getVarByName(f'icvar_{expr_or_node_id}') if data_idx is None else \
                model.getVarByName(f'icvar_{expr_or_node_id}__{data_idx}')
    else:
        return convert2GurobiExpr(expr_or_node_id, model, data_idx, incl_bound=incl_bound)


def callback_cbcuts(model, where):
    """
    Implements the callback for combinatorial Benders'.
    :param model:
    :param where:
    :return:
    """
    mis_lst = None
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        subproblem = model._wrapper._subproblem

        # Set up the subproblem with activated constraints according to the current MIP solution
        subproblem.build(vals)

        # Solve the SP
        status, mis_lst = subproblem.optimize_or_return_mis()
        if mis_lst is not None:
            # Add CB cuts as lazy constraints!
            masterproblem = model._wrapper._masterproblem
            for mis in mis_lst:
                cut = masterproblem.compute_cb_cut(mis, vals)
                model.cbLazy(cut)
                masterproblem._num_added_cuts += 1

    elif where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        # When an optimal of the relaxed problem is solved
        if status == GRB.Status.OPTIMAL and node_count < 50:
            subproblem = model._wrapper._subproblem

            # Check for the integrality of the solution
            vals = model.cbGetNodeRel(model._vars)
            vals = check_integrality(vals)

            subproblem.build(vals)
            status, mis_lst = subproblem.optimize_or_return_mis()

            if mis_lst is not None:
                # Add CB cuts as lazy constraints!
                masterproblem = model._wrapper._masterproblem
                for mis in mis_lst:
                    cut = masterproblem.compute_cb_cut(mis, vals)
                    model.cbCut(cut)
                    masterproblem._num_added_cuts += 1




def is_val(x, val):
    """Checks whether `x' equals to `val' up to the integrality tolerance"""
    return val - IntFeasTol <= x <= val + IntFeasTol


def check_nonzero(arr):
    if isinstance(arr, gp.tupledict):
        return gp.tupledict({k: v for k, v in arr.items() if not is_val(v, 0)})
    else:
        return gp.tupledict({i: v for i, v in enumerate(arr) if not is_val(v, 0)})


def check_integrality(td: gp.tupledict, val=None):
    """
    Given a tupledict object whose values are Gurobi variables,
    check whether they are integers.
    If `val == 0' or `val == 1', then return the indices of the variables whose value matches;
    otherwise (`val' is None), return all indices of integer variables.

    Args:
        td (tupledict):
        val (optional, int):
    """
    res = gp.tupledict()
    for key, v in td.items():
        if val is None:
            zero, one = is_val(v, 0), is_val(v, 1)
            if zero or one:
                res[key] = v
        else:
            if is_val(v, val):
                res[key] = v
    return res


def callback_spo(extracb=None):
    """
    Returns the base callback function to be used within the SPO framework.
    If `extracb' is specified, it is also called upon.

    Args:
         extracb (callback): An extra callback function to be called along with the base callback
    """
    def base_callback(model, where):
        parameter_dir = path.join(path.curdir, 'experiments', 'parameters')
        os.makedirs(parameter_dir, exist_ok=True)

        def log_memory_usage():
            used_memory = psutil.virtual_memory().used / (1024.**3)
            logger.info(f'Used Memory: {round(used_memory, 2)}')

        if where == GRB.Callback.MIPSOL:
            time = model.cbGet(GRB.Callback.RUNTIME)
            best = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            curr_opt_param = model.cbGetSolution(model._param_array.ravel())
            model._curr_opt_param = curr_opt_param
            model._best_obj = best
            model._best_bound = bound
            if time - model._time_log > model._time_interval:
                model._time_log = time
                row, col = model._param_array.shape
                f_direc = path.join(parameter_dir, f'{model.ModelName}_param_array_runtime_{int(time)}_obj_{best:.4f}'
                                                   f'_bound_{bound:.4f}.npy')
                np.save(f_direc, np.array(curr_opt_param).reshape(row, col))
                logger.info(f'Runtime: {time}\tBest objective: {best}\tBound: {bound}')
                log_memory_usage()

        elif where == GRB.Callback.MIP:
            time = model.cbGet(GRB.Callback.RUNTIME)
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            if time - model._time_log > model._time_interval and best < GRB.INFINITY:
                model._time_log = time
                best = model._best_obj
                bound = model._best_bound
                row, col = model._param_array.shape
                curr_opt_param = model._curr_opt_param
                f_direc = path.join(parameter_dir, f'{model.ModelName}_param_array_runtime_{int(time)}_obj_{best:.4f}'
                                                   f'_bound_{bound:.4f}.npy')
                np.save(f_direc, np.array(curr_opt_param).reshape(row, col))
                logger.info(f'Runtime: {time}\tBest objective: {best}\tBound: {bound}')
                log_memory_usage()

        # Call the extra callback function if specified
        try:
            extracb(model, where)
        except TypeError:       # No extra callback function passed
            pass
    return base_callback


def callback(extracb=None):
    def base_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            time = model.cbGet(GRB.Callback.RUNTIME)
            best = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            model._best_obj = best
            model._best_bound = bound
            if time - model._time_log > model._time_interval:
                model._time_log = time
                logger.info(f'Runtime: {time}\tBest objective: {best}\tBound: {bound}')

        elif where == GRB.Callback.MIP:
            time = model.cbGet(GRB.Callback.RUNTIME)
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            if time - model._time_log > model._time_interval and best < GRB.INFINITY:
                model._time_log = time
                best = model._best_obj
                bound = model._best_bound
                logger.info(f'Runtime: {time}\tBest objective: {best}\tBound: {bound}')

        try:
            extracb(model, where)
        except TypeError:       # No extra callback provided
            pass
    return base_callback
