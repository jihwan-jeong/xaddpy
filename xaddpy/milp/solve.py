"""
Functions that are used to solve some problems using the MILP encoding.
"""

import os
import os.path as path
import time
from typing import Optional, Tuple

import pulp as pl

from xaddpy.milp.build import build_milp
from xaddpy.milp.util import callback
from xaddpy.utils.logger import logger
from xaddpy.utils.lp_util import (GRB_SOL_STATUS_TO_PULP_STATUS, GurobiModel,
                                  Model, convert_to_pulp_expr,
                                  set_equality_constraint)
from xaddpy.xadd import xadd


def solve_milp(
        context: xadd.XADD,
        m: Model,
        dec_vars: list,
        eq_constr_dict,
        verbose: bool = False,
        timeout = None,
        time_interval: int = 0,
        args = None,
) -> dict:
    # Compile the MILP starting from the root node
    stime = time.time()
    _build_milp(dec_vars,
               context,
               m,
               eq_constr_dict)
    etime = time.time()
    time_modeling = etime - stime

    stime = etime

    # Set some parameters & attributes of the MILP model
    m.setAttr('_time_interval', time_interval)
    m.setParam('TimeLimit', timeout)
    if verbose or args.verbose:
        log_file = f"{args.log_dir}/{args.model_name}_{args.date_time}_solver.log"
        logger.info(f"MILP solver outputs will be saved in {log_file}")
        log_dir = path.join(path.curdir, log_file)
        m.setParam('OutputFlag', 1)
        m.setParam('LogFile', log_dir)
        m.setParam('LogToConsole', 0)
    
    # Optimize the model
    status = m.solve(callback=callback())
    logger.info(f"Done solving the MILP model: status = {status}")
    obj_val = None
    if status == pl.LpStatusOptimal:
        obj_val = m.objVal
        logger.info(f"Objective: {obj_val}")
    elif status == pl.LpStatusInfeasible:
        logger.info("The problem is infeasible!")
    etime = time.time()
    time_milp = etime - stime

    info = {}
    if status != pl.LpStatusInfeasible:
        for v in m.getVars():
            name, val = v.name, v.value()
            assert val is not None
            if verbose or args.verbose:
                logger.info(f"{name} {val}")
            info[name] = val   
    
    info.update(dict(
        time_modeling=time_modeling,
        time_milp=time_milp,
        obj_value=obj_val,
    ))
    return info


def _build_milp(
        dec_vars: list,
        context: xadd.XADD,
        m: Model,
        eq_constr_dict: dict,
        obj = None,
        node_id: Optional[int] = None,
        binary: bool = False,
):
    # Create decision variables
    for v in dec_vars:  
        lb, ub = context._var_to_bound[v]
        is_binary = v in context._bool_var_set
        x = m.addVar(lb=lb, ub=ub,
                     vtype=pl.LpBinary if is_binary else pl.LpContinuous,
                     name=str(v))
        m.sympy_to_pulp[v] = x

    # Handle equality constraints first
    # Then, substitute these variables to eliminate them
    for v, rhs in eq_constr_dict.items():
        x = m.getVarByName(str(v))
        rhs = convert_to_pulp_expr(rhs, m, incl_bound=True)
        set_equality_constraint(x, rhs, m, incl_bound=True)
        assert v not in dec_vars        # TODO: ??
    
    # Build the MILP model
    if context._additive_obj:
        obj = 0
        obj_dict = context.get_objective()
        for i, obj_i_node_id in obj_dict.items():
            obj_i = m.addVar(lb=float('-inf'), ub=float('inf'),
                             vtype=pl.LpContinuous, name=f"f_{i}")
            
            # If the same node already added, add an equality constraint
            if f"icvar_{obj_i_node_id}" in m._name_to_var:
                m.addConstr(obj_i == m.getVarByName(f'icvar_{obj_i_node_id}'),
                            name=f'f_{i}_eq_icvar_{obj_i_node_id}')
            else:
                m._name_to_var[f'icvar_{obj_i_node_id}'] = obj_i
            
            build_milp(context,
                       node_id=obj_i_node_id,
                       m=m,
                       dec_partition=[],
                       binary=False)
            obj += obj_i
        
        # Set the objective
        m.setObjective(obj)
    elif obj is None:
        obj_node_id = context.get_objective()
        assert obj_node_id is not None
        obj = m.addVar(lb=float('-inf'), ub=float('inf'), vtype=pl.LpContinuous, name='obj')
        m._name_to_var[f"icvar_{obj_node_id}"] = obj

        build_milp(context,
                   node_id=obj_node_id,
                   m=m,
                   dec_partition=[],
                   binary=False)
        
        # Set the objective
        m.setObjective(obj)
    else:
        # Objective is already defined, so only encode the given XADD
        assert node_id is not None
        build_milp(context,
                   node_id=node_id,
                   m=m,
                   dec_partition=[],
                   binary=binary)
