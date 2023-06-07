"""
Defines helper functions for compiling an XADD into a MILP. 
Note that it is required to have Gurobi installed in order to run these.
"""
from typing import Union, cast

import pulp as pl
import sympy as sp

from xaddpy.utils.global_vars import EPSILON
from xaddpy.utils.lp_util import (GurobiModel, convert_to_pulp_expr,
                                  set_equality_constraint)
from xaddpy.xadd import xadd
from xaddpy.xadd.node import XADDINode, XADDTNode


def build_milp(
        context: xadd.XADD,
        node_id: int,
        m: GurobiModel,
        dec_partition: list,
        lb: float = float('-inf'),
        ub: float = float('inf'),
        binary: bool = False,
) -> Union[int, sp.Basic]:
    """
    Given an XADD, encode it as MILP.

    Args:
        context (XADD)
        node_id (int)
        m(GurobiModel)
        dec_partition (list)
        lb (float)
        ub (float)
        binary (bool)
    """
    node = context.get_exist_node(node_id)

    if node.is_leaf():
        node = cast(XADDTNode, node)
        expr = node.expr

        # Handle infeasibility
        if expr == sp.oo or expr == -sp.oo:
            handle_infeasibility(m, dec_partition)
        return expr
    
    node = cast(XADDINode, node)
    dec = node.dec
    expr = context._id_to_expr[dec]
    assert expr.rhs == 0, "RHS should always be 0 for a canonical XADD"

    # Add variables associated with decision expressions (indicator) and nodes
    indicator_var_name = f"ind_{dec}"
    indicator = m.getVarByName(indicator_var_name)     # Reuse ind_dec var
    if indicator is None:
        m.addVar(vtype=pl.LpBinary, name=indicator_var_name)
    
    # Each unique node is associated with one continuous variable
    icvar_node_name = f"icvar_{node_id}"
    cvar_node = m.getVarByName(icvar_node_name)
    if cvar_node is None:
        m.addVar(lb=lb, ub=ub, vtype=pl.LpContinuous, name=icvar_node_name)
    
    # Recursively build the MILP model: low branch -> high branch
    dec_partition.append(-dec)
    low = build_milp(context,
                     node.low,
                     m,
                     dec_partition,
                     lb=lb, ub=ub, binary=binary)
    dec_partition.remove(-dec)

    dec_partition.append(dec)
    high = build_milp(context,
                      node.high,
                      m,
                      dec_partition,
                      lb=lb, ub=ub, binary=binary)
    dec_partition.remove(dec)

    # Check for duplicated constraints and add
    m.addDecNodeIndicatorConstr(dec, expr)
    m.addIntNodeIndicatorConstr(dec, node_id, low=low, high=high)

    return node_id
    

def handle_infeasibility(
        m: GurobiModel,
        dec_partition: list,
):
    """
    Enforces a set of constraints such that the infeasible partition is accounted for.
    For example, suppose we have `dec_partition=[d1, d2, d3]` whose function 
    value is \infty. Then, it implies that `d1 and d2 => not d3` because if d1, d2, 
    and d3 are all true, that leads to infeasibility.

    Encoding such a logical condition can be done by defining additional 
    binary variables. In the previous example, if `i1, i2, i3` are the associated 
    binary variables, respectively, then we enforce
        (1 - i1) + (1 - i2) >= i3
        Note that if (i1, i2) = (1, 1), then i3 = 0 should hold
        otherwise, i3 can have either 0 or 1... not constrained.
    On the other hand, if `dec_partition=[d1, not d2, d3]`, then
        (1 - i1) + i2 >= i3
        e.g., (i1, i2) = (1, 0) => i3 = 0
    and so on and so forth...

    Args:
          m GurobiModel)
          dec_partition (list)  The list containing decisions (conditionals) 
          that have been encountered until reaching the current infeasible partition
    """
    # Decisions and their indicator variables up until the last decision
    ind_vars = [None] * len(dec_partition)
    for i, dec in enumerate(dec_partition):
        ind_v = m.getVarByName(f"ind_{dec if dec > 0 else -dec}")
        assert ind_v is not None, "Binary variables associated with parent nodes\
            should have been defined already!"
        ind_vars[i] = ind_v
    
    # Create the constraint
    try:
        constr_name = f"Feasibility_({'_'.join(map(lambda x: x.replace('-', 'n'), map(str, dec_partition[:-1])))})" \
                      f"_impl_({str(-dec_partition[-1]).replace('-', 'n')})"
    except IndexError:
        raise RuntimeError
    constr = m.get_constraint_by_name(constr_name)
    if constr is None:
        if len(dec_partition) == 1:
            m.addConstr(ind_vars[0] == 0 if dec_partition[0] > 0 else ind_vars[0] == 1,
                        name=constr_name)
        else:
            lhs, rhs = 0, 0
            for i, (v, dec) in enumerate(zip(ind_vars, dec_partition)):
                if dec < 0:
                    if i == len(dec_partition) - 1:
                        rhs += 1
                    lhs += v
                else:
                    if i < len(dec_partition) - 1:
                        rhs -= 1
                    lhs -= v
            m.addConstr(lhs >= rhs, name=constr_name)


def check_redundancy_and_add_constraint(
        m,
        constr_cache,
        dec,
        boolean,
        *args, **kwargs
):
    assert isinstance(m, GurobiModel)
    dec_id = kwargs.get('dec_id', None)
    assert dec_id is not None
    num_data = kwargs.get('num_data', None)     # Used for EMSPO
    ind_var_name = f'ind_{dec}'

    # Indicator = 1 or 0 -> Equality constraint
    if len(args) == 2:
        lhs, rhs = args
        constr_name = f'GC_({ind_var_name})_({boolean})_({lhs})_eq_({rhs})'
        check = constr_name in constr_cache
        if not check:
            if num_data is None:
                indicator = m.getVarByName(ind_var_name)
                lhs = m.getVarByName(f'icvar_{lhs}')
                if isinstance(rhs, sp.core.numbers.Number):
                    rhs = float(rhs)
                elif isinstance(rhs, int):
                    rhs = m.getVarByName(f'icvar_{rhs}')
                else:
                    rhs = convert_to_pulp_expr(rhs, m)
                m.addGenConstrIndicator(
                    indicator, boolean, lhs == rhs,
                    name=f"GC_({ind_var_name})_({boolean})_({lhs})_eq_({rhs})")
            else:
                for i in range(num_data):
                    indicator = m.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = m.getVarByName(f'icvar_{lhs}__{i}')
                    if isinstance(rhs, sp.core.numbers.Number):
                        rhs_i = float(rhs)
                    elif isinstance(rhs, int):
                        rhs_i = m.getVarByName(f'icvar_{rhs}__{i}')
                    else:
                        rhs_i = convert_to_pulp_expr(rhs, m, data_idx=i)
                    m.addGenConstrIndicator(
                        indicator, boolean, lhs_i == rhs_i,
                        name=f"GC_({ind_var_name})_({boolean})_({lhs})_eq_({rhs})__{i}")
            constr_cache.add(constr_name)

    # Indicator = 1 or 0 -> inequality constraint (for decision nodes)
    elif len(args) == 3:
        lhs, rel, rhs = args
        constr_name = f'GC_({ind_var_name})_({boolean})_({dec_id})'

        check = constr_name in constr_cache
        if not check:
            # To handle the case when the decision holds in equality... slightly perturb
            epsilon = kwargs.get('epsilon', EPSILON)
            rhs = rhs - epsilon if rel == '<' else rhs + epsilon

            if num_data is None:
                indicator = m.getVarByName(ind_var_name)
                lhs = convert_to_pulp_expr(lhs, m)
                m.addGenConstrIndicator(indicator, boolean, lhs, rel, rhs, name=constr_name)
            else:
                for i in range(num_data):
                    indicator = m.getVarByName(f'{ind_var_name}__{i}')
                    lhs_i = convert_to_pulp_expr(lhs, m, i)
                    m.addGenConstrIndicator(indicator, boolean, lhs_i, rel, rhs, name=f'{constr_name}__{i}')
            constr_cache.add(constr_name)
