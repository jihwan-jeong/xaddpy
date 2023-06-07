import argparse
import json
from typing import Tuple

import gurobipy as gp
import sympy as sp
import sympy.core.relational as relational
from gurobipy import GRB, quicksum

TYPE_CONVERTER = {sp.Integer: int, sp.Float: float, sp.core.numbers.Zero: int, sp.core.numbers.NegativeOne: int,
                 sp.core.numbers.One: int, sp.core.numbers.Rational: float, sp.core.numbers.Half: float,
                 sp.core.numbers.Infinity: float, sp.core.numbers.NegativeInfinity: float}
REL_CONVERTER = {relational.GreaterThan: GRB.GREATER_EQUAL, relational.StrictGreaterThan: GRB.GREATER_EQUAL,
                relational.LessThan: GRB.LESS_EQUAL, relational.StrictLessThan: GRB.LESS_EQUAL,
                relational.Eq: GRB.EQUAL}


def construct_dblp_gurobi_model(args: argparse.Namespace) -> Tuple[gp.Model, dict]:
    m = gp.Model(args.model_name)
    try:
        with open(args.json_file, 'r') as f:
            prob_instance = json.load(f)
    except:
        raise FileNotFoundError(f"File {args.json_file} not found")
    
    # Create Sympy symbols for cvariables and bvariables
    if len(prob_instance['cvariables0']) == 1 and isinstance(prob_instance['cvariables0'][0], int):
        cvariables0 = sp.symbols('x1:%s' % (prob_instance['cvariables0'][0]+1))
    else:
        cvariables0 = []
        for v in prob_instance['cvariables0']:
            cvariables0.append(sp.symbols(v))
    if len(prob_instance['cvariables1']) == 1 and isinstance(prob_instance['cvariables1'][0], int):
        cvariables1 = sp.symbols('y1:%s' % (prob_instance['cvariables1'][0]+1))
    else:
        cvariables1 = []
        for v in prob_instance['cvariables1']:
            cvariables1.append(sp.symbols(v))
    cvariables = cvariables0 + cvariables1
    cvar_dim = len(cvariables)
    ns = {str(v): v for v in cvariables}

    min_vals = prob_instance['min-values']
    max_vals = prob_instance['max-values']

    if len(min_vals) == 1 and len(min_vals) == len(max_vals):
        min_vals = min_vals * cvar_dim
        max_vals = max_vals * cvar_dim
    assert len(min_vals) == len(max_vals) and len(min_vals) == cvar_dim,\
        "Bound information mismatch!"
    
    bound_dict = {}
    for i, (lb, ub) in enumerate(zip(min_vals, max_vals)):
        lb, ub = sp.S(lb), sp.S(ub)
        bound_dict[cvariables[i]] = (float(lb), float(ub))
    
    # Add variables
    sp_to_grb = {}
    for v in cvariables:
        v_grb = m.addVar(lb=bound_dict[v][0], ub=bound_dict[v][1], name=str(v), vtype=GRB.CONTINUOUS)
        sp_to_grb[v] = v_grb
    m.update()

    # Get constraints
    ineq_constrs = []
    eq_constr_dict = {}
    for const in prob_instance['ineq-constr']:
        const = sp.sympify(const)
        lhs, rel, rhs = const.lhs, type(const), const.rhs
        lhs = lhs - rhs
        g_lhs = convert_sp_to_grb_expr(m, lhs, sp_to_grb, bound_dict)
        rel = REL_CONVERTER[rel]
        if rel == GRB.LESS_EQUAL:
            constr = g_lhs <= 0
        elif rel == GRB.GREATER_EQUAL:
            constr = g_lhs >= 0
        else:
            raise ValueError(f"Unsupported relational operator {rel}")
        m.addConstr(constr)
    m.update()

    # Set the objective
    obj = prob_instance['objective']
    obj = sp.expand(sp.sympify(obj, locals=ns))
    obj = convert_sp_to_grb_expr(m, obj, sp_to_grb, bound_dict)
    m.setObjective(obj, sense=GRB.MINIMIZE)

    # Set some parameters
    m.setParam('NonConvex', 2)
    m.setParam('OutputFlag', 1 if args.verbose else 0)

    info_dict = dict(
        num_dec_vars_0=len(cvariables0),
        num_dec_vars_1=len(cvariables1),
        num_constraints=len(m.getConstrs()),
    )
    return m, info_dict


def convert_sp_to_grb_expr(
        m: gp.Model,
        expr: sp.Basic, 
        sp_to_grb: dict, 
        var_to_bound: dict,
        binary: bool = False, 
        incl_bound: bool = True,
):
    if expr in sp_to_grb:
        return sp_to_grb[expr]
    
    # Recursively convert Sympy expression to Gurobi expression
    if isinstance(expr, sp.Number) and not isinstance(expr, sp.core.numbers.NaN):
        return TYPE_CONVERTER[type(expr)](expr)
    elif isinstance(expr, sp.core.numbers.NaN):
        return float('inf')
    elif isinstance(expr, sp.Symbol):
        var_str = str(expr)
        v = None
        if m.getVars():
            v = m.getVarByName(var_str)
        if v is not None:
            return v
        
        if binary:
            v = m.addVar(name=var_str, vtype=GRB.BINARY)
        elif incl_bound:
            bound = var_to_bound.get(expr, (float('-inf'), float('inf')))
            lb, ub = bound
            v = m.addVar(lb=lb, ub=ub, name=var_str, vtype=GRB.CONTINUOUS)
        else:
            v = m.addVar(lb=float('-inf'), ub=float('inf'), name=var_str, vtype=GRB.CONTINUOUS)
        return v
    
    res = [convert_sp_to_grb_expr(m, arg, sp_to_grb, var_to_bound, binary, incl_bound)
           for arg in expr.args]
    
    # Operation between args0 and args1 is either Add or Mul
    if isinstance(expr, sp.Add):
        ret = quicksum(res)
    elif isinstance(expr, sp.Mul):
        ret = 1
        for r in res:
            ret *= r
    else:
        raise NotImplementedError(f"Operation {expr.func} not supported")
    
    # Store in cache
    sp_to_grb[expr] = ret
    return ret
