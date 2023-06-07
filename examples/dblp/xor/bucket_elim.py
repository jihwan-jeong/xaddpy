from typing import Optional

import sympy as sp

from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.utils.logger import logger


def run_symbolic_bucket_elimination(
        context: XADD,
        buckets: dict,
        solver_type: int = 0,
):
    keys = sorted(buckets.keys())
    for i in keys:
        var_i, f_j_lst = buckets[i]

        # For bucket i, add all functions in the bucket
        g_i = context.ZERO
        for j in range(len(f_j_lst)):
            g_i = context.apply(g_i, f_j_lst[j], 'add')
            g_i = context.reduce_lp(g_i)

        # Eliminate var_i from the resulting function
        if solver_type == 0:
            h_i = context.min_or_max_multi_var(g_i, [var_i], is_min=True)
        else:
            h_i = context.min_or_max_var(g_i, var_i, is_min=True)
        n_i = context.get_exist_node(h_i)

        # Assign h_i to a bucket (if a constant, add it to the last bucket)
        if n_i._is_leaf and isinstance(n_i.expr, sp.core.numbers.Number):
            buckets[keys[-1]][1].append(h_i)
        elif i == keys[-1]:
            buckets[i][1].append(h_i)
        else:
            buckets[i + 1][1].append(h_i)


def run_symbolic_bucket_elimination_multi(
        context: XADD,
        buckets: dict,
        solver_type: int = 0,
):
    res_dict = {}
    keys = sorted(buckets.keys())
    for i in keys:
        logger.info(f"Eliminate bucket {i}/{len(keys)} containing variables {buckets[i][0]}")
        variables, f_j_lst = buckets[i]

        # For bucket i, add all functions in the bucket
        g_i = context.ZERO
        for j in range(len(f_j_lst)):
            g_i = context.apply(g_i, f_j_lst[j], 'add')
            g_i = context.reduce_lp(g_i)
        
        # Eliminate var_i from the resulting function
        if solver_type == 0:
            h_i = context.min_or_max_multi_var(g_i,
                                               variables,
                                               is_min=True,
                                               annotate=False)
        else:
            # TODO: min out one variable at a time (full functionality not ported over yet)
            h_i = g_i
            for j in range(len(variables)):
                h_i = context.min_or_max_var(h_i, variables[j], is_min=True)

        # Store the result for each bucket
        res_dict[i] = h_i
    return res_dict
