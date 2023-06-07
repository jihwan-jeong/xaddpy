from argparse import Namespace
from time import time
from typing import Tuple, Union

from examples.dblp.util import build_xadd_from_json
from examples.xadd_for_milp.xadd_milp import XADD
from xaddpy.utils.logger import logger


def solve(
        solver_type: int = 0,
        save_xadd: bool = False,
        verbose: bool = False,
        fname_xadd: str = None,
        args: Union[dict, Namespace] = None,
) -> Tuple[XADD, dict, float]:
    if isinstance(args, Namespace):
        args = vars(args)
    context = XADD(args)
    fname_json = fname_xadd.replace('.xadd', '.json')

    variables, dblp_xadd, eq_constr_dict = build_xadd_from_json(context, fname_json)

    # SVE: iteratively min out variables
    vars_to_min = variables['min_var_list']
    logger.info("Start SVE")
    if verbose:
        logger.info(f"The objective: \n{context.get_repr(dblp_xadd)}\n")
        logger.info(f"Variables to minimize: {vars_to_min}\nSolver type: {solver_type}\n")
    
    stime = time()
    if solver_type == 0:
        res = context.min_or_max_multi_var(dblp_xadd, vars_to_min, is_min=True)
    
    elif solver_type == 1:
        for v in vars_to_min:
            logger.info(f"\tEliminate variable {str(v)}")
            res = context.min_or_max_var(dblp_xadd, v, is_min=True)
            dblp_xadd = res
    etime = time()
    time_sve = etime - stime
    logger.info(f"Done SVE: time = {time_sve:.4f}")

    # Export the XADDs
    if save_xadd:
        context.export_min_or_max_xadd(res, fname_xadd)
    context.set_objective(res)
    return context, eq_constr_dict, time_sve
