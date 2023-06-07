try:
    from gurobipy import GRB
except:
    pass

from xaddpy.utils.logger import logger


# Callback for Gurobi solver
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