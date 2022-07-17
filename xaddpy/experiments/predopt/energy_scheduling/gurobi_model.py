from math import ceil
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from xaddpy.utils.logger import logger
import sympy as sp

"""
The helper class and functions necessary for solving the MILP defined in the ICON Challenge.
The code is copied from 
    https://github.com/JayMan91/NeurIPSIntopt/blob/main/EnergyCost/ICON.py
"""


def read_config(filename, args):
    with open(filename) as f:
        mylist = f.read().splitlines()

    DEFAULT_Q = 30
    q = int(60 * (24 / args.sample_per_day))    # Time lapse between samples in minutes
    num_resources = int(mylist[1])
    num_machines = int(mylist[2])
    IDLE = [None] * num_machines
    UP = [None] * num_machines
    DOWN = [None] * num_machines
    MAX_CAPACITY = [None] * num_machines

    # Note: starting from the 4th line (index=3) of day01.txt, the following coefficients are defined per machine
    # There are `num_machines` number of machines, which is also specified in `day01.txt` in the third line.
    for m in range(num_machines):
        l = mylist[2 * m + 3].split()
        IDLE[m] = int(l[1])
        UP[m] = float(l[2])
        DOWN[m] = float(l[3])
        MAX_CAPACITY[m] = list(map(int, mylist[2 * (m+2)].split()))
    lines_read = 2 * num_machines + 2

    # Coefficients for tasks
    num_tasks = int(mylist[lines_read + 1])
    RESOURCE_USE = [None] * num_tasks       # Resource usage for task
    DURATION = [None] * num_tasks           # Duration for task
    E_START = [None] * num_tasks            # Earliest start time for task
    L_END = [None] * num_tasks              # Latest end time for task
    POWER_USE = [None] * num_tasks          # Power usage for task
    POWER_USE_FRAC = [None] * num_tasks     # Power usage for task in fractions
    for f in range(num_tasks):
        l = mylist[2 * f + lines_read + 2].split()
        DURATION[f] = ceil(int(l[1]) * DEFAULT_Q / q)  # TODO: Only allow even DURATION values? What if N is 12 rather than 24?
        E_START[f] = ceil(int(l[2]) * DEFAULT_Q / q)
        L_END[f] = int(l[3]) * DEFAULT_Q // q
        POWER_USE[f] = float(l[4])
        POWER_USE_FRAC[f] = sp.Rational(str(float(l[4]))).limit_denominator(1000)
        RESOURCE_USE[f] = list(map(int, mylist[2 * f + lines_read + 3].split()))

    presolve = args.presolve

    return dict(
        num_machines=num_machines,
        num_tasks=num_tasks,
        num_resources=num_resources,
        MAX_CAPACITY=MAX_CAPACITY,
        RESOURCE_USE=RESOURCE_USE,
        DURATION=DURATION,
        E_START=E_START,
        L_END=L_END,
        POWER_USE=POWER_USE,
        POWER_USE_FRAC=POWER_USE_FRAC,
        IDLE=IDLE,
        UP=UP,
        DOWN=DOWN,
        q=q,
        presolve=presolve,
    )


class GurobiICON:
    def __init__(
            self,
            num_machines=None,
            num_tasks=None,
            num_resources=None,
            MAX_CAPACITY=None,
            RESOURCE_USE=None,
            DURATION=None,
            E_START=None,
            L_END=None,
            POWER_USE=None,
            IDLE=None,
            UP=None,
            DOWN=None,
            q=None,
            reset=None,
            presolve=False,
            relax=None,
            verbose=False,
            warmstart=False,
            method=-1,
            **h
    ):
        self.num_machines = num_machines
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.MAX_CAPACITY = MAX_CAPACITY
        self.RESOURCE_USE = RESOURCE_USE
        self.DURATION = DURATION
        self.E_START = E_START
        self.L_END = L_END
        self.POWER_USE = POWER_USE
        self.IDLE = IDLE
        self.UP = UP
        self.DOWN = DOWN
        self.q = q
        self.relax = True if presolve else relax  # use relax if presolve True
        self.presolve = presolve
        self.verbose = verbose
        self._presolved = False

        self.method = method
        self.sol_hist = []
        self.reset = reset
        self.warmstart = warmstart

    def make_model(self, model_name=None):
        MACHINES = range(self.num_machines)
        TASKS = range(self.num_tasks)
        RESOURCES = range(self.num_resources)

        MAX_CAPACITY = self.MAX_CAPACITY
        RESOURCE_USE = self.RESOURCE_USE
        DURATION = self.DURATION
        E_START = self.E_START
        L_END = self.L_END
        POWER_USE = self.POWER_USE
        IDLE = self.IDLE
        UP = self.UP
        DOWN = self.DOWN
        relax = self.relax
        q = self.q
        N = 1440 // q           # min per day / min per sample = num samples per day

        model_name = 'icon' if model_name is None else model_name
        m = gp.Model(model_name)
        # if not self.verbose:
        #     m.setParam('OutputFlag', 0)
        if relax:
            x = m.addVars(TASKS, MACHINES, range(N), lb=0., ub=1., vtype=GRB.CONTINUOUS, name='x')
        else:
            x = m.addVars(TASKS, MACHINES, range(N), vtype=GRB.BINARY, name='x')

        m.addConstrs(x.sum(j, '*', range(E_START[j])) == 0 for j in TASKS)  # INTOPT ERROR: E_START \in [1, 24], WHILE HERE TIME \in [0, 23]. NOT CORRECTED BECAUSE BASELINES USE THE SAME CONSTRAINTS.
        m.addConstrs(x.sum(j, '*', range(L_END[j] - DURATION[j] + 1, N)) == 0 for j in TASKS)
        m.addConstrs((quicksum(x[(j, mc, t)] for t in range(N) for mc in MACHINES) == 1 for j in TASKS))

        # Capacity requirement
        for r in RESOURCES:
            for mc in MACHINES:
                for t in range(N):
                    m.addConstr(
                        quicksum(quicksum(x[(j, mc, t1)] for t1 in range(max(0, t - DURATION[j] + 1), t + 1)) *
                                RESOURCE_USE[j][r] for j in TASKS)
                        <= MAX_CAPACITY[mc][r]
                    )

        # Presolve to make the model more compact
        if self.presolve:
            # self.model_before_presolve = m
            m = m.presolve()
        else:
            m.update()
        self.model = m

        self.x = dict()
        for var in m.getVars():
            name = var.varName
            if name.startswith('x['):
                j, mc, t = map(int, name[2:-1].split(','))
                self.x[(j, mc, t)] = var

    def solve_model(self, price, timelimit=None):
        m = self.model
        DURATION = self.DURATION
        POWER_USE = self.POWER_USE
        IDLE = self.IDLE
        UP = self.UP
        DOWN = self.DOWN
        q = self.q
        N = 1440 // q
        newcut = None
        if self.reset:
            m.reset()

        verbose = self.verbose
        x = self.x
        num_machines = self.num_machines
        num_tasks = self.num_tasks
        num_resources = self.num_resources
        MACHINES = range(self.num_machines)
        TASKS = range(self.num_tasks)
        RESOURCES = range(self.num_resources)

        obj_expr = quicksum([
            x[(j, mc, t)] * np.sum(price[t: t + DURATION[j]]) * POWER_USE[j] * q / 60
            for j in TASKS for t in range(N - DURATION[j] + 1) for mc in MACHINES if (j, mc, t) in x]
        )

        if self.warmstart:
            bestval = np.inf
            if len(self.sol_hist) > 0:
                for i, sol in enumerate(self.sol_hist):
                    pvars, dcons, vbasis, cbasis, sol_vec = sol
                    val = sum((
                        sum(sol_vec[j, mc, t] for mc in MACHINES) * np.sum(price[t: t + DURATION[j]]) * POWER_USE[j] * q / 60)
                        for j in TASKS for t in range(N - DURATION[j] + 1
                    ))
                    if val < bestval:
                        val = bestval
                        ind = i

                pvars, dconst, vbasis, cbasis, sol_vec = self.sol_hist[ind]
                for i, var in enumerate(m.getVars()):
                    var.Pstart = pvars[i]
                    var.VBasis = vbasis[i]
                for i, cons in enumerate(m.getConstrs()):
                    cons.Dstart = dcons[i]
                    cons.CBasis = cbasis[i]

        m.setObjective(obj_expr, GRB.MINIMIZE)

        if timelimit:
            m.setParam('TimeLimit', timelimit)
        m.setParam('Method', self.method)
        m.optimize()

        solver = np.zeros(N)
        if m.status in [GRB.Status.OPTIMAL, 9]:
            try:
                task_on = np.zeros((num_tasks, num_machines, N))
                for ((j, mc, t), var) in x.items():
                    try:
                        task_on[j, mc, t] = var.X
                    except AttributeError:
                        task_on[j, mc, t] = 0.
                        logger.error("AttributeError: b' Unable to retrieve attribute 'X'")
                        logger.error("__________________Something WRONG_____________________")

                if verbose:
                    logger.info(f'\nCost: {m.objVal}')
                    logger.info(f'\nExecution time: {m.RunTime}')

                for t in range(N):
                    solver[t] = np.sum(
                        np.sum(task_on[j, :, max(0, t - DURATION[j] + 1): t+1]) * POWER_USE[j] for j in TASKS
                    )
                solver = solver * q / 60

                if self.warmstart:
                    pvars = m.getAttr(GRB.Attr.X, m.getVars())
                    dcons = m.getAttr(GRB.Attr.Pi, m.getConstrs())
                    vbasis = m.getAttr(GRB.Attr.VBasis, m.getVars())
                    cbasis = m.getAttr(GRB.Attr.CBasis, m.getConstrs())
                    self.sol_hist.append((pvars, dconst, vbasis, cbasis, task_on))
                    if len(self.sol_hist) > 10:
                        self.sol_hist.pop(0)
                self.presolved_solver = solver
                return solver, m.Runtime
            except NameError:
                logger.error('\n__________Something wrong_______ \n')
                if newcut is not None:
                    m.remove(newcut)
                    newcut = None
                return solver, m.Runtime

        elif m.status == GRB.Status.INF_OR_UNBD:
            logger.info('Model is infeasible or unbounded')
        elif m.status == GRB.Status.INFEASIBLE:
            logger.info('Model is infeasible')
        elif m.status == GRB.Status.UNBOUNDED:
            logger.info('Model is unbounded')
        else:
            logger.info(f'Optimization ended with status {m.status}')
        return solver, m.Runtime

    def __call__(self, price, *args, **kwargs):
        """
        Given an array of prices for 24-hour time window, compute the optimal solutions and objective.
        """
        return self.solve_model(price)

    def compute_sols(self, prices, presolve=False):
        self.model.Params.OutputFlag = 0
        res = []
        for i in range(len(prices)):
            prices_i = prices[i]
            sol, _ = self(prices_i)
            res.append(sol)
        res = np.array(res)

        if presolve:
            self._presolved = True
            self.presolved_solution = res
        else:
            return res

    def compute_regret(self, y_pred, y_true):

        actual_cost = []
        optimal_cost = []
        regret = []

        for i in range(len(y_pred)):
            y_pred_i = y_pred[i]
            y_true_i = y_true[i]

            sol_pred, _ = self(y_pred_i)
            sol_true, _ = self(y_true_i)

            actual_cost.append(sum(y_true_i * sol_pred))
            optimal_cost.append(sum(y_true_i * sol_true))
            regret.append(actual_cost[i] - optimal_cost[i])
        return actual_cost, optimal_cost, regret

