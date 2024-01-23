from typing import Dict, FrozenSet, Optional, Set, cast
from gurobipy import GRB

import psutil
import pulp as pl
from pulp import const
import symengine.lib.symengine_wrapper as core

from xaddpy.utils.global_vars import LP_BACKEND, REL_NEGATED, REL_TYPE
from xaddpy.utils.logger import logger
from xaddpy.utils.lp_util import Model, GurobiModel, convert_to_pulp_expr, pulp_constr_to_gurobi_constr
from xaddpy.utils.symengine import BooleanVar
from xaddpy.xadd.node import Node, XADDINode, XADDTNode

default_check_redundancy = True
TEST_SLACK = True


class ReduceLPContext:
    """Context for reduce_lp method from XADD."""
    def __init__(self, context, **kwargs):
        self.LPContext = context
        self.set_to_implied_set: Dict[FrozenSet[int], Set[int]] = {}
        self.set_to_nonimplied_set: Dict[FrozenSet[int], Set[int]] = {}
        self.local_reduce_lp = None
        self.kwargs = kwargs

    def reduce_lp(self, node_id, redun=None):
        if redun is None:
            redun = default_check_redundancy

        if self.local_reduce_lp is None:
            self.local_reduce_lp = LocalReduceLP(context=self.LPContext, reduce_lp_context=self, **self.kwargs)

        return self.local_reduce_lp.reduce_lp(node_id, redun)

    def flush_caches(self):
        self.flush_implications()
        try:
            self.local_reduce_lp.flush_caches()
        except AttributeError as e:
            logger.error(e)
            pass

    def flush_implications(self):
        """Flushes the caches of implied and non-implied sets"""
        self.set_to_implied_set.clear()
        self.set_to_nonimplied_set.clear()
        

class LocalReduceLP:
    """Actual workhorse for reduce_lp method from XADD."""
    def __init__(self, context, reduce_lp_context: ReduceLPContext, **kwargs):
        self._context = context
        self._expr_to_linear_check = self._context._expr_to_linear_check
        self._expr_id_to_linear_check = self._context._expr_id_to_linear_check
        self.reduce_lp_context = reduce_lp_context
        self.lp = None
        self.verbose = kwargs.get('verbose', False)

    def flush_caches(self):
        self.lp._lhs_expr_to_pulp.clear()
        self.lp._sym_to_pulp.clear()

    def reduce_lp_v2(
            self,
            node_id: int,
            test_dec: Set[int],
            redundancy: bool) -> int:
        """Performs reduce_lp on the given node and returns the reduced node ID.

        Args:
            node_id (int): The ID of the node to which 'reduce_lp' is applied.
            test_dec (set): The set containing IDs of decisions.
            redundancy (bool): Check for redundancy.

        Returns:
            int: Reduced XADD node ID.
        """
        avail_mem = (
            psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        )
        if avail_mem < 10:
            logger.info('freeing up cache of reduce_lp')
            self.reduce_lp_context.flush_caches()

        node: Node = self.get_exist_node(node_id)

        # A leaf node should be reduced (and cannot be restricted).
        if node.is_leaf():
            return node_id

        node = cast(XADDINode, node)
        dec_id = node.dec

        # Skip non-linear expressions (any higher order terms including bilinear).
        if not self._expr_id_to_linear_check[dec_id]:
            low = self.reduce_lp_v2(node.low, test_dec, redundancy)
            high = self.reduce_lp_v2(node.high, test_dec, redundancy)
            return self.get_internal_node(dec_id, low, high)

        # Full branch implication test:
        # If `node.dec` is implied by `test_dec`, then replace `node` with `node.high`.
        if self.is_test_implied(test_dec, dec_id):
            return self.reduce_lp_v2(node.high, test_dec, redundancy)
        # If the negated decision is implied by `test_dec`, 
        # then replace `node` with `node.low`.
        elif self.is_test_implied(test_dec, -dec_id):
            return self.reduce_lp_v2(node.low, test_dec, redundancy)

        # Make subtree reduced before redundancy check.
        test_dec.add(-dec_id)
        low = self.reduce_lp_v2(node.low, test_dec, redundancy)
        try:
            test_dec.remove(-dec_id)
        except KeyError as e:
            logger.error(str(e))
            logger.info("It is likely that the node is not canonical")
            exit(1)

        test_dec.add(dec_id)
        high = self.reduce_lp_v2(node.high, test_dec, redundancy)
        test_dec.remove(node.dec)

        # After reducing subtrees, check if this node became redundant.
        if redundancy:
            # 1) Check if true branch is implied in the low branch
            # if current decision is true.
            test_dec.add(dec_id)
            low_replace = self.is_result_implied(test_dec, low, high)
            test_dec.remove(dec_id)

            if low_replace: return low

            # 2) Check if false branch is implied in the true branch
            # if current decision is false.
            test_dec.add(-dec_id)
            high_replace = self.is_result_implied(test_dec, high, low)
            test_dec.remove(-dec_id)

            if high_replace: return high

        # Standard reduce: getINode will handle the case of low == high.
        return self.get_internal_node(dec_id, low, high)

    def reduce_lp(self, node_id: int, redundancy: bool) -> int:
        test_dec = set()
        node_id = self.reduce_lp_v2(node_id, test_dec, redundancy)
        return node_id

    def is_test_implied(self, test_dec: Set[int], dec: int) -> bool:
        """Checks whether 'dec' is implied by 'test_dec'.

        Args:
            test_dec (Set[int]): The set of decisions.
            dec (int): The ID of the decision that we want to test.

        Returns:
            bool: True if 'dec' is implied by 'test_dec' set; False otherwise.
        """
        implied_set = self.reduce_lp_context.set_to_implied_set.get(frozenset(test_dec.copy()), None)
        if implied_set is not None and dec in implied_set:
            # When dec can easily be checked as implied (using impliedSet)
            return True

        non_implied_set = self.reduce_lp_context.set_to_nonimplied_set.get(frozenset(test_dec.copy()), None)
        if non_implied_set is not None and dec in non_implied_set:
            return False

        if -dec in test_dec:
            logger.warning(f"Checking if decision implies its negation! - {test_dec}")
        # If adding the negation of `dec` to `test_dec` makes it infeasible,
        # then `test_dec` implies `dec`
        test_dec.add(-dec)
        implied = self.is_infeasible(test_dec)
        test_dec.remove(-dec)
        if implied:
            if implied_set is None:
                implied_set = set()
                self.reduce_lp_context.set_to_implied_set[frozenset(test_dec.copy())] = implied_set
            implied_set.add(dec)
        else:
            if non_implied_set is None:
                non_implied_set = set()
                self.reduce_lp_context.set_to_nonimplied_set[frozenset(test_dec.copy())] = non_implied_set
            non_implied_set.add(dec)
        return implied

    def is_infeasible(self, test_dec: Set[int]) -> bool:
        """Checks whether a set of decisions contained in 'test_dec' is infeasible.

        Args:
            test_dec (Set[int]): The set contatining IDs of decisions.

        Returns:
            bool: True if infeasible; False otherwise.
        """
        infeasible = False

        # Based on decisions, solve an LP with those constraints,
        # and determine if feasible or infeasible.
        self.lp = LP(self.context) if not self.lp else self.lp
        lp = self.lp
        m = lp.model

        # Check for constraints to remove or add.
        constrs = lp.check_constraints(test_dec, lp.added_decisions[m])

        # Add new constraints.
        for dec in constrs['add']:
            lp.add_decision(dec=dec, m=m, require_rebuild=lp.require_rebuild)
        # Remove old constraints.
        for dec in constrs['remove']:
            lp.remove_decision(dec=dec, m=m)
        # Optimize the model to see if infeasible.
        try:
            status = lp.solve()
        except Exception as e:
            logger.error(str(e))
            exit(1)

        if status == pl.LpStatusUndefined:
            logger.warning("Undefined status during Test 1?")

        if status == pl.LpStatusInfeasible:
            infeasible = True
        if infeasible:
            return infeasible

        ## Test 2: test slack.
        if TEST_SLACK:
            infeasible = lp.test_slack(test_dec)
        return infeasible
    
    def is_result_implied(self, test_dec: Set[int], subtree: int, goal: int) -> bool:
        """Checks whether 'goal' can be reached by `subtree`.

        Args:
            test_dec (Set[int]): The set of decisions.
            subtree (int): The ID of the subtree.
            goal (int): The ID of the goal node.

        Returns:
            bool: Whether `subtree` always reaches `goal` given `test_dec`.
        """
        if subtree == goal:
            return True
        subtree_node = self.get_exist_node(subtree)
        goal_node = self.get_exist_node(goal)

        if not subtree_node.is_leaf():
            # subtree is an internal node.
            if not goal_node.is_leaf():
                # Both subtree and goal are internal nodes.
                subtree_node = cast(XADDINode, subtree_node)
                goal_node = cast(XADDINode, goal_node)
                
                # Use variable ordering to stop pointless searches.
                if subtree_node.dec >= goal_node.dec:
                    return False

            # If decisions down to the current node imply
            # the negation of the `subtree_node.dec`?
            if self.is_test_implied(test_dec, -subtree_node.dec):
                # Then, check for the low branch.
                return self.is_result_implied(test_dec, subtree_node.low, goal)
            # Otherwise, they imply `subtree_node.dec`.
            if self.is_test_implied(test_dec, subtree_node.dec):
                # Then, check for the high branch.
                return self.is_result_implied(test_dec, subtree_node.high, goal)

            # Now, recurse starting from the low branch.
            test_dec.add(-subtree_node.dec)
            implied_in_low = self.is_result_implied(test_dec, subtree_node.low, goal)
            test_dec.remove(-subtree_node.dec)

            # If one branch failed, no need to test the other one.
            if not implied_in_low: return False

            # Check the other branch.
            test_dec.add(subtree_node.dec)
            implied_in_high = self.is_result_implied(test_dec, subtree_node.high, goal)
            test_dec.remove(subtree_node.dec)

            return implied_in_high
        return False    # If XADDTNode, '==' check can make it True.

    @property
    def context(self):
        return self._context

    def get_internal_node(self, dec: int, low: int, high: int) -> int:
        return self._context.get_internal_node(dec, low, high)
    
    def get_exist_node(self, node_id: int) -> Node:
        return self.context.get_exist_node(node_id)    


class LP:
    def __init__(self, context):
        self._context = context
        kwargs = dict(
            name='LPReduce',
            backend=LP_BACKEND,
            sense=pl.LpMaximize,
            msg=False,
        )
        if LP_BACKEND == 'gurobi':
            kwargs['backend'] = 'gurobi_custom'
            self.model = GurobiModel(**kwargs)
            self.slack_model = GurobiModel(**kwargs)
        else:
            self.model = Model(**kwargs)
            self.slack_model = Model(**kwargs)

        # Set some model params.
        self.model.setObjective(1)      # We are only checking for feasibility.
        self.model.setAttr('_var_to_bound', context._var_to_bound)
        self.model.set_sym_to_pulp_dict(self.context._sym_to_pulp)

        # Define a positive slack variable and set it as the objective for maximization.
        S = self.slack_model.addVar(lb=0, name='Slack')
        self.slack_model.setObjective(S)
        self.slack_model.setAttr('_var_to_bound', context._var_to_bound)
        self.slack_model.set_sym_to_pulp_dict(self.context._sym_to_pulp)
        self.S = S
        self.slack_model.require_rebuild = True

        # Cache some useful variables.
        self._sym_to_pulp = self.context._sym_to_pulp
        self._lhs_expr_to_pulp = {}
        self._name_to_solver_constr = {
            self.model: {}, self.slack_model: {}
        }

        # Constraint management.
        self.added_decisions: Dict[Set[int]] = {
            self.model: set(), self.slack_model: set()
        }

    @property
    def require_rebuild(self):
        if hasattr(self.model._model, 'require_rebuild'):
            return self.model._model.require_rebuild
        return True

    @property
    def context(self):
        return self._context

    def reset(self):
        self.model.reset()

    def check_constraints(
            self, test_dec: Set[int], added_decisions: Set[int],
    ) -> Dict[str, Set[int]]:
        """Checks the constraints to remove or add.
        
        Args:
            test_dec (Set[int]): The set of decisions.
            added_decisions (Set[int]): The set of decisions
                that have already been added to the model.

        Returns:
            Dict[str, Set[int]]: The dictionary of constraints to remove or add.
        """
        constrs_to_add = test_dec.difference(added_decisions)
        constrs_to_remove = added_decisions.difference(test_dec)
        constrs_to_keep = added_decisions.intersection(test_dec)
        return dict(
            add=constrs_to_add,
            remove=constrs_to_remove,
            keep=constrs_to_keep,
        )

    def add_decision(
            self, dec: int, m: Model = None, require_rebuild: bool = True
    ):
        if m is None:
            m = self.model
        # Check if the constraint has already been added to the model
        constraint = m.get_constraint_by_name(f'dec({dec})'.replace('-', 'm'))
        if not require_rebuild and constraint is not None:
            return
        if dec > 0:
            self.add_constraint(m, dec, True, require_rebuild)
        else:
            self.add_constraint(m, -dec, False, require_rebuild)

    def add_constraint(
            self,
            m: Model,
            dec: int,
            is_true: bool,
            require_rebuild: bool,
            add_slack: bool = False,
    ):
        """
        Given an integer id for a decision expression (and whether it's true or false),
            add the expression to LP problem.
            1) Need to create Variable objects for each of SymEngine variables
                (if already created, retrieve from cache).
            2) Need to convert SymEngine dec_expr to optlang Constraint format
                e.g. c1 = Constraint(x1 + x2 + x3, ub=10)
                    for x1 + x2 + x3 <= 10

        Args:
            m (Model):      The LP model.
            dec (int):      Decision ID.
            is_true (bool): Whether the decision is true or false.
            require_rebuild (bool): Whether to rebuild the model.
            add_slack (bool, optional): Whether to add slack variable.
                Defaults to False.
        """
        dec_expr = self.context._id_to_expr[dec]
        dec = dec if is_true else -dec
        name = f'dec({dec})'.replace('-', 'm')
        S = self.S if add_slack else 0

        # Handle relational conditionals.
        if isinstance(dec_expr, core.Rel):
            lhs, rhs = dec_expr.args
            rel = REL_TYPE[type(dec_expr)]
            if not is_true:
                rel = REL_NEGATED[rel]

            assert rhs == 0, (
                "RHS of a relational expression should always be 0 by construction!"
            )
            lhs_pulp = self.convert_expr(lhs)   # Convert lhs to pulp expression (rhs=0)

            if rel == '>' or rel == '>=':
                m.addConstr(lhs_pulp - S >= 0, name=name)
            elif rel == '<' or rel == '<=':
                m.addConstr(lhs_pulp + S <= 0, name=name)

        # Handle Boolean decisions.
        elif dec_expr.is_Boolean:
            assert not add_slack
            bool_pulp = self.convert_expr(dec_expr, binary=True)
            if is_true:
                m.addConstr(bool_pulp == 1, name=name)
            else:
                m.addConstr(bool_pulp == 0, name=name)
        else:
            raise NotImplementedError("Decision expression not supported")

        # TODO: we are assuming Gurobi solver here.
        # If we have a solver model, add the constraint to the model.
        if not require_rebuild:
            assert hasattr(m._model, 'solverModel'), (
                "Underlying solver model not found!"
            )
            solver_constr = self._name_to_solver_constr[m].get(name)
            if solver_constr is None:
                pl_constr = m._constr_cache[name]
                # Add any variables not in the solver model.
                for v in pl_constr.keys():
                    check = m._model.solverModel._var_name_to_var.get(v.name)
                    if check is not None:
                        continue
                    lb, ub = v.lowBound, v.upBound
                    lb = -GRB.INFINITY if lb is None else lb
                    ub = GRB.INFINITY if ub is None else ub
                    vtype = GRB.CONTINUOUS
                    if v.cat == const.LpInteger:
                        vtype = GRB.BINARY
                    solver_var = m._model.solverModel.addVar(
                        lb=lb, ub=ub, vtype=vtype, name=v.name
                    )
                    m._model.solverModel._var_name_to_var[v.name] = solver_var

                solver_constr, lhs_rel_rhs = pulp_constr_to_gurobi_constr(
                    m._model.solverModel, name, pl_constr)
                self._name_to_solver_constr[m][name] = lhs_rel_rhs
            else:
                m._model.solverModel.addConstr(*solver_constr, name=name)

        # Keep track of added decisions.
        self.added_decisions[m].add(dec)

    def remove_decision(self, dec: int, m: Model):
        """Removes the constraint corresponding to the given decision ID."""
        m.removeConstrByName(f'dec({dec})'.replace('-', 'm'))
        self.added_decisions[m].remove(dec)

    def solve(self) -> Optional[int]:
        """
        Solve the LP defined by all added constraints, variables and objective. We only care about if it's
        infeasible or feasible. So, return status.
        :return:        (str)
        """
        try:
            status = self.model.solve()
            return status
        except Exception as e:
            raise e

    def set_objective(self, obj):
        self.model.setObjective(obj)

    def convert_expr(self, expr, binary: bool = False):
        """Returns the PuLP expression.

        Args:
            expr (core.Basic): The SymEngine expression.
            binary (bool, optional): Whether the expression is binary. Defaults to False.

        Returns:
            pl.LpAffineExpression: The PuLP expression.
        """
        if expr in self._lhs_expr_to_pulp:
            return self._lhs_expr_to_pulp[expr]
        else:
            pulp_expr = convert_to_pulp_expr(
                expr,
                model=self.model,
                incl_bound=True,
                binary=binary,
            )
            self._lhs_expr_to_pulp[expr] = pulp_expr
            return pulp_expr

    def test_slack(self, test_dec: Set[int]):
        """Checks the value of slack variable, S.

        For each constraint a.T @ w + b >= 0, the slack is the greatest value S > 0 s.t. a.T @ w + b - S >= 0
        For each constraint a.T @ w + b <= 0, the slack is the greatest value S > 0 s.t. a.T @ w + b + S <= 0

        """
        infeasible = False

        # Check if test_slack is turned off by context (this happens at the time of arg substitution)
        if not self.context._prune_equality:
            return infeasible

        m = self.slack_model        

        # Check for constraints to remove or add.
        constrs = self.check_constraints(test_dec, self.added_decisions[m])

        # Add new constraints.
        for dec in constrs['add']:
            # Skip Boolean decisions.
            dec_expr = (
                self.context._id_to_expr[-dec] if dec < 0 else 
                self.context._id_to_expr[dec]
            )
            if not isinstance(dec_expr, core.Rel):
                continue

            self.add_constraint(
                m, abs(dec), dec > 0, require_rebuild=m.require_rebuild, add_slack=True)

        # Remove old constraints.
        for dec in constrs['remove']:
            self.remove_decision(dec, m)

        if len(m.get_constraints()) == 0:
            return infeasible

        # Optimize the model to see if infeasible
        try:
            status = m.solve()
            m.require_rebuild = False
        except Exception as e:
            logger.error(e)
            exit(1)

        # Due to dual reduction in presolve of Gurobi, INF_OR_UNBD status can be returned
        # Turn off the functionality and resolve
        # TODO: How to handle dualreductions for other solvers than Gurobi?
        if status in (pl.LpStatusInfeasible, pl.LpStatusUnbounded, pl.LpStatusUndefined):
            m.toggle_direct_solver_on()
            m.setParam('DualReductions', 0)
            status = m.solve()
            m.setParam('DualReductions', 1)
            m.toggle_direct_solver_off()

        opt_obj = m.objVal if status == pl.LpStatusOptimal else 1e100

        if status == pl.LpStatusInfeasible:
            logger.warning("Infeasibility at test 2 should not have occurred!")
            infeasible = True
        elif status != pl.LpStatusUnbounded and opt_obj < 1e-40:
            infeasible = True

        return infeasible
