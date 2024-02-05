"""Implements the main XADD class and its helper classes & functions."""

import abc
from pathlib import Path
from typing import cast, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
from symengine import oo
import symengine.lib.symengine_wrapper as core
import sympy as sp

try:
    from xaddpy.utils.graph import Graph
except ImportError:
    warnings.warn("[Module: xaddpy.xadd.xadd] Import error: pygraphviz not installed")
    class Graph:
        def __init__(self, *args, **kwargs):
            pass

from xaddpy.utils.global_vars import (
    REL_TYPE, OP_TYPE, UNARY_OP, RELATIONAL_OPERATOR, ACCEPTED_RV_TYPES
)
import xaddpy.utils.util as xadd_util
from xaddpy.utils.util import check_sym_boolean, sample_rvs, check_expr_linear
from xaddpy.utils.logger import logger
from xaddpy.utils.symengine import BooleanVar, RandomVar
from xaddpy.xadd.node import Node, XADDINode, XADDTNode
from xaddpy.xadd.reduce_lp import ReduceLPContext
from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar


USE_APPLY_GET_INODE_CANON = False
LARGE_INTEGER = 10000
VAR_TYPE = core.Symbol | BooleanVar | RandomVar
DECISION_TYPE = core.Rel | BooleanVar | RandomVar   # RandomVar?


def default_ordering(context, expr: core.Basic) -> int:
    num_unique_expr = len(context._expr_to_id)

    # Decision expression consisting of continuous variables
    if isinstance(expr, core.Rel):
        is_linear = check_expr_linear(expr)
        if is_linear:
            index = num_unique_expr + LARGE_INTEGER
        else:
            index = num_unique_expr + LARGE_INTEGER ** 2
    # Boolean decisions
    else:
        index = num_unique_expr
    return index


class XADD:

    _func_var_index = default_ordering

    def __init__(
            self,
            args: dict = {},
            perform_reduce_lp: bool = True,
    ):
        # XADD variable maintenance
        self._cvar_to_id: Dict[core.Symbol, int] = {}
        self._id_to_cvar: Dict[int, core.Symbol] = {}
        self._bvar_to_id: Dict[BooleanVar, int] = {}
        self._id_to_bvar: Dict[int, BooleanVar] = {}
        self._rv_to_id: Dict[core.Symbol, int] = {}
        self._id_to_rv: Dict[int, core.Symbol] = {}
        self._str_var_to_var: Dict[str, core.Symbol] = {}
        self._cont_var_set: Set[core.Symbol] = set()
        self._bool_var_set: Set[BooleanVar] = set()
        self._random_var_set: Set[RandomVar] = set()
        self._rv_to_params: Dict[RandomVar, Any] = {}
        self._rv_to_type: Dict[RandomVar, str] = {}

        self._sym_to_pulp = {}
        self._opt_var = None
        self._opt_var_lst = None
        self._eliminated_var = []
        self._decisionVars = set()
        self._min_var_set = set()
        self._free_var_set = set()
        self._name_space: Dict[str, VAR_TYPE] = {}

        # Bound maintenance (need to be passed from the output of parser function)
        self._var_to_bound = {}
        self._temp_ub_lb_cache = set()

        # Decision expression maintenance
        self._expr_to_id: Dict[core.Basic, int] = {}
        self._id_to_expr: Dict[int, core.Basic] = {}
        self._expr_to_linear_check: Dict[core.Basic, bool] = {}
        self._expr_id_to_linear_check: Dict[int, bool] = {}

        # XADD node maintenance
        self._id_to_node: Dict[int, Node] = {}
        self._node_to_id: Dict[Node, int] = {}
        self._var_to_anno: Dict[core.Symbol, int] = {}     # annotation dictionary for argmin / argmax

        # Flush
        self._special_nodes = set()
        self._node_to_id_new = {}
        self._id_to_node_new = {}
        self._id_to_expr_new = {}
        self._expr_to_id_new = {}

        # Reduce & Apply caches
        self._reduce_cache: Dict[Tuple[int, int, str], int] = {}
        self._reduce_leafop_cache: Dict[Tuple[int, Any], int] = {}
        self._reduce_canon_cache = {}

        self._apply_cache = {}
        self._apply_caches = {}
        self._inode_to_vars: Dict[Tuple[int, int, int], Set[VAR_TYPE]] = {}
        self._factor_cache = {}

        # Reduce LP
        self.perform_reduce_lp = perform_reduce_lp
        self.RLPContext = ReduceLPContext(self, **args)

        # Node maintenance
        self._nodeCounter = 0

        # temporary nodes
        self._tempINode = XADDINode(-1, -1, -1, context=self)
        self._temp_term_node = XADDTNode(core.S(-1), context=self)

        # Ensure that the 0th decision ID is invalid
        null = NullDec()
        self._id_to_expr[0] = null
        self._expr_to_id[null] = 0

        # Create standard nodes
        self.create_standard_nodes()

        # How to handle decisions that hold with equality
        self._prune_equality = True

        # Do or do not reorder XADD after substitution
        self._direct_substitution = False

        # Other attributes
        self._args: dict = args

        # Related to MILP compilation
        self._obj = None
        self._additive_obj = False

    def set_variable_ordering_func(self, func: Callable):
        XADD._func_var_index = func
    
    def add_random_var(self, var: RandomVar, **kwargs):
        if not var in self._random_var_set:
            assert kwargs.get('params') is not None
            assert kwargs.get('type') is not None and kwargs['type'] in ACCEPTED_RV_TYPES
            num_existing_rv = len(self._random_var_set)
            self._random_var_set.add(var)
            self._str_var_to_var[str(var)] = var
            self._rv_to_id[var] = num_existing_rv
            self._id_to_rv[num_existing_rv] = var

            self._rv_to_params[var] = kwargs['params']
            self._rv_to_type[var] = kwargs['type']
            self._name_space[str(var)] = var

    def add_continuous_var(self, var: core.Symbol):
        if not var in self._cont_var_set:
            num_existing_cvar = len(self._cont_var_set)
            self._cont_var_set.add(var)
            self._str_var_to_var[str(var)] = var
            self._cvar_to_id[var] = num_existing_cvar
            self._id_to_cvar[num_existing_cvar] = var
            self._name_space[str(var)] = var

    def add_boolean_var(self, var: Union[core.Symbol, BooleanVar]):
        if not (
            isinstance(var, BooleanVar) or (
                isinstance(var, RandomVar) and str(var).startswith('Bernoulli')
            )
        ):
            logger.info(f"The type of boolean variable {var} is not correctly set.")
            var = BooleanVar(var)
        if var not in self._bool_var_set:
            num_existing_bvar = len(self._bool_var_set)
            self._bool_var_set.add(var)
            self._str_var_to_var[str(var)] = var
            self._bvar_to_id[var] = num_existing_bvar
            self._id_to_bvar[num_existing_bvar] = var
            self._name_space[str(var)] = var

    def get_var_from_name(self, name: str) -> VAR_TYPE:
        return self._name_space.get(name)

    def create_standard_nodes(self):
        """Creates and stores standard nodes."""
        self.ZERO = self.get_leaf_node(core.S(0))
        self.ONE = self.get_leaf_node(core.S(1))
        self.TRUE = self.get_leaf_node(core.true)
        self.FALSE = self.get_leaf_node(core.false)
        self.oo = self.get_leaf_node(oo)
        self.NEG_oo = self.get_leaf_node(-oo)
        self.NAN = self.get_leaf_node(core.nan)

    def add_eliminated_var(self, var):
        self._eliminated_var.append(var)

    """Managing variables"""
    def update_bounds(self, bound_dict):
        self._var_to_bound.update(bound_dict)

    def update_name_space(self, ns):
        self._name_space = ns

    def convert_func_to_xadd(self, term, **kwargs):
        args = term.as_two_terms()
        xadd1 = self.convert_to_xadd(args[0], **kwargs)
        xadd2 = self.convert_to_xadd(args[1], **kwargs)
        op = OP_TYPE[type(term)]
        return self.apply(xadd1, xadd2, op)

    def convert_to_xadd(self, term: core.Basic, **kwargs):
        if isinstance(term, VAR_TYPE) or \
            isinstance(term, core.Number) or \
                isinstance(term, core.BooleanAtom):
            if isinstance(term, BooleanVar):
                dec, is_reversed = self.get_dec_expr_index(term, create=True, **kwargs)
                low, high = self.FALSE, self.TRUE
                if is_reversed:
                    low, high = high, low
                return self.get_internal_node(dec, low, high)
            return self.get_leaf_node(term, **kwargs)
        else:
            return self.convert_func_to_xadd(term, **kwargs)

    def build_initial_xadd(
            self, xadd_as_list: List[core.Basic], to_canonical: bool = True
    ):
        """
        Given decisions and leaf values in a list, 
        recursively build initial XADD and return the id of the root node.

        Args:
            xadd_as_list (List[core.Basic])
        Returns:
            int: The id of the root node
        """
        if len(xadd_as_list) == 1:
            return self.get_leaf_node(xadd_as_list[0])

        dec_expr = xadd_as_list[0]
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)

        high = self.build_initial_xadd(xadd_as_list[1])
        low = self.build_initial_xadd(xadd_as_list[2])

        # swap low and high branches if reversed
        if is_reversed:
            low, high = high, low
        if to_canonical:
            return self.get_inode_canon(dec, low, high)
        else:
            return self.get_internal_node(dec, low, high)

    def make_canonical(self, node_id: int) -> int:
        self._reduce_canon_cache.clear()
        return self.make_canonical_int(node_id)

    def make_canonical_int(self, node_id: int) -> int:
        n = self.get_exist_node(node_id)

        # A terminal node should be reduced by default
        if n.is_leaf():
            return self.get_leaf_node_from_node(node=n)

        n = cast(XADDINode, n)

        # Check to see if this node has already been made canonical
        ret = self._reduce_canon_cache.get(node_id, None)
        if ret is not None:
            return ret

        # Recursively ensure canonicity for subdiagrams
        low = self.make_canonical_int(n.low)
        high = self.make_canonical_int(n.high)

        # Enforce canonicity via the 'apply trick' at this level.
        dec = n.dec
        dec_expr = self._id_to_expr[dec]
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)

        # If reversed, swap low and high branches
        if is_reversed:
            high, low = low, high

        ret = self.get_inode_canon(dec, low, high)

        # Error check
        self.check_local_ordering_and_exit_on_error(ret)

        # Put return value in cache and return
        self._reduce_canon_cache[node_id] = ret
        return ret

    def check_local_ordering_and_exit_on_error(self, node_id: int) -> None:
        """
        :param node_id:         (int)
        """
        node = self.get_exist_node(node_id)
        if not node.is_leaf():
            dec_id = node.dec
            low_n = self.get_exist_node(node.low)
            if not low_n.is_leaf():
                if dec_id >= low_n.dec:
                    # compare local order
                    logger.info(
                        (
                            f"Reordering problem: {dec_id} >= {low_n.dec}\n"
                            f"{dec_id}: {self._id_to_expr[dec_id]}\n"
                            f"{low_n.dec}: {self._id_to_expr[low_n.dec]}"
                        )
                    )
                    raise ValueError
            high_n = self.get_exist_node(node.high)
            if not high_n.is_leaf():
                if dec_id >= high_n.dec:
                    # compare local order
                    logger.info(
                        (
                            f"Reordering problem: {dec_id} >= {high_n.dec}\n"
                            f"{dec_id}: {self._id_to_expr[dec_id]}\n"
                            f"{high_n.dec}: {self._id_to_expr[high_n.dec]}"
                        )
                    )
                    raise ValueError

    def contains_node_id(self, root: int, target: int) -> bool:
        """
        Returns True if the diagram in root contains a node with the target ID
        :param root:            (int)
        :param target:          (int)
        :return:
        """
        visited = set()
        return self.contains_node_id_int(root, target, visited)

    def contains_node_id_int(self, id: int, target: int, visited: set) -> bool:
        if id == target:
            return True
        if id in visited:
            return False
        visited.add(id)
        node = self.get_exist_node(id)
        if not node.is_leaf():
            if self.contains_node_id_int(node.low, target, visited):
                return True
            if self.contains_node_id_int(node.high, target, visited):
                return True
        return False

    def get_inode_canon(self, dec: int, low: int, high: int) -> int:
        if dec <= 0:
            logger.info(
                (
                    "Warning: Canonizing Negative Decision: "
                    f"{dec} -> {self._id_to_expr[abs(dec)]}"
                )
            )
        result1 = self.get_inode_canon_apply_trick(dec, low, high)
        result2 = self.get_inode_canon_insert(dec, low, high)

        if result1 != result2 and not self.contains_node_id(result1, self.NAN):
            logger.info("Warning: Canonical error (difference not on NAN)")
        return result2

    def get_inode_canon_insert(self, dec: int, low: int, high: int) -> int:
        false_half = self.reduce_insert_node(low, dec, self.ZERO, True)
        true_half = self.reduce_insert_node(high, dec, self.ZERO, False)
        return self.apply_int(true_half, false_half, 'add')

    def reduce_insert_node(
            self, orig: int, dec: int, node_to_insert_on_dec_value: int, dec_value: bool
    ) -> int:
        insertNodeCache = {}
        return self.reduce_insert_node_int(
            orig, dec, node_to_insert_on_dec_value, dec_value, insertNodeCache)

    def reduce_insert_node_int(
            self, orig: int, dec: int, insertNode: int, dec_value: bool, insertNodeCache: dict
    ) -> int:
        ret = insertNodeCache.get(orig, None)
        if ret is not None:
            return ret

        node = self.get_exist_node(orig)
        if node.is_leaf() or (not node.is_leaf() and dec < node.dec):
            ret = self.get_internal_node(dec, orig, insertNode) if dec_value else self.get_internal_node(dec, insertNode, orig)
        else:
            node = cast(XADDINode, node)
            if not node.dec >= dec:
                low = self.reduce_insert_node_int(node.low, dec, insertNode, dec_value, insertNodeCache)
                high = self.reduce_insert_node_int(node.high, dec, insertNode, dec_value, insertNodeCache)
                ret = self.get_internal_node(node.dec, low, high)
            else:
                if dec_value:
                    ret = self.reduce_insert_node_int(node.low, dec, insertNode, dec_value, insertNodeCache)
                else:
                    ret = self.reduce_insert_node_int(node.high, dec, insertNode, dec_value, insertNodeCache)

        insertNodeCache[orig] = ret
        return ret

    def get_inode_canon_apply_trick(self, dec: int, low: int, high: int) -> int:
        ind_true = self.get_internal_node(dec, self.ZERO, self.ONE)
        ind_false = self.get_internal_node(dec, self.ONE, self.ZERO)
        true_half = self.apply_int(ind_true, high, 'prod')
        false_half = self.apply_int(ind_false, low, 'prod')
        result = self.apply_int(true_half, false_half, 'add')
        return result
    
    def evaluate_decision(
            self,
            dec_expr: DECISION_TYPE,
            bool_assign: Dict[core.Symbol, Union[core.BooleanAtom, bool]], 
            cont_assign: Dict[core.Symbol, Union[int, float]], 
    ) -> Union[bool, None]:
        if isinstance(dec_expr, core.BooleanAtom):
            return bool(dec_expr)
        # if a decision expression is a single symbol, it should be a boolean decision
        elif isinstance(dec_expr, BooleanVar):
            return bool_assign.get(dec_expr)
        # Inequality decision
        elif isinstance(dec_expr, core.Rel):
            # if any of the variables in dec_expr is not evaluated, returns None
            var_set = dec_expr.free_symbols
            non_assigned_vars = var_set.difference(cont_assign.keys())
            if len(non_assigned_vars) > 0:
                return None

            dec_expr = dec_expr.xreplace({sub_out: core.S(sub_in) for sub_out, sub_in in cont_assign.items()})
            assert isinstance(dec_expr, core.BooleanAtom) or isinstance(dec_expr, bool)
            return bool(dec_expr)
        else:
            return None
        
    def evaluate(
            self, 
            node_id: int, 
            bool_assign: Dict[core.Symbol, Union[core.BooleanAtom, bool]], 
            cont_assign: Dict[core.Symbol, Union[int, float]],
            primitive_type: bool = False,
    ) -> Union[float, int, bool, None]:
        """
        Evaluates a given node by inserting boolean and real values into variables.
        """
        # Get the root node
        node = self.get_exist_node(node_id)

        # Traverse the decision diagram until leaf is found
        while isinstance(node, XADDINode):
            branch_high: bool = self.evaluate_decision(self._id_to_expr[node.dec], bool_assign, cont_assign)

            # Not all required variables were assigned
            if branch_high is None:
                return
            
            # Advance down to the next node
            node = self.get_exist_node(node.high if branch_high else node.low)

        # Now at a terminal node; evaluate the expression
        t: XADDTNode = cast(XADDTNode, node)
        expr = t.expr
        expr = expr.xreplace({sub_out: core.S(sub_in) for sub_out, sub_in in cont_assign.items()})
        
        # Not all required variables were assigned
        if len(expr.free_symbols) > 0:
            return
        
        # Return python primitive type
        if primitive_type and isinstance(expr, core.BooleanAtom):
            return bool(expr)
        elif primitive_type:
            return float(expr)
        
        # Otherwise, return symengine instance
        return expr
    
    def unary_op(self, node_id: int, op: str, *args) -> int:
        """Applies a unary operation to a given node.
        This operation only affects the leaf values of a node.

        Args:
            node_id (int): The id of the operand node
            op (str): The unary operation to apply

        Returns:
            int: The ID of the resulting node
        """
        ret = self.unary_op_int(node_id, op, *args) 
        ret = self.make_canonical(ret)
        return ret
    
    def unary_op_int(self, node_id: int, op: str, *args) -> int:
        node = self.get_exist_node(node_id)

        if node.is_leaf():
            node = cast(XADDTNode, node)
            expr = node.expr
            if op == 'sgn':
                return self.sgn_op(node_id)
            elif op == '~':
                assert check_sym_boolean(expr)
                if isinstance(expr, core.BooleanAtom):
                    expr = core.logical_not(expr)
                    return self.get_leaf_node(expr, node._annotation)
                else:
                    assert isinstance(expr, BooleanVar)
                    return self.get_dec_node(expr, core.false, core.true)
            elif op == 'abs' and not isinstance(expr, core.Number):
                return self.abs_op(node_id)
            
            sp_op = UNARY_OP.get(op, None)
            if sp_op is None:
                raise ValueError(f"Unary operation {op} not recognized")
            expr = core.expand(sp_op(expr, *args))    # TODO: Need to simplify? that can take a while
            annotation = node._annotation
            return self.get_leaf_node(expr, annotation)
        
        # Handle an internal node
        node = cast(XADDINode, node)
        low = self.unary_op_int(node.low, op)
        high = self.unary_op_int(node.high, op)
        
        dec = node.dec
        ret = self.get_internal_node(dec, low, high)
        
        # TODO: do we need to cache the result?
        return ret

    def abs_op(self, node_id: int) -> int:
        """Implements the absolute value function.
        That is,

        abs(x) = x if x >= 0
                 -x otherwise

        In XADD, this is equivalent to
            ([x >= 0]
                ([x])
                ([-x])
            )
        """
        node = self.get_exist_node(node_id)
        assert node.is_leaf()
        
        node = cast(XADDTNode, node)
        expr = node.expr
        
        dec_expr = expr >= 0
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
        low =  self.unary_op(node_id, '-')
        high = node_id

        if is_reversed:
            low, high = high, low
        
        ret = self.get_internal_node(dec, low, high)
        ret = self.make_canonical(ret)
        return ret
    
    def sgn_op(self, node_id: int) -> int:
        """Implements the sign function...
        That is,

        sgn(x) = 1  if x > 0
                 0  if x == 0
                 -1 otherwise
        In XADD, this is equivalent to
            ([x <= 0]
                ([1])
                ([x == 0]
                    ([0])
                    ([-1])
                )
            )
        which requires introducing a decision that checkes equality.
        """
        node = self.get_exist_node(node_id)
        assert node.is_leaf()
        
        node = cast(XADDTNode, node)
        expr = node.expr
        
        dec_expr1 = expr <= 0
        dec1, is_reversed = self.get_dec_expr_index(dec_expr1, create=True)
        dec_expr2 = core.Eq(expr, 0)
        low = self.ONE        
        high = self.get_dec_node(dec_expr2, core.S(-1), core.S(0))
        
        if is_reversed:
            high, low = low, high
        
        ret = self.get_internal_node(dec1, low, high)
        ret = self.make_canonical(ret)
        return ret

    def scalar_op(self, node_id: int, val: float, op: str) -> int:
        scalar_node = self.get_leaf_node(core.S(val))
        return self.apply(node_id, scalar_node, op)

    def apply(self, id1: int, id2: int, op: str, annotation=None) -> int:
        """Recursively apply op(node1, node2)

        Args:
            id1 (int): ID of the first node
            id2 (int): ID of the second node
            op (str): (str) 'max', 'min', 'add', 'subtract', 'prod', 'div' (non-Boolean)
                                  'or', 'and' (Boolean)
                                  '!=', '==', '>', '>=', '<', '<=' (Relational)
            annotation (_type_, optional): _description_. Defaults to None.

        Returns:
            int: The resulting node ID
        """
        ret = self.apply_int(id1, id2, op, annotation)
        if op in ('min', 'max', '!=', '==', '>', '>=', '<', '<=', 'or', 'and'):
            ret = self.make_canonical(ret)
        return ret

    def get_apply_cache(self) -> dict:
        """This method is not used in this branch"""
        assert self._opt_var is not None
        hm = self._apply_caches.get(self._opt_var, None)
        if hm is not None:
            return hm
        else:
            hm = {}
            self._apply_caches[self._opt_var] = hm
            return hm

    def apply_int(self, id1: int, id2: int, op: str, annotation=None) -> int:
        """Recursively apply op(node1, node2)

        Args:
            id1 (int): ID of the first node
            id2 (int): ID of the second node
            op (str): (str) 'max', 'min', 'add', 'subtract', 'prod', 'div' (non-Boolean)
                                  'or', 'and' (Boolean)
                                  '!=', '==', '>', '>=', '<', '<=' (Relational)
            annotation (_type_, optional): _description_. Defaults to None.

        Returns:
            int: The resulting node ID
        """
        # Check apply cache and return if found
        if annotation is None:
            _tempApplyKey = (id1, id2, op)
            ret = self._apply_cache.get(_tempApplyKey, None)
        elif self._opt_var is None:
            _tempApplyKey = (id1, id2, op, annotation[0], annotation[1])
            ret = self._apply_cache.get(_tempApplyKey, None)
        else:
            _tempApplyKey2 = (id1, id2, op, annotation[0], annotation[1])
            ret = self.get_apply_cache().get(_tempApplyKey2, None)

        if ret is not None:
            return ret

        # If not found, compute..
        n1 = self.get_exist_node(id1)
        n2 = self.get_exist_node(id2)
        ret = self.compute_leaf_node(id1, n1, id2, n2, op, annotation)

        if ret is None:
            # Determine the new decision expression
            if not n1.is_leaf():
                if not n2.is_leaf():
                    if n2.dec >= n1.dec:
                        dec = n1.dec
                    else:
                        dec = n2.dec
                else:
                    dec = n1.dec
            else:
                dec = n2.dec

            # Determine next recursion for n1
            if (not n1.is_leaf()) and (n1.dec == dec):
                low1, high1 = n1.low, n1.high
            else:
                low1, high1 = id1, id1

            # Determine next recursion for n2
            if (not n2.is_leaf()) and (n2.dec == dec):
                low2, high2 = n2.low, n2.high
            else:
                low2, high2 = id2, id2

            low = self.apply_int(low1, low2, op, annotation)
            high = self.apply_int(high1, high2, op, annotation)

            ret = self.get_internal_node(dec, low, high)

        # Add result to apply cache
        if annotation is None:
            self._apply_cache[(id1, id2, op)] = ret
        elif self._opt_var is None:
            self._apply_cache[(id1, id2, op, annotation[0], annotation[1])] = ret
        else:
            self.get_apply_cache()[(id1, id2, op, annotation[0], annotation[1])] = ret
        return ret

    def compute_leaf_node(
            self, id1: int, n1: Node, id2: int, n2: Node, op: str, annotation: Union[tuple, None]
    ) -> Optional[int]:
        """_summary_

        Args:
            id1 (int): ID of the first node
            n1 (Node): The first node
            id2 (int): ID of the second node
            n2 (Node): The second node
            op (str): 'max', 'min', 'add', 'subtract', 'prod', 'div' (non-Boolean)
                                  'or', 'and' (Boolean)
                                  '!=', '==', '>', '>=', '<', '<=' (Relational)
            annotation (Union[tuple, None]): _description_

        Returns:
            (Optional) int: None if not terminal nodes; integer ID if terminal nodes
        """
        assert op in {
            'max', 'min', 'add', 'prod', 'subtract', 'div', 
            'and', 'or', 
            '!=', '==', '>', '>=', '<', '<='
        }

        # NaN cannot become valid by operations
        # But, this would not occur unless we intended..
        # Hence, just deal with NaNs when two leaf nodes are compared
        if ((id1 == self.NAN) or (id2 == self.NAN)) and (n1.is_leaf() and n2.is_leaf()):
            return self.NAN
        elif (n1.is_leaf() and id2 == self.ZERO and op == 'div'):       # Division by zero results in NAN
            return self.NAN
        elif (id1 == self.NAN) or (id2 == self.NAN):
            return None
        
        # 0 * x = 0
        if op == 'prod' and (id1 == self.ZERO or id2 == self.ZERO):
            return self.ZERO

        # Identities (1 * x = x, 0 + x = x, x / 1 = x, x - 0 = 0) (Note: annotations are ignored for these ops)
        if (op == 'add' and id1 == self.ZERO) or (op == 'prod' and id1 == self.ONE):
            return id2
        if (op == 'div' and id2 == self.ONE) or \
            (op == 'prod' and id2 == self.ONE) or \
            ((op == 'add' or op == 'subtract') and id2 == self.ZERO):
            return id1
        
        # Infinity identities  
        if n1.is_leaf() and n1.expr == oo:
            n1 = cast(XADDTNode, n1)
            if not n2.is_leaf():
                if op in ('max', 'add', 'subtract'):
                    return self.get_leaf_node_from_node(n1)                 # To retain annotation (if exists)
                elif op == 'min':
                    return id2
                elif op == '>=' or op == '>':
                    return self.TRUE
            else:
                n2 = cast(XADDTNode, n2)
                if n2.expr != oo and op in ('max', 'add', 'subtract'):
                    return self.get_leaf_node_from_node(n1)                 # To retain annotation (if exists)
                elif n2.expr != oo and op == 'min':
                    return id2 if annotation is None else self.get_leaf_node(n2.expr, annotation[1])
                elif n2.expr == oo and op in ('max', 'min', 'add', 'prod'):
                    ret_node = self.get_leaf_node(oo, annotation=self.NAN)
                    # self.add_special_node(ret_node)
                    return ret_node
            if op == '==':
                return self.FALSE                   # cannot evaluate equivalence of oo with other values
            elif op == '!=':
                return self.TRUE                    # (+-)oo will always be different than other values 

        elif n1.is_leaf() and n1.expr == -oo:
            n1 = cast(XADDTNode, n1)
            if not n2.is_leaf():
                if op in ('add', 'subtract', 'min'):
                    return self.get_leaf_node_from_node(n1)
                elif op == 'max':
                    return id2
                elif op == '<=' or op == '<':
                    return self.TRUE
            else:
                n2 = cast(XADDTNode, n2)
                if n2.expr != -oo and (op in ('add', 'min', 'subtract')):
                    return self.get_leaf_node_from_node(n1)
                elif n2.expr != -oo and op == 'max':
                    return id2 if annotation is None else self.get_leaf_node(n2.expr, annotation[1])
                elif n2.expr == -oo and (op in ('max', 'min', 'add')):
                    ret_node = self.get_leaf_node(-oo, annotation=self.NAN)
                    return ret_node
                elif n2.expr == -oo and op == 'prod':
                    ret_node = self.get_leaf_node(oo, annotation=self.NAN)
                    return ret_node
                elif op == '<=' or op == '<':
                    return self.TRUE
            if op == '==':
                return self.FALSE                   # cannot evaluate equivalence of oo with other values
            elif op == '!=':
                return self.TRUE                    # (+-)oo will always be different than other values 

        if n2.is_leaf() and n2.expr == oo:
            # n1 cannot be oo or -oo at this point..
            n2 = cast(XADDTNode, n2)
            if op == 'add' or op == 'max':
                return self.get_leaf_node_from_node(n2)
            elif op == 'min':
                if annotation is None:      # If annotation is given, need to annotate them at leaf nodes
                    return id1
                if n1.is_leaf():
                    return self.get_leaf_node(n1.expr, annotation[0])
            elif op == 'subtract':
                return self.NEG_oo
            elif op == 'div':
                return self.ZERO
            elif op == '>=' or op == '>':
                return self.FALSE
            elif op == '<=' or op == '<':
                return self.TRUE
            elif op == '==':
                return self.FALSE                   # cannot evaluate equivalence of oo with other values
            elif op == '!=':
                return self.TRUE                    # (+-)oo will always be different than other values 
            
        elif n2.is_leaf() and n2.expr == -oo:
            n2 = cast(XADDTNode, n2)
            if op == 'add' or op == 'min':
                return self.get_leaf_node_from_node(n2)
            elif op == 'max':
                if not n1.is_leaf():
                    return id1
                else:
                    return id1 if annotation is None else self.get_leaf_node(n1.expr, annotation[0])
            elif op == 'subtract':
                return self.oo
            elif op == 'div':
                return self.ZERO
            elif op == '<=' or op == '<':
                return self.FALSE
            elif op == '>=' or op == '>':
                return self.TRUE
            elif op == '==':
                return self.FALSE                   # cannot evaluate equivalence of oo with other values
            elif op == '!=':
                return self.TRUE                    # (+-)oo will always be different than other values 

        # Handle 'and' and 'or' operations when it can be immediately evaluated
        if op == 'or' or op == 'and':
            if n1.is_leaf() and isinstance(n1.expr, core.BooleanAtom):
                if op == 'or' and n1.expr:
                    return self.TRUE
                if op == 'and' and not n1.expr:
                    return self.FALSE
            if n2.is_leaf() and isinstance(n2.expr, core.BooleanAtom):
                if op == 'or' and n2.expr:
                    return self.TRUE
                if op == 'and' and not n2.expr:
                    return self.FALSE

        if n1.is_leaf() and n2.is_leaf():
            n1 = cast(XADDTNode, n1); n2 = cast(XADDTNode, n2)

            if id1 == self.NAN or id2 == self.NAN:
                return self.NAN
            
            # Operations: +, -, *, /
            # No need to take care of annotations for these operations
            # When an arithmetic operation is applied to boolean expression(s),
            # need to create a decision node
            n1_expr, n2_expr = n1.expr, n2.expr
            if op in ('add', 'subtract', 'prod', 'div') and \
                (isinstance(n1_expr, BooleanVar) or 
                 isinstance(n1_expr, core.BooleanAtom) or
                 isinstance(n2_expr, BooleanVar) or 
                 isinstance(n2_expr, core.BooleanAtom) 
            ):
                if isinstance(n1_expr, BooleanVar):
                    dec_node1 = self.get_dec_node(n1_expr, core.S(0), core.S(1))
                elif isinstance(n1_expr, core.BooleanAtom):
                    dec_node1 = self.ONE if n1_expr else self.ZERO
                else:
                    dec_node1 = id1
                if isinstance(n2_expr, BooleanVar):
                    dec_node2 = self.get_dec_node(n2_expr, core.S(0), core.S(1))
                elif isinstance(n2_expr, core.BooleanAtom):
                    dec_node2 = self.ONE if n2_expr else self.ZERO
                else:
                    dec_node2 = id2
                return self.apply(dec_node1, dec_node2, op)

            if op in ('add', 'subtract', 'prod', 'div', 'and', 'or'):

                if op == 'add':
                    result = n1_expr + n2_expr
                elif op == 'subtract':
                    result = n1_expr - n2_expr
                elif op == 'prod':
                    result = core.expand(n1_expr * n2_expr)
                elif op == 'div':
                    result = core.expand(n1_expr / n2_expr)
                else:
                    assert check_sym_boolean(n1_expr)
                    assert check_sym_boolean(n2_expr)
                    dec_n2 = self.get_dec_node(n2_expr, core.false, core.true)
                    dec_n1_id, is_reversed = self.get_dec_expr_index(n1_expr, create=True)
                    if op == 'and':
                        high, low = dec_n2, self.FALSE
                    elif op == 'or':
                        high, low = self.TRUE, dec_n2
                    if is_reversed:
                        high, low = low, high
                    result = self.get_internal_node(dec_n1_id, low, high)
                    return result
                return self.get_leaf_node(result)

            # The canonical form of decision expression: (lhs - rhs 'rel' 0)
            # Handle relational operations
            lhs = (n1.expr - n2.expr).expand()
            if op in RELATIONAL_OPERATOR:
                expr = RELATIONAL_OPERATOR[op](lhs, 0)      # can handle '==' (Equality), '!=' (Unequality), '>', '>=', '<', '<='
            # Handle min, max operations
            else:
                expr = core.LessThan(lhs, 0)

            # handle tautological cases
            if expr == core.true:        
                if op == 'min':             # n1 <= n2 holds
                    return self.get_leaf_node(n1.expr, annotation[0]) if annotation is not None else id1
                elif op == 'max':           # n1 <= n2 holds
                    return self.get_leaf_node(n2.expr, annotation[1]) if annotation is not None else id2
                else:                       # n1 (rel) n2 holds
                    return self.TRUE
            elif expr == core.false:
                if op == 'min':             # n1 > n2 holds
                    return self.get_leaf_node(n2.expr, annotation[1]) if annotation is not None else id2
                elif op == 'max':           # n1 > n2 holds
                    return self.get_leaf_node(n1.expr, annotation[0]) if annotation is not None else id1
                else:                       # n1 (rel) n2 does not hold
                    return self.FALSE
            
            if annotation is not None:
                id1 = self.get_leaf_node(n1.expr, annotation[0])
                id2 = self.get_leaf_node(n2.expr, annotation[1])
            else:
                id1 = self.get_leaf_node(n1.expr, n1._annotation)
                id2 = self.get_leaf_node(n2.expr, n2._annotation)

            expr_index, is_reversed = self.get_dec_expr_index(expr, create=True)
            
            # min / max operations
            if op == 'min' or op == 'max':
                # Swap low and high branches if reversed
                if is_reversed:
                    id1, id2 = id2, id1
                low = id1 if op == 'max' else id2
                high = id2 if op == 'max' else id1
            # relational operations
            else:
                high, low = self.TRUE, self.FALSE
                if is_reversed:
                    high, low = low, high
            return self.get_internal_node(expr_index, low=low, high=high)
        return None

    def substitute(
            self,
            node_id: int,
            subst_dict: Dict[core.Symbol, Union[core.Basic, float, int]]
    ) -> int:
        """
        Symbolic substitution method.

        Args:
            node_id (int): The ID of the node to be updated.
            subst_dict (dict): A dictionary of substitutions.
                A key is a variable and a value is an expression.
        Returns:
            int: The ID of the resulting node.
        """
        subst_cache = {}
        return self.reduce_sub(node_id, subst_dict, subst_cache)

    def reduce_sub(        
            self, 
            node_id: int,
            subst_dict: Dict[core.Symbol, Union[core.Basic, float, int]], 
            subst_cache: Dict[int, int],
    ) -> int:
        """Recursively perform substitution.

        Args:
            node_id (int): The ID of the node to be updated.
            subst_dict (dict): A dictionary of substitutions.
                A key is a variable and a value is an expression.
        Returns:
            int: The ID of the resulting node.
        """
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced by default
        if node.is_leaf():
            node = cast(XADDTNode, node)
            expr = node.expr
            if len(expr.free_symbols.intersection(set(subst_dict.keys()))) > 0:
                expr = expr.xreplace({sub_out: core.S(sub_in) for sub_out, sub_in in subst_dict.items()})
                expr = expr.expand()
            annotation = node._annotation
            return self.get_leaf_node(expr, annotation)

        # If it's an internal node, check the reduce cache
        ret = subst_cache.get(node_id, None)
        if ret is not None:
            return ret

        # Handle an internal node
        node = cast(XADDINode, node)
        low = self.reduce_sub(node.low, subst_dict, subst_cache)
        high = self.reduce_sub(node.high, subst_dict, subst_cache)

        dec = node.dec
        dec_expr = self._id_to_expr[dec]
        # When a Boolean variable is replaced
        if isinstance(dec_expr, VAR_TYPE) and dec_expr.is_Boolean:
            sub_in = subst_dict.get(dec_expr, None)
            is_reversed = False
            if sub_in is not None:
                # Handle tautologies
                if sub_in == core.true:
                    subst_cache[node_id] = high
                    return high
                elif sub_in == core.false:
                    subst_cache[node_id] = low
                    return low
                dec, is_reversed = self.get_dec_expr_index(sub_in, create=True)
        else:
            lhs = dec_expr.args[0]
            if len(lhs.free_symbols.intersection(set(subst_dict.keys()))) > 0:
                lhs = lhs.xreplace({sub_out: core.S(sub_in) for sub_out, sub_in in subst_dict.items()})
            
            # Check if the expression holds in equality and the true branch is NaN
            # Assuming canonical expression.. rhs is always 0. Hence, lhs == 0 iff dec_expr == core.true.
            # In this case, set dec_expr = False, so that false branch can be chosen instead.
            if lhs == 0 and high == self.NAN:
                dec_expr = core.false
            elif lhs == 0 and low == self.NAN:
                dec_expr = core.true
            else:
                dec_expr = lhs <= 0

            # # Handle tautologies
            if dec_expr == core.true:
                subst_cache[node_id] = high
                return high
            elif dec_expr == core.false:
                subst_cache[node_id] = low
                return low

            dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)

        # Swap low and high branches if reversed
        if is_reversed:
            high, low = low, high

        # # Substitution could have affected variable ordering.
        if not self._direct_substitution:
            ret = self.get_inode_canon(dec, low, high)
            self.check_local_ordering_and_exit_on_error(ret)
        else:
            ret = self.get_internal_node(dec, low, high)

        # Put return value in cache and return
        subst_cache[node_id] = ret
        return ret

    def substitute_bool_vars(self, node_id: int, subst_dict: dict) -> int:
        """
        Symbolic substitution method for bool variables.
        
        Args:
            node_id (int): The ID of the node to be updated.
            subst_dict (dict): A dictionary of substitutions.
                A key is a Boolean variable and a value is a Boolean value.

        Returns:
            int: The ID of the resulting node.
        """
        assert all(
                    map(lambda x: isinstance(x, core.BooleanAtom) or isinstance(x, bool), subst_dict.values())
                ), "All values of the `subst_dict` should be boolean type"
        var_set = self.collect_vars(node_id)
        for var in subst_dict:
            if var in var_set:
                dec_id, _ = self.get_dec_expr_index(var, create=False)
                if subst_dict[var]:
                    node_id = self.op_out(node_id, dec_id, "restrict_high")
                else:
                    node_id = self.op_out(node_id, dec_id, "restrict_low")
        return node_id

    def op_out(self, node_id: int, dec_id: int, op: str) -> int:
        """Implements variable elimination of a Boolean decision variable.
        
        Args:
            node_id (int): The ID of the node to which the operation is applied.
            dec_id (int): The ID of the decision variable to be eliminated.
            op (str): The operation to be applied.
                'restrict_low': Eliminate the decision variable by setting it to False.
                'restrict_high': Eliminate the decision variable by setting it to True.
                'add': Sum out the decision variable.
                'prod': Product out the decision variable.
        
        Returns:
            int: The ID of the resulting node.
        """
        # Check if the node contains the decision variable.
        v = self._id_to_expr[dec_id]
        assert isinstance(v, BooleanVar), (
            "`op_out` can only be applied to Boolean variables"
            f"but the given expression is {v}"
        )
        # If does not contain the variable, apply the `op` to the node.
        var_set = self.collect_vars(node_id)
        if v not in var_set:
            return self.apply(node_id, node_id, op)

        # Otherwise, reduce.
        ret = self.reduce_op(node_id, dec_id, op)

        # Operations like sum and product may get decisions out of order.
        if op == 'add' or op == 'prod':
            return self.make_canonical(ret)
        else:
            return ret

    def reduce_op(self, node_id: int, dec_id: int, op: str) -> int:
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced (and cannot be restricted).
        if node.is_leaf():
            return node_id

        # If it's an internal node, check the reduce cache.
        temp_reduce_key = (node_id, dec_id, op)
        ret = self._reduce_cache.get(temp_reduce_key)
        if ret is not None:
            return ret

        node = cast(XADDINode, node)
        if (op != "restrict_high") or (dec_id != node.dec):
            low = self.reduce_op(node.low, dec_id, op)
        if (op != "restrict_low") or (dec_id != node.dec):
            high = self.reduce_op(node.high, dec_id, op)
        if (dec_id != -1) and (dec_id == node.dec):
            # ReduceOp.
            if op == "restrict_low":
                ret = low
            elif op == "restrict_high":
                ret = high
            elif op == "add" or op == "prod":
                ret = self.apply(low, high, op)
            else:
                raise NotImplementedError
        else:
            ret = self.get_internal_node(node.dec, low, high)

        # Put return value in cache and return.
        self._reduce_cache[temp_reduce_key] = ret
        return ret

    def collect_vars(self, node_id: int) -> Set[VAR_TYPE]:
        node = self.get_exist_node(node_id)
        var_set = node.collect_vars()
        return var_set

    def reduced_arg_min_or_max(self, node_id: int, var: VAR_TYPE) -> int:
        arg_id = self.get_arg(node_id)
        arg_id = self.reduce_lp(arg_id)
        self.update_anno(var, arg_id)
        return arg_id

    def get_arg(self, node_id: int, is_min: bool = True) -> int:
        self._is_min = is_min
        ret = self.get_arg_int(node_id)
        return self.make_canonical(ret)

    def get_arg_int(self, node_id: int) -> int:
        """
        node_id is the id of a min/max XADD node.
        var is the decision variable that was max(min)imized out to result in node_id. 
        Recursively build the annotation XADD for var.
        
        Args:
            node_id:    (int) XADD node ID.
        Returns:
            int: The resulting XADD node ID.
        """
        node = self.get_exist_node(node_id)

        if node.is_leaf():
            annotation = node._annotation
            if annotation is None and (node.expr == oo or node.expr == -oo):    # Annotate infeasibility
                annotation = self.NAN
            return annotation
        else:
            low = self.get_arg_int(node.low)
            high = self.get_arg_int(node.high)

            dec = node.dec
            dec_expr = self._id_to_expr[dec]
            dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
            # Swap low and high branches if reversed
            if is_reversed:
                high, low = low, high
            return self.get_inode_canon(dec, low, high)

    def get_all_anno(self) -> Dict[core.Symbol, int]:
        return self._var_to_anno

    def get_annotation(self, var: core.Symbol) -> int:
        return self._var_to_anno[var]

    def update_anno(self, var: VAR_TYPE, anno: int):
        if not hasattr(self, '_var_to_anno'):
            self._var_to_anno: Dict[core.Symbol, int] = {}
        self._var_to_anno[var] = anno

    def get_node(self, node_id: int) -> Node:
        """
        Retrieve a XADD node from cache.
        
        Args:
            node_id:    (int) XADD node ID.
        Returns:
            Node: The corresponding XADD node.
        """
        return self._id_to_node[node_id]

    def min_or_max_multi_var(
            self,
            node_id: int,
            var_lst: List[core.Symbol],
            is_min: bool = True,
            annotate: bool = True,
    ) -> int:
        """
        Given an XADD root node 'node_id', minimize (or maximize) variables in 'var_lst'.
        Supports only continuous variables.
        """
        decisions, decision_values = [], []
        min_or_max = XADDLeafMultivariateMinOrMax(
            var_lst,
            is_max=False if is_min else True,
            bound_dict=self._var_to_bound,
            context=self,
            annotate=annotate,
        )
        _ = self.reduce_process_xadd_leaf(node_id, min_or_max, decisions, decision_values)
        res = min_or_max._running_result
        return res

    def min_or_max_var(
            self,
            node_id: int,
            var: VAR_TYPE,
            is_min: bool = True,
            annotate: bool = False,
    ) -> int:
        """
        Given an XADD root node 'node_id', minimize (or maximize) 'var' out.
        Can also handle Boolean variables.

        Args:
            node_id:    (int) XADD node ID.
            var:        (VAR_TYPE) variable to minimize (or maximize).
            is_min:     (bool) True if minimize, False if maximize.
            annotate:   (bool) True if annotate the argmax(min), False otherwise.
        
        Returns:
            int: The resulting XADD node ID.
        """
        # Check if Boolean.
        if var in self._bool_var_set or isinstance(var, BooleanVar):
            op = "min" if is_min else "max"
            self._opt_var = var
            subst_high = {var: True}
            subst_low = {var: False}
            restrict_high = self.substitute_bool_vars(node_id, subst_high)
            restrict_low = self.substitute_bool_vars(node_id, subst_low)
            annotation = (self.TRUE, self.FALSE) if annotate else None
            running_result = self.apply(restrict_high, restrict_low, op=op, annotation=annotation)
            running_result = self.reduce_lp(running_result)
        # Continuous variables.
        else:
            decisions, decision_values = [], []
            min_or_max = XADDLeafMinOrMax(
                var,
                is_max=False if is_min else True,
                bound_dict=self._var_to_bound,
                context=self,
                annotate=annotate,
            )
            _ = self.reduce_process_xadd_leaf(node_id, min_or_max, decisions, decision_values)
            running_result = min_or_max._running_result
        return running_result

    def substitute_xadd_for_var_in_expr(
            self, leaf_val: core.Basic, var: core.Symbol, xadd: int
    ) -> int:
        """
        Substitute XADD into 'var' that occurs in 'val' (a SymEngine expression). 
        This is only called for leaf expressions.

        Args:
            leaf_val:    (core.Basic) symengine expression.
            var:         (core.Symbol) variable to substitute.
            xadd:        (int) integer that indicates the XADD to substitute into 'var'.
        Returns:
            int: The resulting XADD node ID.
        """
        # Get the root node.
        node = self.get_exist_node(xadd)

        # Handle leaf node cases: simply substitute leaf expression into 'var' in leaf_val.
        if node.is_leaf():
            node = cast(XADDTNode, node)
            xadd_leaf_expr = node.expr
            expr = leaf_val.xreplace({var: xadd_leaf_expr})
            expr = core.expand(expr)

            # Special treatment for oo, -oo.
            try:
                args = expr.args
                if len(args) > 0 and isinstance(expr.args[0], core.Number):
                    if args[0] == core.oo:
                        expr = core.oo
                    elif args[0] == -core.oo:
                        expr = -core.oo
            except AttributeError as e:
                pass
            except Exception as e:
                logger.error(e)
                exit(1)
            node_id = self.get_leaf_node(expr, annotation=None)
            return node_id

        # Internal nodes: get low and high branches and do recursion.
        low, high = node.low, node.high
        low = self.substitute_xadd_for_var_in_expr(leaf_val, var, low)
        high = self.substitute_xadd_for_var_in_expr(leaf_val, var, high)

        # Get the node id for a (sub)XADD and return it.
        node_id = self.get_internal_node(node.dec, low=low, high=high)

        return node_id

    def compute_definite_integral(self, node_id: int, var: core.Symbol) -> int:
        """Computes a definite integral over a variable in an XADD."""
        integrator = XADDLeafDefIntegral(var, self)
        _ = self.reduce_process_xadd_leaf(node_id, integrator, [], [])
        return integrator.running_sum

    def get_repr(self, node_id: int) -> str:
        # For printing out the representation.
        node = self._id_to_node[node_id]
        return repr(node)

    def get_leaf_node_from_node(self, node: XADDTNode) -> int:
        """
        Get the node ID of the leaf node with the same expression as the given node.

        Args:
            node:    (XADDTNode) XADD terminal node.
        Returns:
            int: The resulting XADD node ID.
        """
        expr, annotation = node.expr, node._annotation
        return self.get_leaf_node(expr, annotation)

    def get_leaf_node(
            self, expr: core.Basic, annotation: Optional[int] = None, **kwargs
    ) -> int:
        """Returns the ID of the leaf node with given SymEngine expression and annotation.

        Note that if a new random variable is added within this method,
        kwargs should have the necessary parameters to specify the random variable.
        
        For example, for a uniform random variable, we need
            {'params': [lb, ub]} where lb and ub are the lower and upper bounds of the uniform 
            distribution, respectively.
        
        If this information was not provided, this method will result in an assertion error.
        
        Args:
            expr (core.Basic): The SymEngine expression associated with the leaf node.
            annotation (Optional[int], optional): The node ID of the annotation.
        
        Returns:
            int: The resulting XADD node ID.
        """
        self._temp_term_node.set(expr, annotation)
        node_id = self._node_to_id.get(self._temp_term_node, None)
        if node_id is None:
            # Node not in cache, so create.
            node_id = self._nodeCounter
            node = XADDTNode(expr, annotation, context=self)
            self._id_to_node[node_id] = node
            self._node_to_id[node] = node_id
            self._nodeCounter += 1

            # Add in all new variables.
            vars_in_expr = expr.free_symbols.copy()
            diff_vars = vars_in_expr.difference(self._cont_var_set).difference(self._bool_var_set)
            for v in diff_vars:
                if isinstance(v, BooleanVar) or (
                    isinstance(v, RandomVar) and str(v).startswith('Bernoulli')
                ):
                    assert len(vars_in_expr) == 1, (
                        f'BooleanVar {v} should be the only variable in the expression.'
                    )
                    self.add_boolean_var(v)
                else:
                    self.add_continuous_var(v)
                if isinstance(v, RandomVar):
                    assert kwargs.get('params') is not None
                    assert kwargs.get('type') is not None and kwargs['type'] in ACCEPTED_RV_TYPES
                    self.add_random_var(v, **kwargs)
        return node_id

    def get_dec_node(
            self, 
            dec_expr: DECISION_TYPE,
            low_val: core.Basic, 
            high_val: core.Basic
    ) -> int:
        """
        Get decision node with relational expression having dec, 
            whose low and high values are also given.
        
        Args:
            dec_expr:   (DECISION_TYPE) decision expression.
            low_val:    (core.Basic) low branch expression.
            high_val:   (core.Basic) high branch expression.
        
        Returns:
            int: The resulting XADD node ID.
        """
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
        low = self.get_leaf_node(low_val)
        high = self.get_leaf_node(high_val)
        # Swap low and high branches if reversed.
        if is_reversed:
            high, low = low, high
        return self.get_internal_node(dec, low, high)
    
    def canonical_dec_expr(
            self, expr: core.Basic
    ) -> Tuple[Union[Tuple[core.Basic, core.Basic], core.Basic], bool]:
        """
        Return canonical form of an expression.
        It should always take the following form: expr.args[0] <= 0.
        
        Args:
            expr:  (core.Basic) SymEngine expression.
        
        Returns:
            Tuple[Union[Tuple[core.Basic, core.Basic], core.Basic], bool]: 
                The canonical form of the given expression 
                    and whether the expression is reversed.
        """
        is_reversed = False

        # Handle tautology: simply return without doing anything.
        if expr == core.true:
            return expr, is_reversed
        elif expr == core.false:
            return expr, is_reversed

        # Handle boolean expressions.
        if not isinstance(expr, core.Rel):
            if not isinstance(expr, BooleanVar) and expr.is_symbol:
                if not (isinstance(expr, RandomVar) and str(expr).startswith('Bernoulli')):
                    expr = BooleanVar(expr)
                if expr not in self._bool_var_set:
                    logger.info(f'Random variable {expr} will be treated as Boolean')
                if expr in self._cont_var_set:
                    self._cont_var_set.remove(expr)
                self.add_boolean_var(expr)
            else:
                assert expr.is_Boolean and expr.is_symbol, (
                    f'We only support a single Boolean variable as a'
                    f' decision variable, but got {expr}.'
                )
            return expr, is_reversed
        
        # Always make 'lhs - rhs <= 0' as canonical expression.
        lhs, rhs, rel = expr.args[0], expr.args[1], REL_TYPE[type(expr)]
        lhs = (lhs - rhs).expand()

        if rel == '>=' or rel == '>':
            is_reversed = True
            rel = '<=' if rel == '>=' else '<'

        # Divide lhs by the coefficient of the first term and make it positive.
        if not lhs.is_Symbol:
            sorted_args = tuple(
                sorted(lhs.args, 
                    key=lambda x: ''.join([str(x) for x in x.free_symbols]) 
                            if x.free_symbols
                            else str(x)
                ) 
            )
            coeff_first_term = sorted_args[0]
            if isinstance(coeff_first_term, core.Number):
                # TODO: is this always canonical?. Need to check.
                coeff_first_term = sorted_args[1]

            if isinstance(coeff_first_term, core.Mul):
                arg1 = coeff_first_term.args[0]
                if isinstance(arg1, core.Number):
                    lhs = (lhs / arg1).expand()
                    # Divided by a negative number changes the direction of inequality.
                    if arg1 < 0 and rel in ('<=', '<', '>', '>='):
                        is_reversed = True if not is_reversed else False

            # If possible, use integer coefficients.
            lhs = sum([(int(c) * t) if (int(c) == float(c))
                       else (c * t)
                       for t, c in lhs.as_coefficients_dict().items()])
        expr = RELATIONAL_OPERATOR[rel](lhs, 0)
        return expr, is_reversed

    def get_dec_expr_index(
            self, expr: core.Basic, create: bool, canon: bool = False, **kwargs
    ) -> Tuple[Union[core.BooleanAtom, int], bool]:
        """Given a symbolic expression 'expr', returns its index and `is_reversed` flag.
        
        Note that if a new random variable is included in the expression,
            kwargs should have the necessary parameters to specify the random variable.
        
        For example, for a uniform random variable, we need
            {'params': [lb, ub]} where lb and ub are the lower and upper bounds
            of the uniform distribution, respectively.
        
        If this information was not provided, this method will result in an assertion error.
        
        Args:
            expr (core.Basic): The expression to be used as a decision. This can be a relational
                expression or just a boolean variable.
            create (bool): Whether to assign a new ID for the given expression.
            canon (bool, optional): Deprecated... TODO: check whether this can safely removed.

        Returns:
            Tuple[int, bool]: The index of the given expression and `is_reversed` flag.
        """
        is_reversed = False
        if not canon:
            expr, is_reversed = self.canonical_dec_expr(expr)

        index = self._expr_to_id.get(expr, None)

        if index is None:
            index = 0

        # If found, and not create.
        if index != 0 or not create:
            return index, is_reversed
        # If nothing's found, create one and store.
        else:
            index = XADD._func_var_index(self, expr)
            self._expr_to_id[expr] = index
            self._id_to_expr[index] = expr
            
            # Check whether the expression is at most linear in free variables.
            is_linear = check_expr_linear(expr)
            self._expr_to_linear_check[expr] = is_linear
            self._expr_id_to_linear_check[index] = is_linear
            
            # Add in all new variables.
            # (1) If a single symbol, it has to be Boolean.
            if isinstance(expr, VAR_TYPE):
                logger.info(
                    f'Variable {expr} of type ({type(expr)}) is used as a decision variable.'
                )
                self.add_boolean_var(expr)
            # (2) Otherwise, all continuous variables, including RVs.
            else:
                vars_in_expr = expr.free_symbols.copy()
                diff_vars = vars_in_expr.difference(self._cont_var_set)
                for v in diff_vars:
                    self.add_continuous_var(v)

                    if isinstance(v, RandomVar):
                        assert kwargs.get('params') is not None
                        assert kwargs.get('type') is not None and kwargs['type'] in ACCEPTED_RV_TYPES
                        self.add_random_var(v, **kwargs)
        return index, is_reversed

    def get_exist_node(self, node_id: int) -> Node:
        """Returns the XADD node with the given ID."""
        node = self._id_to_node.get(node_id, None)
        if node is None:
            logger.info("Unexpected Missing node: " + node_id)
        return node

    def get_internal_node(self, dec_id: int, low: int, high: int) -> int:
        """Returns the ID of the internal node.

        Args:
            dec_id (int): The ID of the decision expression.
            low (int): The ID of the low branch.
            high (int): The ID of the high branch.
        Returns:
            int: The ID of the internal node.
        """
        if dec_id < 0:
            high, low = low, high
            dec_id = -dec_id

        # Check if low == high.
        if low == high:
            return low

        # Handle tautological cases.
        dec_expr = self._id_to_expr.get(dec_id, None)
        if dec_expr == core.true:
            return high
        elif dec_expr == core.false:
            return low

        # Retrieve XADDINode (create if it does not exist).
        self._tempINode.set(dec_id, low, high)
        node_id = self._node_to_id.get(self._tempINode, None)
        if node_id is None:
            node_id = self._nodeCounter
            node = XADDINode(dec_id, low, high, context=self)
            self._node_to_id[node] = node_id
            self._id_to_node[node_id] = node
            self._nodeCounter += 1
        return node_id

    def reduce_sample(
            self, 
            node_id: int, 
            use_expectation: bool = False, 
            rng: np.random.Generator = None
    ) -> int:
        """Samples all random variables existing in the given node.

        Args:
            node_id (int): The node to reduce.
            use_expectation (bool): Whether to use the expected value instead of sampling.
                Defaults to False.
            rng (np.random.Generator): The random number generator to use.

        Returns:
            int: The ID of the reduce node after sampling.
        """
        if rng is None:
            rng = np.random.default_rng()

        node = self.get_exist_node(node_id)

        if node.is_leaf():
            node = cast(XADDTNode, node)
            expr = node.expr
            rvs = [v for v in expr.free_symbols if v in self._random_var_set]
            if len(rvs) == 0:
                return node_id
            types = [self._rv_to_type[rv] for rv in rvs]
            params = [self._rv_to_params[rv] for rv in rvs]
            samples = sample_rvs(rvs, types, params, rng, use_expectation)
            expr = expr.xreplace(samples)
            return self.get_leaf_node(expr, annotation=node._annotation)
        
        # Handle an internal node.
        node = cast(XADDINode, node)
        low = self.reduce_sample(node.low, use_expectation, rng)
        high = self.reduce_sample(node.high, use_expectation, rng)

        dec = node.dec
        dec_expr = self._id_to_expr[dec]
        rvs = [v for v in dec_expr.free_symbols if v in self._random_var_set]
        if len(rvs) != 0:
            types = [self._rv_to_type[rv] for rv in rvs]
            params = [self._rv_to_params[rv] for rv in rvs]
            samples = sample_rvs(rvs, types, params, rng, use_expectation)
            dec_expr = dec_expr.xreplace(samples)
            dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
            if is_reversed:
                low, high = high, low

        return self.get_internal_node(dec, low, high)

    def reduce_process_xadd_leaf(
            self, 
            node_id: int, 
            leaf_op,    # XADDLeafOperation
            decisions: List[DECISION_TYPE], 
            decision_values: List[bool],
    ) -> int:
        """Recursively process the XADD node with the given leaf operation.

        Args:
            node_id (int): The node to process.
            leaf_op (XADDLeafOperation): The leaf operation to apply.
            decisions: (List[DECISION_TYPE]) List of decisions.
            decision_values: (List[bool]) List of decision values.
        Returns:
            int: The resulting XADD node ID.
        """
        node = self.get_exist_node(node_id)
        if node.is_leaf():
            return leaf_op.process_xadd_leaf(decisions, decision_values, node.expr)

        # Internal node
        dec_expr = self._id_to_expr.get(node.dec)

        # Recurse the False branch.
        decisions.append(dec_expr)
        decision_values.append(False)
        low = self.reduce_process_xadd_leaf(node.low, leaf_op, decisions, decision_values)

        # Recurse the True branch.
        decision_values[-1] = True
        high = self.reduce_process_xadd_leaf(node.high, leaf_op, decisions, decision_values)

        decisions.pop()
        decision_values.pop()

        ret = self.get_internal_node(node.dec, low, high)
        if leaf_op._require_canonical:
            ret = self.make_canonical(ret)

        # # Put return value in cache and return.
        # Different leaf_op has different properties... skipping now.
        # self._reduce_leafop_cache[(node_id, leaf_op.__class__.__name__)] = ret
        return ret
    
    """
    Verifying feasibility and redundancy of all paths in the XADD.
    """
    def reduce_lp(self, node_id: int) -> int:
        """Consistency and redundancy checking.
        
        Args:
            node_id (int): The node to check.
        Returns:
            int: The resulting XADD node ID.
        """
        if self.perform_reduce_lp:
            return self.RLPContext.reduce_lp(node_id)
        return node_id

    @property
    def perform_reduce_lp(self):
        return self._perform_reduce_lp

    @perform_reduce_lp.setter
    def perform_reduce_lp(self, val: bool):
        self._perform_reduce_lp = val

    """
    Related to MILP compilation of XADD.
    """
    def set_objective(self, obj: Union[int, dict]):
        self._obj = obj
        if isinstance(obj, dict):
            self._additive_obj = True

    def get_objective(self):
        return self._obj

    """
    Cache maintenance.
    """
    def clear_special_nodes(self):
        self._special_nodes.clear()

    def add_special_node(self, n: int):
        try:
            if n is None:
                raise ValueError("add_sepcial_node: None")
        except ValueError as error:
            logger.error(error)
            exit(1)
        self._special_nodes.add(n)

    def remove_special_node(self, n: int):
        self._special_nodes.discard(n)

    def flush_caches(self):
        logger.info(f"[FLUSHING CACHES...  {len(self._node_to_id) + len(self._id_to_node)}, nodes -> ")

        # Can always clear these
        self._reduce_cache.clear()
        self._reduce_canon_cache.clear()
        self._reduce_leafop_cache.clear()
        self._apply_cache.clear()
        for applyCache in self._apply_caches.values():
            applyCache.clear()
        self._inode_to_vars.clear()
        self._temp_ub_lb_cache.clear()
        self.RLPContext.flush_implications()

        # Set up temporary alternates to these HashMaps
        self._node_to_id_new = {}
        self._id_to_node_new = {}

        # Copy over 'special' nodes then set new dict
        for node_id in self._special_nodes:
            self.copy_in_new_cache_node(node_id)

        self._node_to_id = self._node_to_id_new
        self._id_to_node = self._id_to_node_new
        self.create_standard_nodes()

        logger.info(f"{len(self._node_to_id)+len(self._id_to_node)} nodes")

    def copy_in_new_cache_node(self, node_id: int):
        if node_id in self._id_to_node_new:
            return
        node = self.get_exist_node(node_id)
        if not node.is_leaf():
            # Copy node and node id
            self._id_to_node_new[node_id] = node
            self._node_to_id_new[node] = node_id
            # Recurse
            self.copy_in_new_cache_node(node.high)
            self.copy_in_new_cache_node(node.low)
        else:
            self._id_to_node_new[node_id] = node
            self._node_to_id_new[node] = node_id

    """
    Export and import XADDs.
    """
    def export_xadd(
            self,
            node_id: int,
            fname: str,
            append: bool = False,
            include_node_info: bool = False
    ):
        """Export the XADD node to a file.
        If append is True, then open the file in the append mode.

        Args:
            node_id (int): The ID of the node to be exported.
            fname (str): The file name to export the XADD node.
            append (bool, optional): Whether to append to the file.
                Defaults to False.
            include_node_info (bool, optional): Whether to include node
                information. Defaults to False.
        """
        # Firstly, turn off printing node info.
        node: Node = self._id_to_node.get(node_id, None)
        if node is None:
            raise KeyError(f'There is no node with id {node_id}')
        if not include_node_info:
            node.turn_off_print_node_info()

        if append:
            with open(fname, 'a+') as f:
                f.write('\n')
                f.write(str(node))
        else:
            with open(fname, 'w+') as f:
                f.write(str(node))

        # Turn the printing mode back on.
        node.turn_on_print_node_info()

    def import_xadd(
            self, 
            fname: Optional[str] = None,
            xadd_str: Optional[str] = None,
            to_canonical: bool = True,
    ) -> int:
        """Import the XADD node from a file or a string.

        Args:
            fname (Optional[str], optional): The file name
                to import the XADD node. Defaults to None.
            xadd_str (Optional[str], optional): The string
                to import the XADD node. Defaults to None.
            to_canonical (bool, optional): Whether to make the XADD canonical.
        
        Returns:
            int: The ID of the imported XADD node.
        """
        assert (
            (fname is not None and xadd_str is None) or 
            (fname is None and xadd_str is not None)
        ), "Specify either a file name or a string, not both."

        if fname is not None:
            with open(fname, 'r') as f:
                xadd_str = f.read().replace('\n', '')

        # Note: when it is just a leaf expression: not supported.
        if xadd_str.rfind('(') == 0 and xadd_str.rfind('[') == 2:
            xadd_as_list = [core.sympify(xadd_str.strip('( [] )'))]
        else:
            xadd_as_list = parse_xadd_grammar(xadd_str)[1][0]
        node_id = self.build_initial_xadd(xadd_as_list, to_canonical=to_canonical)
        return node_id

    """
    Graph visualization.
    """
    def get_graph(self, node_id: int, name: str = '') -> Graph:
        """Creates a graph view of a given node."""
        try:
            graph = Graph(
                name=name, 
                directed=True,
            )
            root = self.get_exist_node(node_id)
            root.to_graph(graph, node_id)
            return graph
        except AttributeError:
            warnings.warn("You need to install 'pygraphviz' to construct graph visualization")
            raise Exception

    def save_graph(self, node_id: int, file_name: str):
        """Saves the graph visualization of a given node."""
        graph = self.get_graph(node_id)
        graph.configure()

        f_dir = Path('./tmp')
        f_dir.mkdir(exist_ok=True, parents=True)
        if file_name.endswith('png'):
            file_name = file_name.replace('png', 'pdf')
        elif not file_name.endswith('pdf'):
            file_name = file_name + '.pdf'
        f_path = f_dir / file_name

        graph.draw(f_path, prog='dot')



def get_xadd_bounds(
        context: XADD,
        v: core.Symbol,
        decisions: List[core.Basic],
        decision_values: List[bool],
        lb: float | core.Number = -oo,
        ub: float | core.Number = oo,
) -> Tuple[int, int, List[Tuple[core.Basic, bool]]]:
    """Returns the lower and upper bound XADDs for a given variable."""
    lower_bound = [core.S(lb)]
    upper_bound = [core.S(ub)]

    # Independent decisions.
    var_indep_decisions = []

    for dec_expr, is_true in zip(decisions, decision_values):
        # Check boolean decisions or if `v` is in `dec_expr`.
        if isinstance(dec_expr, BooleanVar) or v not in dec_expr.atoms():
            var_indep_decisions.append((dec_expr, is_true))
            continue

        lhs, rhs = dec_expr.args[0], dec_expr.args[1]
        lt = isinstance(dec_expr, core.LessThan)
        lt = (lt and is_true) or (not lt and not is_true)
        expr = lhs <= rhs if lt else lhs >= rhs

        # Get bounds over `v`.
        bound_expr, upper = xadd_util.get_bound(v, expr)
        if upper:
            upper_bound.append(bound_expr)
        else:
            lower_bound.append(bound_expr)

    # lower bound over `v` is the maximum among lower bounds.
    xadd_lower_bound = -1
    for e in lower_bound:
        xadd_lower_bound = (
            context.get_leaf_node(e) if xadd_lower_bound == -1
            else context.apply(xadd_lower_bound, context.get_leaf_node(e), op='max'))

    # upper bound over `v` is the minimum among upper bounds.
    xadd_upper_bound = -1
    for e in upper_bound:
        xadd_upper_bound = (
            context.get_leaf_node(e) if xadd_upper_bound == -1
            else context.apply(xadd_upper_bound, context.get_leaf_node(e), op='min'))

    # Reduce lower and upper bound xadds.
    xadd_lower_bound = context.reduce_lp(xadd_lower_bound)
    xadd_upper_bound = context.reduce_lp(xadd_upper_bound)

    # Ensure lower bounds are smaller than upper bounds.
    for e1 in lower_bound:
        for e2 in upper_bound:
            comp = core.LessThan((e1 - e2).expand(), 0)     # lb - ub <= 0.
            if comp == core.true or e2 == oo or e1 == -oo:
                continue
            var_indep_decisions.append((comp, True))

    return xadd_lower_bound, xadd_upper_bound, var_indep_decisions


class NullDec:
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, NullDec)


class XADDLeafOperation(metaclass=abc.ABCMeta):
    def __init__(self, context: XADD):
        self._context: XADD = context
        self._require_canonical = False

    @abc.abstractmethod
    def process_xadd_leaf(
        self, decisions: List[DECISION_TYPE], decision_values: List[bool], leaf_val: core.Basic
    ) -> int:
        pass


class ControlFlow(XADDLeafOperation):
    def __init__(
            self,
            true_branch: int,
            false_branch: int,
            context: XADD,
    ):
        """Returns a node that implements a control flow.
        
        That is, if a leaf node evaluates to true, the true branch will be attached,
        otherwise the false branch is attached.

        This operation needs to call make_canonical to correct the variable ordering.

        Args:
            true_branch (int): The node id for the true branch
            false_branch (int): The node id for the false branch
            context (XADD): The XADD context manager
        """
        super().__init__(context)
        self._true_branch = true_branch
        self._false_branch = false_branch

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        assert check_sym_boolean(leaf_val) or leaf_val == 1 or leaf_val == 0
        
        if isinstance(leaf_val, core.BooleanAtom):
            if leaf_val:
                return self._true_branch
            else:
                return self._false_branch
        
        dec, is_reversed = self._context.get_dec_expr_index(leaf_val, create=True)
        low = self._false_branch
        high = self._true_branch
        if is_reversed:
            high, low = low, high
        ret = self._context.get_internal_node(dec, low, high)
        ret = self._context.make_canonical(ret)
        return ret


class DeltaFunctionSubstitution(XADDLeafOperation):
    def __init__(
            self,
            sub_var: core.Symbol,
            xadd_sub_at_leaves: int,
            context: XADD,
            is_linear: bool = True,
    ):
        """
        From the case statement of 'xadd_sub_at_leaves', 
        all occurrences of sub_var will be replaced.
        """
        super().__init__(context)
        self._require_canonical = True
        self._leafSubs: Dict[core.Symbol, Union[float, int, bool]] = {}
        self._xadd_sub_at_leaves: int = xadd_sub_at_leaves
        self._subVar = sub_var
        self._is_linear = is_linear

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        self._leafSubs = {}
        
        if leaf_val == core.nan:
            return self._context.NAN
        # If boolean variable, handle differently
        elif self._subVar in self._context._bool_var_set:
            self._leafSubs[self._subVar] = True \
                if leaf_val == 1 or \
                    ((isinstance(leaf_val, core.BooleanAtom) or isinstance(leaf_val, bool)) 
                        and leaf_val)\
                else False
            return self._context.substitute_bool_vars(self._xadd_sub_at_leaves, self._leafSubs)
        # Continuous variable
        else:
            self._leafSubs[self._subVar] = leaf_val
            ret = self._context.substitute(self._xadd_sub_at_leaves, self._leafSubs)
            if self._is_linear:
                ret = self._context.reduce_lp(ret)
            return ret


class XADDLeafIndefIntegral(XADDLeafOperation):
    """Class for indefinite integral."""

    def __init__(
        self,
        var: core.Symbol,
        context: XADD,
    ):
        super().__init__(context)
        self.var = var

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        # Return an XADD for the resulting expression.
        sympy_expr = leaf_val._sympy_()
        sympy_var = self.var._sympy_()
        sympy_integral = sp.integrate(sympy_expr, sympy_var)
        integral = core.sympify(sympy_integral)
        return self._context.get_leaf_node(integral)


class XADDLeafDefIntegral(XADDLeafIndefIntegral):
    """Class for definite integral."""

    def __init__(
        self,
        var: core.Symbol,
        context: XADD,
    ):
        super().__init__(var, context)
        self.running_sum = context.ZERO
        if var in context._var_to_bound:
            lb, ub = context._var_to_bound[var]
            if lb == float('-inf'):
                lb = -oo
            if ub == float('inf'):
                ub = oo
            self.lower_bound, self.upper_bound = lb, ub
        else:
            self.lower_bound, self.upper_bound = -oo, oo

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        """Process the leaf node to compute a definite integral.
        
        * Determine if this will be a delta integral or not:
            i. if we encounter a delta function here that contains
                the variable then one of them has to be linear in
                the variable, otherwise we exit.
            ii. if find delta linear in variable then we extract substitution
                and make it to all remaining terms -- delta and non-delta --
                and return that result.
            iii. if delta's but do not contain variable then factor
                these out for multiplication in at the end.
            iv. what to do on encountering summation?  breaks into
                individual subproblems of the above, all results summed together!
        """
        # Bound management.
        xadd_lower_bound, xadd_upper_bound, target_var_indep_decisions = get_xadd_bounds(
            self._context, self.var, decisions, decision_values, self.lower_bound, self.upper_bound)
        
        # Compute the integral of this leaf w.r.t. the integration variable using SymPy.
        sympy_leaf = leaf_val._sympy_()
        sympy_var = self.var._sympy_()
        sympy_integral = sp.integrate(sympy_leaf, sympy_var)
        leaf_integral = core.sympify(sympy_integral)

        # Compute: leaf_integral{int_var = xadd_upper_bound} - leaf_integral{int_var = xadd_lower_bound}.
        int_eval_lower = self._context.substitute_xadd_for_var_in_expr(
            leaf_integral, var=self.var, xadd=xadd_lower_bound)
        int_eval_upper = self._context.substitute_xadd_for_var_in_expr(
            leaf_integral, var=self.var, xadd=xadd_upper_bound)
        int_eval = self._context.apply(int_eval_upper, int_eval_lower, 'subtract')

        # Finally, multiply in boolean decisions and irrelevant comparisons.
        for dec, is_true in target_var_indep_decisions:
            high_val = core.S(1) * int(is_true)
            low_val = core.S(1) * (1 - int(is_true))
            int_eval = self._context.apply(
                int_eval, self._context.get_dec_node(dec, low_val, high_val), 'prod')
        self.running_sum = self._context.apply(self.running_sum, int_eval, 'add')

        # All return information is stored in _runningSum so no need to return anything.
        # Just keep the original diagram as-is.
        return self._context.get_leaf_node(leaf_val)


class XADDLeafMultivariateMinOrMax(XADDLeafOperation):
    def __init__(
            self, 
            var_lst: List[core.Symbol],
            is_max: bool,
            bound_dict: Dict[core.Symbol, tuple],
            context: XADD,
            annotate: bool
    ):
        super().__init__(context)
        self._var_lst: List[core.Symbol] = var_lst
        self._context._opt_var_lst: List[core.Symbol] = var_lst
        self._is_max: bool = is_max
        self.bound_dict: Dict[core.Symbol, tuple] = bound_dict
        self._running_result: int = -1
        self._annotate: bool = annotate

    @property
    def _var(self) -> core.Symbol:
        return self._var_lst[0]

    @property
    def _lower_bound(self) -> Union[float, int]:
        lb = self.bound_dict[self._var][0]
        if lb == float('-inf'):
            return -oo
        return lb

    @property
    def _upper_bound(self) -> Union[float, int]:
        ub = self.bound_dict[self._var][1]
        if ub == float('inf'):
            return oo
        return ub

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        # Check if below computation is unnecessary.
        # min(oo, oo) = oo; max(oo, oo) = oo;
        # min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases,
        # and so we simply annotate them with NaN.
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(
                leaf_val, annotation=self._context.NAN)

            # Compare with the running result.
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management.
        xadd_lower_bound, xadd_upper_bound, target_var_indep_decisions = get_xadd_bounds(
            self._context, self._var, decisions, decision_values, self._lower_bound, self._upper_bound)

        # Substitute lower and upper bounds into leaf node.
        eval_lower = self._context.substitute_xadd_for_var_in_expr(
            leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(
            leaf_val, var=self._var, xadd=xadd_upper_bound)

        # Take casemin / casemax of eval_lower and eval_upper.
        """
        If `leaf_val` is bilinear, then we know that a leaf value of `eval_upper - eval_lower` will factorize as 
            (ub_vj - lb_vj) * (d_vj + \sum_i x_i Q_ij) and that (ub_vj - lb_vj) >= 0
        Therefore, we simply need to add the following conditional:
            ( [d_vj + \sum_i x_i Q_ij <= 0]
                ( [eval_upper] )
                ( [eval_lower] ))
        This can be done via the following trick:
            Let A = ( [d_vj + \sum_i x_i Q_ij <= 0], and B = ( [d_vj + \sum_i x_i Q_ij <= 0]
                        ( [1] )                                  ( [0] )
                        ( [0] ))                                 ( [1] ))
            Then, consider ``C = A \oprod `eval_upper`` and ``D = B \oprod `eval_lower``.
            The desired result can be obtained by 
                C \oplus D
            Then, we should canonicalize the resulting node. 
        """
        is_bilinear = xadd_util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`
            expr = xadd_util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == core.true:
                min_max_eval = eval_upper
            elif dec_expr == core.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(
                    dec, self._context.ZERO, self._context.ONE)      # Note: need to use ZERO_ig for annotating purpose... 
                ind_false = self._context.get_internal_node(         # but this is skipped in this branch.
                    dec, self._context.ONE, self._context.ZERO)
                upper_half = self._context.apply(
                    ind_true if not is_reversed else ind_false,
                    eval_upper,
                    'prod')
                lower_half = self._context.apply(
                    ind_false if not is_reversed else ind_true,
                    eval_lower,
                    'prod')
                min_max_eval = self._context.apply(
                    upper_half, lower_half, 'add',
                    annotation=(xadd_upper_bound, xadd_lower_bound)
                        if self._annotate else None
                )
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound,
            # while 2nd argument is lower bound.
            min_max_eval = self._context.apply(
                eval_upper,
                eval_lower,
                'max' if self._is_max else 'min',
                annotation=(xadd_upper_bound, xadd_lower_bound)
                    if self._annotate else None
            )

        # Reduce LP.
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions.
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below:
            # ensuring non-valid paths result in infinite penalty.
            min_max_eval = self._context.apply(
                indep_constraint,
                min_max_eval,
                'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce.
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        """
        Min(max)imize out remaining variables.
        """
        if len(self._var_lst) > 1:
            min_or_max = XADDLeafMultivariateMinOrMax(
                self._var_lst[1:],
                is_max=self._is_max,
                bound_dict=self.bound_dict,
                context=self._context,
                annotate=self._annotate,
            )
            decisions, decision_values = [], []
            _ = self._context.reduce_process_xadd_leaf(
                min_max_eval,
                min_or_max,
                decisions,
                decision_values)
            min_max_eval = min_or_max._running_result

        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(
                self._running_result,
                min_max_eval,
                'max' if self._is_max else 'min')
            self._running_result = self._context.reduce_lp(self._running_result)

        return self._context.get_leaf_node(leaf_val)


class XADDLeafMinOrMax(XADDLeafOperation):
    def __init__(
            self, 
            var: core.Symbol,
            is_max: bool,
            bound_dict: Dict[core.Symbol, tuple],
            context: XADD,
            annotate: bool = False,
    ):
        super().__init__(context)
        self._var: core.Symbol = var
        self._context._opt_var: core.Symbol = var
        self._is_max: bool = is_max
        self._running_result: int = -1
        self.annotate = annotate
        if var in bound_dict:
            lb = bound_dict[var][0]
            ub = bound_dict[var][1]
            if lb == float('-inf'):
                lb = -oo
            if ub == float('inf'):
                ub = oo
            self._lower_bound, self._upper_bound = lb, ub
        else:
            logger.info(
                (
                    f"No domain bounds over {str(var)} are provided..."
                     " Using -oo and oo as lower and upper bounds."
                )
            )
            self._lower_bound: Union[int, float, core.Number] = -oo
            self._upper_bound: Union[int, float, core.Number] = oo

    def process_xadd_leaf(
            self,
            decisions: List[DECISION_TYPE],
            decision_values: List[bool],
            leaf_val: core.Basic
    ) -> int:
        # Check if below computation is unnecessary.
        # min(oo, oo) = oo; max(oo, oo) = oo;
        # min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases,
        # and so we simply annotate them with NaN.
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(leaf_val, annotation=self._context.NAN)

            # Compare with the running result
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management.
        xadd_lower_bound, xadd_upper_bound, target_var_indep_decisions = get_xadd_bounds(
            self._context, self._var, decisions, decision_values, self._lower_bound, self._upper_bound)

        # Substitute lower and upper bounds into leaf node.
        eval_lower = self._context.substitute_xadd_for_var_in_expr(
            leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(
            leaf_val, var=self._var, xadd=xadd_upper_bound)
        annotation = (xadd_upper_bound, xadd_lower_bound) if self.annotate else None

        # Take casemin / casemax of eval_lower and eval_upper.
        """
        If `leaf_val` is bilinear, then we know that a leaf value of `eval_upper - eval_lower` will factorize as 
            (ub_vj - lb_vj) * (d_vj + \sum_i x_i Q_ij) and that (ub_vj - lb_vj) >= 0
        Therefore, we simply need to add the following conditional:
            ( [d_vj + \sum_i x_i Q_ij <= 0]
                ( [eval_upper] )   anno: xadd_upper_bound
                ( [eval_lower] )   anno: xadd_lower_bound
            )
        This can be done via the following trick:
            Let A = ( [d_vj + \sum_i x_i Q_ij <= 0], and B = ( [d_vj + \sum_i x_i Q_ij <= 0]
                        ( [1] )                                  ( [0] )
                        ( [0] ))                                 ( [1] ))
            Then, consider ``C = A \oprod `eval_upper`` and ``D = B \oprod `eval_lower``.
            The desired result can be obtained by 
                C \oplus D
            Then, we should canonicalize the resulting node. 
        """
        is_bilinear = xadd_util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`.
            expr = xadd_util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == core.true:
                min_max_eval = eval_upper
            elif dec_expr == core.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(
                    dec, self._context.ZERO, self._context.ONE)      # Note: need to use ZERO_ig for annotating purpose... 
                ind_false = self._context.get_internal_node(         # but this is skipped in this branch.
                    dec, self._context.ONE, self._context.ZERO)
                upper_half = self._context.apply(
                    ind_true if not is_reversed else ind_false,
                    eval_upper,
                    'prod')
                lower_half = self._context.apply(
                    ind_false if not is_reversed else ind_true,
                    eval_lower,
                    'prod')
                min_max_eval = self._context.apply(
                    upper_half,
                    lower_half,
                    'add',
                    annotation=annotation)
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound,
            # while 2nd argument is lower bound.
            min_max_eval = self._context.apply(
                eval_upper,
                eval_lower,
                'max' if self._is_max else 'min',
                annotation=annotation)

        # Reduce LP.
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions.
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below:
            # ensuring non-valid paths result in infinite penalty.
            min_max_eval = self._context.apply(
                indep_constraint,
                min_max_eval,
                'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce.
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Compare with the running result.
        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(
                self._running_result,
                min_max_eval,
                'max' if self._is_max else 'min')

        # Reduce running result.
        self._running_result = self._context.reduce_lp(self._running_result)

        return self._context.get_leaf_node(leaf_val)
