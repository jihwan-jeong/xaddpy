from sympy.logic import boolalg
import sympy
import sympy.core.numbers as numbers
import sympy.core.relational as relational
from sympy import oo, S
import numpy as np

import xaddpy.utils.util
from xaddpy.xadd.node import Node, XADDINode, XADDTNode
from xaddpy.xadd.reduce_lp import ReduceLPContext
import abc
from xaddpy.utils.global_vars import (
    REL_TYPE, OP_TYPE, UNARY_OP, RELATIONAL_OPERATOR, ACCEPTED_RV_TYPES
)
from xaddpy.utils.util import check_sympy_boolean, sample_rvs, check_expr_linear
from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar
from xaddpy.utils.logger import logger

from typing import Callable, Dict, Tuple, Union, cast, List, Optional

USE_APPLY_GET_INODE_CANON = False
LARGE_INTEGER = 10000


def default_ordering(context, expr: sympy.Basic) -> int:
    num_unique_expr = len(context._expr_to_id)

    # Decision expression consisting of continuous variables
    if isinstance(expr, relational.Rel):
        poly = expr.lhs.as_poly()
        deg = poly.total_degree()    
        if deg > 1:
            index = num_unique_expr + LARGE_INTEGER ** 2
        else:
            index = num_unique_expr + LARGE_INTEGER
    # Boolean decisions
    else:
        index = num_unique_expr
    return index


class XADD:

    _func_var_index = default_ordering

    def __init__(self, args: dict = {}):
        # XADD variable maintenance
        self._cvar_to_id = {}
        self._id_to_cvar = {}
        self._bvar_to_id = {}
        self._id_to_bvar = {}
        self._rv_to_id = {}
        self._id_to_rv = {}
        self._str_var_to_var = {}
        self._cont_var_set = set()
        self._bool_var_set = set()
        self._random_var_set = set()
        self._rv_to_params = {}
        self._rv_to_type = {}

        self._sympy_to_pulp = {}
        self._opt_var = None
        self._opt_var_lst = None
        self._eliminated_var = []
        self._decisionVars = set()
        self._min_var_set = set()
        self._free_var_set = set()
        self._name_space = None

        # Bound maintenance (need to be passed from the output of parser function)
        self._var_to_bound = {}
        self._temp_ub_lb_cache = set()

        # Decision expression maintenance
        self._expr_to_id: Dict[sympy.Basic, int] = {}
        self._id_to_expr: Dict[int, sympy.Basic] = {}
        self._expr_to_linear_check: Dict[sympy.Basic, bool] = {}
        self._expr_id_to_linear_check: Dict[int, bool] = {}

        # XADD node maintenance
        self._id_to_node: Dict[int, Node] = {}
        self._node_to_id: Dict[Node, int] = {}
        self._var_to_anno: Dict[sympy.Symbol, int] = {}     # annotation dictionary for argmin / argmax

        # Flush
        self._special_nodes = set()
        self._node_to_id_new = {}
        self._id_to_node_new = {}
        self._id_to_expr_new = {}
        self._expr_to_id_new = {}

        # Reduce & Apply caches
        self._reduce_cache = {}
        self._reduce_leafop_cache = {}
        self._reduce_canon_cache = {}

        self._apply_cache = {}
        self._apply_caches = {}
        self._inode_to_vars = {}
        self._factor_cache = {}

        # Reduce LP
        self.RLPContext = ReduceLPContext(self, **args)
        
        # Node maintenance
        self._nodeCounter = 0

        # temporary nodes
        self._tempINode = XADDINode(-1, -1, -1, context=self)
        self._temp_term_node = XADDTNode(sympy.S(-1), context=self)

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

    def set_variable_ordering_func(self, func: Callable):
        XADD._func_var_index = func
    
    def add_random_var(self, var: sympy.Symbol, **kwargs):
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

    def add_continuous_var(self, var: sympy.Symbol):
        if not var in self._cont_var_set:
            num_existing_cvar = len(self._cont_var_set)
            self._cont_var_set.add(var)
            self._str_var_to_var[str(var)] = var
            self._cvar_to_id[var] = num_existing_cvar
            self._id_to_cvar[num_existing_cvar] = var

    def add_boolean_var(self, var: sympy.Symbol):
        if not var._assumptions.get('bool', False):
            print(f"The type of boolean variable {var} is not correctly set..")
            var._assumptions['bool'] = True
        if var not in self._bool_var_set:
            num_existing_bvar = len(self._bool_var_set)
            self._bool_var_set.add(var)
            self._str_var_to_var[str(var)] = var
            self._bvar_to_id[var] = num_existing_bvar
            self._id_to_bvar[num_existing_bvar] = var
            
    def create_standard_nodes(self):
        """
        Create and store standard nodes and generate indices, which can be frequently used.
        """
        self.ZERO = self.get_leaf_node(sympy.S(0))
        self.ONE = self.get_leaf_node(sympy.S(1))
        self.TRUE = self.get_leaf_node(sympy.true)
        self.FALSE = self.get_leaf_node(sympy.false)
        self.oo = self.get_leaf_node(oo)
        self.NEG_oo = self.get_leaf_node(-oo)
        self.NAN = self.get_leaf_node(sympy.nan)

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

    def convert_to_xadd(self, term: sympy.Basic, **kwargs):
        if isinstance(term, sympy.Symbol) or \
            isinstance(term, numbers.Number) or \
                isinstance(term, boolalg.BooleanAtom):
            if term._assumptions.get('bool', False):
                dec, is_reversed = self.get_dec_expr_index(term, create=True, **kwargs)
                low, high = self.FALSE, self.TRUE
                if is_reversed:
                    low, high = high, low
                return self.get_internal_node(dec, low, high)
            return self.get_leaf_node(term, **kwargs)
        else:
            return self.convert_func_to_xadd(term, **kwargs)

    def build_initial_xadd(self, xadd_as_list: List, to_canonical=True):
        """
        Given decisions and leaf values in a list, 
        recursively build initial XADD and return the id of the root node.
        :param xadd_as_list:        (list)
        :return:
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
                    print("Reordering problem: {} >= {}\n{}: {}\n{}: {}".
                          format(dec_id, low_n.dec, dec_id, self._id_to_expr[dec_id], low_n.dec, self._id_to_expr[low_n.dec]))
                    raise ValueError
            high_n = self.get_exist_node(node.high)
            if not high_n.is_leaf():
                if dec_id >= high_n.dec:
                    # compare local order
                    print("Reordering problem: {} >= {}\n{}: {}\n{}: {}".
                          format(dec_id, high_n.dec, dec_id, self._id_to_expr[dec_id], high_n.dec,
                                 self._id_to_expr[high_n.dec]))
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
            print(f"Warning: Canonizing Negative Decision: {dec} -> {self._id_to_expr[abs(dec)]}")
        result1 = self.get_inode_canon_apply_trick(dec, low, high)
        result2 = self.get_inode_canon_insert(dec, low, high)

        if result1 != result2 and not self.contains_node_id(result1, self.NAN):
            print("Canonical error (difference not on NAN)")
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
            dec_expr: sympy.Basic,
            bool_assign: Dict[sympy.Symbol, Union[boolalg.BooleanAtom, bool]], 
            cont_assign: Dict[sympy.Symbol, Union[int, float]], 
    ) -> Union[bool, None]:
        if isinstance(dec_expr, boolalg.BooleanAtom):
            return bool(dec_expr)
        # if a decision expression is a single symbol, it should be a boolean decision
        elif dec_expr._assumptions.get('bool', False):
            return bool_assign.get(dec_expr)
        # Inequality decision
        elif isinstance(dec_expr, relational.Rel):
            # if any of the variables in dec_expr is not evaluated, returns None
            var_set = dec_expr.free_symbols
            non_assigned_vars = var_set.difference(cont_assign.keys())
            if len(non_assigned_vars) > 0:
                return None

            dec_expr = dec_expr.xreplace({sub_out: sympy.S(sub_in) for sub_out, sub_in in cont_assign.items()})
            assert isinstance(dec_expr, boolalg.BooleanAtom) or isinstance(dec_expr, bool)
            return bool(dec_expr)
        else:
            return None
        
    def evaluate(
            self, 
            node_id: int, 
            bool_assign: Dict[sympy.Symbol, Union[boolalg.BooleanAtom, bool]], 
            cont_assign: Dict[sympy.Symbol, Union[int, float]],
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
        expr = expr.xreplace({sub_out: sympy.S(sub_in) for sub_out, sub_in in cont_assign.items()})
        
        # Not all required variables were assigned
        if len(expr.free_symbols) > 0:
            return
        
        # Return python primitive type
        if primitive_type and isinstance(expr, boolalg.BooleanAtom):
            return bool(expr)
        elif primitive_type:
            return float(expr)
        
        # Otherwise, return sympy type
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
                assert check_sympy_boolean(expr)
                expr = ~expr
                if isinstance(expr, boolalg.BooleanAtom):
                    return self.get_leaf_node(expr, node._annotation)
                else:
                    return self.get_dec_node(expr, sympy.false, sympy.true)
            
            sp_op = UNARY_OP.get(op, None)
            if sp_op is None:
                raise ValueError(f"Unary operation {op} not recognized")
            expr = sympy.expand(sp_op(expr, *args))    # TODO: Need to simplify? that can take a while
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
        dec_expr2 = sympy.Eq(expr, 0)
        low = self.ONE        
        high = self.get_dec_node(dec_expr2, sympy.S(-1), sympy.S(0))
        
        if is_reversed:
            high, low = low, high
        
        ret = self.get_internal_node(dec1, low, high)
        ret = self.make_canonical(ret)
        return ret

    def apply(self, id1: int, id2: int, op: str, annotation=None) -> int:
        """
        Recursively apply op(node1, node2). 
        op can be 
                'min', 'max', 'sum', 'minus', 'prod', 'div' for non-Boolean operations
            or  'or', 'and' for Boolean operations.
        :param id1:
        :param id2:
        :param op:
        :param annotation:          (tuple)
        :return:
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
        """
        Recursively apply op(node1, node2).
        :param id1:         (int) index of node 1
        :param id2:         (int) index of node 2
        :param op:          (str) 'max', 'min', 'add', 'subtract', 'prod', 'div' (non-Boolean)
                                  'or', 'and' (Boolean)
                                  '!=', '==', '>', '>=', '<', '<=' (Relational)
        :return:
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
    ):
        """
        :param id1:         (int)
        :param n1:          (Node)
        :param id2:         (int)
        :param n2:          (Node)
        :param op:          (str) 'max', 'min', 'add', 'subtract', 'prod', 'div' (non-Boolean)
                                  'or', 'and' (Boolean)
                                  '!=', '==', '>', '>=', '<', '<=' (Relational)
        :param annotation:  (tuple)
        :return:
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
            if n1.is_leaf() and isinstance(n1.expr, boolalg.BooleanAtom):
                if op == 'or' and n1.expr:
                    return self.TRUE
                if op == 'and' and not n1.expr:
                    return self.FALSE
            if n2.is_leaf() and isinstance(n2.expr, boolalg.BooleanAtom):
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
                (n1_expr._assumptions.get('bool', False) or 
                 isinstance(n1_expr, boolalg.BooleanAtom) or
                 n2_expr._assumptions.get('bool', False) or 
                 isinstance(n2_expr, boolalg.BooleanAtom) 
            ):
                if n1_expr._assumptions.get('bool', False):
                    dec_node1 = self.get_dec_node(n1_expr, sympy.S(0), sympy.S(1))
                elif isinstance(n1_expr, boolalg.BooleanAtom):
                    dec_node1 = self.ONE if n1_expr else self.ZERO
                else:
                    dec_node1 = id1
                if n2_expr._assumptions.get('bool', False):
                    dec_node2 = self.get_dec_node(n2_expr, sympy.S(0), sympy.S(1))
                elif isinstance(n2_expr, boolalg.BooleanAtom):
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
                    result = sympy.expand(n1_expr * n2_expr)
                elif op == 'div':
                    result = sympy.expand(n1_expr / n2_expr)
                else:
                    assert check_sympy_boolean(n1_expr)
                    assert check_sympy_boolean(n2_expr)
                    dec_n2 = self.get_dec_node(n2_expr, sympy.false, sympy.true)
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
            if op in RELATIONAL_OPERATOR:
                lhs = n1.expr - n2.expr
                expr = RELATIONAL_OPERATOR[op](lhs, 0)      # can handle '==' (Equality), '!=' (Unequality), '>', '>=', '<', '<='
            # Handle min, max operations
            else:
                lhs = n1.expr - n2.expr
                rhs = 0
                expr = lhs <= rhs

            # handle tautological cases
            if expr == sympy.S.true:        
                if op == 'min':             # n1 <= n2 holds
                    return self.get_leaf_node(n1.expr, annotation[0]) if annotation is not None else id1
                elif op == 'max':           # n1 <= n2 holds
                    return self.get_leaf_node(n2.expr, annotation[1]) if annotation is not None else id2
                else:                       # n1 (rel) n2 holds
                    return self.TRUE
            elif expr == sympy.S.false:
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

    def substitute(self, node_id: int, subst_dict: dict) -> int:
        """
        Symbolic substitution method.
        :param node_id:             (int)
        :param subst_dict:          (dict)
        :return:
        """
        subst_cache = {}
        return self.reduce_sub(node_id, subst_dict, subst_cache)

    def reduce_sub(        
            self, 
            node_id: int,
            subst_dict: Dict[sympy.Symbol, Union[sympy.Basic, float, int]], 
            subst_cache: Dict[int, int],
    ) -> int:
        """

        :param node_id:
        :param subst_dict:
        :param subst_cache:
        :return:
        """
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced by default
        if node.is_leaf():
            node = cast(XADDTNode, node)
            expr = node.expr
            if len(expr.free_symbols.intersection(set(subst_dict.keys()))) > 0:
                expr = expr.xreplace({sub_out: sympy.S(sub_in) for sub_out, sub_in in subst_dict.items()})
                expr = sympy.expand(expr)
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
        if isinstance(dec_expr, sympy.Symbol):
            sub_in = subst_dict.get(dec_expr, None)
            is_reversed = False
            if sub_in is not None:
                # Handle tautologies
                if sub_in == S.true:
                    subst_cache[node_id] = high
                    return high
                elif sub_in == S.false:
                    subst_cache[node_id] = low
                    return low
                dec, is_reversed = self.get_dec_expr_index(sub_in, create=True)
        else:
            lhs = dec_expr.lhs
            if len(lhs.free_symbols.intersection(set(subst_dict.keys()))) > 0:
                lhs = lhs.xreplace({sub_out: sympy.S(sub_in) for sub_out, sub_in in subst_dict.items()})
            
            # Check if the expression holds in equality and the true branch is NaN
            # Assuming canonical expression.. rhs is always 0. Hence, lhs == 0 iff dec_expr == S.true.
            # In this case, set dec_expr = False, so that false branch can be chosen instead.
            if lhs == 0 and high == self.NAN:
                dec_expr = S.false
            elif lhs == 0 and low == self.NAN:
                dec_expr = S.true
            else:
                dec_expr = lhs <= 0

            # # Handle tautologies
            if dec_expr == S.true:
                subst_cache[node_id] = high
                return high
            elif dec_expr == S.false:
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
        :param node_id:             (int)
        :param subst_dict:          (dict)
        :return:
        """
        assert all(
                    map(lambda x: isinstance(x, boolalg.BooleanAtom) or isinstance(x, bool), subst_dict.values())
                ), "All values of the `subst_dict` should be boolean type"
        varSet = self.collect_vars(node_id)
        for var in subst_dict:
            if var in varSet:
                dec, _ = self.get_dec_expr_index(var, create=False)
                if subst_dict[var]:
                    node_id = self.op_out(node_id, dec, "restrict_high")
                else:
                    node_id = self.op_out(node_id, dec, "restrict_low")
        return node_id

    def op_out(self, node_id: int, dec_id: int, op: str):
        ret = self.reduce_op(node_id, dec_id, op)

        # Operations like sum and product may get decisions out of order
        if op == 'add' or op == 'prod':
            return self.make_canonical(ret)
        else:
            return ret

    def reduce_op(self, node_id: int, dec_id: int, op: str) -> int:
        node = self.get_exist_node(node_id)

        # A terminal node should be reduced (and cannot be restricted)
        # by default if hashing and equality testing are working in getLeafNode
        if node.is_leaf():
            return node_id      # Assuming that to have a node id means canonical

        # If its an internal node, check the reduce cache
        temp_reduce_key = (node_id, dec_id, op)
        ret = self._reduce_cache.get(temp_reduce_key, None)
        if ret is not None:
            return ret
        
        node = cast(XADDINode, node)
        if (op != "restrict_high") or (dec_id != node.dec):
            low = self.reduce_op(node.low, dec_id, op)
        if (op != "restrict_low") or (dec_id != node.dec):
            high = self.reduce_op(node.high, dec_id, op)
        #if (op != -1 & & var_id != -1 & & var_id == inode._var) {
        if (dec_id != -1) and (dec_id == node.dec):
            # ReduceOp
            if op == "restrict_low":
                ret = low
            elif op == "restrict_high":
                ret = high
            elif op == "sum" or op == "prod":
                ret = self.apply(low, high, op)
            else:
                raise NotImplementedError
        else:
            ret = self.get_internal_node(node.dec, low, high)

        # Put return value in cache and return
        self._reduce_cache[temp_reduce_key] = ret
        return ret

    def collect_vars(self, node_id: int) -> set:
        node = self.get_exist_node(node_id)
        var_set = set()
        node.collect_vars_(var_set)
        return var_set

    def reduced_arg_min_or_max(self, node_id: int, var) -> int:
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
        :param node_id:             (int)
        :return:
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

    def get_all_anno(self) -> Dict[sympy.Symbol, int]:
        return self._var_to_anno

    def get_annotation(self, var: sympy.Symbol) -> int:
        return self._var_to_anno[var]

    def update_anno(self, var: sympy.Symbol, anno: int):
        if not hasattr(self, '_var_to_anno'):
            self._var_to_anno: Dict[sympy.Symbol, int] = {}
        self._var_to_anno[var] = anno

    def get_node(self, node_id: int) -> Node:
        """
        Retrieve a XADD node from cache.
        :param node_id:             (int)
        :return:
        """
        return self._id_to_node[node_id]

    def min_or_max_multi_var(
            self, node_id: int, var_lst: List[sympy.Symbol], is_min: bool = True, annotate: bool = True,
    ):
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

    def min_or_max_var(self, node_id, var, is_min=True):
        """
        Given an XADD root node 'node_id', minimize (or maximize) 'var' out.
        :param node_id:      (int)
        :param var:                 (sympy.Symbol)
        :return:                    (int)
        """
        # Check if binary variable
        if var in self._bool_var_set:
            op = "min" if is_min else "max"
            self._opt_var = var
            subst_high = {var: True}
            subst_low = {var: False}
            restrict_high = self.substitute_bool_vars(node_id, subst_high)
            restrict_low = self.substitute_bool_vars(node_id, subst_low)
            running_result = self.apply(restrict_high, restrict_low, op=op, annotation=(self.ONE, self.ZERO))
            running_result = self.reduce_lp(running_result)
        # Continuous variables
        else:
            decisions, decision_values = [], []
            min_or_max = XADDLeafMinOrMax(var, is_max=False if is_min else True, bound_dict=self._var_to_bound, context=self)
            _ = self.reduce_process_xadd_leaf(node_id, min_or_max, decisions, decision_values)
            running_result = min_or_max._running_result
        return running_result

    def substitute_xadd_for_var_in_expr(
            self, leaf_val: sympy.Basic, var: sympy.Symbol, xadd: int
    ) -> int:
        """
        Substitute XADD into 'var' that occurs in 'val' (a Sympy expression). 
        This is only called for leaf expressions.

        :param leaf_val:    (sympy.Basic) sympy expression
        :param var:         (sympy.Symbol) variable to substitute
        :param xadd:        (int) integer that indicates the XADD to substitute into 'var'
        :return:
        """
        # Get the root node
        node = self.get_exist_node(xadd)

        # Handle leaf node cases: simply substitute leaf expression into 'var' in leaf_val
        if node.is_leaf():
            node = cast(XADDTNode, node)
            xadd_leaf_expr = node.expr
            # expr = leaf_val.subs(var, xadd_leaf_expr)
            expr = leaf_val.xreplace({var: xadd_leaf_expr})
            expr = sympy.expand(expr)

            # Special treatment for oo, -oo
            try:
                two_terms = expr.as_two_terms()
                if isinstance(two_terms[0], sympy.core.Number):
                    if two_terms[0] == sympy.oo:
                        expr = sympy.oo
                    elif two_terms[0] == -sympy.oo:
                        expr = -sympy.oo
            except AttributeError as e:
                pass
            except Exception as e:
                logger.error(e)
                exit(1)
            node_id = self.get_leaf_node(expr, annotation=None)
            return node_id

        # Internal nodes: get low and high branches and do recursion
        low, high = node.low, node.high
        low = self.substitute_xadd_for_var_in_expr(leaf_val, var, low)
        high = self.substitute_xadd_for_var_in_expr(leaf_val, var, high)

        # Get the node id for a (sub)XADD and return it
        node_id = self.get_internal_node(node.dec, low=low, high=high)

        return node_id

    def get_repr(self, node_id: int) -> str:
        # For printing out the representation
        node = self._id_to_node[node_id]
        return repr(node)

    def get_leaf_node_from_node(self, node: XADDTNode) -> int:
        """
        :param node:            (Node) If Node object is passed.. also take annotation into consideration
        :return:
        """
        expr, annotation = node.expr, node._annotation
        return self.get_leaf_node(expr, annotation)

    def get_leaf_node(
            self, expr: sympy.Basic, annotation: Optional[int] = None, **kwargs
    ) -> int:
        """Returns the ID of the leaf node with given SymPy expression and annotation.

        Note that if a new random variable is added within this method,
        kwargs should have the necessary parameters to specify the random variable.
        
        For example, for a uniform random variable, we need
            {'params': [lb, ub]} where lb and ub are the lower and upper bounds of the uniform 
            distribution, respectively.
        
        If this information was not provided, this method will result in an assertion error.
        
        Args:
            expr (sympy.Basic): The SymPy expression associated with the leaf node.
            annotation (Optional[int], optional): The node ID of the annotation.
        
        Returns:
            int: _description_
        """
        self._temp_term_node.set(expr, annotation)
        node_id = self._node_to_id.get(self._temp_term_node, None)
        if node_id is None:
            # node not in cache, so create
            node_id = self._nodeCounter
            node = XADDTNode(expr, annotation, context=self)
            self._id_to_node[node_id] = node
            self._node_to_id[node] = node_id
            self._nodeCounter += 1

            # add in all new variables
            vars_in_expr = expr.free_symbols.copy()
            diff_vars = vars_in_expr.difference(self._cont_var_set).difference(self._bool_var_set)
            for v in diff_vars:
                if v._assumptions.get('bool', False):
                    self.add_boolean_var(v)
                else:
                    self.add_continuous_var(v)
                if v._assumptions.get('random', False):
                    assert kwargs.get('params') is not None
                    assert kwargs.get('type') is not None and kwargs['type'] in ACCEPTED_RV_TYPES
                    self.add_random_var(v, **kwargs)
        return node_id

    def get_dec_node(
            self, 
            dec_expr: Union[relational.Rel, sympy.Symbol, boolalg.BooleanFunction],
            low_val: sympy.Basic, 
            high_val: sympy.Basic
    ) -> int:
        """
        Get decision node with relational expression having dec, whose low and high values are also given.
        :param dec_expr:            (sympy.relational.Rel)
        :param low_val:             (float)
        :param high_val:            (float)
        :return:
        """
        dec, is_reversed = self.get_dec_expr_index(dec_expr, create=True)
        low = self.get_leaf_node(low_val)
        high = self.get_leaf_node(high_val)
        # Swap low and high branches if reversed
        if is_reversed:
            high, low = low, high
        return self.get_internal_node(dec, low, high)
    
    def canonical_dec_expr(
            self, expr: sympy.Basic
    ) -> Tuple[Union[Tuple[sympy.Basic, sympy.Basic], sympy.Basic], bool]:
        """
        Return canonical form of an expression.
        It should always take either one of the two forms: expr.lhs <= 0 or expr.lhs >= 0.
        Args:
            expr:

        Returns:

        """
        # sympy expr is already alphabetically ordered
        is_reversed = False

        # Handle tautology: simply return without doing anything
        if expr == sympy.S.true:
            return expr, is_reversed
        elif expr == sympy.S.false:
            return expr, is_reversed

        # Handle boolean expressions
        if not isinstance(expr, relational.Rel):
            if isinstance(expr, boolalg.Not):
                expr = ~expr
                is_reversed = True
            if not expr._assumptions.get('bool', False):
                print(f"Expression {expr} will be treated as Boolean")
                expr._assumptions['bool'] = True
                if expr in self._cont_var_set:
                    self._cont_var_set.remove(expr)
                if expr not in self._bool_var_set:
                    self._bool_var_set.add(expr)
            return expr, is_reversed
        
        # Always make 'lhs - rhs <= 0' as canonical expression
        lhs, rhs, rel = expr.lhs, expr.rhs, REL_TYPE[type(expr)]
        lhs = lhs - rhs

        if rel == '>=' or rel == '>':
            is_reversed = True
            rel = '<=' if rel == '>=' else '<'

        # Divide lhs by the coefficient of the first term (cannot be a negative number)
        coeff_first_term = lhs.as_ordered_terms()[0]
        if isinstance(coeff_first_term, sympy.core.Number):
            coeff_first_term = lhs.as_ordered_terms()[1]

        if isinstance(coeff_first_term, sympy.core.Mul):
            arg1 = coeff_first_term.args[0]
            if isinstance(arg1, sympy.core.Number) and arg1 > 0:
                lhs = lhs / arg1
            elif isinstance(arg1, sympy.core.Number) and arg1 < 0:
                lhs = lhs / arg1
                # divided by a negative number changes the direction of inequality
                if rel in ('<=', '<', '>', '>='):
                    is_reversed = True if not is_reversed else False
        
        expr = relational.Relational(lhs, 0, rel)
        return expr, is_reversed

    def get_dec_expr_index(
            self, expr: sympy.Basic, create: bool, canon: bool = False, **kwargs
    ) -> Tuple[Union[boolalg.BooleanAtom, int], bool]:
        """Given a symbolic expression 'expr', return the index of the expression in XADD._id_to_expr.
        
        Note that if a new random variable is included in the expression,
        kwargs should have the necessary parameters to specify the random variable.
        
        For example, for a uniform random variable, we need
            {'params': [lb, ub]} where lb and ub are the lower and upper bounds of the uniform 
            distribution, respectively.
        
        If this information was not provided, this method will result in an assertion error.
        
        Args:
            expr (sympy.Basic): The expression to be used as a decision. This can be a relational
                expression or just a boolean variable.
            create (bool): Whether to assign a new ID for the given expression.
            canon (bool, optional): Deprecated... TODO: check whether this can safely removed.

        Returns:
            Tuple[Union[boolalg.BooleanAtom, int], bool]: _description_
        """
        is_reversed = False
        if not canon:
            expr, is_reversed = self.canonical_dec_expr(expr)

        index = self._expr_to_id.get(expr, None)

        if index is None:
            index = 0

        # If found, and not create
        if index != 0 or not create:
            return index, is_reversed
        # If nothing's found, create one and store
        else:
            index = XADD._func_var_index(self, expr)
            self._expr_to_id[expr] = index
            self._id_to_expr[index] = expr
            
            # Check whether the expression is at most linear in free variables
            is_linear = check_expr_linear(expr)
            self._expr_to_linear_check[expr] = is_linear
            self._expr_id_to_linear_check[index] = is_linear
            
            # Add in all new variables
            vars_in_expr = expr.free_symbols.copy()
            diff_vars = vars_in_expr.difference(self._cont_var_set).\
                difference(self._bool_var_set).difference(self._random_var_set)
            for v in diff_vars:
                if v._assumptions.get('bool', False):
                    self.add_boolean_var(v)
                else:
                    self.add_continuous_var(v)
                if v._assumptions.get('random', False):
                    assert kwargs.get('params') is not None
                    assert kwargs.get('type') is not None and kwargs['type'] in ACCEPTED_RV_TYPES
                    self.add_random_var(v, **kwargs)
        return index, is_reversed

    def get_exist_node(self, node_id: int) -> Node:
        node = self._id_to_node.get(node_id, None)
        if node is None:
            print("Unexpected Missing node: " + node_id)
        return node

    def get_internal_node(self, dec_id: int, low: int, high: int) -> int:
        """

        :param dec_id:      (int) id of decision expression
        :param low:         (int) id of low branch node
        :param high:        (int) id of high branch node
        :return:            (int) return id of node
        """
        if dec_id < 0:
            high, low = low, high
            dec_id = -dec_id

        # Check if low == high
        if low == high:
            return low

        # Handle tautological cases
        dec_expr = self._id_to_expr.get(dec_id, None)
        if dec_expr == sympy.S.true:
            return high
        elif dec_expr == sympy.S.false:
            return low

        # Retrieve XADDINode (create if it does not exist)
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
        """Samples all random variables existing in the given node and return the
        ID of the reduced node with all random values instantiated 

        Args:
            node_id (int): The XADD node
            use_expectation (bool, optional): Whether to use the expected value instead of sampling
            rng (np.random.Generator, optional): The random number generator to use

        Returns:
            int: The ID of the reduce node after sampling
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
        
        # Handle an internal node
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
            leaf_op,        # XADDLeafOperation
            decisions: list, 
            decision_values: list
    ) -> int:
        """

        :param node_id:
        :param leaf_op:
        :param decisions:
        :param decision_values:
        :return:
        """
        node = self.get_exist_node(node_id)
        if node.is_leaf():
            return leaf_op.process_xadd_leaf(decisions, decision_values, node.expr)

        # Internal node
        dec_expr = self._id_to_expr.get(node.dec)

        # Recurse the False branch
        decisions.append(dec_expr)
        decision_values.append(False)
        low = self.reduce_process_xadd_leaf(node.low, leaf_op, decisions, decision_values)

        # Recurse the True branch
        decision_values[-1] = True
        high = self.reduce_process_xadd_leaf(node.high, leaf_op, decisions, decision_values)

        decisions.pop()
        decision_values.pop()

        ret = self.get_internal_node(node.dec, low, high)
        if isinstance(leaf_op, DeltaFunctionSubstitution):
            ret = self.make_canonical(ret)

        # Put return value in cache and return  # TODO: this does not distinguish different leaf operations
        self._reduce_leafop_cache[node_id] = ret
        return ret
    
    """
    Verifying feasibility and redundancy of all paths in the XADD
    """
    def reduce_lp(self, node_id: int) -> int:
        """
        Consistency and redundancy checking
        :param node_id:     (int) Node id
        :return:
        """
        return self.RLPContext.reduce_lp(node_id)

    """
    Cache maintenance
    """
    def clear_special_nodes(self):
        self._special_nodes.clear()

    def add_special_node(self, n: int):
        try:
            if n is None:
                raise ValueError("add_sepcial_node: None")
        except ValueError as error:
            print(error)
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
        # self._factor_cache.clear()
        self._temp_ub_lb_cache.clear()
        # self._reduce_annotate_cache.clear()
        self.RLPContext.flush_implications()

        # Set up temporary alternates to these HashMaps
        self._node_to_id_new.clear()
        self._id_to_node_new.clear()
        # self._id_to_expr_new.clear()
        # self._expr_to_id_new.clear()

        # Copy over 'special' nodes then set new dict
        for node_id in self._special_nodes:
            self.copy_in_new_cache_node(node_id)

        self._node_to_id = self._node_to_id_new.copy()
        self._id_to_node = self._id_to_node_new.copy()
        # self._expr_to_id = self._expr_to_id_new.copy()
        # self._id_to_expr = self._id_to_expr_new.copy()

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

            # Copy decision expr and its id

            # Recurse
            self.copy_in_new_cache_node(node._high)
            self.copy_in_new_cache_node(node._low)
        else:
            self._id_to_node_new[node_id] = node
            self._node_to_id_new[node] = node_id

    """
    Export and import XADDs
    """
    def export_xadd(self, node_id: int, fname: str, append: bool = False):
        """
        Export the XADD node to a file.
        If append is True, then open the file in the append mode.
        """
        # Firstly, turn off printing node info
        node: Node = self._id_to_node.get(node_id, None)
        if node is None:
            raise KeyError(f'There is no node with id {node_id}')
        node.turn_off_print_node_info()

        if append:
            with open(fname, 'a+') as f:
                f.write('\n')
                f.write(str(node))
        else:
            with open(fname, 'w+') as f:
                f.write(str(node))

        # Turn the printing mode back on
        node.turn_on_print_node_info()

    def import_xadd(
            self, 
            fname: Optional[str] = None,
            xadd_str: Optional[str] = None,
            locals: Optional[dict] = None,
            to_canonical: bool = True
    ) -> int:
        """
        Import the XADD node defined in an input file or in a string.
        """
        assert (fname is not None and xadd_str is None) or (fname is None and xadd_str is not None),\
            "Specify either a file name or a string, not both"

        if fname is not None:
            with open(fname, 'r') as f:
                xadd_str = f.read().replace('\n', '')
        
        # Note: when it is just a leaf expression: not supported
        if xadd_str.rfind('(') == 0 and xadd_str.rfind('[') == 2:
            xadd_as_list = [sympy.sympify(xadd_str.strip('( [] )'), locals=locals)]
        else:
            xadd_as_list = parse_xadd_grammar(xadd_str, ns=locals if locals is not None else {})[1][0]
        node_id = self.build_initial_xadd(xadd_as_list, to_canonical=to_canonical)
        return node_id


class NullDec:
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, NullDec)


class XADDLeafOperation(metaclass=abc.ABCMeta):
    def __init__(self, context: XADD):
        self._context: XADD = context

    @abc.abstractmethod
    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
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

    def process_xadd_leaf(self, decisions: list, decision_values: list, leaf_val: sympy.Basic):
        assert check_sympy_boolean(leaf_val) or leaf_val == 1 or leaf_val == 0
        
        if isinstance(leaf_val, boolalg.BooleanAtom):
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
            sub_var: sympy.Symbol,
            xadd_sub_at_leaves: int,
            context: XADD
    ):
        """
        From the case statement of 'xadd_sub_at_leaves', 
        all occurrences of sub_var will be replaced.
        """
        super().__init__(context)
        self._leafSubs: Dict[sympy.Symbol, Union[float, int, bool]] = {}
        self._xadd_sub_at_leaves: int = xadd_sub_at_leaves
        self._subVar = sub_var

    def process_xadd_leaf(self, decisions, decision_values, leaf_val):
        self._leafSubs = {}
        
        if leaf_val == sympy.nan:
            return self._context.NAN
        # If boolean variable, handle differently
        elif self._subVar in self._context._bool_var_set:
            self._leafSubs[self._subVar] = True \
                if leaf_val == 1 or \
                    ((isinstance(leaf_val, boolalg.BooleanAtom) or isinstance(leaf_val, bool)) 
                        and leaf_val)\
                else False
            return self._context.substitute_bool_vars(self._xadd_sub_at_leaves, self._leafSubs)
        # Continuous variable
        else:
            self._leafSubs[self._subVar] = leaf_val
            ret = self._context.substitute(self._xadd_sub_at_leaves, self._leafSubs)
            ret = self._context.reduce_lp(ret)
            return ret


class XADDLeafMultivariateMinOrMax(XADDLeafOperation):
    def __init__(
            self, 
            var_lst: List[sympy.Symbol],
            is_max: bool,
            bound_dict: Dict[sympy.Symbol, tuple],
            context: XADD,
            annotate: bool
    ):
        super().__init__(context)
        self._var_lst: List[sympy.Symbol] = var_lst
        self._context._opt_var_lst: List[sympy.Symbol] = var_lst
        self._is_max: bool = is_max
        self.bound_dict: Dict[sympy.Symbol, tuple] = bound_dict
        self._running_result: int = -1
        self._annotate: bool = annotate

    @property
    def _var(self) -> sympy.Symbol:
        return self._var_lst[0]

    @property
    def _lower_bound(self) -> Union[float, int]:
        return self.bound_dict[self._var][0]

    @property
    def _upper_bound(self) -> Union[float, int]:
        return self.bound_dict[self._var][1]

    def process_xadd_leaf(
            self, decisions: list, decision_values: list, leaf_val: sympy.Basic
    ):
        """

        :param decisions:
        :param decision_values:
        :param leaf_val:        (sympy.Basic) leaf expression
        :return:
        """
        # Check if below computation is unnecessary
        # min(oo, oo) = oo; max(oo, oo) = oo; min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases, and so we simply annotate them with NaN
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(leaf_val, annotation=self._context.NAN)

            # Compare with the running result
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management
        lower_bound = []
        upper_bound = []
        lower_bound.append(sympy.S(self._lower_bound))
        upper_bound.append(sympy.S(self._upper_bound))

        # Independent decisions (incorporated later): [(dec_expr, bool)]
        target_var_indep_decisions = []

        # Get lower and upper bounds over the variable
        for dec_expr, is_true in zip(decisions, decision_values):
            # Check boolean decisions or if self._var in dec_expr
            if (dec_expr in self._context._bool_var_set) or (self._var not in dec_expr.atoms()):
                target_var_indep_decisions.append((dec_expr, is_true))
                continue

            lhs, rhs, gt = dec_expr.lhs, dec_expr.rhs, isinstance(dec_expr, relational.GreaterThan)
            gt = (gt and is_true) or (not gt and not is_true)
            expr = lhs >= rhs if gt else lhs <= rhs

            # Get bounds over 'var'
            bound_expr, upper = xaddpy.utils.util.get_bound(self._var, expr)
            if upper:
                upper_bound.append(bound_expr)
            else:
                lower_bound.append(bound_expr)

        # lower bound over 'var' is the maximum among lower bounds
        xadd_lower_bound = -1
        for e in lower_bound:
            xadd_lower_bound = self._context.get_leaf_node(e) if xadd_lower_bound == -1 \
                               else self._context.apply(xadd_lower_bound, self._context.get_leaf_node(e), op='max')

        xadd_upper_bound = -1
        for e in upper_bound:
            xadd_upper_bound = self._context.get_leaf_node(e) if xadd_upper_bound == -1 \
                else self._context.apply(xadd_upper_bound, self._context.get_leaf_node(e), op='min')

        # Reduce lower and upper bound xadds for potential computational gains
        xadd_lower_bound = self._context.reduce_lp(xadd_lower_bound)
        xadd_upper_bound = self._context.reduce_lp(xadd_upper_bound)

        # Ensure lower bounds are smaller than upper bounds
        for e1 in lower_bound:
            for e2 in upper_bound:
                comp = (e2 - e1 >= 0)   # ub - lb
                if comp == sympy.S.true or \
                        e2 == oo or e1 == -oo:
                    continue
                target_var_indep_decisions.append((comp, True))
                assert isinstance(comp, relational.GreaterThan)

        # Substitute lower and upper bounds into leaf node
        eval_lower = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_upper_bound)

        # Take casemin / casemax of eval_lower and eval_upper
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
        is_bilinear = xaddpy.utils.util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`
            expr = xaddpy.utils.util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == sympy.S.true:
                min_max_eval = eval_upper
            elif dec_expr == sympy.S.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(dec, self._context.ZERO, self._context.ONE)      # Note: need to use ZERO_ig for annotating purpose... 
                ind_false = self._context.get_internal_node(dec, self._context.ONE, self._context.ZERO)     # but this is skipped in this branch
                upper_half = self._context.apply(ind_true if not is_reversed else ind_false, eval_upper, 'prod')
                lower_half = self._context.apply(ind_false if not is_reversed else ind_true, eval_lower, 'prod')
                min_max_eval = self._context.apply(upper_half, lower_half, 'add',
                                                   annotation=(xadd_upper_bound, xadd_lower_bound) if self._annotate else None)
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound, while 2nd argument is lower bound
            min_max_eval = self._context.apply(eval_upper, eval_lower, 'max' if self._is_max else 'min',
                                               annotation=(xadd_upper_bound, xadd_lower_bound) if self._annotate else None)

        # Reduce LP
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below: ensuring non-valid paths result in infinite penalty
            min_max_eval = self._context.apply(indep_constraint, min_max_eval, 'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        """
        Min(max)imize out remaining variables
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
            _ = self._context.reduce_process_xadd_leaf(min_max_eval, min_or_max, decisions, decision_values)
            min_max_eval = min_or_max._running_result

        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(self._running_result, min_max_eval, 'max' if self._is_max else 'min')
            self._running_result = self._context.reduce_lp(self._running_result)
        
        return self._context.get_leaf_node(leaf_val)


class XADDLeafMinOrMax(XADDLeafOperation):
    def __init__(
            self, 
            var: sympy.Symbol,
            is_max: bool,
            bound_dict: Dict[sympy.Symbol, tuple],
            context: XADD
    ):
        super().__init__(context)
        self._var: sympy.Symbol = var
        self._context._opt_var: sympy.Symbol = var
        self._is_max: bool = is_max
        self._running_result: int = -1
        if var in bound_dict:
            self._lower_bound: Union[int, float, numbers.Number] = bound_dict[var][0]
            self._upper_bound: Union[int, float, numbers.Number] = bound_dict[var][1]
        else:
            print("No domain bounds over {} are provided... using -oo and oo as lower and upper bounds.".format(var))
            self._lower_bound: Union[int, float, numbers.Number] = -oo
            self._upper_bound: Union[int, float, numbers.Number] = oo

    def process_xadd_leaf(self, decisions: list, decision_values: list, leaf_val: sympy.Basic):
        """

        :param decisions:
        :param decision_values:
        :param leaf_val:        (sympy.Basic) leaf expression
        :return:
        """
        # Check if below computation is unnecessary
        # min(oo, oo) = oo; max(oo, oo) = oo; min(-oo, -oo) = -oo; max(-oo, -oo) = -oo;
        # But, argmax and argmin are ambiguous in these cases, and so we simply annotate them with NaN
        if leaf_val == oo or leaf_val == -oo:
            min_max_eval = self._context.get_leaf_node(leaf_val, annotation=self._context.NAN)

            # Compare with the running result
            if self._running_result == -1:
                self._running_result = min_max_eval
            return self._context.get_leaf_node(leaf_val)

        # Bound management
        lower_bound = []
        upper_bound = []
        lower_bound.append(sympy.S(self._lower_bound))
        upper_bound.append(sympy.S(self._upper_bound))

        # Independent decisions (incorporated later): [(dec_expr, bool)]
        target_var_indep_decisions = []

        # Get lower and upper bounds over the variable
        for dec_expr, is_true in zip(decisions, decision_values):
            # Check boolean decisions or if self._var in dec_expr
            if (dec_expr in self._context._bool_var_set) or (self._var not in dec_expr.atoms()):
                target_var_indep_decisions.append((dec_expr, is_true))
                continue

            lhs, rhs, gt = dec_expr.lhs, dec_expr.rhs, isinstance(dec_expr, relational.GreaterThan)
            gt = (gt and is_true) or (not gt and not is_true)
            expr = lhs >= rhs if gt else lhs <= rhs

            # Get bounds over 'var'
            bound_expr, upper = xaddpy.utils.util.get_bound(self._var, expr)
            if upper:
                upper_bound.append(bound_expr)
            else:
                lower_bound.append(bound_expr)

        # lower bound over 'var' is the maximum among lower bounds
        xadd_lower_bound = -1
        for e in lower_bound:
            xadd_lower_bound = self._context.get_leaf_node(e) if xadd_lower_bound == -1 \
                               else self._context.apply(xadd_lower_bound, self._context.get_leaf_node(e), op='max')

        xadd_upper_bound = -1
        for e in upper_bound:
            xadd_upper_bound = self._context.get_leaf_node(e) if xadd_upper_bound == -1 \
                else self._context.apply(xadd_upper_bound, self._context.get_leaf_node(e), op='min')

        # Reduce lower and upper bound xadds for potential computational gains
        xadd_lower_bound = self._context.reduce_lp(xadd_lower_bound)
        xadd_upper_bound = self._context.reduce_lp(xadd_upper_bound)

        # Ensure lower bounds are smaller than upper bounds
        for e1 in lower_bound:
            for e2 in upper_bound:
                comp = (e2 - e1 >= 0)   # ub - lb
                if comp == sympy.S.true or \
                        e2 == oo or e1 == -oo:
                    continue
                target_var_indep_decisions.append((comp, True))
                assert isinstance(comp, relational.GreaterThan)
                # comp_lhs, is_reversed = self._context.clean_up_expr(comp.lhs, factor=True)
                # self._context._temp_ub_lb_cache.add(comp_lhs if not is_reversed else -comp_lhs)

        # Substitute lower and upper bounds into leaf node
        eval_lower = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_lower_bound)
        eval_upper = self._context.substitute_xadd_for_var_in_expr(leaf_val, var=self._var, xadd=xadd_upper_bound)

        # Take casemin / casemax of eval_lower and eval_upper
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
        is_bilinear = xaddpy.utils.util.is_bilinear(leaf_val)
        expr = 0
        if is_bilinear:
            # Get the expression multiplied to `self._var`
            expr = xaddpy.utils.util.get_multiplied_expr(leaf_val, self._var)
        if is_bilinear and expr != 0:
            dec_expr = expr <= 0
            if dec_expr == sympy.S.true:
                min_max_eval = eval_upper
            elif dec_expr == sympy.S.false:
                min_max_eval = eval_lower
            else:
                dec, is_reversed = self._context.get_dec_expr_index(dec_expr, create=True)
                ind_true = self._context.get_internal_node(dec, self._context.ZERO, self._context.ONE)      # Note: need to use ZERO_ig for annotating purpose... 
                ind_false = self._context.get_internal_node(dec, self._context.ONE, self._context.ZERO)     # but this is skipped in this branch
                upper_half = self._context.apply(ind_true if not is_reversed else ind_false, eval_upper, 'prod')
                lower_half = self._context.apply(ind_false if not is_reversed else ind_true, eval_lower, 'prod')
                min_max_eval = self._context.apply(upper_half, lower_half, 'add',
                                                   annotation=(xadd_upper_bound, xadd_lower_bound))
                min_max_eval = self._context.make_canonical(min_max_eval)
        else:
            # Note: always 1st argument should be upper bound, while 2nd argument is lower bound
            min_max_eval = self._context.apply(eval_upper, eval_lower, 'max' if self._is_max else 'min',
                                               annotation=(xadd_upper_bound, xadd_lower_bound))
        # self._context._temp_ub_lb_cache.clear()

        # Reduce LP
        min_max_eval = self._context.reduce_lp(min_max_eval)
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = False

        # Incorporate independent decisions
        for d, b in target_var_indep_decisions:
            high_val = oo if (b and self._is_max) or (not b and not self._is_max) \
                else -oo
            low_val = -oo if (b and self._is_max) or (not b and not self._is_max) \
                else oo
            indep_constraint = self._context.get_dec_node(d, low_val, high_val)
            # Note 'min' and 'max' are swapped below: ensuring non-valid paths result in infinite penalty
            min_max_eval = self._context.apply(indep_constraint, min_max_eval, 'min' if self._is_max else 'max')
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Reduce
        if self._context._args.get("leaf_minmax_no_prune", False):
            self._context._prune_equality = True
            min_max_eval = self._context.reduce_lp(min_max_eval)

        # Compare with the running result
        if self._running_result == -1:
            self._running_result = min_max_eval
        else:
            self._running_result = self._context.apply(self._running_result, min_max_eval, 'max' if self._is_max else 'min')

        # Reduce running result
        self._running_result = self._context.reduce_lp(self._running_result)

        return self._context.get_leaf_node(leaf_val)
