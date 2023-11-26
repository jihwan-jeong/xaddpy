import warnings
import sympy.core.relational as relational
import sympy
from sympy import oo
import abc

try:
    from xaddpy.utils.graph import Graph
except ImportError:
    warnings.warn("[Module: xaddpy.xadd.node] Import error: pygraphviz not installed")
    class Graph:
        def __init__(self, *args, **kwargs):
            pass


class Node(metaclass=abc.ABCMeta):
    def __init__(self, context):
        self._context = context
        self._print_node_info = True
        self._is_leaf = False
        self._inode_to_vars = self._context._inode_to_vars

    def __str__(self):
        return

    @abc.abstractmethod
    def turn_off_print_node_info(self):
        pass

    @abc.abstractmethod
    def turn_on_print_node_info(self):
        pass        

    @abc.abstractmethod
    def collect_vars(self):
        pass

    def collect_nodes(self, nodes=None):
        """
        :param nodes:       (set)
        :return:
        """
        pass

    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    @abc.abstractmethod
    def low(self) -> int:
        pass
    
    @property
    @abc.abstractmethod
    def high(self) -> int:
        pass

    @abc.abstractmethod
    def to_graph(self, graph: Graph, node_id: int):
        pass
    

class XADDTNode(Node):
    def __init__(self, expr: sympy.Basic, annotation=None, context=None):
        """
        A leaf XADD node implementation. Annotation can be tracked. Need to provide integer ids for
        leaf expression, node id, and annotation (if not None).
        :param expr:            (sympy.Basic) XADDTNode receives symbolic expression not integer id
        :param annotation:        (int)
        """
        # Link the node with XADD
        assert context is not None, "XADD should be passed when instantiating nodes!"
        super().__init__(context)

        # Set the expression associated with the leaf node
        self.expr = expr

        # Set the annotation
        self._annotation = annotation

        # Flag the node as leaf
        self._is_leaf = True

    def turn_off_print_node_info(self):
        self._print_node_info = False

    def turn_on_print_node_info(self):
        self._print_node_info = True

    def set(self, expr, annotation):
        """
        Set expression and annotation.
        :param expr:        (sympy.Basic) Symbolic expression
        :param annotation:    (int) id of annotation
        """
        self.expr = expr
        self._annotation = annotation

    @property
    def expr(self) -> sympy.Basic:
        return self._expr

    @expr.setter
    def expr(self, expr: sympy.Basic):
        assert isinstance(expr, sympy.Basic), "expr should be a Sympy object for XADDTNode!"
        self._expr = expr

    @property
    def _annotation(self):
        return self.__annotation

    @_annotation.setter
    def _annotation(self, annotation):
        self.__annotation = annotation

    def collect_vars(self):
        """
        Updates the set containing all sympy Symbols.
        """
        return self.expr.free_symbols.copy()

    def collect_nodes(self, nodes=None):
        if nodes is not None:
            nodes.add(self)
        else:
            nodes = set()
            nodes.add(self)

    def __hash__(self):
        if self._annotation is None:
            return hash(str(self.expr))
        else:
            return hash((str(self.expr), self._annotation))

    def __eq__(self, other):
        if self._annotation is None:
            return other._annotation is None and str(self.expr) == str(other.expr)
        else:
            return str(self.expr) == str(other.expr) and self._annotation == other._annotation

    def __str__(self, level=0):
        # curr_node_expr = self.expr
        str_expr = f"( [{self.expr}] )"
        str_node_id = f" node_id: {self._context._node_to_id.get(self)}"
        str_anno = f" anno: {self._annotation}" if self._annotation is not None else ""
        if self._print_node_info:
            return str_expr + str_node_id + str_anno
        else:
            return str_expr
        #
        # ret = "\t" * level + str(self.expr) + "\n"
        # if not self._is_leaf:
        #     low = cache.id_to_node[self.low]
        #     high = cache.id_to_node[self.high]
        #     ret += high.__str__(level + 1)
        #     ret += low.__str__(level+1)
        # else:
        #
        #     res_str = "{}".format(str(self.expr))
        # return ret

    def __repr__(self, level=0):
        # curr_node_expr = self.expr
        str_expr = f"( [{self.expr}] )"
        str_node_id = f" node_id: {self._context._node_to_id.get(self)}"
        str_anno = f" anno: {self._annotation}" if self._annotation is not None else ""
        if self._print_node_info:
            return str_expr + str_node_id + str_anno
        else:
            return str_expr

    def to_graph(self, graph: Graph, node_id: int):
        this_node = str(node_id)
        graph.add_node(this_node)
        str_expr = f"( [{self.expr}] )"
        if self._annotation is None:
            graph.add_node_label(this_node, str_expr)
        else:
            graph.add_node_label(this_node, f"{str_expr}\n[ {self._annotation} ]")
        graph.add_node_color(this_node, color='lightsalmon')
        graph.add_node_shape(this_node, shape='box')
        graph.add_node_style(this_node, style='filled')
    
    @property
    def high(self) -> int:
        raise NotImplementedError
    
    @property
    def low(self) -> int:
        raise NotImplementedError


class XADDINode(Node):
    def __init__(self, dec, low=None, high=None, context=None):
        """
        Basic decision node of a tree case function.
        The value is a Sympy inequality expression, and the low and the high branches correspond to
        False and True, respectively. Each node will have a unique identifier (integer) stored in a dictionary
        as an attribute in a XADD object.
        :param dec:     (int) Decision expression in a canonical form (rhs is a number, lhs contains variables)
        :param low:     (int) False branch
        :param high:    (int) True branch
        """
        # Link the node with XADD
        assert context is not None, "XADD should be passed when instantiating nodes!"
        super().__init__(context)

        self.dec = dec
        self.low = low
        self.high = high

    def turn_off_print_node_info(self):
        self._print_node_info = False
        high = self._context._id_to_node.get(self.high, None)
        if high is not None:
            high.turn_off_print_node_info()

        low = self._context._id_to_node.get(self.low, None)
        if low is not None:
            low.turn_off_print_node_info()

    def turn_on_print_node_info(self):
        self._print_node_info = True
        high = self._context._id_to_node.get(self.high, None)
        if high is not None:
            high.turn_on_print_node_info()

        low = self._context._id_to_node.get(self.low, None)
        if low is not None:
            low.turn_on_print_node_info()

    def set(self, dec_id, low, high):
        self.dec = dec_id
        self.low = low
        self.high = high

    @property
    def dec(self):
        return self._dec

    @dec.setter
    def dec(self, dec):
        assert isinstance(dec, int)
        self._dec = dec

    @property
    def high(self) -> int:
        return self._high
    
    @property
    def low(self) -> int:
        return self._low
    
    @high.setter
    def high(self, high):
        self._high = high
    
    @low.setter
    def low(self, low):
        self._low = low

    def collect_nodes(self, nodes: set = None):
        if self in nodes:
            return
        nodes.add(self)
        self._context.get_exist_node(self._low).collect_nodes(nodes)
        self._context.get_exist_node(self._high).collect_nodes(nodes)

    def collect_vars(self):
        # Check cache
        var_set = self._inode_to_vars.get((self.dec, self.low, self.high), None)
        if var_set is not None:
            return var_set.copy()

        var_set = set()
        low = self._context.get_exist_node(self.low)
        high = self._context.get_exist_node(self.high)
        low_vars = low.collect_vars()
        high_vars = high.collect_vars()
        expr = self._context._id_to_expr[self.dec]
        var_set.update(expr.free_symbols)
        var_set.update(low_vars)
        var_set.update(high_vars)

        self._inode_to_vars[(self.dec, self.low, self.high)] = var_set.copy()
        return var_set.copy()

    def __hash__(self):
        """
        Note that the terminal node and internal node are used differently in comparing keys in dictionary.
        """
        return hash((self.dec, self.low, self.high))

    def __eq__(self, other):
        if isinstance(other, XADDINode):
            return (self.dec == other.dec) and (self.low == other.low) and (self.high == other.high)
        else:
            return False

    def __str__(self, level=0):
        ret = ""
        ret += f"( [{self._context._id_to_expr[self.dec]}]"

        # print node id
        if self._print_node_info:
            ret += f" \t(dec, id): {self.dec}, {self._context._node_to_id.get(self)}"

        # Node level cache
        high = self._context._id_to_node.get(self.high, None)
        if high is not None:
            ret += "\n" + "\t"*(level+1) + f"{high.__str__(level+1)} "
        else:
            ret += "h:[None] "

        low = self._context._id_to_node.get(self.low, None)
        if low is not None:
            ret += "\n" + "\t" * (level + 1) + f"{low.__str__(level + 1)} "
        else:
            ret += "l:[None] "
        ret += "\n" + "\t" * level + ") "
        return ret

    def __repr__(self, level=0):
        ret = ""
        ret += f"( [{self._context._id_to_expr[self.dec]}]"

        # print node id
        if self._print_node_info:
            ret += f" \t(dec, id): {self.dec}, {self._context._node_to_id.get(self)}"

        # Node level cache
        high = self._context._id_to_node.get(self.high, None)
        if high is not None:
            ret += "\n" + "\t"*(level+1) + f"{high.__str__(level+1)} "
        else:
            ret += "h:[None] "

        low = self._context._id_to_node.get(self.low, None)
        if low is not None:
            ret += "\n" + "\t" * (level + 1) + f"{low.__str__(level + 1)} "
        else:
            ret += "h:[None] "
        ret += "\n" + "\t" * level + ") "
        return ret

    def to_graph(self, graph: Graph, node_id: int):
        # Decision node
        this_node = str(node_id)
        graph.add_node(this_node)
        dec_id = self.dec
        dec_expr = self._context._id_to_expr[dec_id]
        str_expr = str(dec_expr)
        graph.add_node_label(this_node, str_expr)
        graph.add_node_color(this_node, color='lightblue')
        graph.add_node_shape(this_node, shape='ellipse')
        graph.add_node_style(this_node, style='filled')

        # Children
        low: Node = self._context.get_exist_node(self.low)
        if low is not None:
            low_node = str(self.low)
            graph.add_edge(this_node, low_node, color='black', style='dashed', label='')
            low.to_graph(graph, self.low)
        
        high: Node = self._context.get_exist_node(self.high)
        if high is not None:
            high_node = str(self.high)
            graph.add_edge(this_node, high_node, color='black', style='solid', label='')
            high.to_graph(graph, self.high)
