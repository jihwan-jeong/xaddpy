import symengine.lib.symengine_wrapper as core

from xaddpy.utils.symengine import BooleanVar
from xaddpy.xadd.xadd import XADD


def test_xadd():
    context = XADD()
    x, y = core.S('x'), core.S('y')
    """
    Create a node 
        ([x - y - 5 <= 0]
            ([x ** 2])              # when the decision expression holds true
            ([10])                  # otherwise
        )
    """
    dec_expr1 = x - y <= 5
    
    xadd_as_list1 = [dec_expr1, [x ** 2], [core.S(10)]]       # constant numbers should be passed through core.S()
    node1: int = context.build_initial_xadd(xadd_as_list1)       # This method recursively builds an XADD node given a nested list of expressions 
    print(f"Node 1:\n{context.get_exist_node(node1)}")
    """
    Create another node 
        ([x + 2 * y <= 0]
            ([-2 * y])              # when the decision expression holds true
            ([3 * x])               # otherwise
        )
    """
    dec_expr2 = x + 2 * y <= 0
    dec_id2, is_reversed = context.get_dec_expr_index(dec_expr2, create=True)
    high: int = context.get_leaf_node(core.S(- 2) * y)        # You can instantiate a leaf node by passing the expression
    low: int = context.get_leaf_node(core.S(3) * x)
    if is_reversed:                                     # In case the canonical expression associated with `dec_id` is reversed,
        tmp = low; low = high; high = tmp               # swap low and high
    node2: int = context.get_internal_node(dec_id=dec_id2, low=low, high=high)
    print(f"Node 2:\n{context.get_exist_node(node2)}")

    # Examples of some basic operations between the two XADDs
    node_sum = context.apply(node1, node2, op='add')
    node_prod = context.apply(node1, node2, op='prod')
    node_case_min = context.apply(node1, node2, op='min')
    node_case_max = context.apply(node1, node2, op='max')

    """
    Additional notes: 
        this repo selectively implemented necessary components from the original Java XADD code.
        So, there should be some missing functionalities and some operations may not be supported in the current form.
    """
    return


def test_op_out():
    context = XADD()

    # Load the joint probability as XADD
    p_b1b2 = context.import_xadd('xaddpy/tests/ex/bool_prob.xadd')

    # Get the decision index of `b2`
    b2 = BooleanVar(core.Symbol('b2'))
    b2_dec_id, _ = context.get_dec_expr_index(b2, create=False)

    # Marginalize out `b2`
    p_b1 = context.op_out(node_id=p_b1b2, dec_id=b2_dec_id, op='add')
    print(f"P(b1): \n{context.get_repr(p_b1)}")


if __name__ == "__main__":
    test_xadd()
    test_op_out()
