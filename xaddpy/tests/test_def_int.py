import symengine.lib.symengine_wrapper as core

from xaddpy.utils.symengine import BooleanVar
from xaddpy.xadd.xadd import XADD, XADDLeafDefIntegral


def test_uni_int():
    context = XADD()

    # Load the XADD.
    fname = 'xaddpy/tests/ex/uni_int.xadd'
    xadd = context.import_xadd(fname)

    # Get the variable.
    x = context.get_var_from_name('x')

    # Integrate the XADD.
    leaf_op = XADDLeafDefIntegral(x, context)
    _ = context.reduce_process_xadd_leaf(xadd, leaf_op, [], [])
    res = leaf_op.running_sum
    print(f"Original XADD:\n{context.get_repr(xadd)}")
    print(f"Integral XADD:\n{context.get_repr(res)}")


def test_multi_int():
    context = XADD()

    # Load the XADD.
    fname = 'xaddpy/tests/ex/multi_int.xadd'
    xadd = context.import_xadd(fname)
    print(f"Original XADD:\n{context.get_repr(xadd)}")

    # Get the variables.
    x1, x2, x3 = list(map(context.get_var_from_name, ['x1', 'x2', 'x3']))

    # Integrate the XADD.
    int_id = xadd
    for v in [x1, x2, x3]:
        leaf_op = XADDLeafDefIntegral(v, context)
        _ = context.reduce_process_xadd_leaf(int_id, leaf_op, [], [])
        int_id = leaf_op.running_sum
        print(f"Integrated out {str(v)}:\n{context.get_repr(int_id)}")


if __name__ == "__main__":
    test_uni_int()
    test_multi_int()
