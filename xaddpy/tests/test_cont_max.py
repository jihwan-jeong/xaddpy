import symengine.lib.symengine_wrapper as core

from xaddpy.utils.symengine import BooleanVar
from xaddpy.xadd.xadd import XADD


def test_continuous_max():
    context = XADD()

    # Load the reward function as XADD
    reward_dd = context.import_xadd('xaddpy/tests/ex/inventory.xadd')

    # Update the bound information over variables of interest
    a, x = core.Symbol('a'), core.Symbol('x')
    context.update_bounds({a: (0, 500), x: (-1000, 1000)})

    # Max out the demand
    d = context.get_var_from_name('d')
    max_reward_a_x = context.min_or_max_var(reward_dd, d, is_min=False, annotate=True)
    print(f"Maximize over a: \n{context.get_repr(max_reward_a_x)}")

    # Max out the order quantity
    max_reward_d_x = context.min_or_max_var(reward_dd, a, is_min=False, annotate=True)
    print(f"Maximize over a: \n{context.get_repr(max_reward_d_x)}")

    # Get the argmax over a
    argmax_a_id = context.reduced_arg_min_or_max(max_reward_d_x, a)
    print(f"Argmax over a: \n{context.get_repr(argmax_a_id)}")

    # Max out the inventory level
    max_reward_d = context.min_or_max_var(max_reward_d_x, x, is_min=False, annotate=True)
    print(f"Maximize over x: \n{context.get_repr(max_reward_d)}")

    # Get the argmax over x
    argmax_x_id = context.reduced_arg_min_or_max(max_reward_d, x)
    print(f"Argmax over x: \n{context.get_repr(argmax_x_id)}")
    return


if __name__ == "__main__":
    test_continuous_max()
