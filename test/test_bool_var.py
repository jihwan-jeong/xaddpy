from pathlib import Path
import sympy as sp

from xaddpy.xadd.xadd import XADD



def test_bvar_subs():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "test/ex/boolxor.xadd"
    
    orig_dd = context.import_xadd(fname)
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")

    b1, b2, b3 = sp.symbols('b1 b2 b3')
    replace = {b1: True, b2: True, b3: False}
    subs_dd = context.substitute_bool_vars(orig_dd, replace)
    print(f"After substitution: \n{context.get_exist_node(subs_dd)}")


def test_mixed_subs():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "test/ex/bool_cont_mixed.xadd"
    
    orig_dd = context.import_xadd(fname)
    var_set = context.collect_vars(orig_dd)
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")
    
    # Get the symbols
    b1, b2, b3, x, y = sp.symbols('b1 b2 b3 x y')

    # Define substitutions
    subst_bool = {b1: b2, b2: b1}
    subst_cont = {x: y, y: x}

    res1 = context.substitute(orig_dd, subst_dict=subst_bool)
    res1 = context.reduce_lp(res1)
    print(f"After boolean substitution of {subst_bool}: \n{context.get_repr(res1)}")

    res2 = context.substitute(orig_dd, subst_dict=subst_cont)
    res2 = context.reduce_lp(res2)
    print(f"After substitution of continuous vars {subst_cont}: \n{context.get_repr(res2)}")

    subst_all = subst_bool.copy()
    subst_all.update(subst_cont)
    res3 = context.substitute(orig_dd, subst_dict=subst_all)
    res3 = context.reduce_lp(res3)
    print(f"After substitution of all vars {subst_all}: \n{context.get_repr(res3)}")


def test_mixed_eval():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "test/ex/bool_cont_mixed.xadd"
    
    orig_dd = context.import_xadd(fname)
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")
    
    # Get the symbols
    b1, b2, b3, x, y = sp.symbols('b1 b2 b3 x y')
    
    # Evaluate the node
    bool_assign = {b1: True, b2: False, b3: True}
    cont_assign = {x: 3.0, y: -3}

    res = context.evaluate(orig_dd, bool_assign=bool_assign, cont_assign=cont_assign)

    if res is None:
        print("Not every variable is assigned a value")
    else:
        print(f"Evaluated result: {res}")


def test_reduce_lp():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "test/ex/bool_cont_mixed.xadd"
    
    orig_dd = context.import_xadd(fname, to_canonical=False)
    print(f"Original XADD (not canonical form): \n{context.get_repr(orig_dd)}")
    orig_dd = context.reduce_lp(orig_dd)
    print(f"After calling reduce_lp: \n{context.get_repr(orig_dd)}")   
    
    # Get the symbols
    b1, b2, b3, x, y = sp.symbols('b1 b2 b3 x y')

    # Substitute x = 2 * y + 5
    subst = {x: 2 * y + 5}
    subst_dd = context.substitute(orig_dd, subst)
    print(f"Substitute `x = 2 * y + 5`\nResult (before reduce_lp): \n{context.get_repr(subst_dd)}")
    res = context.reduce_lp(subst_dd)
    print(f"\nAfter reduce_lp: \n{context.get_repr(res)}")


if __name__ == "__main__":
    # test_bvar_subs()
    # test_mixed_subs()
    # test_mixed_eval()
    test_reduce_lp()
    # test_bvar_eval()

