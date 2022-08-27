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
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")
    
    # Get the symbols
    b1, b2, b3, x, y = sp.symbols('b1 b2 b3 x y')
    b4, b5, a, b = sp.symbols('b4 b5 a b')

    # Define substitutions
    subst_bool = {b1: b4, b2: b5}
    subst_cont = {x: a, y: b}

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
    
    orig_dd = context.import_xadd(fname)
    
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")
    
    # Make canonical
    can_dd = context.make_canonical(orig_dd)
    
    # Get the symbols
    b1, b2, b3, x, y = sp.symbols('b1 b2 b3 x y')

    # Substitute x = 2 * y
    subst = {x: 2 * y}
    subst_dd = context.substitute()

def test_bvar_eval():
    context = XADD()
    return


if __name__ == "__main__":
    # test_bvar_subs()
    test_mixed_subs()
    # test_mixed_eval()
    # test_bvar_eval()

