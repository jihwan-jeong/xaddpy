from pathlib import Path
import sympy as sp

from xaddpy import XADD


def test_bvar_subs():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "xaddpy/tests/ex/boolxor.xadd"
    
    orig_dd = context.import_xadd(fname)
    print(f"Original XADD: \n{context.get_repr(orig_dd)}")

    b1, b2, b3 = sp.symbols('b1 b2 b3')
    replace = {b1: True, b2: True, b3: False}
    subs_dd = context.substitute_bool_vars(orig_dd, replace)
    print(f"After substitution: \n{context.get_exist_node(subs_dd)}")


def test_mixed_subs():
    context = XADD()
    cwd = Path.cwd()
    fname = cwd / "xaddpy/tests/ex/bool_cont_mixed.xadd"
    
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
    fname = cwd / "xaddpy/tests/ex/bool_cont_mixed.xadd"
    
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
        
    # Define the variables
    b1, b2, b3 = sp.symbols('b1 b2 b3', bool=True)
    x, y = sp.symbols('x y')
    dec_b1, is_reversed_b1 = context.get_dec_expr_index(b1, create=True)
    dec_b2, is_reversed_b2 = context.get_dec_expr_index(b2, create=True)

    # 2x + y <= 0 => x, otherwise y
    e1 = 2 * x + y <= 0
    id1 = context.get_dec_node(e1, y, x)
    
    # b3 => 2x, otherwise 2y
    id2 = context.get_dec_node(b3, 2*y, 2*x)
    
    # b3 => 3x, otherwise 3y
    id3 = context.get_dec_node(b3, 3*y, 3*x)

    # b3 => 4x, otherwise 4y
    id4 = context.get_dec_node(b3, 4*y, 4*x)

    # b3 => 5x, otherwise 5y
    id5 = context.get_dec_node(b3, 5*y, 5*x)

    # b3 => 6x, otherwise 6y
    id6 = context.get_dec_node(b3, 6*y, 6*x)

    e2 = x - y <= 0
    dec2, is_reversed = context.get_dec_expr_index(e2, create=True)
    high1 = context.get_internal_node(dec2, id1 if is_reversed else id2, id2 if is_reversed else id1)
    
    e3 = -3 * x - 5 * y <= 0
    dec3, is_reversed = context.get_dec_expr_index(e3, create=True)
    low1 = context.get_internal_node(dec3, id3 if is_reversed else id4, id4 if is_reversed else id3)

    high = context.get_internal_node(
        dec_b2, 
        high1 if is_reversed_b2 else low1, 
        low1 if is_reversed_b2 else high1)
    low = context.get_internal_node(
        dec_b2, 
        id5 if is_reversed_b2 else id6,
        id6 if is_reversed_b2 else id5)
        
    orig_dd = context.get_internal_node(
        dec_b1,
        high if is_reversed_b1 else low,
        low if is_reversed_b1 else high)
    orig_dd = context.make_canonical(orig_dd)

    print(f"Original XADD (not canonical form): \n{context.get_repr(orig_dd)}")
    orig_dd = context.reduce_lp(orig_dd)
    print(f"After calling reduce_lp: \n{context.get_repr(orig_dd)}")   
    
    # Substitute x = 2 * y + 5
    subst = {x: 2 * y + 5}
    subst_dd = context.substitute(orig_dd, subst)
    print(f"Substitute `x = 2 * y + 5`\nResult (before reduce_lp): \n{context.get_repr(subst_dd)}")
    res = context.reduce_lp(subst_dd)
    print(f"\nAfter reduce_lp: \n{context.get_repr(res)}")


if __name__ == "__main__":
    test_bvar_subs()
    test_mixed_subs()
    test_mixed_eval()
    test_reduce_lp()
