# Python Implementation of XADD

This repository implements the Python version of XADD (eXtended Algebraic Decision Diagrams) which was first introduced in [Sanner et al. (2011)](https://arxiv.org/pdf/1202.3762.pdf); you can find the original Java implementation from [here](https://github.com/ssanner/xadd-inference). 

Our Python XADD code uses [SymEngine](https://github.com/symengine/symengine.py) for symbolically maintaining all variables and related operations, and [PULP](https://github.com/coin-or/pulp) is used for pruning unreachable paths.  Note that we only check linear conditionals.  If you have Gurobi installed and configured in the conda environment, then PULP will use Gurobi for solving (MI)LPs; otherwise, the default solver ([CBC](https://github.com/coin-or/Cbc)) is going to be used. However, we do not actively support optimizers other than Gurobi for now.

Note that the implementation for [EMSPO](https://proceedings.mlr.press/v162/jeong22a/jeong22a.pdf) --- Exact symbolic reduction of linear Smart Predict+Optimize to MILP (Jeong et al., ICML-22) --- has been moved to the branch [emspo](https://github.com/jihwan-jeong/xaddpy/tree/emspo).

You can find the implementation for the [CPAIOR-23](https://ssanner.github.io/papers/cpaior23_dblpsve.pdf) work --- A Mixed-Integer Linear Programming Reduction of Disjoint Bilinear Programs via Symbolic
Variable Elimination --- in [examples/dblp](examples/dblp).

## Installation

**Load your Python virtual environment then type the following commands for package installation**

```shell
pip install xaddpy

# Optional: if you want to use Gurobi for the 'reduce_lp' method
# that prunes out unreachable partitions using LP solvers
pip install gurobipy    # If you have a license
```

## Installing pygraphviz for visualization
With `pygraphviz`, you can visualize a given XADD in a graph format, which can be very useful. Here, we explain how to install the package.

To begin with, you need to install the following:

- graphviz
- pygraphviz

Make sure you have activated the right conda environment with `conda activate YOUR_CONDA_ENVIRONMENT`.

### Step 1: Installing graphviz

1. For Ubuntu/Debian users, run the following command.

```shell
sudo apt-get install graphviz graphviz-dev
```

2. For Fedora and Red Hat systems, you can do as follows.

```shell
sudo dnf install graphviz graphviz-devel
```

3. For Mac users, you can use `brew` to install `graphviz`.

```shell
brew install graphviz
```

Unfortunately, we do not provide support for Windows systems, though you can refer to the [pygraphviz documentation](https://pygraphviz.github.io/documentation/stable/install.html) for information.

### Step 2: Installing pygraphviz

1. Linux systems

```shell
pip install pygraphviz
```

2. MacOS

```shell
python -m pip install \
    --global-option=build_ext \
    --global-option="-I$(brew --prefix graphviz)/include/" \
    --global-option="-L$(brew --prefix graphviz)/lib/" \
    pygraphviz
```

Note that due to the default installation location by `brew`, you need to provide some additional options for `pip` installation.

## Using xaddpy

You can find useful XADD usecases in the [xaddpy/tests/test_bool_var.py](xaddpy/tests/test_bool_var.py) and [xaddpy/tests/test_xadd.py](xaddpy/tests/test_xadd.py) files. Here, we will first briefly discuss different ways to build an initial XADD that you want to work with. 

### Loading from a file

If you know the entire structure of an initial XADD, then you can create a text file specifying the XADD and load it using the `XADD.import_xadd` method. It's important that, when you manually write down the XADD you have, you follow the same syntax rule as in the example file shown below.

Below is a part of the XADD written in [xaddpy/tests/ex/bool_cont_mixed.xadd](xaddpy/tests/ex/bool_cont_mixed.xadd):
```
...
        ( [x - y <= 0]
            ( [2 * x + y <= 0]
                ([x])
                ([y])
            )
            ( [b3]
                ([2 * x])
                ([2 * y])
            )
        )
...
```
Here, `[x-y <= 0]` defines a decision expression; its true branch is another node with the decision `[2 * x + y <= 0]`, while the decision of the false branch is a Boolean variable `b3`. Similarly, if `[2 * x + y <= 0]` holds true, then we get the leaf value `[x]`; otherwise, we get `[y]`. _All expressions should be wrapped with brackets, including Boolean variables._ A SymEngine `Symbol` object will be created for each unique variable.

To load this XADD, you can do the following:
```python
from xaddpy import XADD
context = XADD()
fname = 'xaddpy/tests/ex/bool_cont_mixed.xadd'

orig_xadd = context.import_xadd(fname)
```
Following the Java implementation, we call the instantiated XADD object `context`. This object maintains and manages all existing/new nodes and decision expressions. For example, `context._id_to_node` is a Python dictionary that stores mappings from node IDs (`int`) to the corresponding `Node` objects. For more information, please refer to the [constructor of the `XADD` class](xaddpy/xadd/xadd.py#L57).

To check whether you've got the right XADD imported, you can print it out.
```python
print(f"Imported XADD: \n{context.get_repr(orig_xadd)}")
```
The `XADD.get_repr` method will return `repr(node)` and the string representation of each XADD node is implemented in [xaddpy/xadd/node.py](xaddpy/xadd/node.py). Beware that the printing method can be slow for a large XADD.

### Recursively building an XADD
Another way of creating an initial XADD node is by recursively building it with the `apply` method. A very simple example would be something like this:

```python
from xaddpy import XADD
import symengine.lib.symengine_wrapper as core

context = XADD()

x_id = context.convert_to_xadd(core.Symbol('x'))
y_id = context.convert_to_xadd(core.Symbol('y'))

sum_node_id = context.apply(x_id, y_id, op='add')
comp_node_id = context.apply(sum_node_id, y_id, op='min')

print(f"Sum node:\n{context.get_repr(sum_node_id)}\n")
print(f"Comparison node:\n{context.get_repr(comp_node_id)}")
```
You can check that the print output shows
```
Sum node:
( [x + y] ) node_id: 9

Comparison node:
( [x <= 0] (dec, id): 10001, 10
         ( [x + y] ) node_id: 9 
         ( [y] ) node_id: 8 
)
```
which is the expected outcome!

Check out a much more comprehensive example demonstrating the recursive construction of a nontrivial XADD from here: [pyRDDLGym/XADD/RDDLModelXADD.py](https://github.com/ataitler/pyRDDLGym/blob/01955ee7bca2861124709c116f419f2927c04a89/pyRDDLGym/XADD/RDDLModelXADD.py#L124).

### Directly creating an XADD node
Finally, you might want to build a constant node, an arbitrary decision expression, and a Boolean decision directly. To this end, let's consider building the following XADD: 

```
([b]
    ([1])
    ([x + y <= 0]
        ([0])
        ([2])
    )
)
```

To do this, we will first create an internal node whose decision is `[x + y <= 0]`, the low and the high branches are `[0]` and `[2]` (respectively). Using `SymEngine`'s `S` function (or you can use `sympify`), you can turn an algebraic expression involving variables and numerics into a symbolic expression. Given this decision expression, you can get its unique index using `XADD.get_dec_expr_index` method. You can use the decision ID along with the ID of the low and high nodes connected to the decision to create the corresponding decision node, using `XADD.get_internal_node`.

```python
import symengine.lib.symengine_wrapper as core
from xaddpy import XADD

context = XADD()

# Get the unique ID of the decision expression
dec_expr: core.Basic = core.S('x + y <= 0')
dec_id, is_reversed = context.get_dec_expr_index(dec_expr, create=True)

# Get the IDs of the high and low branches: [0] and [2], respectively
high: int = context.get_leaf_node(core.S(0))
low: int = context.get_leaf_node(core.S(2))
if is_reversed:
    low, high = high, low

# Create the decision node with the IDs
dec_node_id: int = context.get_internal_node(dec_id, low=low, high=high)
print(f"Node created:\n{context.get_repr(dec_node_id)}")
```

Note that `XADD.get_dec_expr_index` returns a boolean variable `is_reversed` which is `False` if the canonical decision expression of the given decision has the same inequality direction. If the direction has changed, then `is_reversed=True`; in this case, low and high branches should be swapped.

Another way of creating this node is to use the `XADD.get_dec_node` method. This method can only be used when the low and high nodes are terminal nodes containing leaf expressions.

```python
dec_node_id = context.get_dec_node(dec_expr, low_val=core.S(2), high_val=core.S(0))
```

Note also that you need to wrap constants with the `core.S` function to turn them into `core.Basic` objects.

Now, it remains to create a decision node with the Boolean variable `b` and connect it to its low and high branches. 

```python
from xaddpy.utils.symengine import BooleanVar

b = BooleanVar(core.Symbol('b'))
dec_b_id, _ = context.get_dec_expr_index(b, create=True)
```

First of all, you need to import and instantiate a `BooleanVar` object for a Boolean variable. Otherwise, the variable won't be recognized as a Boolean variable in XADD operations!

Once you have the decision ID, we can finally link this decision node with the node created earlier. 

```python
high: int = context.get_leaf_node(core.S(1))
node_id: int = context.get_internal_node(dec_b_id, low=dec_node_id, high=high)
print(f"Node created:\n{context.get_repr(node_id)}")
```
And we get the following print outputs.
```
Output:
Node created:
( [b]   (dec, id): 2, 9
        ( [1] ) node_id: 1 
        ( [x + y <= 0]  (dec, id): 10001, 8
                ( [0] ) node_id: 0 
                ( [2] ) node_id: 7 
        )  
) 
```

### XADD Operations

#### XADD.apply(id1: int, id2: int, op: str)
You can perform the `apply` operation to two XADD nodes with IDs `id1` and `id2`. Below is the list of the supported operators (`op`):

**Non-Boolean operations**
- 'max', 'min'
- 'add', 'subtract'
- 'prod', 'div'

**Boolean operations**
- 'and'
- 'or'

**Relational operations**
- '!=', '=='
- '>', '>='
- '<', '<='

#### XADD.unary_op(node_id: int, op: str) (unary operations)
You can also apply the following unary operators to a single XADD node recursively (also check `UNARY_OP` in [xaddpy/utils/global_vars.py](xaddpy/utils/global_vars.py)). In this case, an operator will be applied to each and every leaf value of a given node. Hence, the decision expressions will remain unchanged.

- 'sin, 'cos', 'tan'
- 'sinh', 'cosh', 'tanh'
- 'exp', 'log', 'log2', 'log10', 'log1p'
- 'floor', 'ceil'
- 'sqrt', 'pow'
- '-', '+'
- 'sgn' (sign function... sgn(x) = 1 if x > 0; 0 if x == 0; -1 otherwise)
- 'abs'
- 'float', 'int'
- '~' (negation)

The `pow` operation requires an additional argument specifying the exponent.

#### XADD.evaluate(node_id: int, bool_assign: dict, cont_assign: bool, primitive_type: bool)
When you want to assign concrete values to Boolean and continuous variables, you can use this method. An example is provided in the `test_mixed_eval` function defined in [xaddpy/tests/test_bool_var.py](xaddpy/tests/test_bool_var.py).

As another example, let's say we want to evaluate the XADD node defined a few lines above.
```python
x, y = core.symbols('x y')

bool_assign = {b: True}
cont_assign = {x: 2, y: -1}

res = context.evaluate(node_id, bool_assign=bool_assign, cont_assign=cont_assign)
print(f"Result: {res}")
```

In this case, `b=True` will directly leads to the leaf value of `1` regardless of the assignment given to `x` and `y` variables. 

```python
bool_assign = {b: False}
res = context.evaluate(node_id, bool_assign=bool_assign, cont_assign=cont_assign)
print(f"Result: {res}")
```

If we change the value of `b`, we can see that we get `2`. Note that you have to make sure that all symbolic variables get assigned specific values; otherwise, the function will return `None`. 

#### XADD.substitute(node_id: int, subst_dict: dict)
If instead you want to assign values to a subset of symbolic variables while leaving the other variables as-is, you can use the `substitute` method. Similar to `evaluate`, you need to pass in a dictionary mapping SymEngine `Symbol`s to their concrete values.

For example,

```python
subst_dict = {x: 1}
node_id_after_subs = context.substitute(node_id, subst_dict)
print(f"Result:\n{context.get_repr(node_id_after_subs)}")
```
which outputs
```
Result:
( [b]   (dec, id): 2, 16
        ( [1] ) node_id: 1 
        ( [y + 1 <= 0]  (dec, id): 10003, 12
                ( [0] ) node_id: 0 
                ( [2] ) node_id: 7 
        )  
) 
```
as expected.

#### XADD.collect_vars(node_id: int)
If you want to extract all Boolean and continuous symbols existing in an XADD node, you can use this method.

```python
var_set = context.collect_vars(node_id)
print(f"var_set: {var_set}")
```
```
Output:
var_set: {y, b, x}
```

This method can be useful to figure out which variables need to have values assigned in order to evaluate a given XADD node.

#### XADD.make_canonical(node_id: int)
This method gives a canonical order to an XADD that is potentially unordered. Note that the `apply` method already calls `make_canonical` when the `op` is one of `('min', 'max', '!=', '==', '>', '>=', '<', '<=', 'or', 'and')`.


#### Variable Elimination

1. Sum out: `XADD.op_out(node_id: int, dec_id: int, op: str = 'add')`

Let's say we have a joint probability distribution function over Boolean variables `b1, b2`, i.e., `P(b1, b2)` as in the following example. `P(b1, b2)=`

```
( [b1]
    ( [b2] 
        ( [0.25] )
        ( [0.3] )
    )
    ( [b2]
        ( [0.1] )
        ( [0.35] )
    )
)
```
Notice that the values are non-negative and sum up to one, making this a valid probability distribution. Now, one may be interested in marginalizing out a variable `b2` to get `P(b1) = \sum_{b2} P(b1, b2)`. This can be done in XADD by using the `op_out` method.

Let's directly dive into an example:

```python
# Load the joint probability as XADD
p_b1b2 = context.import_xadd('xaddpy/tests/ex/bool_prob.xadd')

# Get the decision index of `b2`
b2 = BooleanVar(core.Symbol('b2'))
b2_dec_id, _ = context.get_dec_expr_index(b2, create=False)

# Marginalize out `b2`
p_b1 = context.op_out(node_id=p_b1b2, dec_id=b2_dec_id, op='add')
print(f"P(b1): \n{context.get_repr(p_b1)}")
```
```
Output: 
P(b1): 
( [b1]  (dec, id): 1, 26
        ( [0.55] ) node_id: 25 
        ( [0.45] ) node_id: 24 
)
```

As expected, the obtained `P(b1)` is a function of only `b1` variable, and the probabilities sum up to `1`.


2. Prod out

Similarly, if we specify `op='prod'`, we can 'prod out' a Boolean variable from a given XADD.

3. Max out (or min out) continuous variables: `XADD.min_or_max_var(node_id: int, var: VAR_TYPE, is_min: bool)`

One of the most interesting and useful applications of symbolic variable elimination is 'maxing out' or 'minning out' **continuous** variable(s) from a symbolic function. See [Jeong et al. (2023)](https://ssanner.github.io/papers/cpaior23_dblpsve.pdf) and [Jeong et al. (2022)](https://proceedings.mlr.press/v162/jeong22a/jeong22a.pdf) for more detailed discussions. Look up the `min_or_max_var` method in the xadd.py file. For now, we only support optimizing a linear or disjointly bilinear expressions at the leaf values and decision expressions.

As a concrete toy example, imagine the problem of inventory management. There is a Boolean variable `d` which denotes the level of demand (i.e., `d=True` if demand is high; otherwise `d=False`). Let's say the current inventory level of a product of interest is `x \in [-1000, 1000]`. Suppose we can place an order of amount `a \in [0, 500]` for this product. And we will have the following reward based on the current demand, inventory level, and the new order:
```
( [d]
    ( [x >= 150]
        ( [150 - 0.1 * a - 0.05 * x ] )
        ( [(x - 150) - 0.1 * a - 0.05 * x] )
    )
    ( [x >= 50]
        ( [50 - 0.1 * a - 0.05 * x] )
        ( [(x - 50) - 0.1 * a - 0.05 * x] )
    )
)
```
Though it is natural to consider multi-step decisions for this kind of problem, let's only focus on optimizing this reward for a single step, for the sake of simplicity and illustration.

So, given this reward, what we might be interested in is the maximum reward we can obtain, subject to the demand level and the current inventory level. That is, we want to compute `max_a reward(a, x, d)`.

```python
# Load the reward function as XADD
reward_dd = context.import_xadd('xaddpy/tests/ex/inventory.xadd')

# Update the bound information over variables of interest
a, x = core.Symbol('a'), core.Symbol('x')
context.update_bounds({a: (0, 500), x: (-1000, 1000)}) 

# Max out the order quantity
max_reward_d_x = context.min_or_max_var(reward_dd, a, is_min=False, annotate=True)
print(f"Maximize over a: \n{context.get_repr(max_reward_d_x)}")
```
```
Output:
Maximize over a: 
( [d]   (dec, id): 1, 82
        ( [-150 + x <= 0]       (dec, id): 10002, 81
                ( [-150 + 0.95*x] ) node_id: 72 anno: 0 
                ( [150 - 0.05*x] ) node_id: 58 anno: 0 
        )  
        ( [-50 + x <= 0]        (dec, id): 10003, 51
                ( [-50 + 0.95*x] ) node_id: 42 anno: 0 
                ( [50 - 0.05*x] ) node_id: 29 anno: 0 
        )  
)
```
To obtain this result, note that we should provide the bound information over the continuous variables. If not, then `(-oo, oo)` will be used as the bounds.

If we want to know which values of `a` will yield the optimal outcomes, we can apply the argmax operation. Specifically,

```python
argmax_a_id = context.reduced_arg_min_or_max(max_reward_d_x, a)
print(f"Argmax over a: \n{context.get_repr(argmax_a_id)}")
```
```
Output:
Argmax over a: 
( [0] ) node_id: 0
```
Trivially in this case, not ordering any new products will maximize the one-step reward, which makes sense. A more interesting case would, of course, be when we have to make sequential decisions taking into account stochastic demands and the product level that changes according to the order amount and the demand. For this kind of problems, we suggest you take a look at [Symbolic Dynamic Programming (SDP)](https://github.com/ataitler/pyRDDLGym/blob/sdp-symengine/run_examples/run_vi.py).

Now, if we further optimize the `max_reward_d_x` over `x` variable, we get the following:
```python
# Max out the inventory level
max_reward_d = context.min_or_max_var(max_reward_d_x, x, is_min=False, annotate=True)
print(f"Maximize over x: \n{context.get_repr(max_reward_d)}")

# Get the argmax over x
argmax_x_id = context.reduced_arg_min_or_max(max_reward_d, x)
print(f"Argmax over x: \n{context.get_repr(argmax_x_id)}")
```

```
Output:
Maximize over x: 
( [d]   (dec, id): 1, 105
        ( [142.5] ) node_id: 102 anno: 99 
        ( [47.5] ) node_id: 89 anno: 85 
)
Argmax over x: 
( [d]   (dec, id): 1, 115
        ( [150] ) node_id: 99 
        ( [50] ) node_id: 85 
)
```
The results tells us that the maximum achievable reward is 142.5 when `d=True, x=150` or 47.5 when `d=False, x=50`.

4. Max (min) out Boolean variables with `XADD.min_or_max_var`

Eliminating Boolean variables with max or min operations can be easily done by using the previously discussed `min_or_max_var` method. You just need to pass the Boolean variable to the method.

#### Definite Integral

Given an XADD node and a symbolic variable, you can integrate out the variable from the node. See [test_def_int.py](xaddpy/tests/test_def_int.py) which provides examples of this operation.

## Citation

Please use the following bibtex for citations:

```
@InProceedings{pmlr-v162-jeong22a,
  title = 	 {An Exact Symbolic Reduction of Linear Smart {P}redict+{O}ptimize to Mixed Integer Linear Programming},
  author =       {Jeong, Jihwan and Jaggi, Parth and Butler, Andrew and Sanner, Scott},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {10053--10067},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}
```