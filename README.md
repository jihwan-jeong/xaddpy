# Python Implementation of XADD

This repository implements the Python version of XADD (eXtended Algebraic Decision Diagrams) which was first introduced in [Sanner et al. (2011)](https://arxiv.org/pdf/1202.3762.pdf); you can find the original Java implementation from [here](https://github.com/ssanner/xadd-inference). 

Our Python XADD code uses [Sympy](https://github.com/sympy/sympy) for symbolically maintaining all variables and related operations, and [PULP](https://github.com/coin-or/pulp) is used for pruning unreachable paths.  Note that we only check linear conditionals.  If you have Gurobi installed and configured in the conda environment, then PULP will use Gurobi for solving (MI)LPs; otherwise, the default solver ([CBC](https://github.com/coin-or/Cbc)) is going to be used.

Note that the implementation for [EMSPO](https://proceedings.mlr.press/v162/jeong22a/jeong22a.pdf) --- Exact symbolic reduction of linear Smart Predict+Optimize to MILP (Jeong et al., ICML-22) --- has been moved to the branch [emspo](https://github.com/jihwan-jeong/xaddpy/tree/emspo). 

## Installation

**Load your Python virtual environment then type the following commands for package installation**

```shell
pip install xaddpy

# Optional: if you want to use Gurobi for the 'reduce_lp' method
# that prunes out unreachable partitions using LP solvers
pip install gurobipy    # If you have a license
```

## Using xaddpy

You can find useful XADD usecases in the [xaddpy/tests/test_bool_var.py](xaddpy/tests/test_bool_var.py) and [xaddpy/tests/test_xadd.py](xaddpy/tests/test_xadd.py) files. Here, we will briefly show two main ways to build an initial XADD that you want to work with. 

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
            (b3
                ([2 * x])
                ([2 * y])
            )
        )
...
```
Here, `[x-y <= 0]` defines a decision expression; its true branch is another node with the decision `[2 * x + y <= 0]`, while the decision of the false branch is a Boolean variable `b3`. Similarly, if `[2 * x + y <= 0]` holds true, then we get the leaf value `[x]`; otherwise, we get `[y]`. Inequality decisions and leaf values are wrapped with brackets, while you can directly put the variable name in the case of a Boolean decision. A Sympy `Symbol` object will be created for each unique variable.

To load this XADD, you can do the following:
```python
from xaddpy import XADD
context = XADD()
fname = 'xaddpy/tests/ex/bool_cont_mixed.xadd'

orig_xadd = context.import_xadd(fname)
```
Following the Java implementation, we call the instantiated XADD object `context`. This object maintains and manages all existing/new nodes and decision expressions. For example, `context._id_to_node` is a Python dictionary that stores mappings from node IDs (int) to the corresponding `Node` objects. For more information, please refer to the constructor of the `XADD` class.

To check whether you've got the right XADD imported, you can print it out.
```python
print(f"Imported XADD: \n{context.get_repr(orig_xadd)}")
```
The `XADD.get_repr` method will return `repr(node)` and the string representation of each XADD node is implemented in [xaddpy/xadd/node.py](xaddpy/xadd/node.py). Beware that the printing method can be slow for a large XADD.

### Recursively building an XADD
Another way of creating an initial XADD node is by recursively building it with the `apply` method. A very simple example would be something like this:

```python
from xaddpy import XADD
import sympy as sp

context = XADD()

x_id = context.convert_to_xadd(sp.Symbol('x'))
y_id = context.convert_to_xadd(sp.Symbol('y'))

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