### Python Implementation of XADD

This repository implements the Python version of XADD (eXtended Algebraic Decision Diagrams) which was first introduced in [Sanner et al. (2011)](https://arxiv.org/pdf/1202.3762.pdf); you can find the original Java implementation from [here](https://github.com/ssanner/xadd-inference). 

Our Python XADD code uses [Sympy](https://github.com/sympy/sympy) for symbolically maintaining all variables and related operations, and [PULP](https://github.com/coin-or/pulp) is used for pruning unreachable paths.  Note that we only check linear conditionals.  If you have Gurobi installed and configured in the conda environment, then PULP will use Gurobi for solving (MI)LPs; otherwise, the default solver ([CBC](https://github.com/coin-or/Cbc)) is going to be used.

Note that the implementation for [EMSPO](https://proceedings.mlr.press/v162/jeong22a/jeong22a.pdf) --- Exact symbolic reduction of linear Smart Predict+Optimize to MILP (Jeong et al., ICML-22) --- has been moved to the branch [emspo](https://github.com/jihwan-jeong/xaddpy/tree/emspo). 

### Installation

#### Load your Python virtual environment then type the following commands for package installation

```shell
pip install gurobipy        # Should your application use linear conditional expressions
pip install tqdm sympy==1.6.2 numpy matplotlib scipy psutil

# Move to the current directory and install xaddpy as a Python package
pip install -e .
```

Note: our implementation relies on the specific version (1.6.2) of SymPy. Make sure you install the right version. Also, you might need Gurobi license to execute the code.