# Exact MILP Reduction of Linear Smart "Predict, then Optimize"

This repository contains the official implementation of EMSPO (exact symbolic MILP reduction of the linear SPO) which is presented in ICML-22 in Baltimore, MD.

### Python Implementation of XADD
The symbolic variable elimination (SVE) requires the use of an efficient data structure to maintain a compact representation of symbolic case functions and their operations. 
We use XADD (eXtended Algebraic Decision Diagrams) which was first introduced in [Sanner et al. (2011)](https://arxiv.org/pdf/1202.3762.pdf); you can find the original Java implementation from [here](https://github.com/ssanner/xadd-inference). 

In our work, we ported the Java XADD code to Python using [Sympy](https://github.com/sympy/sympy) and Gurobi. 

### Installation

#### Load your Python virtual environment then type the following commands for package installation

```shell
pip install gurobipy
pip install tqdm cvxpy sympy==1.6.2 numpy matplotlib pandas scipy psutil scikit-learn qpth block torch

# Move to the current directory and install xaddpy as a Python package
pip install -e .
```

Note: you need Gurobi license to execute the code. Also, our implementation relies on the specific version (1.6.2) of SymPy. Make sure you install the right version.

### Run scripts
Once your environment is set up, follow the steps below to run the experiments. 
Note that we have not updated the selected hyperparameters of TwoStage, IntOpt, QPTL, and SPO+ models that we used to plot graphs in the paper.

We implement three example domains in this repository: 
1. 3 x 3 shortest path problem
2. Reduced energy cost aware scheduling problem
3. Cost-sensitive multi-class classification

To run experiments, execute a corresponding `run_experiments` file as a module with some arguments specified: 
```shell
# Shortest path
python -m xaddpy.experiments.predopt.shortest_path.run_experiments --algo emspo --size 200 --verbose --timeout -1 --nonlinearity 1 --num_edges 12 --num_runs 5

# Classification
python -m xaddpy.experiments.predopt.classification.run_experiments --algo emspo --size 200 --verbose --nonlinearity 1 --timeout -1 

# Energy-aware scheduling
python -m xaddpy.experiments.predopt.energy_scheduling.run_experiments --algo emspo --presolve --prob_config 3 --timeout -1 --verbose --epsilon 0.001
```

By default, each problem already has its argmin solution saved. If you want to solve the argmin step again, e.g., remove the `.xadd` file in `xaddpy/experiments/predopt/shortest_path/prob_instances`.