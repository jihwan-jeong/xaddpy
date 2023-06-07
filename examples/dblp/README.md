# A Mixed-Integer Linear Programming Reduction of Disjoint Bilinear Programs via Symbolic Variable Elimination (CPAIOR-23)

Welcome to our research codebase! This repository hosts our work presented at CPAIOR-23, where we ventured into an innovative use of Symbolic Variable Elimination (SVE) in conjunction with Extended Algebraic Decision Diagrams (XADDs). The novelty lies in transforming Disjointly Constrained Bilinear Programming (DBLP) problems into Mixed Integer Linear Programming (MILP) problems.

What sets our work apart is its pioneering nature - we're the first to provide a constructive conversion of DBLPs into MILPs. At the heart of this conversion is the utilization of symbolic tools, notably XADDs. Our approach hints at a wider application of this concept, suggesting the potential for transforming other optimization problems in a similar fashion, thereby presenting a compelling toolset for mathematical optimization.

One particularly exciting facet of our research is its implications for DBLPs that feature intricate logical constraints. XADDs demonstrate their potential in these scenarios by representing these complex constraints in a compact manner, providing efficiency improvements in certain cases. Yet, it's crucial to acknowledge the broader context in which our work sits - the vast and intricate field of optimization problem-solving, where formidable solvers like Gurobi have an essential role.

We invite you to delve into our repository, to unravel the intricacies, and to join us on this intriguing journey into the realm of mathematical programming using symbolic tools. Our contributions mark a step towards better understanding and manipulating these intricate problems, setting the stage for future explorations and breakthroughs in this domain.

## Running experiments

This repository allows you to recreate the experiments from our CPAIOR-23 paper. We've grouped these into three categories based on the problem types. Here's how you can run each of them:

### DBLP involving XOR combinations of constraints
Run the experiments using the command:

```
python -m examples.dblp.xor.run_experiments --ny 15 --max_n 20 --min_n 1 --use_q --leaf_minmax_no_prune --seed 0 --epsilon 0
```

The generated results will be stored in the path `results/dblp/xor/sve/results.txt`.

### Test instances based on Vicente et al. (1992)
For these instances, use the following command:

```
python -m examples.dblp.test_problems.run_experiments --opt_var 1 --output_file results.txt --seed 0 --ny 15 --leaf_minmax_no_prune --epsilon 0.00001 --cfg 0 3 0 3 3
```

The flag `--opt_var 1` denotes that `y` variables will be symbolically eliminated, while the `x` variables will be optimized by Gurobi after the MILP reduction. Further details about these test instances can be found in the original Vicente et al. (1992) paper. The results from these experiments will be saved at `results/dblp/vincente/sve/results.txt`.

### Randomized test instances with different sizes and sparsity

To generate and solve randomized test instances of varying sizes and sparsity, use this command:

```
python -m examples.dblp.test_problems.run_experiments --opt_var 1 --solver_type 0 --output_file results.txt --seed 0 --density 0.3 --leaf_minmax_no_prune --epsilon 0.00001 --domain random
```

Altering the `--density` parameter allows you to adjust the density of the coefficients in the objective function and constraints. The results from these tests will be stored at `results/dblp/random/sve/results.txt`.

## References

1. Vicente, L.N., Calamai, P.H., Júdice, J.J., J, J.J.: Generation of disjointly constrained bilinear programming test problems. Computational Optimization and Applications 1, 299–306 (1992)
2. Jeong, J., Sanner, S., Kumar, A.: A Mixed-Integer Linear Programming Reduction of Disjoint Bilinear Programs via Symbolic Variable Elimination. CPAIOR-2023. Nice, France.

