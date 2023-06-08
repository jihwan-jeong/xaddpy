# A Mixed-Integer Linear Programming Reduction of Disjoint Bilinear Programs via Symbolic Variable Elimination (CPAIOR-23)

Welcome to our research codebase! This repository is associated with [our work presented at CPAIOR-23](https://ssanner.github.io/papers/cpaior23_dblpsve.pdf) (see also [the previous extended version](https://github.com/jihwan-jeong/xaddpy/tree/main/examples/dblp/DBLP_MILP_SVE_Long_Version.pdf)), wherein we introduce a novel approach for converting Disjointly Constrained Bilinear Programming (DBLP) problems into Mixed Integer Linear Programming (MILP) problems. This transformative approach utilizes symbolic tools like Extended Algebraic Decision Diagrams (XADDs), marking us as the first to perform such a constructive conversion.

While our focus is on DBLPs, the methodologies we've developed potentially extend to other optimization problems as well. In this work, we have found that the MILP reduction is particularly useful for problems that involve complex logical constraints, where XADDs can compactly represent those constraints and be manipulated to provide much more efficient solution.

We encourage you to explore this repository, understand our methods, and see the potential that symbolic tools bring to mathematical programming. This is a stepping stone towards a more nuanced understanding and handling of complex optimization problems, and we hope it spurs further innovation in this field.

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

