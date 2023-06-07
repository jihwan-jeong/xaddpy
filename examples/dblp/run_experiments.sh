#!/bin/bash

# Run the experiments for test instances
## Note: The results are stored in the file results/dblp/vincente/results.txt
## If you want to understand how the code works, it's best to run this with a debugger and go line by line with it 
## Also, the values given for cfg determine the problem definition.
## See the `create_prob_json` function in examples/dblp/util.py for details.
python -m examples.dblp.test_problems.run_experiments --opt_var 1 --solver_type 0 --output_file results.txt --seed 0 --ny 15 --leaf_minmax_no_prune --epsilon 0.00001 --cfg 0 3 0 3 3

# Run the experiments with varying densities
## Note: the results are stored in the file results/dblp/random/results.txt
python -m examples.dblp.test_problems.run_experiments --opt_var 1 --solver_type 0 --output_file results.txt --seed 0 --density 0.3 --leaf_minmax_no_prune --epsilon 0.00001 --domain random

# Run the XOR experiments
## Note: the results are stored in the file results/dblp/xor/results.txt
python -m examples.dblp.xor.run_experiments --ny 15 --max_n 20 --min_n 1 --use_q --leaf_minmax_no_prune --seed 0 --epsilon 0
