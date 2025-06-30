# Numerical Optimization HW02: Interior Point Method

807194
311121677

This project implements an interior point (log-barrier) method for solving small constrained optimization problems in Python, as described in the assignment for Numerical Optimization 2025B.

## Project Structure

```
.
├── src/
│   └── constrained_min.py      # Interior point solver implementation
├── tests/
│   ├── examples.py             # QP/LP examples, plotting, and output
│   └── test_contrained_min.py  # Unit tests for the solver
├── output/                     # Generated plots and results
├── PDF/
│   └── EX02_programming.pdf    # Assignment description
└── README.md                   # This file
```

## How to Run

### 1. Install Requirements
This project uses only standard scientific Python libraries (numpy, scipy, matplotlib). Install them with:

```sh
pip install numpy scipy matplotlib
```

### 2. Run Examples and Generate Plots
To run the QP and LP examples, generate the required plots, and print the results:

```sh
python tests/examples.py
```
- Plots will be saved in the `output/` directory.
- The script prints the final solution, objective, and constraint values for both examples.

### 3. Run Unit Tests
To verify correctness:

```sh
python -m unittest tests/test_contrained_min.py
```

## Assignment Description (Summary)
- Implements an interior point (log-barrier) method for constrained optimization.
- Demonstrates the method on:
  - A quadratic programming (QP) example (closest probability vector to (0,0,-1)).
  - A linear programming (LP) example (maximizing x+y over a polygonal region).
- Plots the feasible region and central path, and objective value vs. iteration.

## Notes
- The initial point for each example must be strictly feasible.
- The code is robust to optimizer failures and ensures all iterates remain strictly feasible.
- The project is structured for clarity and reproducibility.

## Author
- Balthasar Hoettges 
- Itamar Topol

---

For more details, see the assignment PDF in the `PDF/` directory. 