import unittest
import numpy as np
from src.constrained_min import interior_pt
from tests.examples import (
    qp_objective, qp_ineq_constraints, qp_eq_constraints, qp_initial_point,
    lp_objective, lp_ineq_constraints, lp_eq_constraints, lp_initial_point
)

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        x0 = qp_initial_point()
        ineq = qp_ineq_constraints()
        A, b = qp_eq_constraints()
        x_star, history = interior_pt(qp_objective, ineq, A, b, x0)
        # Check that the solution is feasible and objective is finite
        self.assertTrue(np.all([g(x_star) >= -1e-6 for g in ineq]))
        self.assertAlmostEqual(np.sum(x_star), 1.0, places=6)
        self.assertTrue(np.isfinite(qp_objective(x_star)))

    def test_lp(self):
        x0 = lp_initial_point()
        ineq = lp_ineq_constraints()
        A, b = lp_eq_constraints()
        x_star, history = interior_pt(lp_objective, ineq, A, b, x0)
        # Check that the solution is feasible and objective is finite
        self.assertTrue(np.all([g(x_star) >= -1e-6 for g in ineq]))
        self.assertTrue(np.isfinite(lp_objective(x_star)))

if __name__ == '__main__':
    unittest.main()
