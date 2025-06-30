import numpy as np
from scipy.optimize import minimize


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    """
    Interior point method (log-barrier) for constrained optimization.

    Parameters:
        func: callable
            The objective function to minimize. Should take a vector x and return a scalar.
        ineq_constraints: list of callables
            List of inequality constraint functions, each taking x and returning a scalar (should be >= 0).
        eq_constraints_mat: np.ndarray
            Matrix A for affine equality constraints Ax = b.
        eq_constraints_rhs: np.ndarray
            Vector b for affine equality constraints Ax = b.
        x0: np.ndarray
            Initial strictly feasible point.
    Returns:
        x: np.ndarray
            The solution found by the interior point method.
        history: dict
            Dictionary containing the path of iterates and objective values for plotting.
    """
    m = len(ineq_constraints)
    x = np.array(x0, dtype=float)
    t = 1.0
    mu = 10.0
    tol = 1e-8
    max_iter = 50
    history = {'x': [np.copy(x)], 'obj': [func(x)]}  # seed with the initial point

    def barrier_obj(x, t):
        barrier = 0.0
        for g in ineq_constraints:
            val = g(x)
            if val <= 0:
                return np.inf
            barrier -= np.log(val)
        return t * func(x) + barrier

    def approx_grad(f, x, eps=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.array(x, dtype=float)
            x2 = np.array(x, dtype=float)
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (f(x1) - f(x2)) / (2 * eps)
        return grad

    def barrier_grad(x, t):
        # Analytic gradient: t * grad(f) - sum_i grad(g_i)/g_i(x)
        grad = t * approx_grad(func, x)
        for g in ineq_constraints:
            val = g(x)
            if val <= 0:
                return np.full_like(x, np.nan)
            grad -= approx_grad(g, x) / val
        return grad

    # Build scipy constraints
    ineq_cons_scipy = [{'type': 'ineq', 'fun': (lambda x, g=g: g(x))} for g in ineq_constraints]
    eq_cons_scipy = []
    if eq_constraints_mat is not None and eq_constraints_rhs is not None:
        def eq_fun(x):
            return eq_constraints_mat @ x - eq_constraints_rhs
        eq_cons_scipy = [{'type': 'eq', 'fun': eq_fun}]

    for it in range(max_iter):
        res = minimize(
            lambda x: barrier_obj(x, t),
            x,
            method='SLSQP',
            jac=(lambda x: barrier_grad(x, t)),
            constraints=ineq_cons_scipy + eq_cons_scipy,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 100}
        )
        # --- Patch: handle unsuccessful or infeasible steps
        if not res.success:
            x = (x + res.x) / 2.0
        else:
            x = res.x
        # Reject any point that violates a barrier constraint
        if any(g(x) <= 0 for g in ineq_constraints):
            alpha = 0.5
            while any(g((1-alpha)*history['x'][-1] + alpha*x) <= 0 for g in ineq_constraints):
                alpha *= 0.5
            x = (1-alpha)*history['x'][-1] + alpha*x
        history['x'].append(np.copy(x))
        history['obj'].append(func(x))
        # First grow t, then check the stopping criterion
        t *= mu
        if m / t < tol:
            break

    history['x'] = np.array(history['x'])
    history['obj'] = np.array(history['obj'])
    return x, history
