import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.constrained_min import interior_pt

# Quadratic Programming (QP) Example
def qp_objective(x):
    x, y, z = x
    return x**2 + y**2 + (z + 1)**2

def qp_ineq_constraints():
    return [
        lambda x: x[0],  # x >= 0
        lambda x: x[1],  # y >= 0
        lambda x: x[2],  # z >= 0
    ]

def qp_eq_constraints():
    A = np.array([[1.0, 1.0, 1.0]])  # x + y + z = 1
    b = np.array([1.0])
    return A, b

def qp_initial_point():
    return np.array([0.1, 0.2, 0.7])

# Linear Programming (LP) Example
def lp_objective(x):
    x1, y1 = x
    return -(x1 + y1)  # maximize x + y <=> minimize -(x + y)

def lp_ineq_constraints():
    return [
        lambda x: x[1] + x[0] - 1,   # y >= -x + 1  => y + x - 1 >= 0
        lambda x: 1 - x[1],          # y <= 1       => 1 - y >= 0
        lambda x: 2 - x[0],          # x <= 2       => 2 - x >= 0
        lambda x: x[1],              # y >= 0
    ]

def lp_eq_constraints():
    return None, None  # No equality constraints

def lp_initial_point():
    return np.array([0.5, 0.75])

def run_qp_example():
    x0 = qp_initial_point()
    ineq = qp_ineq_constraints()
    print('QP initial point:', x0)
    print('QP initial constraint values:', [g(x0) for g in ineq])
    A, b = qp_eq_constraints()
    x_star, history = interior_pt(qp_objective, ineq, A, b, x0)
    path = np.array(history['x'])
    obj_vals = np.array(history['obj'])

    # Plot 1: Feasible region (triangle) and central path in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])        # equal scaling on all axes (Matplotlib ≥3.3)
    ax.view_init(elev=22, azim=-35)   # nicer viewing angle (optional)
    # Vertices of the triangle (x+y+z=1, x,y,z>=0)
    verts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # Draw the triangle as a 3D polygon
    tri3d = Poly3DCollection([verts], alpha=0.2, facecolor='cyan')
    ax.add_collection3d(tri3d)
    # Central path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', markersize=6, linewidth=1.5, color='red', label='Central Path')
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='green', s=60, label='Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('QP: Feasible Region and Central Path')
    ax.legend()
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/qp_central_path.png')
    plt.close()

    # Plot 2: Objective value vs. outer iteration
    plt.figure()
    plt.plot(np.arange(len(obj_vals)), obj_vals, marker='o')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Objective Value')
    plt.title('QP: Objective Value vs. Outer Iteration')
    plt.tight_layout()
    plt.savefig('output/qp_objective_vs_iter.png')
    plt.close()

    # Print final results
    print('QP Example:')
    print('Final x:', x_star)
    print('Final objective:', qp_objective(x_star))
    print('Constraint values:', [g(x_star) for g in ineq])
    print('Equality constraint (sum):', np.sum(x_star))
    print()

def run_lp_example():
    x0 = lp_initial_point()
    ineq = lp_ineq_constraints()
    print('LP initial point:', x0)
    print('LP initial constraint values:', [g(x0) for g in ineq])
    A, b = lp_eq_constraints()
    x_star, history = interior_pt(lp_objective, ineq, A, b, x0)
    path = np.array(history['x'])  # ensure array for plotting
    obj_vals = -np.array(history['obj'])  # only negate once for max

    # Plot 1: Feasible region (polygon) and central path in 2D
    fig, ax = plt.subplots()
    # Replace region drawing block with polygon
    from matplotlib.patches import Polygon
    verts = np.array([[0,1], [2,1], [2,0], [1,0]])
    poly = Polygon(verts, closed=True, facecolor='cyan', edgecolor='k', alpha=0.2, zorder=0)
    ax.add_patch(poly)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    # Central path
    ax.plot(path[:, 0], path[:, 1], marker='o', markersize=6, linewidth=1.5, color='red', label='Central Path')
    ax.scatter(path[-1, 0], path[-1, 1], color='green', s=60, label='Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('LP: Feasible Region and Central Path')
    ax.legend()
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/lp_central_path.png')
    plt.close()

    # Plot 2: Objective value vs. outer iteration
    plt.figure()
    plt.plot(np.arange(len(obj_vals)), obj_vals, marker='o')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Objective Value (max)')
    plt.title('LP: Objective Value vs. Outer Iteration')
    plt.tight_layout()
    plt.savefig('output/lp_objective_vs_iter.png')
    plt.close()

    # Print final results
    print('LP Example:')
    print('Final x:', x_star)
    print('Final objective (max):', obj_vals[-1])
    print('Constraint values:', [g(x_star) for g in ineq])
    print()

if __name__ == '__main__':
    run_qp_example()
    run_lp_example()
