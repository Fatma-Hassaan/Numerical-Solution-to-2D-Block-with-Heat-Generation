# solver.py
import numpy as np

from boundary_conditions import apply_boundary_conditions

def solve_heat_equation(params):
    """
    Solves 2D steady-state heat conduction using Gauss-Seidel iteration.
    
    Parameters:
    params (dict): Dictionary containing simulation parameters
    
    Returns:
    T (np.ndarray): Final temperature field
    residual_history (list): Residual values per iteration
    """
    # Extract parameters
    Lx = params['Lx']
    Ly = params['Ly']
    Nx = params['Nx']
    Ny = params['Ny']
    k = params.get('k', 1.0)  # Default thermal conductivity to 1.0 if not provided
    q_gen = params.get('q_gen', 0.0)  # Default heat generation to 0.0 if not provided
    tolerance = params['tolerance']
    max_iter = params['max_iter']

    dx = Lx / Nx
    dy = Ly / Ny

    # Initialize temperature field
    T = np.ones((Ny + 1, Nx + 1)) * params.get('T_inf', 0.0)  # Default T_inf to 0.0 if not provided

    # Apply fixed boundaries (Dirichlet or Neumann)
    T[:, 0] = params['T_left']  # Left boundary (Dirichlet)
    
    residual_history = []

    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        T_old = T.copy()

        # Update internal nodes
        for j in range(1, Ny):  # y-direction
            for i in range(1, Nx):  # x-direction
                T[j, i] = (
                    (T[j+1, i] + T[j-1, i]) * dy**2 +
                    (T[j, i+1] + T[j, i-1]) * dx**2 +
                    q_gen * dx**2 * dy**2 / k
                ) / (2 * (dx**2 + dy**2))

        # Apply boundary conditions
        T = apply_boundary_conditions(T, params, dx, dy)

        # Compute residual
        residual = np.max(np.abs(T - T_old))
        residual_history.append(residual)

        # Check convergence
        if residual < tolerance:
            print(f"Converged in {iteration} iterations.")
            break

    return T, residual_history