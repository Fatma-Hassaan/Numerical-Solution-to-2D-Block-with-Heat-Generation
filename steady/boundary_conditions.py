# boundary_conditions.py
import numpy as np

from typing import Dict

def apply_boundary_conditions(T: np.ndarray, params: Dict[str, float], dx: float, dy: float) -> np.ndarray:
    """
    Applies mixed boundary conditions:
    - Left: Heat flux (Dirichlet or Neumann)
    - Right/Top/Bottom: Convection
    """
    k = params.get('k', 1.0)  # Default value for 'k' is set to 1.0
    h = params.get('h', 1.0)  # Default value for 'h' is set to 1.0
    T_inf = params.get('T_inf', 300.0)  # Default value for 'T_inf' is set to 300.0
    q_flux = params.get('q_flux')
    Ny, Nx = T.shape

    # Left boundary (Dirichlet or Neumann)
    if 'T_left' in params:
        T[:, 0] = params['T_left']  # Fixed temperature
    else:
        # Heat flux: -k*dT/dx = q_flux => T[0,j] = T[1,j] - q_flux*dx/k
        T[:, 0] = T[:, 1] - q_flux * dx / k

    # Right boundary (convection)
    T[:, -1] = (T[:, -2] + (h * dx / k) * T_inf) / (1 + h * dx / k)

    # Top boundary (convection)
    T[-1, :] = (T[-2, :] + (h * dy / k) * T_inf) / (1 + h * dy / k)

    # Bottom boundary (convection)
    T[0, :] = (T[1, :] + (h * dy / k) * T_inf) / (1 + h * dy / k)

    return T