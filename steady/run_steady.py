import numpy as np
import matplotlib.pyplot as plt
from solver import solve_heat_equation
from visualization import plot_temperature

# Simulation parameters
params = {
    'Lx': 1.0,
    'Ly': 1.0,
    'Nx': 50,
    'Ny': 50,
    'T_left': 100.0,
    'T_right': 0.0,
    'T_top': 0.0,
    'T_bottom': 0.0,
    'tolerance': 1e-6,
    'max_iter': 10000
}

# Node locations
x = np.linspace(0, params['Lx'], params['Nx'] + 1)
y = np.linspace(0, params['Ly'], params['Ny'] + 1)
X, Y = np.meshgrid(x, y)

# Solve
T, _ = solve_heat_equation(params)

# Analytical solution at (x=0.4, y=0.4)
T_analytical = 16.8568

# Find indices for x=0.4, y=0.4
dx = params['Lx'] / params['Nx']
dy = params['Ly'] / params['Ny']
x_idx = int(0.4 / dx)
y_idx = int(0.4 / dy)

# Get numerical temperature at that point
T_numerical = T[y_idx, x_idx]

# Calculate percentage error
error_percent = abs((T_numerical - T_analytical) / T_analytical) * 100

# Print results
print(f"Analytical Temperature at (0.4, 0.4): {T_analytical:.4f}°C")
print(f"Numerical Temperature at (0.4, 0.4): {T_numerical:.4f}°C")
print(f"Percentage Error: {error_percent:.4f}%")

# Visualize
plot_temperature(T, X, Y, "Temperature Distribution with Error Analysis")