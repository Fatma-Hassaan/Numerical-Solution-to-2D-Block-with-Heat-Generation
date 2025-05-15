import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Lx = 1.0   # Domain length (m)
Ly = 1.0   # Domain height (m)
Nx = 50    # Grid points in x-direction
Ny = 50    # Grid points in y-direction
k = 222    # Thermal conductivity (W/mK)
T_left = 100.0  # Left boundary temperature (°C)
T_right = 0.0   # Right boundary temperature (°C)
T_top = 0.0     # Top boundary temperature (°C)
T_bottom = 0.0  # Bottom boundary temperature (°C)
tolerance = 1e-6
max_iter = 10000

dx = Lx / Nx
dy = Ly / Ny

# Initialize temperature field
T = np.zeros((Ny + 1, Nx + 1))

# Apply fixed boundary temperatures
T[:, 0] = T_left       # Left boundary
T[:, -1] = T_right     # Right boundary
T[0, :] = T_bottom     # Bottom boundary
T[-1, :] = T_top       # Top boundary

# Gauss-Seidel iteration
for _ in range(max_iter):
    T_old = T.copy()
    
    # Update internal nodes
    for j in range(1, Ny):
        for i in range(1, Nx):
            T[j, i] = (
                (T[j+1, i] + T[j-1, i]) * dy**2 +
                (T[j, i+1] + T[j, i-1]) * dx**2
            ) / (2 * (dx**2 + dy**2))
    
    # Compute residual
    residual = np.max(np.abs(T - T_old))
    if residual < tolerance:
        break

# Create grid for plotting
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)
X, Y = np.meshgrid(x, y)

# Generate contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, T.T, levels=50, cmap='jet')
plt.colorbar(contour, label='Temperature (°C)')
plt.title('Contour Plot of Temperature Distribution')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.show()

# Save the figure
plt.savefig('temperature_contour.png', dpi=300)