import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.interpolate import griddata
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, HTML

# =====================================
# PARAMETERS (ADJUST THESE)
# =====================================
Lx, Ly = 2.0, 2.0           # Domain dimensions
nx, ny = 50, 50             # Grid resolution
k = 6.0                     # Thermal conductivity
q_gen = 500.0               # Heat generation rate
h = 100.0                     # Convective coefficient
T_inf = 500.0                # Ambient temperature
q_flux = 500.0              # Heat flux at x = Lx
tolerance = 1e-6            # Convergence tolerance
max_iter = 2000             # Maximum iterations

# Transient analysis parameters
rho = 1000                  # Density (kg/m³)
cp = 1000                   # Specific heat (J/kg·K)
dt = 0.01                 # Time step (s)
time_steps = 800            # Number of time steps

# =====================================
# MESH SETUP
# =====================================
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# =====================================
# SOLVER WITH GAUSS-SEIDEL METHOD
# =====================================
def solve_heat_equation():
    T = np.ones((nx, ny)) * T_inf  # Initial guess
    T_history = []
    residuals = []
    
    for iteration in tqdm(range(max_iter), desc="Solving..."):
        T_old = T.copy()
        
        # Update interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                T_xx = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2
                T_yy = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                T[i, j] = (T_xx + T_yy - q_gen/k) * (dx**2 * dy**2) / (2*(dx**2 + dy**2))
        
        # Boundary conditions
        T[0, :] = (T[1, :] + h*dx/k * T_inf) / (1 + h*dx/k)    # Left (convective)
        T[-1, :] = T[-2, :] + q_flux * dx / k                  # Right (heat flux)
        T[:, 0] = (T[:, 1] + h*dy/k * T_inf) / (1 + h*dy/k)    # Bottom (convective)
        T[:, -1] = (T[:, -2] + h*dy/k * T_inf) / (1 + h*dy/k)  # Top (convective)
        
        # Track convergence
        residual = np.max(np.abs(T - T_old))
        residuals.append(residual)
        if iteration % 10 == 0:
            T_history.append(T.copy())
        
        if residual < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    
    return T, T_history, residuals

T_final, T_history, residuals = solve_heat_equation()

# =====================================
# ADVANCED VISUALIZATIONS & ANALYSIS
# =====================================
# 1. INTERACTIVE 3D PLOT WITH PLOTLY
def plot_3d_interactive():
    fig = go.Figure(data=[go.Surface(z=T_final.T, x=X, y=Y)])
    fig.update_layout(
        title='3D Temperature Distribution',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Temperature (°C)'
        )
    )
    fig.show()

# 2. HEAT FLUX VECTOR FIELD (QUIVER PLOT)
def plot_heat_flux():
    dT_dx, dT_dy = np.gradient(T_final.T, dx, dy)
    q_x = -k * dT_dx
    q_y = -k * dT_dy

    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, q_x, q_y, color='red', scale=50)
    plt.contourf(X, Y, T_final.T, levels=50, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Heat Flux Vectors')
    plt.xlabel('X-axis →')
    plt.ylabel('Y-axis →')
    plt.show()

# 3. TRANSIENT SIMULATION (TIME-DEPENDENT)
def run_transient_simulation():
    T_transient = np.ones((nx, ny)) * T_inf
    alpha = k / (rho * cp)
    transient_history = []

    for t in tqdm(range(time_steps), desc="Transient Simulation"):
        T_old = T_transient.copy()
        
        # Time-stepping using Forward Euler
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                T_xx = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dx**2
                T_yy = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dy**2
                T_transient[i, j] = T_old[i, j] + alpha * dt * (T_xx + T_yy + q_gen/k)
        
        # Boundary conditions
        T_transient[0, :] = (T_transient[1, :] + h*dx/k * T_inf) / (1 + h*dx/k)
        T_transient[-1, :] = T_transient[-2, :] + q_flux * dx / k
        T_transient[:, 0] = (T_transient[:, 1] + h*dy/k * T_inf) / (1 + h*dy/k)
        T_transient[:, -1] = (T_transient[:, -2] + h*dy/k * T_inf) / (1 + h*dy/k)
        
        transient_history.append(T_transient.copy())
    
    return transient_history

# 4. ERROR ESTIMATION (COMPARE WITH ANALYTICAL SOLUTION)
def compute_error():
    # Example analytical solution for validation (no heat generation)
    def analytical_solution(x):
        return T_inf + (q_flux / k) * (Lx - x)
    
    T_analytical = analytical_solution(X)
    error = np.abs(T_final - T_analytical.T)
    return error

# 5. PARAMETER SWEEP (HEAT GENERATION RATE)
def parameter_sweep():
    q_gen_values = [100, 500, 1000]
    plt.figure(figsize=(10, 6))
    
    for q in q_gen_values:
        T_sweep = solve_heat_equation(q_gen=q)[0]
        plt.plot(x, T_sweep[ny//2, :], label=f'q_gen={q} W/m³')
    
    plt.xlabel('X-axis Position')
    plt.ylabel('Temperature (°C)')
    plt.title('Effect of Heat Generation Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. EXPORT TO VTK FOR PARAVIEW
def export_to_vtk():
    from pyevtk.hl import gridToVTK
    gridToVTK(
        "./temperature_field",
        X[:, 0], Y[0, :], np.zeros(nx),
        cellData={'temperature': T_final.T}
    )

# 7. CONVERGENCE STUDY (GRID INDEPENDENCE)
def convergence_study():
    grid_sizes = [10, 20, 50, 100]
    error_norms = []
    
    for size in grid_sizes:
        nx_study, ny_study = size, size
        dx_study = Lx / (nx_study - 1)
        dy_study = Ly / (ny_study - 1)
        T_study = solve_heat_equation(nx=nx_study, ny=ny_study)[0]
        error = compute_error()
        error_norms.append(np.linalg.norm(error))
    
    plt.figure(figsize=(8, 5))
    plt.loglog(grid_sizes, error_norms, 'o-', lw=2)
    plt.xlabel('Grid Size')
    plt.ylabel('L2 Error Norm')
    plt.title('Grid Convergence Study')
    plt.grid(True)
    plt.show()

# 8. INTERACTIVE DASHBOARD (IPYWIDGETS)
def interactive_dashboard():
    from ipywidgets import interact, FloatSlider, Button
    
    def update_plot(q_gen=500, h=5):
        T = solve_heat_equation(q_gen=q_gen, h=h)[0]
        plt.contourf(X, Y, T.T, levels=50, cmap='hot')
        plt.colorbar()
        plt.show()
    
    interact(update_plot, q_gen=(100, 1000, 50), h=(1, 20, 1))

# =====================================
# RUN ANALYSIS
# =====================================
if __name__ == "__main__":
    # Choose which features to run
    plot_3d_interactive()
    plot_heat_flux()
    transient_history = run_transient_simulation()
    error = compute_error()
    parameter_sweep()
    export_to_vtk()
    convergence_study()
    interactive_dashboard()     
