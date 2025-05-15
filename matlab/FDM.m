% Program: SteadyState_2D_HeatConduction.m
% Solves steady-state 2D conduction using finite difference method.
% Assumes constant thermal properties and Dirichlet boundary conditions.
%
% Variable List:
% T = Temperature (deg. Celsius)
% T1 = Boundary condition temperature 1 (deg. Celsius)
% T2 = Boundary condition temperature 2 (deg. Celsius)
% Lx = Plate length in x-direction (m)
% Ly = Plate length in y-direction (m)
% Nx = Number of increments in x-direction
% Ny = Number of increments in y-direction
% dx = Increment size in x-direction (m)
% dy = Increment size in y-direction (m)
% x = x-distance node locations
% y = y-distance node locations
% k = Thermal conductivity (W/mK)
% rho = Density (kg/m^3)
% Cp = Specific heat (J/kgK)
% alpha = Thermal diffusivity (m^2/s)
% Tplot = Stores T at x = y = 0.4 m
% node_id = Mapping from 2D grid to 1D vector
% A = Coefficient matrix
% C = Right-hand side vector
% Tsolve = Solution vector
% T = Temperature matrix
% v = Temperature levels for contours
% Nc = Number of contours for plot

clear, clc

% Geometry and grid
Lx = 1;                 % Plate length in x-direction (m)
Ly = 1;                 % Plate length in y-direction (m)
Nx = 20;                % Number of increments in x-direction
Ny = Nx;                % Number of increments in y-direction
dx = Lx / Nx;           % Increment size in x-direction
dy = Ly / Ny;           % Increment size in y-direction

% Boundary conditions
T1 = 0;                 % Bottom, left, and right side temperature
T2 = 100;               % Top side temperature

% Material properties
k = 222;                % Thermal conductivity (W/mK)
rho = 2800;             % Density (kg/m^3)
Cp = 896;               % Specific heat (J/kgK)
alpha = k / (rho * Cp);% Thermal diffusivity (m^2/s)

% Analytical solution at point (x=0.4, y=0.4)
tol_ans = 16.8568;      % Steady-state temperature at x = y = 0.4m

% Node locations
x = 0:dx:Lx;            % x-distance node locations
y = 0:dy:Ly;            % y-distance node locations

% Node numbering
node_id = zeros(Ny+1, Nx+1);
i = 1;
for iy = 1:Ny+1
    for ix = 1:Nx+1
        node_id(iy, ix) = i;
        i = i + 1;
    end
end

% Initialize system matrix A and RHS vector C
A = zeros((Nx+1)*(Ny+1));
C = zeros((Nx+1)*(Ny+1), 1);

% Build matrix A and vector C
for j = 1:Ny+1
    for i = 1:Nx+1
        if (i == 1) || (i == Nx+1) || (j == 1)
            % Bottom, left, or right boundary
            A(node_id(j,i), node_id(j,i)) = 1;
            C(node_id(j,i)) = T1;
        elseif (j == Ny+1)
            % Top boundary
            A(node_id(j,i), node_id(j,i)) = 1;
            C(node_id(j,i)) = T2;
        else
            % Interior node: Laplace equation
            A(node_id(j,i), node_id(j,i)) = -4;
            A(node_id(j,i), node_id(j+1,i)) = 1;  % North
            A(node_id(j,i), node_id(j-1,i)) = 1;  % South
            A(node_id(j,i), node_id(j,i+1)) = 1;  % East
            A(node_id(j,i), node_id(j,i-1)) = 1;  % West
            C(node_id(j,i)) = 0;
        end
    end
end

% Solve the linear system
Tsolve = A \ C;

% Reshape solution into 2D temperature matrix
T = zeros(Ny+1, Nx+1);
for j = 1:Ny+1
    for i = 1:Nx+1
        T(i,j) = Tsolve(node_id(j,i));
    end
end

% Extract temperature at x = y = 0.4 m
point_temp = T(9,9);  % (i=9, j=9) corresponds to x=0.4m, y=0.4m
diff = abs((point_temp - tol_ans) / tol_ans) * 100;
fprintf('Temperature at (x=0.4, y=0.4): %.4f°C\n', point_temp);
fprintf('Percent difference from analytical: %.4f%%\n', diff);

% Plot the temperature distribution
Nc = 50;                      % Number of contours
dT = (T2 - T1) / Nc;          % Temperature step
v = T1:dT:T2;                 % Contour levels
colormap(jet)
contourf(x, y, T', v, 'LineStyle', 'none');
colorbar;
axis equal tight;
title('Steady-State Temperature Distribution (°C)');
xlabel('x (m)');
ylabel('y (m)');
set(gca, 'XTick', 0:0.1:Lx);
set(gca, 'YTick', 0:0.1:Ly);