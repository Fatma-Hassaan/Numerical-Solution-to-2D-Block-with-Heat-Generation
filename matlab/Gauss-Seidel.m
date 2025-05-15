% Program: SteadyState_2D_HeatConduction_Animated.m
% Solves steady-state 2D conduction using Gauss-Seidel iteration.
% Animates the solution convergence with color contour plot.

clear, clc, close all

% Geometry and grid
Lx = 1;                 % Plate length in x-direction (m)
Ly = 1;                 % Plate length in y-direction (m)
Nx = 20;                % Number of divisions in x-direction
Ny = Nx;                % Number of divisions in y-direction
dx = Lx / Nx;           % Grid spacing in x
dy = Ly / Ny;           % Grid spacing in y

% Boundary conditions
T1 = 0;                 % Bottom, left, and right side temperature
T2 = 100;               % Top side temperature

% Analytical solution at point (x=0.4, y=0.4)
tol_ans = 16.8568;

% Iteration parameters
max_iter = 1000;        % Maximum number of iterations
tolerance = 1e-4;       % Convergence tolerance
omega = 1.5;            % Relaxation factor (for SOR)

% Node locations
x = 0:dx:Lx;
y = 0:dy:Ly;

% Initialize temperature matrix
T = T1 * ones(Ny+1, Nx+1);

% Apply fixed boundary conditions
T(Ny+1, :) = T2;        % Top boundary
T(:, 1) = T1;           % Left boundary
T(:, Nx+1) = T1;        % Right boundary
T(1, :) = T1;           % Bottom boundary

% Animation setup
figure;
colormap(jet);
Nc = 50;
dT = (T2 - T1) / Nc;
v = T1:dT:T2;

% Store temperature at center point for plotting
T_center = zeros(max_iter, 1);
iter = 0;

% Gauss-Seidel Iteration
while iter < max_iter
    iter = iter + 1;
    T_old = T;
    
    % Update interior points
    for j = 2:Ny
        for i = 2:Nx
            T_new = 0.25 * (T(j+1, i) + T(j-1, i) + T(j, i+1) + T(j, i-1));
            T(j, i) = omega * T_new + (1 - omega) * T(j, i);
        end
    end

    % Compute max change for convergence
    max_change = max(abs(T(:) - T_old(:)));
    
    % Store center point temperature
    T_center(iter) = T(9, 9);  % (i=9, j=9) corresponds to x=0.4, y=0.4

    % Plot current temperature field
    contourf(x, y, T', v, 'LineStyle', 'none');
    colorbar;
    title(['Iteration = ', num2str(iter), ...
           ', Temp at (0.4,0.4) = ', num2str(T_center(iter), '%.4f'), ...
           ' °C, Error = ', num2str(abs((T_center(iter) - tol_ans)/tol_ans)*100, '%.4f'), '%']);
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal tight;
    drawnow;
    pause(0.05);  % Control animation speed

    % Check convergence
    if max_change < tolerance
        break;
    end
end

% Final plot
contourf(x, y, T', v, 'LineStyle', 'none');
colorbar;
title(['Steady-State Temperature Distribution (°C), Final Iteration = ', num2str(iter)]);
xlabel('x (m)');
ylabel('y (m)');
axis equal tight;

% Print results
fprintf('Final iteration: %d\n', iter);
fprintf('Temperature at (x=0.4, y=0.4): %.4f°C\n', T_center(iter));
fprintf('Percent error vs analytical: %.4f%%\n', abs((T_center(iter) - tol_ans)/tol_ans)*100);

% Plot temperature evolution at center point
figure;
plot(1:iter, T_center(1:iter), 'b-', 'LineWidth', 1.5);
grid on;
title('Temperature Evolution at (0.4, 0.4) m');
xlabel('Iteration');
ylabel('Temperature (°C)');