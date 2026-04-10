% Project: Large-Scale Reaction-Diffusion Simulation 2025
% Course: Simulation of Complex Adaptive Systems
% Description: Symbolic computation of the Jacobian matrix and eigenvalue 
%              analysis to determine the local asymptotic stability of steady states.

clear; clc;
syms u v F k Du Dv

% 1. Define the non-linear reaction kinetics
f_u = -u*v^2 + F*(1 - u);
f_v = u*v^2 - (F + k)*v;

% 2. Compute the analytical Jacobian Matrix symbolically
J_sym = jacobian([f_u, f_v], [u, v]);
disp('Analytical Jacobian Matrix:');
disp(J_sym);

% 3. Analyze the trivial steady state (u=1, v=0)
u_ss = 1; v_ss = 0;
J_trivial = subs(J_sym, [u, v], [u_ss, v_ss]);

% 4. Compute Eigenvalues for Stability Properties
% If the real parts of all eigenvalues are strictly negative, the state is stable.
eigenvalues = eig(J_trivial);
disp('Eigenvalues at Trivial Steady State (u=1, v=0):');
disp(eigenvalues);

% Parameter Sensitivity: Check stability for specific experimental values
F_val = 0.035; k_val = 0.065;
eigenvalues_eval = double(subs(eigenvalues, [F, k], [F_val, k_val]));
disp('Evaluated Eigenvalues for Simulation Experiment:');
disp(eigenvalues_eval);