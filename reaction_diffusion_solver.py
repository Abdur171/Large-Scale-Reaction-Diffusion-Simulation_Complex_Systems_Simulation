"""
Project: Large-Scale Reaction-Diffusion Simulation 2025
Course: Simulation of Complex Adaptive Systems
Author: Abdur Arshad
Date: September 2025

Description: 
High-performance numerical solver for non-linear reaction-diffusion systems.
Utilizes Numba JIT compilation to achieve C-like performance and memory 
safety by bypassing Python's dynamic array allocation overhead during 
explicit time-stepping.
"""

import numpy as np
import os
import json
import time
from numba import njit

@njit
def compute_explicit_step_neumann(u, v, Du, Dv, F, k, dt, dx):
    """
    JIT-compiled explicit time-stepping loop.
    Calculates the 5-point finite difference Laplacian and reaction kinetics 
    point-by-point to prevent memory thrashing.
    """
    size = u.shape[0]
    u_next = np.empty_like(u)
    v_next = np.empty_like(v)
    dx2 = dx * dx
    
    for i in range(size):
        for j in range(size):
            # Neumann boundary conditions (zero-flux)
            top_u = u[0, j] if i == 0 else u[i - 1, j]
            bot_u = u[size - 1, j] if i == size - 1 else u[i + 1, j]
            lft_u = u[i, 0] if j == 0 else u[i, j - 1]
            rgt_u = u[i, size - 1] if j == size - 1 else u[i, j + 1]
            
            top_v = v[0, j] if i == 0 else v[i - 1, j]
            bot_v = v[size - 1, j] if i == size - 1 else v[i + 1, j]
            lft_v = v[i, 0] if j == 0 else v[i, j - 1]
            rgt_v = v[i, size - 1] if j == size - 1 else v[i, j + 1]
            
            # Spatial Laplacian
            Lu = (top_u + bot_u + lft_u + rgt_u - 4.0 * u[i, j]) / dx2
            Lv = (top_v + bot_v + lft_v + rgt_v - 4.0 * v[i, j]) / dx2
            
            # Non-linear reaction kinetics
            uvv = u[i, j] * v[i, j] * v[i, j]
            
            # Explicit Forward Euler update
            u_next[i, j] = u[i, j] + dt * (Du * Lu - uvv + F * (1.0 - u[i, j]))
            v_next[i, j] = v[i, j] + dt * (Dv * Lv + uvv - (F + k) * v[i, j])
            
    return u_next, v_next

class ComplexReactionDiffusion:
    def __init__(self, size=1024, Du=0.16, Dv=0.08, F=0.035, k=0.065, dt=1.0, dx=1.0):
        self.size = size
        self.Du, self.Dv = Du, Dv
        self.F, self.k = F, k
        self.dt, self.dx = dt, dx
        
        # Initialize massive state matrices
        self.u = np.ones((size, size))
        self.v = np.zeros((size, size))
        self._apply_initial_perturbation()

    def _apply_initial_perturbation(self):
        """Introduces a localized high-concentration perturbation."""
        r = self.size // 20
        center = self.size // 2
        self.u[center-r:center+r, center-r:center+r] = 0.50
        self.v[center-r:center+r, center-r:center+r] = 0.25
        
        np.random.seed(42)
        self.u += np.random.normal(scale=0.01, size=(self.size, self.size))
        self.v += np.random.normal(scale=0.01, size=(self.size, self.size))

    def run_sensitivity_analysis(self, steps=20000, save_interval=1000):
        """Structures outputs into raw binary arrays and JSON configs."""
        base_dir = f"SimOutputs_Sept2025_neumann_F{self.F}_k{self.k}"
        os.makedirs(base_dir, exist_ok=True)
        
        config = {'size': self.size, 'Du': self.Du, 'Dv': self.Dv, 'F': self.F, 'k': self.k, 'bc': 'neumann'}
        with open(f"{base_dir}/experiment_config.json", 'w') as f:
            json.dump(config, f)
            
        print(f"Initiating large-scale simulation ({self.size}x{self.size} grid)...")
        start_time = time.time()
        
        for step in range(steps):
            # The JIT compiled C-loop runs instantaneously
            self.u, self.v = compute_explicit_step_neumann(
                self.u, self.v, self.Du, self.Dv, self.F, self.k, self.dt, self.dx
            )
            
            if step % save_interval == 0:
                np.save(f"{base_dir}/state_v_{step:05d}.npy", self.v)
                elapsed = time.time() - start_time
                print(f"Step {step}/{steps} completed in {elapsed:.2f} seconds.")
                start_time = time.time() # Reset timer for next batch

if __name__ == "__main__":
    # Test case 1: Neumann boundaries simulating a closed physical container
    solver = ComplexReactionDiffusion(size=1024, F=0.035, k=0.065)
    # Compiling Numba takes a few seconds on the very first run
    solver.run_sensitivity_analysis(steps=5000, save_interval=1000)