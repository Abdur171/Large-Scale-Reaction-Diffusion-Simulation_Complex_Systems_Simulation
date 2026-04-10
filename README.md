# Dynamics of Large-Scale Nonlinear Reaction-Diffusion Systems

**Author:** Abdur Arshad  
**Institution:** Chalmers University of Technology  
**Course:** Simulation of Complex Adaptive Systems  

---

## Project Overview

This repository contains a high-performance computational pipeline for simulating and analyzing the **Gray-Scott model**, a classic nonlinear reaction-diffusion system. The project investigates complex dynamical regimes, including:

- Self-replicating patterns  
- Traveling waves  

scaled up to a massive $1024 \times 1024$ spatial grid.

To overcome the severe memory bottlenecks associated with evaluating millions of coupled non-linear equations, the numerical solver bypasses standard interpreted array operations and utilizes **LLVM-based Just-In-Time (JIT) compilation** for C-like memory contiguity and execution speed.

---

## Mathematical Formulation

The system dynamics are governed by two coupled partial differential equations (PDEs) representing the concentrations of two chemical species ($u$ and $v$):

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + F(1-u)
$$

$$
\frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (F+k)v
$$

Where:

- $D_u$, $D_v$ are the diffusion coefficients  
- $F$ is the feed rate  
- $k$ is the kill rate  
- $\nabla^2$ is the spatial Laplacian operator  

The Laplacian is approximated using a highly optimized **5-point finite difference stencil**.

---

## Simulation Results & Visual Analysis

The included visualization demonstrates the temporal evolution of the system under **Neumann (zero-flux) boundary conditions**:

- **Iteration 0**  
  The simulation initializes with a homogeneous steady state, disrupted by a localized, high-concentration square perturbation of Species $V$ at the center of the grid.

- **Iteration 1000 & 2000**  
  Driven by nonlinear reaction kinetics, the initial perturbation destabilizes and undergoes structural breakdown. It splits into four distinct, self-replicating spots — a phenomenon often referred to as *mitosis* in pattern-forming systems.

- **Iteration 4000**  
  The spots propagate diagonally outward as symmetrical traveling waves. The radial symmetry of these wave fronts confirms:
  - Correct spatial discretization  
  - Proper implementation of Neumann boundary conditions  
  - Absence of numerical drift or edge-induced artifacts  

---

## Repository Structure

- **`reaction_diffusion_solver.py`**  
  The core Python engine. It utilizes **Numba JIT compilation** and explicit Forward Euler time-stepping to solve the PDEs efficiently without memory thrashing.

- **`stability_analysis.m`**  
  A MATLAB script used to:
  - Symbolically derive the analytical Jacobian matrix  
  - Compute eigenvalues  
  - Prove local asymptotic stability of steady states prior to simulation  

- **`visualize_results.py`**  
  A post-processing script that ingests raw `.npy` outputs and generates high-resolution progression grids.

---
