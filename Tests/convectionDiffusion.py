"""
This script provides an analytical solution for the 1D steady-state convection-diffusion equation
and plots the temperature distribution along the length of the domain.
"""

from Scripts.fvm_solver import *
import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(x, L, rho, c, u, k):
    Pe = (rho * c * u * L) / k
    # Avoid overflow for large Pe
    if Pe > 50: 
        # For high Pe, T is 0 until the very end (boundary layer)
        return 0.0 if x < L else 1.0 
    return (np.exp(Pe * x / L) - 1) / (np.exp(Pe) - 1)

# Usage
L = 1.0
x_coords = np.linspace(0, L, 100)
k = 0.1
u = 1.0
T_exact = [analytical_solution(x, L, 1.0, 1.0, u, k) for x in x_coords]

# Simple rectangular mesh
dimX, dimY = 200, 3
mesh = np.meshgrid(np.linspace(0, L, dimX), np.linspace(0, 0.1, dimY))
rho = np.ones((dimY, dimX)) * 1.0                                           # Density field
cp = np.ones((dimY, dimX)) * 1.0                                            # Specific heat capacity field
k = np.ones((dimY, dimX)) * k                                             # Thermal conductivity field
u = np.array([[[-u, 0] for x in range(dimX)] for y in range(dimY)])                   # Velocity field

boundary =   ['N', 'N', 'D', 'D'] # [N,S,W,E] : D : Dirichlet, N : Neumann, R : Robin
TD =  [0, 0, 0, 1] # [N,S,W,E]
alpha = 20
Tinf = 90
q = 0
fvm_solver = FVMSolver(mesh[0], mesh[1], boundary, TD, q, alpha, Tinf, k, u, rho, cp)
# Solve the convection-diffusion equation
T_numerical = fvm_solver.solve()
# Get the middle row for comparison
T_numerical_mid = T_numerical[1, :]

plt.figure(figsize=(8, 5))
plt.plot(x_coords, T_exact, label='Analytical Solution', color='black', linewidth=2)
plt.plot(mesh[0][1], T_numerical_mid, 'r--', label='Numerical Solution (FVM)', linewidth=2)
plt.xlabel('x')
plt.ylabel('T')
plt.title('Analytical Solution of 1D Convection-Diffusion Equation')
plt.legend()
plt.grid()
plt.show()