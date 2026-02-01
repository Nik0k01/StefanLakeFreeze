"""
Test case for 2D Convection-Diffusion of a Gaussian Pulse
This script provides an analytical solution for the 2D unsteady convection-diffusion equation
with an initial Gaussian pulse and compares it with the numerical solution obtained via FVM.
"""
from Scripts.fvm_solver import *
import numpy as np
import matplotlib.pyplot as plt

def gaussian_analytical_2d(X, Y, t, u, v, D, x0, y0, sigma0):
    """
    X, Y:   Meshgrid coordinates
    t:      Current time
    u, v:   Velocity components
    D:      Diffusion coefficient (k/rho*cp)
    x0, y0: Initial center position
    sigma0: Initial pulse width (spread)
    """
    # 1. Calculate new spread (variance increases with time)
    # Variance = sigma^2 + 2*D*t
    sigma2_t = sigma0**2 + 2 * D * t
    
    # 2. Calculate new center position
    xc = x0 + u * t
    yc = y0 + v * t
    
    # 3. Calculate Peak Height Decay (Conservation of Energy in 2D)
    # Peak decays as (sigma0^2 / sigma_t^2)
    peak_decay = sigma0**2 / sigma2_t
    
    # 4. Compute Gaussian
    exponent = -((X - xc)**2 + (Y - yc)**2) / (2 * sigma2_t)
    return peak_decay * np.exp(exponent)

# Simple rectangular mesh
dimX, dimY = 40, 40
L = 3.0
mesh = np.meshgrid(np.linspace(0, L, dimX), np.linspace(0, L, dimY))
k = 0.1
u = 1.0
v = 1.0
# Initial Temperature field: Gaussian pulse at (0.2, 0.2)
T_initial = gaussian_analytical_2d(mesh[0], mesh[1], t=0, u=u, v=v, D=k/(1.0*1.0), x0=1, y0=1, sigma0=0.05)

# Get Analytical solution at t_end
T_exact = gaussian_analytical_2d(mesh[0], mesh[1], t=0.5, u=u, v=v, D=k/(1.0*1.0), x0=1, y0=1, sigma0=0.05)

rho = np.ones((dimY, dimX)) * 1.0                                           # Density field
cp = np.ones((dimY, dimX)) * 1.0                                            # Specific heat capacity field
k = np.ones((dimY, dimX)) * k                                               # Thermal conductivity field
u = np.array([[[-u, -v] for x in range(dimX)] for y in range(dimY)])        # Velocity field

boundary =   ['D', 'D', 'D', 'D'] # [N,S,W,E] : D : Dirichlet, N : Neumann, R : Robin
TD =  [0, 0, 0, 0] # [N,S,W,E]
alpha = 20
Tinf = 90
q = 0
fvm_solver = FVMSolver(mesh[0], mesh[1], boundary, TD, q, alpha, Tinf, k, u, rho, cp)
# Solve the convection-diffusion equation
T_numerical = fvm_solver.unsteady_solve(T_initial=T_initial, t_end=0.5, dt=0.01, boundries=boundary)
# Get the last time step
T_numerical = T_numerical[-1, :, :]

# Plot Comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
cf = plt.contourf(T_numerical, levels=20)
cbar = plt.colorbar(cf)
cbar.set_label('Temperature (T)')
plt.title("Numerical (FVM)")

plt.subplot(1, 3, 2)
cf = plt.contourf(T_exact, levels=20)
cbar = plt.colorbar(cf)
cbar.set_label('Temperature (T)')
plt.title("Analytical (Exact)")

plt.subplot(1, 3, 3)
cf = plt.contourf(T_numerical - T_exact, levels=20, cmap='bwr')
cbar = plt.colorbar(cf)
cbar.set_label('Error')
plt.title("Error (Difference)")
plt.show()