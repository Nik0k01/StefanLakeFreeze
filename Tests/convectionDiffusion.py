"""
Verify FVM solver vs analytical 1D steady convection-diffusion
across low / moderate / high Peclet numbers.
"""

from Scripts.fvm_solver import *
import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(x, L, rho, c, u, k):
    Pe = (rho * c * u * L) / k
    # Safer formula for very high Pe to avoid overflow
    if Pe > 200:
        # boundary-layer-like profile near x=L
        return 0.0 if x < L else 1.0
    return (np.exp(Pe * x / L) - 1.0) / (np.exp(Pe) - 1.0)

# Domain / mesh
L = 1.0
dimX, dimY = 40, 3
x_mesh = np.linspace(0, L, dimX)
y_mesh = np.linspace(0, 0.1, dimY)
X, Y = np.meshgrid(x_mesh, y_mesh)

# Fields (constant)
rho0, cp0 = 1.0, 1.0
k0 = 0.1  # keep k fixed, vary u to change Pe
rho = np.ones((dimY, dimX)) * rho0
cp  = np.ones((dimY, dimX)) * cp0
k_field = np.ones((dimY, dimX)) * k0

# BCs (same as your case)
boundary = ['N', 'N', 'N', 'D']  # [N,S,W,E]
TD = [0, 0, 0, 1]                # Dirichlet values (only E used here)
alpha = 20
Tinf = 90
q = [0, 0, 0, 0]

# Choose 3 Peclet numbers (via u): Pe = u*L/k = 10*u (since L=1, k=0.1)
cases = [
    ("Low Pe (<10)", 0.4),     # Pe=4
    ("Moderate Pe (~10)", 1.0),  # Pe=10
    ("High Pe (>10)", 7), # Pe=70
]

plt.figure(figsize=(9, 5))

for label, u0 in cases:
    # velocity field shape: (dimY, dimX, 2)
    u_field = np.array([[[-u0, 0.0] for _ in range(dimX)] for _ in range(dimY)])

    fvm_solver = FVMSolver(X, Y, boundary, TD, q, alpha, Tinf, k_field, u_field, rho, cp)
    T_num = fvm_solver.solve()
    T_num_mid = T_num[1, :]

    # Analytical evaluated at mesh x-locations (middle row)
    T_ex = np.array([analytical_solution(x, L, rho0, cp0, u0, k0) for x in x_mesh])

    Pe = (rho0 * cp0 * u0 * L) / k0

    plt.plot(x_mesh, T_ex, linewidth=2, label=f"{label}  |  Analytical (Pe={Pe:.1f})")
    plt.plot(x_mesh, T_num_mid, "--", linewidth=2, label=f"{label}  |  FVM")

plt.xlabel("x [m]")
plt.ylabel("T")
plt.title("1D Convection–Diffusion Verification Across Peclet Numbers")
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
