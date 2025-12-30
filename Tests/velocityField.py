"""
Unit tests for the velocity field generation in a 1D vertical column.
Verifies the velocity profile against known solutions for specific source terms.
"""

import numpy as np
import matplotlib.pyplot as plt

class MockSimulation:
    def __init__(self, m=20, dy=0.1):
        self.m = m # Rows (vertical)
        self.n = 1 # Columns (just 1 for 1D test)
        self.dy = dy
        self.bottom_flux = np.zeros((m, 1))
        self.velocity_field = np.zeros((m, 1, 2))
        
    def choose_node(self, i, j):
        # Mocking geometry: Area=1, dx=1 (so Flux == Velocity roughly)
        return 1.0, 1.0, 1.0 

    def generate_velocity_field(self, source_term):
        # COPY-PASTE YOUR FUNCTION LOGIC HERE
        # (With the "bottom_flux = top_flux + flux_sum" fix applied)
        for i in range(self.m): # Starting from the top
            for j in range(self.n): # Starting from the left
                area_cell, dx_n, dx_s = self.choose_node(i, j)
                flux_sum = source_term[i, j] * area_cell
                if i == 0:
                    top_flux = 0.0  # No flow through the top wall
                else:
                    top_flux = self.bottom_flux[i-1, j]
                # Bottom flux is equal to the top plus the source term
                bottom_flux = flux_sum + top_flux
                # Update bottom flux 
                self.bottom_flux[i, j] = bottom_flux
                # Calculate velocities at the node
                top_velocity = top_flux / dx_n
                bottom_velocity = bottom_flux / dx_s
                node_velocity = (top_velocity + bottom_velocity) / 2
                self.velocity_field[i, j] = [0.0, node_velocity]  # Assuming vertical flow only
        return self.velocity_field

# --- RUN TESTS ---
sim = MockSimulation(m=10, dy=1.0)

# TEST 1: Uniform Expansion
source_1 = np.ones((10, 1)) * 1.0 
v_field_1 = sim.generate_velocity_field(source_1)
plt.plot(v_field_1[:, 0, 1], label='Test 1: Uniform (Should be Linear)')

# TEST 2: Plug Flow (Source only in middle)
source_2 = np.zeros((10, 1))
source_2[3:6] = 1.0 # Source in cells 3, 4, 5
v_field_2 = sim.generate_velocity_field(source_2)
plt.plot(v_field_2[:, 0, 1], label='Test 2: Plug (Should be Flat-Ramp-Flat)')

# plt.gca().invert_yaxis() # Velocity points down (negative)
plt.legend()
plt.grid(True)
plt.title("Velocity Field Verification")
plt.show()

# TEST 3: Mass Balance
print(f"Total Source Generated: {np.sum(source_2)}")
print(f"Flux out of Bottom:     {sim.bottom_flux[-1, 0]}")
assert np.isclose(np.sum(source_2), sim.bottom_flux[-1, 0]), "Mass Balance FAILED"
print("Mass Balance PASSED")