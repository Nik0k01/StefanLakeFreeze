"""
Interface Tracking Test
Tracks the ice-water interface position over time and plots against time and sqrt(time)
Validates the Stefan problem analytical solution: z_K(t) ∝ sqrt(t)
"""

from Scripts import stefan_simulation
import numpy as np
import matplotlib.pyplot as plt

def formfunction(x, shape):
    """
    Defines the shape of north boundary
    takes an array and shape
    returns an array
    """
    if x.size == 0:
        return np.array([])
        
    h1 = x[-1]           # west boundary height (based on total length)
    h2 = x[-1] / 10 * 4  # east boundary height
    l  = x[-1]           # domain length
    
    if shape == 'linear':
        m = (h2 - h1) / (2 * l)
        b = h1 / 2
        return m * x + b
    
    elif shape == 'rectangular':
        return l * np.ones((x.size))
        
    elif shape == 'quadratic':
        # Avoid division by zero if h1 == h2
        if h1 == h2:
            return l * np.ones((x.size))
        k = 2 * l**2 / (h1 - h2)
        return (x - l)**2 / k + h2 / 2 
    
    elif shape == 'crazy':
        return h1/2 + (h2/2 - h1/2) * x / l + 0.25*(-h1 + h2/2) * np.sin(np.pi*x/l)**2
    
    else:
        raise ValueError('Unknown shape: %s' % shape)

def setUpMesh(nodes_x, nodes_y, length, formfunction, shape):
    """
    Generates a mesh where every cell has approximately the same area.
    This is achieved by clustering x-nodes in taller regions.
    """
    # 1. Create a high-resolution temporary grid to measure the area profile
    # We use 10,000 points to ensure the integral is accurate
    x_fine = np.linspace(0, length, 10000)
    y_fine = formfunction(x_fine, shape)
    
    # 2. Calculate the Cumulative Area (Integral of height dx)
    # Using trapezoidal integration
    dx_fine = length / (len(x_fine) - 1)
    # Area of each tiny slice
    slice_areas = (y_fine[:-1] + y_fine[1:]) / 2 * dx_fine
    # Cumulative sum [0, area_1, area_1+area_2, ...]
    cum_area = np.concatenate(([0], np.cumsum(slice_areas)))
    
    # 3. Determine target areas for our mesh nodes
    # We want 'nodes_x' points equally spaced in terms of Area, not Length
    total_area = cum_area[-1]
    target_areas = np.linspace(0, total_area, nodes_x)
    
    # 4. Inverse Interpolation
    # Find the x-coordinates that correspond to these target areas
    # np.interp(target_y, known_y, known_x)
    X_nodes_1D = np.interp(target_areas, cum_area, x_fine)
    
    # 5. Generate the final 2D Mesh
    # Recalculate the exact boundary height at our new, non-uniform X nodes
    north_shape = formfunction(X_nodes_1D, shape)
    
    # Create 2D X matrix (repeat the 1D array for every row)
    X = np.tile(X_nodes_1D, (nodes_y, 1))
    
    # Create 2D Y matrix
    # For each column, space points linearly from Top (north_shape) to Bottom (0)
    Y = np.zeros((nodes_y, nodes_x))
    for j in range(nodes_x):
        Y[:, j] = np.linspace(north_shape[j], 0, nodes_y)
        
    return X, Y[::-1]


def calculate_interface_position(fl_field, Y, center_x_idx=None):
    """
    Calculate the average y-position of the ice-water interface.
    Interface is defined as cells where 0 < f < 1 (mixture region).
    
    Parameters:
    -----------
    fl_field : 2D array
        Liquid fraction field (0 = pure ice, 1 = pure liquid)
    Y : 2D array
        Y coordinates of mesh points
    center_x_idx : int, optional
        X index to extract interface position. If None, average over all x.
    
    Returns:
    --------
    float
        Average y-coordinate of the interface
    """
    # Get interface cells (where 0 < f < 1)
    interface_mask = (fl_field > 0) & (fl_field < 1)
    
    if center_x_idx is not None:
        # Extract interface along a vertical line at center_x_idx
        interface_y = Y[interface_mask[:, center_x_idx], center_x_idx]
    else:
        # Average interface position across all x locations
        interface_y = Y[interface_mask]
    
    if len(interface_y) == 0:
        # If no interface cells found, return NaN
        return np.nan
    
    # Return mean interface position
    return np.mean(interface_y)


class InterfaceTracker(stefan_simulation.StefanSimulation):
    """Extended StefanSimulation with interface position tracking."""
    
    def __init__(self, X, Y, initial_temp, time_step, steps_no, q, fl_field_init, center_x_idx=None):
        super().__init__(X, Y, initial_temp, time_step, steps_no, q, fl_field_init)
        self.Y = Y
        self.center_x_idx = center_x_idx if center_x_idx is not None else X.shape[1] // 2
        self.interface_positions = []  # Track interface y-positions
        self.times = []  # Track corresponding times
    
    def run(self):
        """Run simulation while tracking interface position."""
        time_steps = int(self.total_time / self.dt)

        if time_steps > 100:
            max_size = 100
        else:
            max_size = time_steps
        
        steps_to_save = {step: idx for idx, step in enumerate(
            np.linspace(0, time_steps, max_size+1, dtype=int)
        )}
        
        self.flHistory = np.ndarray((max_size, self.X.shape[0], self.X.shape[1]))
        self.THistory = np.ndarray((max_size, self.X.shape[0], self.X.shape[1]))
        self.vHistory = np.ndarray((max_size, self.X.shape[0], self.X.shape[1], 2))
        self.timeHistory = np.ndarray((max_size))
        
        for step in range(time_steps):
            print(f"Time Step {step+1}/{time_steps}, Time: {self.current_time:.2f}s")
            
            # 1. SAVE THE PREVIOUS STATE 
            fl_field_old = self.fl_field.flField.copy()
            
            # 2. Initialize guess for the new field
            fl_field_current_guess = fl_field_old.copy()
            self.update_material_properties(fl_field_current_guess)
            rho_cp_old_step = self.fvm_solver.convFVM.rho * self.fvm_solver.convFVM.cp
            
            converged = False
            iteration = 0
            max_iterations = 30
            tolerance = 1e-6

            while not converged and iteration < max_iterations:
                
                self.update_material_properties(fl_field_current_guess)
                rho_cp_new_iter = self.fvm_solver.convFVM.rho * self.fvm_solver.convFVM.cp

                self.fvm_solver.source_term(
                    source_type='stefan',
                    flFieldOld=fl_field_old,
                    flFieldNew=fl_field_current_guess,
                    dt=self.dt
                )
                
                self.velocity_field.velocity_field = self.velocity_field.generate_velocity_field(
                    fl_field_old,
                    fl_field_current_guess,
                    self.fvm_solver.convFVM.rho
                )
                T_initial_conservative = self.T_field * (rho_cp_old_step / rho_cp_new_iter)
                
                # Solve Temperature
                T_field = self.fvm_solver.unsteady_solve(
                    T_initial=T_initial_conservative, 
                    t_end=self.dt, 
                    dt=self.dt,
                    boundries=self.boundary
                )
                T_new = T_field[-1, :, :]
                
                # Phase field correction
                fl_previous_guess = fl_field_current_guess.copy()
                fl_field_current_guess = self.fl_correction(T_new, fl_previous_guess)
                
                # Under-relaxation to prevent oscillations
                relax = 0.4
                fl_field_current_guess = fl_field_old + relax * (fl_field_current_guess - fl_field_old)
                
                # Check convergence
                convergence_error = np.max(np.abs(fl_previous_guess - fl_field_current_guess))
                
                self.fvm_solver.B = 0.0  # Reset for next iteration
                
                iteration += 1
                
                if convergence_error < tolerance:
                    converged = True
            
            # Update fields for next time step
            self.T_field = T_new.copy()
            self.fl_field.flField = fl_field_current_guess.copy()
            self.current_time += self.dt
            
            # Save to history if this step should be saved
            if step in steps_to_save:
                save_idx = steps_to_save[step]
                self.flHistory[save_idx] = fl_field_current_guess.copy()
                self.THistory[save_idx] = T_new.copy()
                self.vHistory[save_idx] = self.velocity_field.velocity_field.copy()
                self.timeHistory[save_idx] = self.current_time
                
                # Track interface position at this time step
                interface_y = calculate_interface_position(
                    fl_field_current_guess, 
                    self.Y, 
                    self.center_x_idx
                )
                self.interface_positions.append(interface_y)
                self.times.append(self.current_time)


# Setup and run simulation
shape = 'rectangular'
l = 0.1

Lx, Ly = 0.1, 0.1
dimX, dimY = 3, 256
X, Y = setUpMesh(dimX, dimY, l, formfunction, shape)

initial_temp = np.ones((dimY, dimX)) * 273.15
initial_temp[int(dimY/2):, :] += 0.1
x = np.linspace(170, 273, int(dimY/2))[:, None]
initial_temp[:int(dimY/2), :] = x
fl_field_init = np.ones((dimY, dimX))
fl_field_init[:int(dimY/2),:] = 0.0

time_step = 10.  # seconds
steps_no = 100   # number of time steps to simulate

# ============================================================================
# CALCULATE PHASE NUMBER (Eq. 16.8)
# Ph = c_E * (T_S - T_K) / L
# ============================================================================

# Physical properties
c_E = 2090.0         # Specific heat of ice (J/kg·K)
T_S = 273.15         # Melting point (K)
L_f = 334000.0       # Latent heat of fusion (J/kg)

# Determine cold boundary temperature
# T_K is the minimum temperature in the initial temperature field
T_K = initial_temp.min()
delta_T = T_S - T_K  # Temperature difference

# Calculate Phase Number
Ph = c_E * delta_T / L_f

print("\n" + "="*70)
print("PHASE NUMBER ANALYSIS (Stefan Problem - Eq. 16.8)")
print("="*70)
print(f"Phase Number: Ph = c_E × (T_S - T_K) / L_f")
print(f"\nPhysical Properties:")
print(f"  c_E (Ice specific heat):     {c_E:.2f} J/kg·K")
print(f"  L_f (Latent heat fusion):    {L_f:.2f} J/kg")
print(f"\nTemperature Conditions:")
print(f"  T_S (Melting point):         {T_S:.2f} K")
print(f"  T_K (Cold boundary):         {T_K:.2f} K")
print(f"  ΔT = T_S - T_K:              {delta_T:.2f} K")
print(f"\n  → Ph = {c_E:.0f} × {delta_T:.2f} / {L_f:.0f} = {Ph:.6f}")
print(f"\nPhysical Regime:")
if Ph > 10:
    print(f"  ✓ HIGH Ph ({Ph:.4f} >> 1)")
    print(f"    ➜ LATENT HEAT DOMINATED")
    print(f"    ➜ Front moves SLOWLY compared to heat diffusion")
    print(f"    ➜ Temperature profile is LINEAR (Quasi-stationary approximation)")
    print(f"    ➜ ∂T/∂t ≈ 0 (negligible time derivative)")
elif Ph > 1:
    print(f"  ≈ MODERATE Ph ({Ph:.4f} ≈ 1)")
    print(f"    ➜ SENSIBLE and LATENT HEAT comparable")
    print(f"    ➜ Intermediate regime")
else:
    print(f"  ✗ LOW Ph ({Ph:.4f} << 1)")
    print(f"    ➜ SENSIBLE HEAT DOMINATED")
    print(f"    ➜ Ice cools 'slowly' relative to front speed")
    print(f"    ➜ Temperature profile is CURVED (Non-linear)")
    print(f"    ➜ ∂T/∂t is NOT negligible (transient effects important)")
print("="*70 + "\n")

# Create tracker with center x index
center_x_idx = dimX // 2
tracker = InterfaceTracker(X, Y, initial_temp, time_step, steps_no, 
                           q=[-5000, 0, 0, 0], fl_field_init=fl_field_init,
                           center_x_idx=center_x_idx)
tracker.run()

# ============================================================================
# PLOTTING: Interface Position Tracking
# ============================================================================

# Convert to numpy arrays for plotting
times = np.array(tracker.times)
interface_positions = np.array(tracker.interface_positions)
sqrt_times = np.sqrt(times)

# Remove NaN values if any
valid_mask = ~np.isnan(interface_positions)
times_valid = times[valid_mask]
interface_valid = interface_positions[valid_mask]
sqrt_times_valid = sqrt_times[valid_mask]

print(f"\nInterface tracking complete!")
print(f"Number of tracked positions: {len(interface_valid)}")
print(f"Interface position range: {interface_valid.min():.6f} to {interface_valid.max():.6f} m")
print(f"Time range: {times_valid.min():.2f} to {times_valid.max():.2f} s")

# Plot 1: Interface Position vs Time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times_valid, interface_valid, 'o-', linewidth=2, markersize=6, 
        label='Interface Position', color='tab:blue')
ax.set_xlabel('Time (s)', fontweight='bold', fontsize=12)
ax.set_ylabel('Interface Position (m)', fontweight='bold', fontsize=12)
ax.set_title('Ice-Water Interface Position vs Time', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('/home/niko/Documents/Python/StefanLakeFreeze/Plots/interface_vs_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Interface Position vs sqrt(Time)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sqrt_times_valid, interface_valid, 's-', linewidth=2, markersize=6,
        label='Interface Position', color='tab:green')
ax.set_xlabel('√Time (√s)', fontweight='bold', fontsize=12)
ax.set_ylabel('Interface Position (m)', fontweight='bold', fontsize=12)
ax.set_title('Ice-Water Interface Position vs √Time (Analytical Solution Validation)', 
             fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.savefig('/home/niko/Documents/Python/StefanLakeFreeze/Plots/interface_vs_sqrt_time.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Comparison plot - both on same figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Left plot: vs Time
ax1.plot(times_valid, interface_valid, 'o-', linewidth=2, markersize=6,
         label='Interface Position', color='tab:blue')
ax1.set_xlabel('Time (s)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Interface Position (m)', fontweight='bold', fontsize=11)
ax1.set_title('vs Time', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=10)

# Right plot: vs sqrt(Time)
ax2.plot(sqrt_times_valid, interface_valid, 's-', linewidth=2, markersize=6,
         label='Interface Position', color='tab:green')
ax2.set_xlabel('√Time (√s)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Interface Position (m)', fontweight='bold', fontsize=11)
ax2.set_title('vs √Time (Stefan Solution)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('/home/niko/Documents/Python/StefanLakeFreeze/Plots/interface_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Check for linearity with sqrt(t)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sqrt_times_valid, interface_valid, 'o', markersize=8, 
        label='Numerical Solution', color='tab:blue', alpha=0.7)

# Fit a line through the data to verify Stefan solution
coeffs = np.polyfit(sqrt_times_valid, interface_valid, 1)
fit_line = np.poly1d(coeffs)
ax.plot(sqrt_times_valid, fit_line(sqrt_times_valid), '--', linewidth=2,
        label=f'Linear Fit: z = {coeffs[0]:.6f}√t + {coeffs[1]:.6f}', color='tab:red')

ax.set_xlabel('√Time (√s)', fontweight='bold', fontsize=12)
ax.set_ylabel('Interface Position (m)', fontweight='bold', fontsize=12)
ax.set_title('Interface Position vs √Time with Linear Fit\n(Validates Eq. 16.7: $z_K(t) \\propto \\sqrt{t}$)',
             fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11)

# Add R² value
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(sqrt_times_valid, interface_valid)
ax.text(0.05, 0.95, f'$R^2$ = {r_value**2:.6f}', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/niko/Documents/Python/StefanLakeFreeze/Plots/interface_linear_fit.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nLinear Fit Results:")
print(f"Slope (Stefan coefficient): {coeffs[0]:.6f} m/√s")
print(f"Intercept: {coeffs[1]:.6f} m")
print(f"R² value: {r_value**2:.6f}")

# ============================================================================
# OPTIONAL: Low Phase Number Case for Comparison
# ============================================================================
print("\n\n" + "="*70)
print("ALTERNATIVE: LOW PHASE NUMBER CASE (For Comparison)")
print("="*70)
print("Creating a case with Ph << 1 by using MUCH colder boundary...")
print()

# Create a very cold initial condition
initial_temp_cold = np.ones((dimY, dimX)) * 273.15
# Much colder boundary: -50 K instead of gradual
initial_temp_cold[int(dimY/2):, :] += 0.1
x_cold = np.linspace(170 - 50, 273, int(dimY/2))[:, None]  # Extremely cold
initial_temp_cold[:int(dimY/2), :] = x_cold
fl_field_init_cold = np.ones((dimY, dimX))
fl_field_init_cold[:int(dimY/2),:] = 0.0

# Calculate Phase Number for cold case
T_K_cold = initial_temp_cold.min()
delta_T_cold = T_S - T_K_cold
Ph_cold = c_E * delta_T_cold / L_f

print(f"COLD BOUNDARY CASE:")
print(f"  T_K (Cold boundary):         {T_K_cold:.2f} K (was {T_K:.2f} K)")
print(f"  ΔT = T_S - T_K:              {delta_T_cold:.2f} K (was {delta_T:.2f} K)")
print(f"\n  → Ph = {c_E:.0f} × {delta_T_cold:.2f} / {L_f:.0f} = {Ph_cold:.6f}")
print(f"\nComparison:")
print(f"  Original Ph:    {Ph:.6f}")
print(f"  Cold case Ph:   {Ph_cold:.6f}")
print(f"  Ph ratio:       {Ph_cold/Ph:.2f}x")

if Ph_cold < 1:
    print(f"\n  ✓ LOW Ph regime achieved!")
    print(f"    ➜ Sensible heat dominates")
    print(f"    ➜ Temperature profile should be curved (non-linear)")
    print(f"    ➜ Transient term ∂T/∂t is significant")
elif Ph_cold > 10:
    print(f"\n  ✗ Still in HIGH Ph regime")
    print(f"    ➜ Latent heat still dominates")
    print(f"    ➜ To achieve low Ph, need even colder boundary")
    suggested_T_K = T_S - (L_f / c_E * 0.5)  # Target Ph = 0.5
    print(f"    ➜ Suggestion: Use T_K ≈ {suggested_T_K:.2f} K to get Ph ≈ 0.5")
else:
    print(f"\n  ≈ MODERATE Ph regime")

print("="*70)

