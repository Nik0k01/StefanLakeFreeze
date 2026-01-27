from Scripts import stefan_simulation
import numpy as np
import matplotlib.pyplot as plt


import numpy as np

f = open('resultsEnergy.txt', 'w')
fl = open('fl.txt', 'w')

np.set_printoptions(linewidth=250)

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


shape =  'rectangular'    # 'rectangular', 'linear',  'quadratic', 'crazy'

l = 0.1

# Example usage
Lx, Ly = 0.1, 0.1
dimX, dimY = 3, 256
X, Y = setUpMesh(dimX, dimY, l, formfunction, shape)
number_of_frozen_cells = int(0.01 / (Ly / dimY))  # 1 cm of ice
initial_temp = np.ones((dimY, dimX)) * 273.15
initial_temp[number_of_frozen_cells:, :] += 0.1
t_ice = np.linspace(257.15, 273.15, number_of_frozen_cells)[:, None]
initial_temp[:number_of_frozen_cells, :] = t_ice
# initial_temp[0, :] = 253.15  # Set bottom boundary to -20C
fl_field_init = np.ones((dimY, dimX))
fl_field_init[:number_of_frozen_cells,:] = 0.0

time_step = 10    # seconds
steps_no = 5    # number of time steps to simulate
q=-0
simulation = stefan_simulation.StefanSimulation(X, Y, initial_temp, time_step, steps_no, q=[q, 0, 0, 0], fl_field_init=fl_field_init)
simulation.update_material_properties(fl_field_init)
simulation.velocity_field.generate_velocity_field(simulation.fl_field.flField, 
                                                  simulation.fl_field.flField,
                                                  simulation.fvm_solver.convFVM.rho)
# Initialize accumulators
total_energy_extracted = 0.0
initial_total_enthalpy = np.sum(simulation.calculate_enthalpy(simulation.T_field) * simulation.velocity_field.cell_areas)
previous_total_enthalpy = initial_total_enthalpy

# Lists for plotting
energy_balance_history = []
step_error_history = []
time_history = []
fl_convergence_history = []  # Track fl field convergence per step

if steps_no > 100:
        max_size = 100                                  # Maximum 100 time steps
else:
    max_size = steps_no

# Create a dictionary with indexes corresponding to time steps to save
steps_to_save = {step: idx for idx, step in enumerate(
    np.linspace(0, steps_no, max_size+1, dtype=int)
)}
simulation.flHistory = np.ndarray((max_size, simulation.X.shape[0], simulation.X.shape[1]))
simulation.THistory = np.ndarray((max_size, simulation.X.shape[0], simulation.X.shape[1]))
simulation.vHistory = np.ndarray((max_size, simulation.X.shape[0], simulation.X.shape[1], 2))
simulation.timeHistory = np.ndarray((max_size))
    
for step in range(steps_no):
    print(f"Time Step {step+1}/{steps_no}, Time: {simulation.current_time:.2f}s")
    
    # 1. SAVE THE PREVIOUS STATE 
    # This must not change during the iterations
    fl_field_old = simulation.fl_field.flField.copy()
    
    # 2. Initialize guess for the new field
    # Start by assuming nothing changes (or use last step's rate)
    fl_field_current_guess = fl_field_old.copy()
    # Update material properties
    simulation.update_material_properties(fl_field_current_guess)
    rho_cp_old_step = simulation.fvm_solver.convFVM.rho * simulation.fvm_solver.convFVM.cp
    
    converged = False
    iteration = 0
    max_iterations = 100 # usually converges fast
    tolerance = 1e-6
    print(f"Step {step}", file=f)
    fl_convergence_step = []  # Track convergence for this step
    while not converged and iteration < max_iterations:
        
        # Update properties based on the current phase guess 
        simulation.update_material_properties(fl_field_current_guess)
        rho_cp_new_iter = simulation.fvm_solver.convFVM.rho * simulation.fvm_solver.convFVM.cp
        
        # --- STEP A: Update Source Term based on GUESS ---
        # Note: We calculate latent heat release if our guess is correct
        simulation.fvm_solver.source_term(
            source_type='stefan',
            flFieldOld=fl_field_old,           # Always reference t=n
            flFieldNew=fl_field_current_guess, # Reference t=n+1 (guess)
            dt=simulation.dt
        )
        # Update Velocity Field based on current phase guess
        simulation.velocity_field.velocity_field = simulation.velocity_field.generate_velocity_field(
            fl_field_old,
            fl_field_current_guess,
            simulation.fvm_solver.convFVM.rho
        )
        
        T_initial_conservative = simulation.T_field * (rho_cp_old_step / rho_cp_new_iter)        
        # --- STEP B: Solve Temperature ---
        # The solver sees the heat released by the guessed freezing
        T_field = simulation.fvm_solver.unsteady_solve(
            T_initial=T_initial_conservative, 
            t_end=simulation.dt, 
            dt=simulation.dt,
            boundries=simulation.boundary
        )
        print(f"Iteration {iteration}, source: {simulation.fvm_solver.B}", file=f)
        # Extract the 2D result 
        current_T_field = T_field[-1, :, :]

        # --- STEP C: Correct Phase Field (The Driver) ---
        # Now we ask: "Given this Temp, what should the phase actually be?"
        
        # Calculate enthalpy or use Temperature directly to find liquid fraction
        # (Assuming you have a function that returns f based on T or H)
        fl_previous_guess = fl_field_current_guess.copy()
        fl_field_current_guess = simulation.fl_correction(current_T_field, fl_previous_guess)
        
        # --- STEP E: Update Guess for next iteration ---
        # Don't just swap them; use under-relaxation to prevent oscillations
        # f_new = f_old + omega * (f_calc - f_old)
        relax = 0.75
        fl_field_current_guess = fl_previous_guess + relax * (fl_field_current_guess - fl_previous_guess)
                
        # --- STEP D: Check Convergence ---
        # Did our guess match the result?
        diff = np.max(np.abs(fl_previous_guess - fl_field_current_guess))
        fl_convergence_step.append(diff)  # Track convergence
        if abs(diff) < tolerance:
            converged = True
            print(f"Converged on iteration {iteration}")
            
        if iteration == (max_iterations - 1):
            print('Failed to converge!')
            print(f"Max difference: {diff}")
        
        
        simulation.fvm_solver.B = 0.0 # Reset the LHS for next iteration
        
        iteration += 1

        # --- End of Time Step ---
        # Commit the final calculated values
        simulation.T_field = current_T_field
        simulation.fl_field.flField = fl_field_current_guess # Update the object state
        if (step in steps_to_save):
            # Get the appropriate index in T_history
            idx = steps_to_save.get(step, max_size)
            # Update field evolution
            simulation.flHistory[idx, :, :] = simulation.fl_field.flField.copy()
            simulation.THistory[idx, :, :] = simulation.T_field.copy()
            simulation.vHistory[idx, :, :] = simulation.velocity_field.velocity_field.copy()
            simulation.timeHistory[idx] = simulation.current_time

    # Store fl convergence history for this step
    fl_convergence_history.append(fl_convergence_step)

    simulation.current_time += simulation.dt
    
    
    # ========================================================================
    # ENERGY CONSERVATION WITH DIRICHLET BOUNDARY CONDITIONS
    # ========================================================================
    # With Dirichlet BC, temperature is fixed at the boundary.
    # We must calculate the heat flux using Fourier's law: q = -k * dT/dy
    
    # 1. Calculate Temperature Gradient at Bottom Boundary (y=0)
    # Using centered difference or forward/backward difference
    dy = Y[1, 0] - Y[0, 0]  # Grid spacing in y-direction
    
    # Temperature gradient at bottom boundary (dT/dy at y=0)
    # Use forward difference: (T[1] - T[0]) / dy
    dT_dy_bottom = (simulation.T_field[1, :] - simulation.T_field[0, :]) / dy
    
    # Thermal conductivity at bottom boundary
    k_bottom = simulation.fvm_solver.diffFVM.lambda_coeff[0, :]
    
    # Heat flux at bottom boundary (positive = heat into domain from bottom)
    # q = -k * dT/dy (negative gradient means heat flowing in, positive q)
    flux_bottom = k_bottom * dT_dy_bottom  # W/m²
    
    # Integrate over the width to get total power (W)
    dx = X[0, 1] - X[0, 0]  # Average grid spacing in x-direction (may vary)
    flux_watts_bottom = np.sum(flux_bottom) * dx  # W (integrated over width)
    
    # # 2. Also calculate heat flux at top boundary for completeness
    # dT_dy_top = (simulation.T_field[-1, :] - simulation.T_field[-2, :]) / dy
    # k_top = simulation.fvm_solver.diffFVM.lambda_coeff[-1, :]
    # flux_top = -k_top * dT_dy_top  # W/m² (positive = heat out)
    # flux_watts_top = np.sum(flux_top) * dx  # W
    
    # 3. Calculate advective heat transport (enthalpy flux)
    # This accounts for heat carried by mass moving across boundaries
    outflow_enthalpy = (simulation.velocity_field.velocity_field[-1, :, 1] * 
                        simulation.fvm_solver.convFVM.rho[-1, :] *
                        (simulation.fvm_solver.convFVM.cp[-1, :] * 
                         (simulation.T_field[-1, :] - 273.15) + 
                         simulation.fl_field.L_f) 
                        ) * simulation.dt * dx
    
    # 4. Total energy input to system
    # = Conductive heat at boundaries + Advective transport
    # (flux_watts_bottom is heat IN, flux_watts_top is heat OUT)
    net_boundary_flux = flux_watts_bottom #- flux_watts_top  # Net conductive heat input
    total_energy_extracted += np.abs(net_boundary_flux * simulation.dt) + np.abs(outflow_enthalpy.sum())
    # 2. Calculate Current System Enthalpy
    current_total_enthalpy = np.sum(simulation.calculate_enthalpy(simulation.T_field) * simulation.velocity_field.cell_areas)
    
    # 3. The Balance Check
    # Energy Lost by System vs Energy Extracted by Boundary
    energy_lost_internal = initial_total_enthalpy - current_total_enthalpy

    # Calculate change between Current and Previous step only
    delta_H_step = previous_total_enthalpy - current_total_enthalpy
    # Flux energy step = net conductive heat IN + advective heat OUT
    flux_energy_step = np.abs(net_boundary_flux * simulation.dt) - np.abs(outflow_enthalpy.sum())
    previous_total_enthalpy = current_total_enthalpy

    # This error should be < 1% even at Step 10
    if flux_energy_step > 1e-12:  # Avoid division by zero
        step_error = abs(delta_H_step - flux_energy_step) / abs(flux_energy_step)
    else:
        step_error = 0.0
    step_error_history.append(step_error)
    
    # Store error %
    if total_energy_extracted > 1e-12:  # Avoid division by zero
        error = abs(energy_lost_internal - total_energy_extracted) / abs(total_energy_extracted)
    else:
        error = 0.0
    energy_balance_history.append(error)

print(*simulation.flHistory, file=fl)
fl.close()
f.close()

# ============================================================================
# ENERGY CONSERVATION ANALYSIS SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ENERGY CONSERVATION ANALYSIS (Dirichlet Boundary Conditions)")
print("="*70)
print(f"\nDomain Setup:")
print(f"  Width (Lx):                {Lx:.4f} m")
print(f"  Height (Ly):               {Ly:.4f} m")
print(f"  Total simulation time:     {simulation.current_time:.2f} s")
print(f"  Time steps:                {steps_no}")
print(f"  Time step size:            {time_step:.4f} s")

print(f"\nEnergy Tracking (Dirichlet BC):")
print(f"  Initial total enthalpy:    {initial_total_enthalpy:.2f} J")
print(f"  Final total enthalpy:      {current_total_enthalpy:.2f} J")
print(f"  Internal enthalpy change:  {initial_total_enthalpy - current_total_enthalpy:.2f} J")
print(f"  Total extracted (boundary): {total_energy_extracted:.2f} J")
print(f"  Relative error:            {energy_balance_history[-1]*100:.4f} %")

print(f"\nBoundary Flux Calculation (Fourier's Law: q = -k·dT/dy):")
print(f"  Method:  Compute temperature gradient at boundaries")
print(f"  Heat IN  (bottom): Conductive flux through bottom")
print(f"  Heat OUT (top):    Conductive flux through top + Advective flux")
print(f"  Net energy balance = Internal change + Boundary extraction")

print("="*70 + "\n")

# Plot with additional statistics
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Cumulative error
ax1 = axes[0]
ax1.plot([step for step in range(steps_no)], energy_balance_history, linewidth=2, color='tab:blue')
ax1.set_xlabel('Time Step', fontweight='bold', fontsize=11)
ax1.set_ylabel('Relative Error (%)', fontweight='bold', fontsize=11)
ax1.set_title('Energy Balance: Relative Error Over Time\n(Dirichlet BC)', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_yscale('log')
ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% error')
ax1.legend()

# Step-by-step error
ax2 = axes[1]
ax2.plot([step for step in range(steps_no)], step_error_history, linewidth=2, color='tab:green')
ax2.set_xlabel('Time Step', fontweight='bold', fontsize=11)
ax2.set_ylabel('Step Relative Error', fontweight='bold', fontsize=11)
ax2.set_title('Step-by-Step Energy Balance\n(Dirichlet BC)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_yscale('log')
ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
ax2.legend()

plt.tight_layout()
plt.savefig('/home/niko/Documents/Python/StefanLakeFreeze/Plots/energy_conservation_dirichlet.png', dpi=300, bbox_inches='tight')
plt.show()