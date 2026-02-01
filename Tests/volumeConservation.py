from Scripts import stefan_simulation
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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
dimX, dimY = 4, 24
X, Y = setUpMesh(dimX, dimY, l, formfunction, shape)
initial_temp = np.ones((dimY, dimX)) * 273.15 # Initial temperature field (in Kelvin)
initial_temp[int(dimY/2):, :] += 0.1
x = np.linspace(230, 273, int(dimY/2))[:, None]
initial_temp[:int(dimY/2), :] = x
fl_field_init = np.ones((dimY, dimX))
fl_field_init[:int(dimY/2),:] = 0.0

time_step = 0.01    # seconds
steps_no = 2000    # number of time steps to simulate
q=-2000
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
mass_error_list = []


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
    mass_old = np.sum(simulation.fvm_solver.convFVM.rho * simulation.velocity_field.cell_areas)
    rho_cp_old_step = simulation.fvm_solver.convFVM.rho * simulation.fvm_solver.convFVM.cp
    
    converged = False
    iteration = 0
    max_iterations = 30 # usually converges fast
    tolerance = 1e-7
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
        relax = 0.3
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

    mass_new = np.sum(simulation.fvm_solver.convFVM.rho * simulation.velocity_field.cell_areas)
    mass_escaped = simulation.velocity_field.velocity_field[-1, :] * Lx * simulation.dt * 1000 # m/s * m * s * kg/m^3 ~ kg/m
    mass_escaped = np.sum(np.abs(mass_escaped))
    mass_error = np.abs(np.abs(mass_new - mass_old) - mass_escaped) / np.abs(mass_escaped + 1e-9)
    mass_error_list.append(mass_error)
    fl_change = fl_field_current_guess - fl_field_old
    # Expansion
    expansion = fl_change * (1000 / 917)
    simulation.current_time += simulation.dt
    
plt.figure(figsize=(8,6))
plt.plot(mass_error_list)
plt.title("Mass error")
plt.xlabel("Step")
plt.ylabel("Relative step error")
plt.grid()
plt.show()
