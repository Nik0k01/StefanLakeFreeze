from Scripts.fvm_solver import FVMSolver
from Scripts.fl_field import FlField
from Scripts.velocity_field import velocityField
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


class StefanSimulation:
    def __init__(self, X, Y, initial_temp, time_step, steps_no, q):
        self.X = X
        self.Y = Y
        self.dt = time_step
        self.steps_no = steps_no
        self.total_time = time_step * steps_no
        self.current_time = 0.0
        # dx = X[0,1] - X[0,0]
        # dy = Y[1,0] - Y[0,0]
        # self.cell_areas = np.ones(X.shape) * dx * dy

        self.T_field = initial_temp.copy()
        self.fl_field = FlField(X, Y)
        self.enthalpy_field = self.calculate_enthalpy(self.T_field)
        self.velocity_field = velocityField(X, Y, dt=self.dt)
        
        self.boundary = ['N', 'N', 'N', 'N']
        self.fvm_solver = FVMSolver(X, Y, boundary=self.boundary, 
                                     TD=[0, 0, 0, 0], q=q, alpha=1.0, 
                                     Tinf=273.15, conductivity=np.ones(X.shape)*0.560,
                                     velocity_field=self.velocity_field.velocity_field,
                                     rho_field=np.ones(X.shape)*1000,
                                     cp_field=np.ones(X.shape)*4181)
        

        self.vHistory = [] # Initialize velocity history
        # self.energyHistory.append(self.calculate_total_domain_energy())
        # self.energyHistory = []
        # self.boundaryFluxHistory = []
    
    def calculate_enthalpy(self, temperature_field):
        """
        Calculate enthalpy using the correct Mixture Properties for sensible heat.
        """
        # 1. Re-calculate the Mixture Cp based on the current Phase Field
        # (Or use self.cp_field if it is already updated)
        current_fl = self.fl_field.flField
        rho_l, cp_l = self.fl_field.rho_l, self.fl_field.cp_l
        rho_s, cp_s = self.fl_field.rho_s, self.fl_field.cp_s
        
        # Effective Cp per cell
        cp_eff = current_fl * cp_l + (1.0 - current_fl) * cp_s
        rho_eff = current_fl * rho_l + (1.0 - current_fl) * rho_s
        
        # 2. Sensible Heat (Relative to T_melt)
        # CRITICAL: Use cp_eff, NOT cp_l
        # This ensures the 'Weight' of the temperature change matches the solver's logic
        H_sensible = rho_eff * cp_eff * (temperature_field - self.fl_field.T_melt)
        
        # 3. Latent Heat
        H_latent = rho_l * self.fl_field.L_f * current_fl
        
        return H_sensible + H_latent
    
    def fl_correction(self, T_current, fl_field_guess):
        """
        Correction of the phase field based on the temperature field.
        """
        cp_l = 4181.0
        cp_s = 2090.0  
        Lf = 334000.0
        
        # Calculate effective Cp based on current guess
        cp_eff = fl_field_guess * cp_l + (1.0 - fl_field_guess) * cp_s
        
        # Calculate temperature deviation from melting point
        delta_T = T_current - 273.15
        
        # Calculate required change in liquid fraction to absorb/release this heat
        # dH = rho * L * dfl  approx  rho * cp * dT
        # Therefore: dfl = (cp * dT) / L
        # Note: This is an approximation for the iterative update
        delta_fl = cp_eff * delta_T / Lf
        
        # REMOVED: The constraints that forced delta_fl < 0
        # We allow delta_fl to be positive (melting) to correct numerical overshoots
        
        # Only update physically meaningful cells (e.g. within [0,1])
        # But usually, just clipping the result is sufficient.
        
        # Apply the change
        fl_new = fl_field_guess + delta_fl
        
        # Hard Clip to [0, 1] acts as the ultimate physical constraint
        return np.clip(fl_new, 0.0, 1.0)
    
    def update_material_properties(self, fl_guess):
        """
         interpolate properties between solid (ice) and liquid (water).
        """
        # Properties for Ice (Solid)
        rho_s = 917    # kg/m^3
        cp_s = 2090    # J/kg·K
        k_s = 2.22     # W/m·K
        
        # Properties for Water (Liquid)
        rho_l = 1000   # kg/m^3
        cp_l = 4181    # J/kg·K
        k_l = 0.56     # W/m·K
        
        # Linear mixture rule: property = f*liquid + (1-f)*solid
        self.fvm_solver.convFVM.rho = fl_guess * rho_l + (1 - fl_guess) * rho_s
        self.fvm_solver.convFVM.cp = fl_guess * cp_l + (1 - fl_guess) * cp_s
        self.fvm_solver.diffFVM.lambda_coeff = fl_guess * k_l + (1 - fl_guess) * k_s
    
    def run(self):
        time_steps = int(self.total_time / self.dt)

        if time_steps > 100:
            max_size = 100                                  # Maximum 100 time steps
        else:
            max_size = time_steps
        # Create a dictionary with indexes corresponding to time steps to save
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
            # This must not change during the iterations
            fl_field_old = self.fl_field.flField.copy()
            
            # 2. Initialize guess for the new field
            # Start by assuming nothing changes (or use last step's rate)
            fl_field_current_guess = fl_field_old.copy()
            rho_cp_old_step = self.fvm_solver.convFVM.rho * self.fvm_solver.convFVM.cp
            
            converged = False
            iteration = 0
            max_iterations = 100 # usually converges fast
            tolerance = 1e-12

            while not converged and iteration < max_iterations:
                
                # Update properties based on the current phase guess 
                self.update_material_properties(fl_field_current_guess)
                rho_cp_new_iter = self.fvm_solver.convFVM.rho * self.fvm_solver.convFVM.cp


                # --- STEP A: Update Source Term based on GUESS ---
                # Note: We calculate latent heat release if our guess is correct
                self.fvm_solver.source_term(
                    source_type='stefan',
                    flFieldOld=fl_field_old,           # Always reference t=n
                    flFieldNew=fl_field_current_guess, # Reference t=n+1 (guess)
                    dt=self.dt
                )
                # Update Velocity Field based on current phase guess
                self.velocity_field.velocity_field = self.velocity_field.generate_velocity_field(
                    fl_field_old,
                    fl_field_current_guess
                )
                T_initial_conservative = self.T_field * (rho_cp_old_step / rho_cp_new_iter)       
                # --- STEP B: Solve Temperature ---
                # The solver sees the heat released by the guessed freezing
                T_field = self.fvm_solver.unsteady_solve(
                    T_initial=T_initial_conservative, 
                    t_end=self.dt, 
                    dt=self.dt,
                    boundries=self.boundary
                )
                # Extract the 2D result 
                current_T_field = T_field[-1, :, :]

                # --- STEP C: Correct Phase Field (The Driver) ---
                # Now we ask: "Given this Temp, what should the phase actually be?"
                
                # Calculate enthalpy or use Temperature directly to find liquid fraction
                # (Assuming you have a function that returns f based on T or H)
                fl_previous_guess = fl_field_current_guess.copy()
                fl_field_current_guess = self.fl_correction(current_T_field, fl_previous_guess)
               
                         
                # --- STEP D: Check Convergence ---
                # Did our guess match the result?
                diff = np.max(np.abs(fl_field_old - fl_field_current_guess))
                if diff < tolerance:
                    converged = True
                
                # --- STEP E: Update Guess for next iteration ---
                # Don't just swap them; use under-relaxation to prevent oscillations
                # f_new = f_old + omega * (f_calc - f_old)
                relax = 0.2 
                fl_field_current_guess = fl_field_old + relax * (fl_field_current_guess - fl_field_old)
                
                self.fvm_solver.B = 0.0 # Reset the LHS for next iteration
                
                iteration += 1

            # --- End of Time Step ---
            # Commit the final calculated values
            self.T_field = current_T_field
            self.fl_field.flField = fl_field_current_guess # Update the object state
            if (step in steps_to_save):
                # Get the appropriate index in T_history
                idx = steps_to_save.get(step, max_size)
                # Update field evolution
                self.flHistory[idx, :, :] = self.fl_field.flField.copy()
                self.THistory[idx, :, :] = self.T_field.copy()
                self.vHistory[idx, :, :] = self.velocity_field.velocity_field.copy()
                self.timeHistory[idx] = self.current_time
            # storing the array 
            # self.vHistory.append(self.velocity_field.velocity_field.copy()) 
            # self.energyHistory.append(self.calculate_total_domain_energy())
            
            self.current_time += self.dt
            
    def plot_energy(self):
 
        # Create a figure with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Subplot 1: Total Energy ---
        ax1.plot(self.timeHistory, self.energyHistory, 'o-', color='tab:blue', markersize=4)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Total Energy (J)")
        ax1.set_title("Total Energy: Expected to be Linearly Decreasing")
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: Relative Energy Change/Error ---
        E0 = self.energyHistory[0]
        # Using a numpy array for cleaner math if available, otherwise list comp is fine
        rel_error = [(E - E0) / abs(E0) for E in self.energyHistory]
        
        ax2.plot(self.timeHistory, rel_error, 'o-', color='tab:red', markersize=4)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Relative Energy Error")
        ax2.set_title("Relative Energy Conservation Error")
        ax2.grid(True, alpha=0.3)

        # Adjust layout to prevent label overlap
        plt.tight_layout()
        plt.show()

    def animate_field(self, interval=1, filename='temperature_evolution.gif'):
            """
            Create an animation of the field evolution over time and save as GIF.
            
            Args:
                T_history: 3D array of temperature fields over time (time, y, x)
                interval: Delay between frames in milliseconds
            """
            
            plt.figure(figsize=(18, 5))
    
            # 1. LIQUID FRACTION
            plt.subplot(1, 3, 1)
            plt.title(f'Liquid Fraction at Step {0}')
        
            cf1 = plt.contourf(self.X, self.Y, self.flHistory[0], 
                            levels=11, 
                            cmap='Blues')
            plt.colorbar(cf1, label='Liquid Fraction')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.gca().invert_yaxis()
            
            # 2. TEMPERATURE
            plt.subplot(1, 3, 2)
            plt.title(f'Temperature at Step {0} (K)')
            cf2 = plt.contourf(simulation.X, simulation.Y, simulation.THistory[0], 
                            levels=20, cmap='inferno')
            plt.colorbar(cf2, label='Temperature (K)')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.gca().invert_yaxis()
            
            # 3. VELOCITY 
            plt.subplot(1, 3, 3)
            plt.title(f'Vertical Velocity at Step {0}')
            
            v_array = simulation.vHistory[0]
            v_u = v_array[:, :, 0]
            v_v = v_array[:, :, 1]
            
            # Calculate max velocity to handle the zero-velocity case safely
            v_max = np.max(np.sqrt(v_u**2 + v_v**2))
            
            cf3 = plt.subplot(1, 3, 3).contourf(simulation.X, simulation.Y, v_v, 
                                                levels=20, cmap='viridis')
            plt.colorbar(cf3, label='V_y (m/s)', format='%.1e')
            
            # Only draw arrows if there is actual movement to avoid DivideByZero
            if v_max > 1e-15:
                # we use a small scale or 'xy' units.
                plt.quiver(simulation.X, simulation.Y, v_u, v_v, 
                            color='white', alpha=0.7, pivot='mid', 
                            scale=v_max*10, scale_units='height') 
            else:
                plt.text(0.5, 0.5, 'Static Field', ha='center', va='center', 
                            transform=plt.gca().transAxes, color='white')
            
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            def update(frame):
                plt.clf()
                
                # 1. LIQUID FRACTION
                plt.subplot(1, 3, 1)
                plt.title(f'Liquid Fraction at Step {frame}')
            
                cf1 = plt.contourf(self.X, self.Y, self.flHistory[frame], 
                                levels=11, 
                                cmap='Blues')
                plt.colorbar(cf1, label='Liquid Fraction')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.gca().invert_yaxis()
                
                # 2. TEMPERATURE
                plt.subplot(1, 3, 2)
                plt.title(f'Temperature at Step {frame} (K)')
                cf2 = plt.contourf(self.X, self.Y, self.THistory[frame], 
                                levels=20, cmap='inferno')
                plt.colorbar(cf2, label='Temperature (K)')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.gca().invert_yaxis()
                
                # 3. VELOCITY 
                plt.subplot(1, 3, 3)
                plt.title(f'Vertical Velocity at Step {frame}')
                
                v_array = simulation.vHistory[frame]
                v_u = v_array[:, :, 0]
                v_v = v_array[:, :, 1]
                
                # Calculate max velocity to handle the zero-velocity case safely
                v_max = np.max(np.sqrt(v_u**2 + v_v**2))
                
                cf3 = plt.subplot(1, 3, 3).contourf(simulation.X, simulation.Y, v_v, 
                                                    levels=20, cmap='viridis')
                plt.colorbar(cf3, label='V_y (m/s)', format='%.1e')
                
                # Only draw arrows if there is actual movement to avoid DivideByZero
                if v_max > 1e-15:
                    # we use a small scale or 'xy' units.
                    plt.quiver(simulation.X, simulation.Y, v_u, v_v, 
                                color='white', alpha=0.7, pivot='mid', 
                                scale=v_max*10, scale_units='height') 
                else:
                    plt.text(0.5, 0.5, 'Static Field', ha='center', va='center', 
                                transform=plt.gca().transAxes, color='white')
                
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.gca().invert_yaxis()

                plt.tight_layout()

            
            writer = FFMpegWriter(fps=10, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
            ani = FuncAnimation(plt.gcf(), update, frames=len(self.THistory), interval=interval)
            ani.save(filename, writer=writer)

if __name__ == "__main__":
    # Example usage
    Lx, Ly = 0.1, 0.1
    shape =  'rectangular'
    dimX, dimY = 4, 4
    q = [-2000, 0, 0, 0]
    X, Y = setUpMesh(dimX, dimY, Lx, formfunction, shape)    
    initial_temp = np.ones((dimY, dimX)) * 273.15 + 0.1  # Initial temperature field (in Kelvin)
    time_step = 1  # seconds
    steps_no = 100    # number of time steps to simulate

    simulation = StefanSimulation(X, Y, initial_temp, time_step, steps_no, q)
    simulation.run()
    simulation.animate_field(interval=200, filename='stefan_simulation.mp4')
