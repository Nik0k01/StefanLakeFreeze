from Scripts.fvm_solver import FVMSolver
from Scripts.fl_field import FlField
from Scripts.velocity_field import velocityField
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class StefanSimulation:
    def __init__(self, X, Y, initial_temp, time_step, steps_no):
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
        
        self.boundary = ['N', 'D', 'N', 'N']
        self.fvm_solver = FVMSolver(X, Y, boundary=self.boundary, 
                                     TD=[0, 278.15, 0, 0], q=[-50, 0, 0, 0], alpha=1.0, 
                                     Tinf=273.15, conductivity=np.ones(X.shape)*0.560,
                                     velocity_field=self.velocity_field.velocity_field,
                                     rho_field=np.ones(X.shape)*1000,
                                     cp_field=np.ones(X.shape)*4181)
        
        self.flHistory = []
        self.THistory = []
        self.vHistory = [] # Initialize velocity history
        self.vHistory.append(self.velocity_field.velocity_field.copy()) # Initial state
        # self.energyHistory.append(self.calculate_total_domain_energy())
        
        self.flHistory.append(self.fl_field.flField.copy())
        self.THistory.append(self.T_field.copy())
        # self.energyHistory = []
        self.timeHistory = []
        self.timeHistory.append(self.current_time)
        # self.boundaryFluxHistory = []
    
    def calculate_enthalpy(self, temperature_field):
        """
        Calculate enthalpy based on temperature field and phase fractions.
        """
        H = np.zeros_like(temperature_field)
        solid_mask = self.fl_field.flField == 0.0
        liquid_mask = self.fl_field.flField == 1.0
        mushy_mask = (self.fl_field.flField > 0.0) & (self.fl_field.flField < 1.0)

        # Solid region
        H[solid_mask] = self.fl_field.rho_s * self.fl_field.cp_s * (
                         temperature_field[solid_mask] - self.fl_field.T_melt)

        # Mushy region
        H[mushy_mask] = self.fl_field.rho_s * self.fl_field.cp_s * (self.fl_field.T_melt - self.fl_field.T_melt) + \
                         self.fl_field.rho_l * self.fl_field.L_f * self.fl_field.flField[mushy_mask]    
                         
        # Liquid region
        H[liquid_mask] = self.fl_field.rho_l * self.fl_field.L_f + \
                         self.fl_field.rho_l * self.fl_field.cp_l * (temperature_field[liquid_mask] - self.fl_field.T_melt)
        return H
    
    def fl_correction(self, T_current, T_init):
        """
        Correction of the phase field based on the temperature field.
        
        :param self: Description
        :param T_field: 2D numpy array of temperature values
        """
        cp = 4200
        Lf = 334000
        
        if np.any(T_current < 273.15):
            print("Warning: Temperature below freezing point detected.")
            # Calculate change in temperature
            delta_T = T_current - T_init
            # Check where is liquid
            liquid_mask = self.fl_field.flField > 0.0
            # Calculate change in liquid fraction
            delta_fl = cp * delta_T / Lf
            # If liquid fraction increases, skip - we only freeze
            delta_fl = np.where(delta_fl < 0.0, delta_fl, 0.0)
            # Only update where there is liquid
            delta_fl[~liquid_mask] = 0.0
            return np.clip(self.fl_field.flField + delta_fl, 0.0, 1.0)
        else:
            return self.fl_field.flField.copy()
    
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
        
    def calculate_total_domain_energy(self):
        """
        Sum of (Density * Enthalpy * Volume) for all cells.
        Since we are in 2D, we use Area instead of Volume.
        """
        # 1. Calculate Enthalpy per unit mass (J/kg)
        
        rho = self.fvm_solver.convFVM.rho
        cp = self.fvm_solver.convFVM.cp
        Lf = 334000
        T_melt = 273.15
        
        # Sensible heat (relative to T_melt)
        sensible = cp * (self.T_field - T_melt)
        
        # Latent heat
        latent = self.fl_field.flField * Lf
        
        # Total Energy (J) = Volume * rho * (sensible + latent)
        
        total_energy = np.sum(self.cell_areas * rho * (sensible + latent))
        
        return total_energy
    
    
    def run(self):
        time_steps = int(self.total_time / self.dt)

        for step in range(time_steps):
            print(f"Time Step {step+1}/{time_steps}, Time: {self.current_time:.2f}s")
            
            # 1. SAVE THE PREVIOUS STATE 
            # This must not change during the iterations
            fl_field_old = self.fl_field.flField.copy()
            
            # 2. Initialize guess for the new field
            # Start by assuming nothing changes (or use last step's rate)
            fl_field_current_guess = fl_field_old.copy()
            
            converged = False
            iteration = 0
            max_iterations = 20 # usually converges fast
            tolerance = 1e-12

            while not converged and iteration < max_iterations:
                
                # Update properties based on the current phase guess 
                self.update_material_properties(fl_field_current_guess)

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
                # Pass updated velocity to FVM solver
                self.fvm_solver.convFVM.velocity_field = self.velocity_field.velocity_field
                             
                # --- STEP B: Solve Temperature ---
                # The solver sees the heat released by the guessed freezing
                T_field = self.fvm_solver.unsteady_solve(
                    T_initial=self.T_field, 
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
                fl_field_current_guess = self.fl_correction(current_T_field, self.T_field)
               
                         
                # --- STEP D: Check Convergence ---
                # Did our guess match the result?
                diff = np.max(np.abs(fl_field_old - fl_field_current_guess))
                if diff < tolerance:
                    converged = True
                
                # --- STEP E: Update Guess for next iteration ---
                # Don't just swap them; use under-relaxation to prevent oscillations
                # f_new = f_old + omega * (f_calc - f_old)
                relax = 0.5 
                fl_field_current_guess = fl_field_old + relax * (fl_field_current_guess - fl_field_old)
                
                self.fvm_solver.B = 0.0 # Reset the LHS for next iteration
                
                iteration += 1

            # --- End of Time Step ---
            # Commit the final calculated values
            self.T_field = current_T_field
            self.fl_field.flField = fl_field_current_guess # Update the object state
            self.flHistory.append(self.fl_field.flField.copy())
            self.THistory.append(self.T_field.copy())
            # storing the array 
            self.vHistory.append(self.velocity_field.velocity_field.copy()) 
            # self.energyHistory.append(self.calculate_total_domain_energy())
            
            self.current_time += self.dt
            
            self.timeHistory.append(self.current_time)

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
                
# Example usage
Lx, Ly = 0.1, 0.1
dimX, dimY = 4, 4
mesh = np.meshgrid(np.linspace(0, Lx, dimX), np.linspace(0, Ly, dimY))
initial_temp = np.ones((dimY, dimX)) * 273.15  # Initial temperature field (in Kelvin)
initial_temp[-1, :] = 278.15  # Top boundary at higher temperature
time_step = 1  # seconds
steps_no = 15    # number of time steps to simulate

simulation = StefanSimulation(mesh[0], mesh[1], initial_temp, time_step, steps_no)
simulation.run()

# Plot fl, Temperature, and Velocity histories
for step in range(len(simulation.flHistory)):
    plt.figure(figsize=(18, 5))
    
    # 1. LIQUID FRACTION
    plt.subplot(1, 3, 1)
    plt.title(f'Liquid Fraction at Step {step}')
   
    cf1 = plt.contourf(simulation.X, simulation.Y, simulation.flHistory[step], 
                       levels=np.linspace(0.9999, 1.0, 11), 
                       cmap='Blues')
    plt.colorbar(cf1, label='Liquid Fraction')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.gca().invert_yaxis()
    
    # 2. TEMPERATURE
    plt.subplot(1, 3, 2)
    plt.title(f'Temperature at Step {step} (K)')
    cf2 = plt.contourf(simulation.X, simulation.Y, simulation.THistory[step], 
                       levels=20, cmap='inferno')
    plt.colorbar(cf2, label='Temperature (K)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.gca().invert_yaxis()

    # 3. VELOCITY 
    plt.subplot(1, 3, 3)
    plt.title(f'Vertical Velocity at Step {step}')
    
    v_array = simulation.vHistory[step]
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
    plt.show()


# --- ENERGY VERIFICATION PLOT ---
# simulation.plot_energy()
