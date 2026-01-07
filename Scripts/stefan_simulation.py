from Scripts.fvm_solver import FVMSolver
from Scripts.fl_field import FlField
from Scripts.velocity_field import velocityField
import numpy as np
import matplotlib.pyplot as plt

class StefanSimulation:
    def __init__(self, X, Y, initial_temp, time_step, steps_no):
        self.X = X
        self.Y = Y
        self.dt = time_step
        self.steps_no = steps_no
        self.total_time = time_step * steps_no
        self.current_time = 0.0
        self.cell_areas = np.zeros(X.shape)

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
        self.flHistory.append(self.fl_field.flField.copy())
        self.THistory.append(self.T_field.copy())
    
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
                
                # --- STEP A: Update Source Term based on GUESS ---
                # Note: We calculate latent heat release if our guess is correct
                self.fvm_solver.source_term(
                    source_type='stefan',
                    flFieldOld=fl_field_old,           # Always reference t=n
                    flFieldNew=fl_field_current_guess, # Reference t=n+1 (guess)
                    dt=self.dt
                )
                
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
            # self.enthalpy_field = new_enthalpy
            
            self.flHistory.append(self.fl_field.flField.copy())
            self.THistory.append(self.T_field.copy())
            
            if not converged:
                print(f"Warning: Step {step} not converged. Residual: {diff}")
            
            self.current_time += self.dt
            
            
# Example usage
Lx, Ly = 0.1, 0.1
dimX, dimY = 4, 4
mesh = np.meshgrid(np.linspace(0, Lx, dimX), np.linspace(0, Ly, dimY))
initial_temp = np.ones((dimY, dimX)) * 273.15  # Initial temperature field (in Kelvin)
initial_temp[-1, :] = 278.15  # Top boundary at higher temperature
time_step = 1  # seconds
steps_no = 10    # number of time steps to simulate

simulation = StefanSimulation(mesh[0], mesh[1], initial_temp, time_step, steps_no)
simulation.run()

# Plot fl and temperature histories
for step in range(len(simulation.flHistory)):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f'Liquid Fraction at Step {step}')
    cf1 = plt.contourf(simulation.X, simulation.Y, simulation.flHistory[step], 
                       levels=np.linspace(0.99999, 1, 11), 
                       vmin=0.99999, vmax=1, 
                       cmap='Blues')
    plt.colorbar(cf1, label='Liquid Fraction', ticks=np.linspace(0, 1, 11))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.title(f'Temperature at Step {step} (K)')
    plt.contourf(simulation.X, simulation.Y, simulation.THistory[step], levels=20, cmap='inferno')
    plt.colorbar(label='Temperature (K)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    