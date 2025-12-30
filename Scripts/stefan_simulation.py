from Scripts.fvm_solver import FVM_Solver
from Scripts.fl_field import FLField
from velocity_field import VelocityField
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
        
        self.T_field = initial_temp.copy()
        self.fl_field = FLField(X, Y)
        self.enthalpy_field = self.calculate_enthalpy(self.T_field)
        self.velocity_field = VelocityField(X, Y)
        
        self.fvm_solver = FVM_Solver(X, Y, boundary=['N', 'D', 'N', 'N'], 
                                     TD=[0, 278.15, 0, 0], q=[-200, 0, 0, 0], alpha=1.0, 
                                     Tinf=273.15, conductivity=np.ones(X.shape)*560,
                                     velocity_field=np.zeros(X.shape),
                                     rho_field=np.ones(X.shape)*1000,
                                     cp_field=np.ones(X.shape)*4181)
    
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
    
    def run(self):
        time_steps = int(self.total_time / self.dt)
        
        # For a given step, first solve the temperature field
        # Then update the phase field and the velocity field
        # Calculate the enthalpy field
        # For new phase and velocity field correct the temperature field
        # Iterate until convergence within the time step
        # Move to the next time step
        
        
        for step in range(time_steps):
            print(f"Time Step {step+1}/{time_steps}, Time: {self.current_time:.2f}s")
            converged = False
            iteration = 0
            max_iterations = 100
            tolerance = 1e-4
            
            while not converged and iteration < max_iterations:
                # Solve temperature field
                T_field_old = self.T_field.copy()
                fl_field_new = self.fl_field.flField + 0.001
                # Update source term
                self.fvm_solver.source_term(source_type='stefan',flFieldOld=self.fl_field.flField,
                                            flFieldNew=fl_field_new,
                                            dt=self.dt)
                self.T_field = self.fvm_solver.unsteady_solve(self.T_field, self.dt, self.enthalpy_field)
                # Update phase field
                self.fl_field.update_phase_field(self.enthalpy_field)
                
                # Update velocity field based on new phase field
                self.velocity_field.update_velocity_field(self.fl_field.flField)
                self.fvm_solver.velocity_field = self.velocity_field.velocity_field
                
                # Calculate new enthalpy field
                new_enthalpy_field = self.calculate_enthalpy(self.T_field)
                
                # Check convergence
                if np.max(np.abs(new_enthalpy_field - self.enthalpy_field)) < tolerance:
                    converged = True
                
                self.enthalpy_field = new_enthalpy_field
                iteration += 1
            
            if not converged:
                print("Warning: Did not converge within max iterations.")
            
            self.current_time += self.dt
            
            
# Example usage
Lx, Ly = 0.1, 0.1
dimX, dimY = 50, 50
mesh = np.meshgrid(np.linspace(0, Lx, dimX), np.linspace(0, Ly, dimY))
initial_temp = np.ones((dimY, dimX)) * 273.15  # Initial temperature field (in Kelvin)
time_step = 1.0  # seconds
steps_no = 10    # number of time steps to simulate

simulation = StefanSimulation(mesh[0], mesh[1], initial_temp, time_step, steps_no)
simulation.run()