

import numpy as np
import matplotlib.pyplot as plt

class VelocityInterfaceValidator:
    def __init__(self, simulation):
        self.sim = simulation
        
        
        self.rho_l = 1000  
        self.rho_s = 917.0    
        self.L_f = 3.33e5      
        self.T_melt = 273.15   
        
        # Identify the cold boundary temperature (Tk) from history
        self.T_cold = np.min([np.min(T) for T in simulation.THistory])

    def get_analytical_uE(self, times):
        """
        Analytical Stefan interface velocity from Equation:
        uE(t) = sqrt( (lambda * (Ts - Tk)) / (2 * rho_s * L_f) ) * (1 / sqrt(t))
        """
        
        lam = self.sim.fvm_solver.diffFVM.lambda_coeff[0,0]
        
        
       
        coeff = np.sqrt(lam * (self.T_melt - self.T_cold) /  (2 * self.rho_s * self.L_f))
        
        # Avoid division by zero at the start of the simulation
        times_safe = np.where(times < 0, 1e-12, times)
        return coeff / np.sqrt(times_safe)

    def get_numerical_uE_from_water(self):
       
    
        uE_numerical = []

        for idx, fl_field in enumerate(self.sim.flHistory):
            # Target cells where freezing is occurring (0 < liquid fraction < 1)
            
            interface_mask = (fl_field < 1.0)

            if np.any(interface_mask):
                # Extract vertical material velocity from history
                v_field = self.sim.vHistory[idx][:, :, 1]  # Index 1 for vertical
                v_interface = np.abs(v_field[interface_mask]).mean()
                
                
                uE_val = (v_interface * self.rho_l) / self.rho_s
                uE_numerical.append(uE_val)
            else:
                uE_numerical.append(0.0)

        return np.array(uE_numerical)

    def plot_validation(self):
        times = np.array(self.sim.timeHistory)

        uE_analytical = self.get_analytical_uE(times)
        uE_numerical = self.get_numerical_uE_from_water()

        plt.figure(figsize=(10, 6))
        plt.plot(times[1:], uE_analytical[1:], 'r--', label='Analytical $u_E$ (dynamic $T_i$)')
        plt.plot(times[1:], uE_numerical[1:], 'bo', markersize=4, label='Numerical $u_E$')
        plt.xlabel("Time (s)")
        plt.ylabel("Interface velocity (m/s)")
        plt.title("Stefan velocity using interface temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()



from Scripts.stefan_simulation import StefanSimulation

Lx, Ly = 0.1, 0.1
dimX, dimY = 4, 4
mesh = np.meshgrid(np.linspace(0, Lx, dimX), np.linspace(0, Ly, dimY))
initial_temp = np.ones((dimY, dimX)) * 273.15  # Initial temperature field (in Kelvin)
initial_temp[-1, :] = 278.15  # Top boundary at higher temperature
time_step = 1  # seconds
steps_no = 100   # number of time steps to simulate

simulation = StefanSimulation(mesh[0], mesh[1], initial_temp, time_step, steps_no)
simulation.run()


validator = VelocityInterfaceValidator(simulation)
validator.plot_validation()