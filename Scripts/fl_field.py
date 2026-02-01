import numpy as np

class FlField():
    def __init__(self, X, Y, fl_field_initial=None, rho_s=917., rho_l=1000., L_f=334000., cp_s=2090, cp_l=4181, T_melt=273.15):
        self.X = X
        self.Y = Y
        self.rho_s = rho_s
        self.rho_l = rho_l
        self.L_f = L_f
        self.cp_s = cp_s
        self.cp_l = cp_l
        self.T_melt = T_melt 
        
        self.flField = fl_field_initial

    def update_phase_field(self, enthalpy_field):
        """
        Updates phase fraction and temperature based on enthalpy.
        Assumptions:
          - H = 0 is defined as Solid at T_melt.
          - H < 0 is super-cooled solid.
          - H > H_liq is super-heated liquid.
        """
        # Define thresholds (Volumetric Enthalpy)
        # H_sol = 0 (Reference point)
        H_sol = 0.0
        H_liq = self.rho_l * self.L_f 

        # Create masks
        solid_mask = enthalpy_field <= H_sol
        liquid_mask = enthalpy_field >= H_liq
        mushy_mask = (enthalpy_field > H_sol) & (enthalpy_field < H_liq)
        flField = self.flField.copy()
        
        # Update Liquid Fraction
        flField[solid_mask] = 0.0
        flField[liquid_mask] = 1.0
        # Lever rule for mushy zone
        flField[mushy_mask] = (enthalpy_field[mushy_mask] - H_sol) / (H_liq - H_sol)

        # # Update Temperature Field
        # temperature_field = np.zeros_like(enthalpy_field)

        # # Solid Temperature Continuity
        # temperature_field[solid_mask] = self.T_melt + (enthalpy_field[solid_mask] / (self.rho_s * self.cp_s))

        # # Mushy Temperature (Isothermal)
        # temperature_field[mushy_mask] = self.T_melt

        # # Liquid Temperature
        # # Subtract latent heat to get sensible heat portion
        # temperature_field[liquid_mask] = self.T_melt + (enthalpy_field[liquid_mask] - H_liq) / (self.rho_l * self.cp_l)

        return flField