import numpy as np
from Scripts.fvm_solver import Coordinate2D

class velocityField():

    def __init__(self, X, Y, flFieldOld, flFieldNew, rho_l, rho_s, dt):
        # X, Y is the mesh
        # boundary conditions - wall at the top, no flow on the sides, open at the bottom
        # flFieldOld - liquid fraction field at previous time step
        # flFieldNew - liquid fraction field at current time step
        # rho_l - density of liquid phase
        # rho_s - density of solid phase
        # dt - time step size

        self.X = X
        self.Y = Y
        self.flFieldOld = flFieldOld
        self.flFieldNew = flFieldNew
        self.rho_l = rho_l
        self.rho_s = rho_s

        # n is the number of points in the first direction
        # m is the number of points in the second direction
        self.m, self.n = X.shape
        # m -> y or i , n -> x or j
        self.velocity_field = np.array([[[0, 0] for x in range(self.n)] for y in range(self.m)])
        self.top_flux = np.zeros((self.m, self.n))
        self.bottom_flux = np.zeros((self.m, self.n))
        
    def calculate_area(self, ul, bl, br, ur):
        # calculate the area of the cell
        # ul (upper left), bl (bottom left), br (bottom right), ur (upper right) are the coordinates of the four vertices of the cell
        # apply Gaussian trapezoidal formula to calculate the areas
        area = 0.5 * ((ur.x * br.y - br.x * ur.y) + (br.x * bl.y - bl.x * br.y) + 
                    (bl.x * ul.y - ul.x * bl.y) + (ul.x * ur.y - ur.x * ul.y))
        return area

    def choose_node(self, i, j):
        # Based on 'i','j' decide if the node is inner or boundary (which boundary?)
        if (i < (self.m - 1)) and (j < (self.n - 1)) and (i > 0) and (j > 0):
            area, dx_n, dx_s = self.build_inner(i, j)
        elif (i == (self.m - 1)) and (j > 0) and (j < (self.n - 1)):
            area, dx_n, dx_s = self.build_south(i, j)
        elif (i == 0) and (j > 0) and (j < (self.n - 1)):
            area, dx_n, dx_s = self.build_north(i, j)
        elif (j == (self.n - 1)) and (i > 0) and (i < (self.m - 1)):
            area, dx_n, dx_s = self.build_east(i, j)
        elif (j == 0) and (i > 0) and (i < (self.m - 1)):
            area, dx_n, dx_s = self.build_west(i, j)
        elif (j == (self.n - 1)) and (i == (self.m - 1)):
            area, dx_n, dx_s = self.build_SE(i, j)
        elif (j == 0) and (i == (self.m - 1)):
            area, dx_n, dx_s = self.build_SW(i, j)
        elif (j == (self.n - 1)) and (i == 0):
            area, dx_n, dx_s = self.build_NE(i, j)
        elif (j == 0) and (i == 0):
            area, dx_n, dx_s = self.build_NW(i, j)
        return area, dx_n, dx_s
    
    def dy(self, a, b):
        # Calculate distance between 'a' and 'b' along the y axis
        return b.y - a.y
        
    def dx(self, a, b):
        # Calculate distance between 'a' and 'b' along the x axis
        return b.x - a.x
        
    def dist(self, a, b):
        # Calculate the euclidean distance between 'a' and 'b'
        return np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)

    def index(self, i, j):
        return i * self.n + j
    
    def generate_velocity_field(self):
        # Generate a velocity field based on the liquid fraction fields
        dtflField = (self.flFieldNew - self.flFieldOld) / self.dt               # Time derivative of liquid fraction
        source_term = (self.rho_s - self.rho_l) / self.rho_l * dtflField        # Source term due to phase change
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
    
    def build_inner(self, i, j):
        # principle node coordinate
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])
        NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])
        SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])
        SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])

        # auxiliary node coordinate
        Nw = Coordinate2D((N.x + NW.x)/2, (N.y + NW.y)/2)
        Ne = Coordinate2D((N.x + NE.x)/2, (N.y + NE.y)/2)
        Sw = Coordinate2D((S.x + SW.x)/2, (S.y + SW.y)/2)
        Se = Coordinate2D((S.x + SE.x)/2, (S.y + SE.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)
        sw = Coordinate2D((Sw.x + w.x)/2, (Sw.y + w.y)/2)
        ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
        nw = Coordinate2D((Nw.x + w.x)/2, (Nw.y + w.y)/2)
        
        # calculate the area of the cell
        S_P = self.calculate_area(ne, se, sw, nw)
        return S_P, self.dx(ne, nw), self.dx(sw, se)
        
    def build_north(self, i, j):
        # principle node coordinate
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])
        SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])

        # auxiliary node coordinate
        Sw = Coordinate2D((S.x + SW.x)/2, (S.y + SW.y)/2)
        Se = Coordinate2D((S.x + SE.x)/2, (S.y + SE.y)/2)
        sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
        sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)

        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)
        sw = Coordinate2D((Sw.x + w.x)/2, (Sw.y + w.y)/2)

        # calculate the area of the cell
        S_ss = self.calculate_area(e, se, sw, w)
        return S_ss, self.dx(e, w), self.dx(sw, se)
    
    def build_south(self, i, j):
        # principle node coordinate
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])
        NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])

        # auxiliary node coordinate
        Nw = Coordinate2D((N.x + NW.x)/2, (N.y + NW.y)/2)
        Ne = Coordinate2D((N.x + NE.x)/2, (N.y + NE.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
        nw = Coordinate2D((Nw.x + w.x)/2, (Nw.y + w.y)/2)

        # calculate the area of the cell
        S_nn = self.calculate_area(e, ne, nw, w)
        return S_nn, self.dx(ne, nw), self.dx(w, e)
         
    def build_east(self, i, j):
        # principle node coordinate
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])
        NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])

        # auxiliary node coordinate
        sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
        nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)

        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)

        nw = Coordinate2D((nW.x + n.x)/2, (nW.y + n.y)/2)
        sw = Coordinate2D((sW.x + s.x)/2, (sW.y + s.y)/2)

        # calculate the area of the cell
        S_ee = self.calculate_area(s, sw, nw, n)
        return S_ee, self.dx(n, nw), self.dx(sw, s)
    
    def build_west(self, i, j):
        # principle node coordinate
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])
        NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])

        # auxiliary node coordinate
        Se = Coordinate2D((S.x + SE.x)/2, (S.y + SE.y)/2)
        Ne = Coordinate2D((N.x + NE.x)/2, (N.y + NE.y)/2)

        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
        se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)

        # calculate the area of the cell
        S_ww = self.calculate_area(s, n, ne, se)
        return S_ww, self.dx(ne, n), self.dx(s, se)
        
    def build_NW(self, i, j):
        # principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])

        # auxiliary node coordinates (Mirrored from NE)
        sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)
        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
        
        # --- Consistent se definition (Mirrored from NE's sw) ---
        se = Coordinate2D((sE.x + s.x)/2, (sE.y + s.y)/2)

        # calculate areas (Mirrored from NE)
        S_nw = self.calculate_area(e, se, s, P) # Main CV 
        return S_nw, self.dx(e, P), self.dx(s, se)
    
    def build_NE(self, i, j):
        # principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])

        # auxiliary node coordinates 
        sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        sw = Coordinate2D((sW.x + s.x)/2, (sW.y + s.y)/2)

        # calculate areas
        S_ne = self.calculate_area(w, P, s, sw) # Main CV
        return S_ne, self.dx(P, w), self.dx(sw, s)
    
    def build_SW(self, i, j):
        # principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])

        # auxiliary node coordinates (Mirrored from SE)
        nE = Coordinate2D((E.x + NE.x)/2, (E.y + NE.y)/2)
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
        
        # --- Consistent ne definition (Mirrored from SE's nw) ---
        ne = Coordinate2D((nE.x + n.x)/2, (nE.y + n.y)/2)

        # calculate areas (Mirrored from SE)
        S_sw = self.calculate_area(P, n, ne, e) # Main CV
        return S_sw, self.dx(ne, n), self.dx(P, e)
    
    def build_SE(self, i, j):
        # principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])

        # auxiliary node coordinates
        nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        
        # --- Consistent nw definition ---
        nw = Coordinate2D((nW.x + n.x)/2, (nW.y + n.y)/2)

        # calculate areas
        S_se = self.calculate_area(w, nw, n, P) # Main CV
        return S_se, self.dx(n, nw), self.dx(w, P)