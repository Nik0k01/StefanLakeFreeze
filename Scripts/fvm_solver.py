import numpy as np
import matplotlib.pyplot as plt


class Coordinate2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
class DiffFVM():
    """
    Diffusive part of the Finite Volume Method solver for stefan problem
    """
    def __init__(self, X, Y, boundary=[], TD=[], q=0.0, alpha=0.0, Tinf=0.0, conductivity=None):
        # i, j is the self.index of the cell
        # X, Y is the mesh
        # boundary is the boundary condition: "R", "D", "N"
        # TD is the Dirichlet Temperature
        # q is the heat flux
        # alpha is the heat transfer coefficient
        # Tinf is the temperature of the surrounding

        self.X = X
        self.Y = Y
        self.boundary = boundary
        self.TD = TD
        self.q = q
        self.alpha = alpha
        self.Tinf = Tinf
        self.lambda_coeff = conductivity  # thermal conductivity

        # n is the number of points in the first direction
        # m is the number of points in the second direction
        self.m, self.n = X.shape
        # m -> y or i , n -> x or j

        self.A = np.ones((self.n*self.m, self.n*self.m))
        self.B = np.ones(self.n*self.m)
        
    def calculate_area(self, ul, bl, br, ur):
        # calculate the area of the cell
        # ul (upper left), bl (bottom left), br (bottom right), ur (upper right) are the coordinates of the four vertices of the cell
        # apply Gaussian trapezoidal formula to calculate the areas
        area = 0.5 * ((ur.x * br.y - br.x * ur.y) + (br.x * bl.y - bl.x * br.y) + 
                    (bl.x * ul.y - ul.x * bl.y) + (ul.x * ur.y - ur.x * ul.y))
        return area

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
    
    def set_stencil(self, i, j):
        # Based on 'i','j' decide if the node is inner or boundary (which boundary?)
        if (i < (self.m - 1)) and (j < (self.n - 1)) and (i > 0) and (j > 0):
            a, b = self.build_inner(i, j)
        elif (i == (self.m - 1)) and (j > 0) and (j < (self.n - 1)):
            a, b = self.build_south(i, j)
        elif (i == 0) and (j > 0) and (j < (self.n - 1)):
            a, b = self.build_north(i, j)
        elif (j == (self.n - 1)) and (i > 0) and (i < (self.m - 1)):
            a, b = self.build_east(i, j)
        elif (j == 0) and (i > 0) and (i < (self.m - 1)):
            a, b = self.build_west(i, j)
        elif (j == (self.n - 1)) and (i == (self.m - 1)):
            a, b = self.build_SE(i, j)
        elif (j == 0) and (i == (self.m - 1)):
            a, b = self.build_SW(i, j)
        elif (j == (self.n - 1)) and (i == 0):
            a, b = self.build_NE(i, j)
        elif (j == 0) and (i == 0):
            a, b = self.build_NW(i, j)
        return a, b
    
    def build_inner(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # % Nomenclature:
        # %
        # %    NW(i-1,j-1)   Nw -  N(i-1,j) -  Ne     NE(i-1,j+1)
        # %
        # %                 |                 |
        # %
        # %       nW - - - - nw ------ n ------ ne - - - nE
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %   W(i, j-1) - - w - - P (i,j) - - e - -  E (i,j+1)
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %      sW - - - - sw ------ s ------ se - - - sE
        # %
        # %                 |                 |
        # %
        # %   SW(i+1,j-1)   Sw  -  S(i+1,j)  - Se      SE(i+1,j+1)
        # %
        # % Indexing of stencil: 

        # %    D_4 - D_1 - D2
        # %     |     |     | 
        # %    D_3 - D_0 - D3
        # %     |     |     | 
        # %    D_2 -  D1 - D4
        
        # First coordinate is in the real x direction, and the second in the y direction, however 
        # changing the self.index i (first one) changes the point up or down and the j (second one) left and right
        
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
        nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)
        nE = Coordinate2D((E.x + NE.x)/2, (E.y + NE.y)/2)
        sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
        sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)

        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)
        sw = Coordinate2D((Sw.x + w.x)/2, (Sw.y + w.y)/2)
        ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
        nw = Coordinate2D((Nw.x + w.x)/2, (Nw.y + w.y)/2)
        
        # calculate the area of the cell
        S_P = np.abs(self.calculate_area(ne, se, sw, nw))
        S_n = np.abs(self.calculate_area(Ne, e, w, Nw))
        S_s = np.abs(self.calculate_area(e, Se, Sw, w))
        S_w = np.abs(self.calculate_area(n, s, sW, nW))
        S_e = np.abs(self.calculate_area(nE, sE, s, n))
        

        D3 = ((self.dx(se, ne) * (self.dx(nE, n)/4 + self.dx(s, sE)/4 + self.dx(sE, nE))) / S_e + 
              (self.dy(se, ne) * (self.dy(nE, n)/4 + self.dy(s, sE)/4 + self.dy(sE, nE))) / S_e + 
              (self.dx(e, Ne) * self.dx(ne, nw)) / (4*S_n) + (self.dx(Se,e) * self.dx(sw,se)) / (4*S_s) + 
              (self.dy(e, Ne) * self.dy(ne, nw)) / (4*S_n) + (self.dy(Se,e) * self.dy(sw,se)) / (4*S_s)) / S_P
        D_3 = ((self.dx(nw, sw) * (self.dx(n, nW) / 4 + self.dx(sW, s) / 4 + self.dx(nW, sW))) / S_w +
               (self.dy(nw, sw) * (self.dy(n, nW) / 4 + self.dy(sW, s) / 4 + self.dy(nW, sW))) / S_w +
               (self.dx(Nw, w) * self.dx(ne, nw)) / (4 * S_n) +
               (self.dx(w, Sw) * self.dx(sw, se)) / (4 * S_s) +
               (self.dy(Nw, w) * self.dy(ne, nw)) / (4 * S_n) +
               (self.dy(w, Sw) * self.dy(sw, se)) / (4 * S_s)) / S_P
        D1 = ((self.dx(sw, se) * (self.dx(Se, e) / 4 + self.dx(w, Sw) / 4 + self.dx(Sw, Se))) / S_s +
              (self.dy(sw, se) * (self.dy(Se, e) / 4 + self.dy(w, Sw) / 4 + self.dy(Sw, Se))) / S_s +
              (self.dx(s, sE) * self.dx(se, ne)) / (4 * S_e) +
              (self.dx(sW, s) * self.dx(nw, sw)) / (4 * S_w) +
              (self.dy(s, sE) * self.dy(se, ne)) / (4 * S_e) +
              (self.dy(sW, s) * self.dy(nw, sw)) / (4 * S_w)) / S_P
        # North
        D_1 = ((self.dx(ne, nw) * (self.dx(e, Ne) / 4 + self.dx(Nw, w) / 4 + self.dx(Ne, Nw))) / S_n +
               (self.dy(ne, nw) * (self.dy(e, Ne) / 4 + self.dy(Nw, w) / 4 + self.dy(Ne, Nw))) / S_n +
               (self.dx(nE, n) * self.dx(se, ne)) / (4 * S_e) +
               (self.dx(n, nW) * self.dx(nw, sw)) / (4 * S_w) +
               (self.dy(nE, n) * self.dy(se, ne)) / (4 * S_e) +
               (self.dy(n, nW) * self.dy(nw, sw)) / (4 * S_w)) / S_P

        # NW
        D_4 = ((self.dx(Nw, w) * self.dx(ne, nw)) / (4 * S_n) +
               (self.dx(n, nW) * self.dx(nw, sw)) / (4 * S_w) +
               (self.dy(Nw, w) * self.dy(ne, nw)) / (4 * S_n) +
               (self.dy(n, nW) * self.dy(nw, sw)) / (4 * S_w)) / S_P

        # NE
        D2 = ((self.dx(nE, n) * self.dx(se, ne)) / (4 * S_e) +
              (self.dx(e, Ne) * self.dx(ne, nw)) / (4 * S_n) +
              (self.dy(nE, n) * self.dy(se, ne)) / (4 * S_e) +
              (self.dy(e, Ne) * self.dy(ne, nw)) / (4 * S_n)) / S_P

        # SW
        D_2 = ((self.dx(w, Sw) * self.dx(sw, se)) / (4 * S_s) +
               (self.dx(sW, s) * self.dx(nw, sw)) / (4 * S_w) +
               (self.dy(w, Sw) * self.dy(sw, se)) / (4 * S_s) +
               (self.dy(sW, s) * self.dy(nw, sw)) / (4 * S_w)) / S_P

        # SE
        D4 = ((self.dx(s, sE) * self.dx(se, ne)) / (4 * S_e) +
              (self.dx(Se, e) * self.dx(sw, se)) / (4 * S_s) +
              (self.dy(s, sE) * self.dy(se, ne)) / (4 * S_e) +
              (self.dy(Se, e) * self.dy(sw, se)) / (4 * S_s)) / S_P

        # Center (P)
        D0 = ((self.dx(se, ne) * (self.dx(n, s) + self.dx(nE, n) / 4 + self.dx(s, sE) / 4)) / S_e +
              (self.dx(ne, nw) * (self.dx(w, e) + self.dx(e, Ne) / 4 + self.dx(Nw, w) / 4)) / S_n +
              (self.dx(sw, se) * (self.dx(e, w) + self.dx(Se, e) / 4 + self.dx(w, Sw) / 4)) / S_s +
              (self.dx(nw, sw) * (self.dx(s, n) + self.dx(n, nW) / 4 + self.dx(sW, s) / 4)) / S_w +
              (self.dy(se, ne) * (self.dy(n, s) + self.dy(nE, n) / 4 + self.dy(s, sE) / 4)) / S_e +
              (self.dy(ne, nw) * (self.dy(w, e) + self.dy(e, Ne) / 4 + self.dy(Nw, w) / 4)) / S_n +
              (self.dy(sw, se) * (self.dy(e, w) + self.dy(Se, e) / 4 + self.dy(w, Sw) / 4)) / S_s +
              (self.dy(nw, sw) * (self.dy(s, n) + self.dy(n, nW) / 4 + self.dy(sW, s) / 4)) / S_w) / S_P
        
        stencil[self.index(i, j)] = D0
        stencil[self.index(i-1, j)] = D_1
        stencil[self.index(i+1, j)] = D1
        stencil[self.index(i, j-1)] = D_3
        stencil[self.index(i, j+1)] = D3
        stencil[self.index(i-1, j-1)] = D_4
        stencil[self.index(i-1, j+1)] = D2
        stencil[self.index(i+1, j-1)] = D_2
        stencil[self.index(i+1, j+1)] = D4
        stencil *= self.lambda_coeff[i, j]
        
        return stencil,b
        
    def build_north(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        if self.boundary[0] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0]
        else: 
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
            S_ss = (self.calculate_area(e, se, sw, w))
            S_s = (self.calculate_area(e, Se, Sw, w))
            S_ssw = (self.calculate_area(P, s, sW, W))
            S_sse = (self.calculate_area(E, sE, s, P))

            
            
            # East
            D3 = (self.dy(sw, se) * (self.dy(Se, e) / 4) / S_s + self.dx(sw, se) * (self.dx(Se, e) / 4) / S_s +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + 3 * self.dy(sE, E) / 4 + self.dy(E, P) / 2) / S_sse +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + 3 * self.dx(sE, E) / 4 + self.dx(E, P) / 2) / S_sse) / S_ss

            # West
            D_3 = (self.dy(w, sw) * (3 * self.dy(W, sW) / 4 + self.dy(sW, s) / 4 + self.dy(P, W) / 2) / S_ssw +
                   self.dx(w, sw) * (3 * self.dx(W, sW) / 4 + self.dx(sW, s) / 4 + self.dx(P, W) / 2) / S_ssw +
                   self.dy(sw, se) * (self.dy(w, Sw) / 4) / S_s + self.dx(sw, se) * (self.dx(w, Sw) / 4) / S_s) / S_ss

            # South
            D1 = (self.dy(w, sw) * (self.dy(sW, s) / 4 + self.dy(s, P) / 4) / S_ssw +
                  self.dx(w, sw) * (self.dx(sW, s) / 4 + self.dx(s, P) / 4) / S_ssw +
                  self.dy(sw, se) * (self.dy(w, Sw) / 4 + self.dy(Sw, Se) + self.dy(Se, e) / 4) / S_s +
                  self.dx(sw, se) * (self.dx(w, Sw) / 4 + self.dx(Sw, Se) + self.dx(Se, e) / 4) / S_s +
                  self.dy(se, e) * (self.dy(P, s) / 4 + self.dy(s, sE) / 4) / S_sse +
                  self.dx(se, e) * (self.dx(P, s) / 4 + self.dx(s, sE) / 4) / S_sse) / S_ss

            # SW
            D_2 = (self.dy(w, sw) * (self.dy(W, sW) / 4 + self.dy(sW, s) / 4) / S_ssw +
                   self.dx(w, sw) * (self.dx(W, sW) / 4 + self.dx(sW, s) / 4) / S_ssw +
                   self.dy(sw, se) * (self.dy(w, Sw) / 4) / S_s + self.dx(sw, se) * (self.dx(w, Sw) / 4) / S_s) / S_ss

            # SE
            D4 = (self.dy(sw, se) * (self.dy(Se, e) / 4) / S_s + self.dx(sw, se) * (self.dx(Se, e) / 4) / S_s +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + self.dy(sE, E) / 4) / S_sse +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + self.dx(sE, E) / 4) / S_sse) / S_ss
            
            coefficient = 0.0
            if self.boundary[0] == 'N':
                coefficient = 0.0
                b = self.q[0] * self.dist(e, w) / S_ss
            elif self.boundary[0] == 'R':
                coefficient = - self.alpha
                b = - self.alpha * self.Tinf * self.dist(e, w) / S_ss
            else:
                raise ValueError('Unknown boundary type: %s' % self.boundary[0])
            
            D0 = (coefficient * self.dist(e, w) +
                self.dy(w, sw) * (self.dy(sW, s) / 4 + 3 * self.dy(s, P) / 4 + self.dy(P, W) / 2) / S_ssw +
                self.dx(w, sw) * (self.dx(sW, s) / 4 + 3 * self.dx(s, P) / 4 + self.dx(P, W) / 2) / S_ssw +
                self.dy(sw, se) * (self.dy(w, Sw) / 4 + self.dy(Se, e) / 4 + self.dy(e, w)) / S_s +
                self.dx(sw, se) * (self.dx(w, Sw) / 4 + self.dx(Se, e) / 4 + self.dx(e, w)) / S_s +
                self.dy(se, e) * (3 * self.dy(P, s) / 4 + self.dy(s, sE) / 4 + self.dy(E, P) / 2) / S_sse +
                self.dx(se, e) * (3 * self.dx(P, s) / 4 + self.dx(s, sE) / 4 + self.dx(E, P) / 2) / S_sse) / S_ss
            
            stencil[self.index(i, j)] = D0
            stencil[self.index(i+1, j)] = D1
            stencil[self.index(i, j-1)] = D_3
            stencil[self.index(i, j+1)] = D3
            stencil[self.index(i+1, j-1)] = D_2
            stencil[self.index(i+1, j+1)] = D4
            stencil *= self.lambda_coeff[i, j]

        return stencil,b
    
    def build_south(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        if self.boundary[1] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1]
        else: 
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
            nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)
            nE = Coordinate2D((E.x + NE.x)/2, (E.y + NE.y)/2)

            n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
            w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
            e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

            ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
            nw = Coordinate2D((Nw.x + w.x)/2, (Nw.y + w.y)/2)

            # calculate the area of the cell
            N_nn = self.calculate_area(e, ne, nw, w)
            N_n = self.calculate_area(e, Ne, Nw, w)
            N_nnw = self.calculate_area(P, n, nW, W)
            N_nne = self.calculate_area(E, nE, n, P)

            # East
            D3 = (self.dy(nw, ne) * (self.dy(Ne, e) / 4) / N_n + self.dx(nw, ne) * (self.dx(Ne, e) / 4) / N_n +
                self.dy(ne, e) * (self.dy(n, nE) / 4 + 3 * self.dy(nE, E) / 4 + self.dy(E, P) / 2) / N_nne +
                self.dx(ne, e) * (self.dx(n, nE) / 4 + 3 * self.dx(nE, E) / 4 + self.dx(E, P) / 2) / N_nne) / N_nn

            # West
            D_3 = (self.dy(w, nw) * (3 * self.dy(W, nW) / 4 + self.dy(nW, n) / 4 + self.dy(P, W) / 2) / N_nnw +
                self.dx(w, nw) * (3 * self.dx(W, nW) / 4 + self.dx(nW, n) / 4 + self.dx(P, W) / 2) / N_nnw +
                self.dy(nw, ne) * (self.dy(w, Nw) / 4) / N_n + self.dx(nw, ne) * (self.dx(w, Nw) / 4) / N_n) / N_nn

            # North
            D_1 = (self.dy(w, nw) * (self.dy(nW, n) / 4 + self.dy(n, P) / 4) / N_nnw +
                self.dx(w, nw) * (self.dx(nW, n) / 4 + self.dx(n, P) / 4) / N_nnw +
                self.dy(nw, ne) * (self.dy(w, Nw) / 4 + self.dy(Nw, Ne) + self.dy(Ne, e) / 4) / N_n +
                self.dx(nw, ne) * (self.dx(w, Nw) / 4 + self.dx(Nw, Ne) + self.dx(Ne, e) / 4) / N_n +
                self.dy(ne, e) * (self.dy(P, n) / 4 + self.dy(n, nE) / 4) / N_nne +
                self.dx(ne, e) * (self.dx(P, n) / 4 + self.dx(n, nE) / 4) / N_nne) / N_nn

            # NW
            D2 = (self.dy(w, nw) * (self.dy(W, nW) / 4 + self.dy(nW, n) / 4) / N_nnw +
                self.dx(w, nw) * (self.dx(W, nW) / 4 + self.dx(nW, n) / 4) / N_nnw +
                self.dy(nw, ne) * (self.dy(w, Nw) / 4) / N_n + self.dx(nw, ne) * (self.dx(w, Nw) / 4) / N_n) / N_nn

            # NE
            D_4 = (self.dy(nw, ne) * (self.dy(Ne, e) / 4) / N_n + self.dx(nw, ne) * (self.dx(Ne, e) / 4) / N_n +
                self.dy(ne, e) * (self.dy(n, nE) / 4 + self.dy(nE, E) / 4) / N_nne +
                self.dx(ne, e) * (self.dx(n, nE) / 4 + self.dx(nE, E) / 4) / N_nne) / N_nn
            
            coefficient = 0.0
            if self.boundary[1] == 'N':
                coefficient = 0.0
                b = self.q[1] * self.dist(e, w) / N_nn
            elif self.boundary[1] == 'R':
                coefficient = - self.alpha
                b = - self.alpha * self.Tinf * self.dist(e, w) / N_nn
            else:
                raise ValueError('Unknown boundary type: %s' % self.boundary[0])
            
            D0 = (coefficient * self.dist(e, w) +
                self.dy(w, nw) * (self.dy(nW, n) / 4 + 3 * self.dy(n, P) / 4 + self.dy(P, W) / 2) / N_nnw +
                self.dx(w, nw) * (self.dx(nW, n) / 4 + 3 * self.dx(n, P) / 4 + self.dx(P, W) / 2) / N_nnw +
                self.dy(nw, ne) * (self.dy(w, Nw) / 4 + self.dy(Ne, e) / 4 + self.dy(e, w)) / N_n +
                self.dx(nw, ne) * (self.dx(w, Nw) / 4 + self.dx(Ne, e) / 4 + self.dx(e, w)) / N_n +
                self.dy(ne, e) * (3 * self.dy(P, n) / 4 + self.dy(n, nE) / 4 + self.dy(E, P) / 2) / N_nne +
                self.dx(ne, e) * (3 * self.dx(P, n) / 4 + self.dx(n, nE) / 4 + self.dx(E, P) / 2) / N_nne) / N_nn
            
            stencil[self.index(i, j)] = D0
            stencil[self.index(i-1, j)] = D_1
            stencil[self.index(i, j-1)] = D_3
            stencil[self.index(i, j+1)] = D3
            stencil[self.index(i-1, j-1)] = D2
            stencil[self.index(i-1, j+1)] = D_4
            stencil *= self.lambda_coeff[i, j]

        return stencil,b
         
    def build_east(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        if self.boundary[3] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[3]
        else: 
            # principle node coordinate
            P = Coordinate2D(self.X[i, j], self.Y[i, j])
            S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
            W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
            N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
            SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])
            NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])

            # auxiliary node coordinate
            Sw = Coordinate2D((S.x + SW.x)/2, (S.y + SW.y)/2)
            Nw = Coordinate2D((N.x + NW.x)/2, (N.y + NW.y)/2)
            sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
            nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)

            s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
            w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
            n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)

            nw = Coordinate2D((nW.x + n.x)/2, (nW.y + n.y)/2)
            sw = Coordinate2D((sW.x + s.x)/2, (sW.y + s.y)/2)

            # calculate the area of the cell
            S_ee = self.calculate_area(s, sw, nw, n)
            S_e = self.calculate_area(s, sW, nW, n)
            S_ees = self.calculate_area(S, Sw, w, P)
            S_een = self.calculate_area(N, P, w, Nw)

            # North
            D_1 = (self.dy(nw, sw) * (self.dy(n, nW) / 4) / S_e + self.dx(nw, sw) * (self.dx(n, nW) / 4) / S_e +
                self.dy(n, nw) * (self.dy(Nw, w) / 4 + 3 * self.dy(N, Nw) / 4 + self.dy(P, N) / 2) / S_een +
                self.dx(n, nw) * (self.dx(Nw, w) / 4 + 3 * self.dx(N, Nw) / 4 + self.dx(P, N) / 2) / S_een) / S_ee
            
            # South 
            D1 = (self.dy(nw, sw) * (self.dy(sW, s) / 4) / S_e + self.dx(nw, sw) * (self.dx(sW, s) / 4) / S_e +
                self.dy(sw, s) * (self.dy(w, Sw) / 4 + 3 * self.dy(Sw, S) / 4 + self.dy(S, P) / 2) / S_ees +
                self.dx(sw, s) * (self.dx(w, Sw) / 4 + 3 * self.dx(Sw, S) / 4 + self.dx(S, P) / 2) / S_ees) / S_ee

            # West
            D_3 = (self.dy(sw, s) * (self.dy(w, Sw) / 4 + self.dy(P, w) / 4) / S_ees +
                self.dx(sw, s) * (self.dx(w, Sw) / 4 + self.dx(P, w) / 4) / S_ees +
                self.dy(nw, sw) * (self.dy(sW, s) / 4 + self.dy(nW, sW) + self.dy(n, nW) / 4) / S_e +
                self.dx(nw, sw) * (self.dx(sW, s) / 4 + self.dx(nW, sW) + self.dx(n, nW) / 4) / S_e +
                self.dy(n, nw) * (self.dy(w, P) / 4 + self.dy(Nw, w) / 4) / S_een +
                self.dx(n, nw) * (self.dx(w, P) / 4 + self.dx(Nw, w) / 4) / S_een) / S_ee

            # NW
            D_4 = (self.dy(n, nw) * (self.dy(N, Nw) / 4 + self.dy(Nw, w) / 4) / S_een +
                self.dx(n, nw) * (self.dx(N, Nw) / 4 + self.dx(Nw, w) / 4) / S_een +
                self.dy(nw, sw) * (self.dy(n, nW) / 4) / S_e + self.dx(nw, sw) * (self.dx(n, nW) / 4) / S_e) / S_ee

            # SW
            D_2 = (self.dy(nw, sw) * (self.dy(sW, s) / 4) / S_e + self.dx(nw, sw) * (self.dx(sW, s) / 4) / S_e +
                self.dy(sw, s) * (self.dy(Sw, S) / 4 + self.dy(w, Sw) / 4) / S_ees +
                self.dx(sw, s) * (self.dx(Sw, S) / 4 + self.dx(w, Sw) / 4) / S_ees) / S_ee
            
            coefficient = 0.0
            if self.boundary[3] == 'N':
                coefficient = 0.0
                b = self.q[3] * self.dist(n, s) / S_ee
            elif self.boundary[3] == 'R':
                coefficient = - self.alpha
                b = - self.alpha * self.Tinf * self.dist(n, s) / S_ee
            else:
                raise ValueError('Unknown boundary type: %s' % self.boundary[0])
            
            # calculate the area of the cell
            S_ee = self.calculate_area(s, sw, nw, n)
            S_e = self.calculate_area(s, sW, nW, n)
            S_ees = self.calculate_area(S, Sw, w, P)
            S_een = self.calculate_area(N, P, w, Nw)
            # w->s, e->n,
            D0 = (coefficient * self.dist(nw, sw) +
                self.dy(sw, s) * (self.dy(w, Sw) / 4 + 3 * self.dy(P, w) / 4 + self.dy(S, P) / 2) / S_ees +
                self.dx(sw, s) * (self.dx(w, Sw) / 4 + 3 * self.dx(P, w) / 4 + self.dx(S, P) / 2) / S_ees +
                self.dy(nw, sw) * (self.dy(sW, s) / 4 + self.dy(n, nW) / 4 + self.dy(s, n)) / S_e +
                self.dx(nw, sw) * (self.dx(sW, s) / 4 + self.dx(n, nW) / 4 + self.dx(s, n)) / S_e +
                self.dy(n, nw) * (3 * self.dy(w, P) / 4 + self.dy(Nw, w) / 4 + self.dy(P, N) / 2) / S_een +
                self.dx(n, nw) * (3 * self.dx(w, P) / 4 + self.dx(Nw, w) / 4 + self.dx(P, N) / 2) / S_een) / S_ee

            stencil[self.index(i, j)] = D0
            stencil[self.index(i-1, j)] = D_1
            stencil[self.index(i+1, j)] = D1
            stencil[self.index(i, j-1)] = D_3
            stencil[self.index(i-1, j-1)] = D_4
            stencil[self.index(i+1, j-1)] = D_2
            stencil *= self.lambda_coeff[i, j]

        return stencil,b        
    
    def build_west(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        if self.boundary[2] == 'D':  # Using self.index 2 for west boundary
            stencil[self.index(i, j)] = 1.0
            b = self.TD[2]
        else: 
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
            sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)
            nE = Coordinate2D((E.x + NE.x)/2, (E.y + NE.y)/2)

            s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
            n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
            e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

            ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)
            se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)

            # calculate the area of the cell
            S_ww = self.calculate_area(s, n, ne, se)
            S_w = self.calculate_area(s, n, nE, sE)
            S_wws = self.calculate_area(S, P, e, Se)
            S_wwn = self.calculate_area(P, N, Ne, e)

            # North
            D_1 = (self.dy(se, ne) * (self.dy(nE, n) / 4) / S_w + self.dx(se, ne) * (self.dx(nE, n) / 4) / S_w +
                self.dy(ne, n) * (self.dy(e, Ne) / 4 + 3 * self.dy(Ne, N) / 4 + self.dy(N, P) / 2) / S_wwn +
                self.dx(ne, n) * (self.dx(e, Ne) / 4 + 3 * self.dx(Ne, N) / 4 + self.dx(N, P) / 2) / S_wwn) / S_ww

            # South
            D1 = (self.dy(se, ne) * (self.dy(s, sE) / 4) / S_w + self.dx(se, ne) * (self.dx(s, sE) / 4) / S_w +
                self.dy(s, se) * (self.dy(Se, e) / 4 + 3 * self.dy(S, Se) / 4 + self.dy(P, S) / 2) / S_wws +
                self.dx(s, se) * (self.dx(Se, e) / 4 + 3 * self.dx(S, Se) / 4 + self.dx(P, S) / 2) / S_wws) / S_ww

            # East
            D3 = (self.dy(s, se) * (self.dy(Se, e) / 4 + self.dy(e, P) / 4) / S_wws +
                self.dx(s, se) * (self.dx(Se, e) / 4 + self.dx(e, P) / 4) / S_wws +
                self.dy(se, ne) * (self.dy(s, sE) / 4 + self.dy(sE, nE) + self.dy(nE, n) / 4) / S_w +
                self.dx(se, ne) * (self.dx(s, sE) / 4 + self.dx(sE, nE) + self.dx(nE, n) / 4) / S_w +
                self.dy(ne, n) * (self.dy(P, e) / 4 + self.dy(e, Ne) / 4) / S_wwn +
                self.dx(ne, n) * (self.dx(P, e) / 4 + self.dx(e, Ne) / 4) / S_wwn) / S_ww

            # NE
            D2 = (self.dy(ne, n) * (self.dy(e, Ne) / 4 + self.dy(Ne, N) / 4) / S_wwn +
                  self.dx(ne, n) * (self.dx(e, Ne) / 4 + self.dx(Ne, N) / 4) / S_wwn +
                  self.dy(se, ne) * (self.dy(nE, n) / 4) / S_w + self.dx(se, ne) * (self.dx(nE, n) / 4) / S_w) / S_ww

            # SE
            D4 = (self.dy(se, ne) * (self.dy(s, sE) / 4) / S_w + self.dx(se, ne) * (self.dx(s, sE) / 4) / S_w +
                  self.dy(s, se) * (self.dy(Se, e) / 4 + self.dy(S, Se) / 4) / S_wws +
                  self.dx(s, se) * (self.dx(Se, e) / 4 + self.dx(S, Se) / 4) / S_wws) / S_ww
        
            
            coefficient = 0.0
            if self.boundary[2] == 'N':
                coefficient = 0.0
                b = self.q[2] * self.dist(n, s) / S_ww
            elif self.boundary[2] == 'R':
                coefficient = - self.alpha
                b = - self.alpha * self.Tinf * self.dist(n, s) / S_ww
            else:
                raise ValueError('Unknown boundary type: %s' % self.boundary[3])
            
            D0 = (coefficient * self.dist(ne, se) +
                self.dy(s, se) * (self.dy(Se, e) / 4 + 3 * self.dy(e, P) / 4 + self.dy(P, S) / 2) / S_wws +
                self.dx(s, se) * (self.dx(Se, e) / 4 + 3 * self.dx(e, P) / 4 + self.dx(P, S) / 2) / S_wws +
                self.dy(se, ne) * (self.dy(s, sE) / 4 + self.dy(nE, n) / 4 + self.dy(n, s)) / S_w +
                self.dx(se, ne) * (self.dx(s, sE) / 4 + self.dx(nE, n) / 4 + self.dx(n, s)) / S_w +
                self.dy(ne, n) * (3 * self.dy(P, e) / 4 + self.dy(e, Ne) / 4 + self.dy(N, P) / 2) / S_wwn +
                self.dx(ne, n) * (3 * self.dx(P, e) / 4 + self.dx(e, Ne) / 4 + self.dx(N, P) / 2) / S_wwn) / S_ww
            
            stencil[self.index(i, j)] = D0
            stencil[self.index(i-1, j)] = D_1
            stencil[self.index(i+1, j)] = D1
            stencil[self.index(i, j+1)] = D3
            stencil[self.index(i-1, j+1)] = D2
            stencil[self.index(i+1, j+1)] = D4
            stencil *= self.lambda_coeff[i, j]

        return stencil,b
        
    def build_NW(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        
        # For NW corner, we need to consider both North and West boundary conditions
        if self.boundary[0] == 'D' or self.boundary[2] == 'D':  # If either boundary is Dirichlet
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0] if self.boundary[0] == 'D' else self.TD[2]
        else:
            # principle node coordinates
            P = Coordinate2D(self.X[i, j], self.Y[i, j])
            S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
            E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
            SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])

            # auxiliary node coordinates (Mirrored from NE)
            Se = Coordinate2D((S.x + SE.x)/2, (S.y + SE.y)/2)
            sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)
            s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
            e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
            
            # --- Consistent se definition (Mirrored from NE's sw) ---
            se = Coordinate2D((sE.x + s.x)/2, (sE.y + s.y)/2)

            # calculate areas (Mirrored from NE)
            S_nw = self.calculate_area(e, se, s, P) # Main CV (Mirrored from NE's corrected S_ne)
            S_nws = self.calculate_area(P, e, Se, S) # South sub-volume
            S_nwe = self.calculate_area(P, E, sE, s) # East sub-volume

            # --- D1 (South) coefficient (Mirrored from NE's D1) ---
            D1 = (self.dy(s, se) * (self.dy(Se, e) / 4 + 3 * self.dy(S, Se) / 4 + self.dy(P, S) / 2) / S_nws +
                  self.dx(s, se) * (self.dx(Se, e) / 4 + 3 * self.dx(S, Se) / 4 + self.dx(P, S) / 2) / S_nws +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + self.dy(P, s) / 4) / S_nwe +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + self.dx(P, s) / 4) / S_nwe
                 ) / S_nw

            # --- D3 (East) coefficient (Mirrored from NE's D_3) ---
            D3 = (self.dy(s, se) * (self.dy(Se, e) / 4 + self.dy(e, P) / 4) / S_nws +
                  self.dx(s, se) * (self.dx(Se, e) / 4 + self.dx(e, P) / 4) / S_nws +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + 3 * self.dy(sE, E) / 4 + self.dy(E, P) / 2) / S_nwe +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + 3 * self.dx(sE, E) / 4 + self.dx(E, P) / 2) / S_nwe
                 ) / S_nw

            # --- D4 (Southeast) coefficient (Mirrored from NE's D_2) ---
            D4 = (self.dy(s, se) * (self.dy(S, Se) / 4 + self.dy(Se, e) / 4) / S_nws +
                  self.dx(s, se) * (self.dx(S, Se) / 4 + self.dx(Se, e) / 4) / S_nws +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + self.dy(sE, E) / 4) / S_nwe +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + self.dx(sE, E) / 4) / S_nwe
                 ) / S_nw

            # Calculate boundary contributions
            coef_n = 0.0
            coef_w = 0.0
            b_n = 0.0
            b_w = 0.0

            # North boundary contribution (face e-P)
            if self.boundary[0] == 'N':
                b_n = self.q[0] * self.dist(e, P) / S_nw
            elif self.boundary[0] == 'R':
                coef_n = -self.alpha
                b_n = -self.alpha * self.Tinf * self.dist(e, P) / S_nw

            # West boundary contribution (face P-s)
            if self.boundary[2] == 'N':
                b_w = self.q[2] * self.dist(P, s) / S_nw
            elif self.boundary[2] == 'R':
                coef_w = -self.alpha
                b_w = -self.alpha * self.Tinf * self.dist(P, s) / S_nw

            # --- D0 (Center) coefficient (Mirrored from NE's D0) ---
            D0 = ((coef_n * self.dist(e, P) + coef_w * self.dist(P, s)) +
                  self.dy(s, se) * (self.dy(Se, e) / 4 + 3 * self.dy(e, P) / 4 + self.dy(P, S) / 2) / S_nws +
                  self.dx(s, se) * (self.dx(Se, e) / 4 + 3 * self.dx(e, P) / 4 + self.dx(P, S) / 2) / S_nws +
                  self.dy(se, e) * (self.dy(s, sE) / 4 + 3 * self.dy(P, s) / 4 + self.dy(E, P) / 2) / S_nwe +
                  self.dx(se, e) * (self.dx(s, sE) / 4 + 3 * self.dx(P, s) / 4 + self.dx(E, P) / 2) / S_nwe
                 ) / S_nw

            b = b_n + b_w

            # Assemble the stencil
            stencil[self.index(i, j)] = D0
            stencil[self.index(i+1, j)] = D1
            stencil[self.index(i, j+1)] = D3
            stencil[self.index(i+1, j+1)] = D4
            stencil *= self.lambda_coeff[i, j]

        return stencil, b
    
    def build_NE(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        
        # For NE corner, we need to consider both North and East boundary conditions
        if self.boundary[0] == 'D' or self.boundary[3] == 'D':  # If either boundary is Dirichlet
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0] if self.boundary[0] == 'D' else self.TD[3]
        else:
            # principle node coordinates
            P = Coordinate2D(self.X[i, j], self.Y[i, j])
            S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
            W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
            SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1])

            # auxiliary node coordinates 
            Sw = Coordinate2D((S.x + SW.x)/2, (S.y + SW.y)/2)
            sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
            s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
            w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
            sw = Coordinate2D((sW.x + s.x)/2, (sW.y + s.y)/2)

            # calculate areas
            S_ne = self.calculate_area(w, P, s, sw) # Main CV
            S_nes = self.calculate_area(P, S, Sw, w) # South sub-volume
            S_new = self.calculate_area(P, s, sW, W) # West sub-volume

            # --- D1 (South) coefficient ---
            D1 = (self.dy(sw, s) * (self.dy(w, Sw) / 4 + 3 * self.dy(Sw, S) / 4 + self.dy(S, P) / 2) / S_nes +
                  self.dx(sw, s) * (self.dx(w, Sw) / 4 + 3 * self.dx(Sw, S) / 4 + self.dx(S, P) / 2) / S_nes +
                  self.dy(w, sw) * (self.dy(sW, s) / 4 + self.dy(s, P) / 4) / S_new +
                  self.dx(w, sw) * (self.dx(sW, s) / 4 + self.dx(s, P) / 4) / S_new
                 ) / S_ne

            # --- D_3 (West) coefficient ---
            D_3 = (self.dy(sw, s) * (self.dy(w, Sw) / 4 + self.dy(P, w) / 4) / S_nes +
                   self.dx(sw, s) * (self.dx(w, Sw) / 4 + self.dx(P, w) / 4) / S_nes +
                   self.dy(w, sw) * (self.dy(sW, s) / 4 + 3 * self.dy(W, sW) / 4 + self.dy(P, W) / 2) / S_new +
                   self.dx(w, sw) * (self.dx(sW, s) / 4 + 3 * self.dx(W, sW) / 4 + self.dx(P, W) / 2) / S_new
                  ) / S_ne

            # --- D_2 (Southwest) coefficient ---
            D_2 = (self.dy(sw, s) * (self.dy(Sw, S) / 4 + self.dy(w, Sw) / 4) / S_nes +
                   self.dx(sw, s) * (self.dx(Sw, S) / 4 + self.dx(w, Sw) / 4) / S_nes +
                   self.dy(w, sw) * (self.dy(sW, s) / 4 + self.dy(W, sW) / 4) / S_new +
                   self.dx(w, sw) * (self.dx(sW, s) / 4 + self.dx(W, sW) / 4) / S_new
                  ) / S_ne

            # Calculate boundary contributions 
            coef_n = 0.0
            coef_e = 0.0
            b_n = 0.0
            b_e = 0.0

            # North boundary contribution (face P-w)
            if self.boundary[0] == 'N':
                b_n = self.q[0] * self.dist(P, w) / S_ne
            elif self.boundary[0] == 'R':
                coef_n = -self.alpha
                b_n = -self.alpha * self.Tinf * self.dist(P, w) / S_ne

            # East boundary contribution (face s-P)
            if self.boundary[3] == 'N':
                b_e = self.q[3] * self.dist(s, P) / S_ne
            elif self.boundary[3] == 'R':
                coef_e = -self.alpha
                b_e = -self.alpha * self.Tinf * self.dist(s, P) / S_ne

            # --- D0 (Center) coefficient ---
            D0 = ((coef_n * self.dist(P, w) + coef_e * self.dist(s, P)) +
                  self.dy(sw, s) * (self.dy(w, Sw) / 4 + 3 * self.dy(P, w) / 4 + self.dy(S, P) / 2) / S_nes +
                  self.dx(sw, s) * (self.dx(w, Sw) / 4 + 3 * self.dx(P, w) / 4 + self.dx(S, P) / 2) / S_nes +
                  self.dy(w, sw) * (self.dy(sW, s) / 4 + 3 * self.dy(s, P) / 4 + self.dy(P, W) / 2) / S_new +
                  self.dx(w, sw) * (self.dx(sW, s) / 4 + 3 * self.dx(s, P) / 4 + self.dx(P, W) / 2) / S_new
                 ) / S_ne

            b = b_n + b_e

            # Assemble the stencil (These were correct)
            stencil[self.index(i, j)] = D0
            stencil[self.index(i+1, j)] = D1
            stencil[self.index(i, j-1)] = D_3
            stencil[self.index(i+1, j-1)] = D_2
            stencil *= self.lambda_coeff[i, j]

        return stencil, b
    
    def build_SW(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        
        # For SW corner, we need to consider both South and West boundary conditions
        if self.boundary[1] == 'D' or self.boundary[2] == 'D':  # If either boundary is Dirichlet
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1] if self.boundary[1] == 'D' else self.TD[2]
        else:
            # principle node coordinates
            P = Coordinate2D(self.X[i, j], self.Y[i, j])
            N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
            E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
            NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])

            # auxiliary node coordinates (Mirrored from SE)
            Ne = Coordinate2D((N.x + NE.x)/2, (N.y + NE.y)/2)
            nE = Coordinate2D((E.x + NE.x)/2, (E.y + NE.y)/2)
            n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
            e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
            
            # --- Consistent ne definition (Mirrored from SE's nw) ---
            ne = Coordinate2D((nE.x + n.x)/2, (nE.y + n.y)/2)

            # calculate areas (Mirrored from SE)
            S_sw = self.calculate_area(P, n, ne, e) # Main CV
            S_swn = self.calculate_area(P, N, Ne, e) # North sub-volume
            S_swe = self.calculate_area(P, n, nE, E) # East sub-volume

            # --- D_1 (North) coefficient (Mirrored from SE's D_1) ---
            D_1 = (self.dy(e, ne) * (self.dy(nE, n) / 4 + self.dy(n, P) / 4) / S_swe + 
                   self.dx(e, ne) * (self.dx(nE, n) / 4 + self.dx(n, P) / 4) / S_swe +
                   self.dy(ne, n) * (self.dy(e, Ne) / 4 + 3 * self.dy(Ne, N) / 4 + self.dy(N, P) / 2) / S_swn +
                   self.dx(ne, n) * (self.dx(e, Ne) / 4 + 3 * self.dx(Ne, N) / 4 + self.dx(N, P) / 2) / S_swn
                  ) / S_sw

            # --- D3 (East) coefficient (Mirrored from SE's D_3) ---
            D3 = (self.dy(e, ne) * (self.dy(P, E) / 2 + 3 * self.dy(E, nE) / 4 + self.dy(nE, n) / 4) / S_swe + 
                  self.dx(e, ne) * (self.dx(P, E) / 2 + 3 * self.dx(E, nE) / 4 + self.dx(nE, n) / 4) / S_swe +
                  self.dy(ne, n) * (self.dy(P, e) / 4 + self.dy(e, Ne) / 4) / S_swn +
                  self.dx(ne, n) * (self.dx(P, e) / 4 + self.dx(e, Ne) / 4) / S_swn
                 ) / S_sw

            # --- D2 (Northeast) coefficient (Mirrored from SE's D_4) ---
            D2 = (self.dy(ne, n) * (self.dy(Ne, N) / 4 + self.dy(e, Ne) / 4) / S_swn +
                  self.dx(ne, n) * (self.dx(Ne, N) / 4 + self.dx(e, Ne) / 4) / S_swn +
                  self.dy(ne, e) * (self.dy(nE, n) / 4 + self.dy(E, nE) / 4) / S_swe +
                  self.dx(ne, e) * (self.dx(nE, n) / 4 + self.dx(E, nE) / 4) / S_swe
                 ) / S_sw

            # Calculate boundary contributions
            coef_s = 0.0
            coef_w = 0.0
            b_s = 0.0
            b_w = 0.0

            # South boundary contribution (face e-P)
            if self.boundary[1] == 'N':
                b_s = self.q[1] * self.dist(e, P) / S_sw
            elif self.boundary[1] == 'R':
                coef_s = -self.alpha
                b_s = -self.alpha * self.Tinf * self.dist(e, P) / S_sw

            # West boundary contribution (face P-n)
            if self.boundary[2] == 'N':
                b_w = self.q[2] * self.dist(P, n) / S_sw
            elif self.boundary[2] == 'R':
                coef_w = -self.alpha
                b_w = -self.alpha * self.Tinf * self.dist(P, n) / S_sw

            # --- D0 (Center) coefficient (Mirrored from SE's D0) ---
            D0 = ((coef_s * self.dist(e, P) + coef_w * self.dist(P, n)) + # Boundary terms
                  self.dy(e, ne) * (self.dy(P, E) / 2 + self.dy(nE, n) / 4 + 3 * self.dy(n, P) / 4) / S_swe +
                  self.dx(e, ne) * (self.dx(P, E) / 2 + self.dx(nE, n) / 4 + 3 * self.dx(n, P) / 4) / S_swe +
                  self.dy(ne, n) * (3 * self.dy(P, e) / 4 + self.dy(e, Ne) / 4 + self.dy(N, P) / 2) / S_swn +
                  self.dx(ne, n) * (3 * self.dx(P, e) / 4 + self.dx(e, Ne) / 4 + self.dx(N, P) / 2) / S_swn
                 ) / S_sw

            b = b_s + b_w

            # Assemble the stencil
            stencil[self.index(i, j)] = D0
            stencil[self.index(i-1, j)] = D_1
            stencil[self.index(i, j+1)] = D3
            stencil[self.index(i-1, j+1)] = D2 # Corresponds to NE node
            stencil *= self.lambda_coeff[i, j]

        return stencil, b
    
    def build_SE(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        
        # For SE corner, we need to consider both South and East boundary conditions
        if self.boundary[1] == 'D' or self.boundary[3] == 'D':  # If either boundary is Dirichlet
            stencil[self.index(i, j)] = 1.0
            # This logic might need a check if both are 'D'
            b = self.TD[1] if self.boundary[1] == 'D' else self.TD[3] 
        else:
            # principle node coordinates
            P = Coordinate2D(self.X[i, j], self.Y[i, j])
            N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
            W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
            NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])

            # auxiliary node coordinates
            Nw = Coordinate2D((N.x + NW.x)/2, (N.y + NW.y)/2)
            nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)
            n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
            w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
            
            # --- Consistent nw definition ---
            nw = Coordinate2D((nW.x + n.x)/2, (nW.y + n.y)/2)

            # calculate areas
            S_se = self.calculate_area(w, nw, n, P) # Main CV
            S_sen = self.calculate_area(N, P, w, Nw) # North sub-volume
            S_sew = self.calculate_area(P, W, nW, n) # West sub-volume (logic adapted from build_north S_ssw)
                                               # (build_east S_e definition was for a full cell)

            # --- D_1 (North) coefficient ---
            # (Flux from West face 'nw-w' + Flux from North face 'n-nw')
            D_1 = (self.dy(nw, w) * (self.dy(n, nW) / 4 + self.dy(P, n) / 4) / S_sew + 
                   self.dx(nw, w) * (self.dx(n, nW) / 4 + self.dx(P, n) / 4) / S_sew +
                   self.dy(n, nw) * (self.dy(Nw, w) / 4 + 3 * self.dy(N, Nw) / 4 + self.dy(P, N) / 2) / S_sen +
                   self.dx(n, nw) * (self.dx(Nw, w) / 4 + 3 * self.dx(N, Nw) / 4 + self.dx(P, N) / 2) / S_sen
                  ) / S_se

            # --- D_3 (West) coefficient ---
            # (Flux from West face 'nw-w' + Flux from North face 'n-nw')
            D_3 = (self.dy(nw, w) * (self.dy(W, P) / 2 + 3 * self.dy(nW, W) / 4 + self.dy(n, nW) / 4) / S_sew + 
                   self.dx(nw, w) * (self.dx(W, P) / 2 + 3 * self.dx(nW, W) / 4 + self.dx(n, nW) / 4) / S_sew +
                   self.dy(n, nw) * (self.dy(w, P) / 4 + self.dy(Nw, w) / 4) / S_sen +
                   self.dx(n, nw) * (self.dx(w, P) / 4 + self.dx(Nw, w) / 4) / S_sen
                  ) / S_se

            # --- D_4 (NW) coefficient ---
            # (Flux from West face 'nw-w' + Flux from North face 'n-nw')
            D_4 = (self.dy(n, nw) * (self.dy(N, Nw) / 4 + self.dy(Nw, w) / 4) / S_sen +
                   self.dx(n, nw) * (self.dx(N, Nw) / 4 + self.dx(Nw, w) / 4) / S_sen +
                   self.dy(nw, w) * (self.dy(n, nW) / 4 + self.dy(nW, W) / 4) / S_sew +
                   self.dx(nw, w) * (self.dx(n, nW) / 4 + self.dx(nW, W) / 4) / S_sew
                  ) / S_se

            # Calculate boundary contributions
            coef_s = 0.0
            coef_e = 0.0
            b_s = 0.0
            b_e = 0.0

            # South boundary contribution (face w-P)
            if self.boundary[1] == 'N':
                b_s = self.q[1] * self.dist(w, P) / S_se
            elif self.boundary[1] == 'R':
                coef_s = -self.alpha
                b_s = -self.alpha * self.Tinf * self.dist(w, P) / S_se

            # East boundary contribution (face P-n)
            if self.boundary[3] == 'N':
                b_e = self.q[3] * self.dist(P, n) / S_se
            elif self.boundary[3] == 'R':
                coef_e = -self.alpha
                b_e = -self.alpha * self.Tinf * self.dist(P, n) / S_se

            # --- D0 (Center) coefficient ---
            # (Boundary terms + Flux from West face 'nw-w' + Flux from North face 'n-nw')
            D0 = ((coef_s * self.dist(w, P) + coef_e * self.dist(P, n)) + # Boundary terms
                  self.dy(nw, w) * (self.dy(W, P) / 2 + self.dy(n, nW) / 4 + 3 * self.dy(P, n) / 4) / S_sew +
                  self.dx(nw, w) * (self.dx(W, P) / 2 + self.dx(n, nW) / 4 + 3 * self.dx(P, n) / 4) / S_sew +
                  self.dy(n, nw) * (3 * self.dy(w, P) / 4 + self.dy(Nw, w) / 4 + self.dy(P, N) / 2) / S_sen +
                  self.dx(n, nw) * (3 * self.dx(w, P) / 4 + self.dx(Nw, w) / 4 + self.dx(P, N) / 2) / S_sen
                 ) / S_se

            b = b_s + b_e

            # Assemble the stencil (This was correct)
            stencil[self.index(i, j)] = D0
            stencil[self.index(i-1, j)] = D_1
            stencil[self.index(i, j-1)] = D_3
            stencil[self.index(i-1, j-1)] = D_4
            stencil *= self.lambda_coeff[i, j]

        return stencil, b
       
class ConvectiveFVM(DiffFVM):
    
    def __init__(self, X, Y, boundary=[], TD=[], q=0, alpha=0, Tinf=0, conductivity=None, velocity_field=None, rho_field=None, cp_field=None):
        super().__init__(X, Y, boundary, TD, q, alpha, Tinf, conductivity)
        self.velocity_field = velocity_field  # velocity_field should be an array indexed by (x,y) and returning (vx, vy)
        self.rho = rho_field  # rho_field should be an array indexed by (x,y) and returning density at that point
        self.cp = cp_field  # cp_field should be an array indexed by (x,y) and returning specific heat at that point
        
    def build_inner(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # % Nomenclature:
        # %
        # %    NW(i-1,j-1)   Nw -  N(i-1,j) -  Ne     NE(i-1,j+1)
        # %
        # %                 |                 |
        # %
        # %       nW - - - - nw ------ n ------ ne - - - nE
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %   W(i, j-1) - - w - - P (i,j) - - e - -  E (i,j+1)
        # %                 |                 |
        # %       |         |        |        |       |
        # %                 |                 |
        # %      sW - - - - sw ------ s ------ se - - - sE
        # %
        # %                 |                 |
        # %
        # %   SW(i+1,j-1)   Sw  -  S(i+1,j)  - Se      SE(i+1,j+1)
        # %
        # % Indexing of stencil: 

        # %    D_4 - D_1 - D2
        # %     |     |     | 
        # %    D_3 - D_0 - D3
        # %     |     |     | 
        # %    D_2 -  D1 - D4
        
        # First coordinate is in the real x direction, and the second in the y direction, however 
        # changing the self.index i (first one) changes the point up or down and the j (second one) left and right
        
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

        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]
        
        # East
        # x-direction velocity acorss the eastern face
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        # y-direction velocity acorss the eastern face
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(se, ne) * eastern_velocity_x - self.dx(se, ne) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_P
        
        # West
        # x-direction velocity acorss the western face
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        # y-direction velocity acorss the western face
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(nw, sw) * western_velocity_x - self.dx(nw, sw) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_P
        
        # South
        # x-direction velocity acorss the southern face
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        # y-direction velocity acorss the southern face
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(sw, se) * southern_velocity_x - self.dx(sw, se) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_P
        
        # North
        # x-direction velocity acorss the northern face
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        # y-direction velocity acorss the northern face
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        # Flux across the northern face
        F_N = rho * cp * (self.dy(ne, nw) * northern_velocity_x - self.dx(ne, nw) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_P

        # Center (P)
        D0 = (np.maximum(0, F_E) + 
              np.maximum(0, F_W) + 
              np.maximum(0, F_S) + 
              np.maximum(0, F_N)) / S_P
        
        stencil[self.index(i, j)] = D0
        stencil[self.index(i-1, j)] = -D_1
        stencil[self.index(i+1, j)] = -D1
        stencil[self.index(i, j-1)] = -D_3
        stencil[self.index(i, j+1)] = -D3

        return stencil,b
    
    def build_north(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # 1. Handle Dirichlet (Fixed Temperature Wall/Inlet)
        if self.boundary[0] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0]
            return stencil, b
    
        # 2. General Flux Handling (Neumann / Mixed / Wall)
        
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

        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)

        se = Coordinate2D((Se.x + e.x)/2, (Se.y + e.y)/2)
        sw = Coordinate2D((Sw.x + w.x)/2, (Sw.y + w.y)/2)

        # calculate the area of the cell
        S_ss = self.calculate_area(e, se, sw, w)

        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]

        # East
        # x-direction velocity acorss the eastern face
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        # y-direction velocity acorss the eastern face
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(se, e) * eastern_velocity_x - self.dx(se, e) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_ss

        # West
        # x-direction velocity acorss the western face
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        # y-direction velocity acorss the western face
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(w, sw) * western_velocity_x - self.dx(w, sw) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_ss

        # South
        # x-direction velocity acorss the southern face
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        # y-direction velocity acorss the southern face
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(sw, se) * southern_velocity_x - self.dx(sw, se) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_ss

        # --- NORTH FACE (The Boundary) ---
        # We need the velocity AT the boundary line (ne -> nw).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        northern_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        northern_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_N = rho * cp * (self.dy(e, w) * northern_velocity_x - self.dx(e, w) * northern_velocity_y)
        
        # Center (P)
        D0 = (np.maximum(0, F_E) + 
              np.maximum(0, F_W) + 
              np.maximum(0, F_S) +
              np.maximum(0, F_N)) / S_ss
        
        # --- RHS Source (Inlet Case) ---
        # If F_N < 0 (Inflow), we are bringing in enthalpy from outside.
        # We need the temperature of that incoming fluid (T_inf or T_boundary).
        if F_N < 0:
            T_inlet = self.TD[0] # Assuming boundary temp is stored here
            b += (-F_N * T_inlet) / S_ss
        
        stencil[self.index(i, j)] = D0
        stencil[self.index(i+1, j)] = -D1
        stencil[self.index(i, j-1)] = -D_3
        stencil[self.index(i, j+1)] = -D3
 
        return stencil,b
    
    def build_south(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # 1. Handle Dirichlet (Fixed Temperature Wall/Inlet)
        if self.boundary[1] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[1]
            return stencil, b
        # 2. General Flux Handling (Neumann / Mixed / Wall)
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
        S_nn = self.calculate_area(ne, e, w, nw)

        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]

        # East
        # x-direction velocity acorss the eastern face
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        # y-direction velocity acorss the eastern face
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(e, ne) * eastern_velocity_x - self.dx(e, ne) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_nn

        # West
        # x-direction velocity acorss the western face
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        # y-direction velocity acorss the western face
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(nw, w) * western_velocity_x - self.dx(nw, w) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_nn

        # North
        # x-direction velocity acorss the northern face
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        # y-direction velocity acorss the northern face
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        F_N = rho * cp * (self.dy(ne, nw) * northern_velocity_x - self.dx(ne, nw) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_nn

        # --- SOUTH FACE (The Boundary) ---
        # We need the velocity AT the boundary line (w -> e).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        southern_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        southern_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_S = rho * cp * (self.dy(w, e) * southern_velocity_x - self.dx(w, e) * southern_velocity_y)
        
        # Center (P)
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_nn
        
        # --- RHS Source (Inlet Case) ---
        # If F_N < 0 (Inflow), we are bringing in enthalpy from outside.
        # We need the temperature of that incoming fluid (T_inf or T_boundary).
        if F_S < 0:
            T_inlet = self.TD[0] # Assuming boundary temp is stored here
            b += (-F_S * T_inlet) / S_nn
            
        stencil[self.index(i, j)] = D0
        stencil[self.index(i-1, j)] = -D_1
        stencil[self.index(i, j-1)] = -D_3
        stencil[self.index(i, j+1)] = -D3

        return stencil,b
    
    def build_east(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # 1. Handle Dirichlet (Fixed Temperature Wall/Inlet)
        if self.boundary[3] == 'D':
            stencil[self.index(i, j)] = 1.0
            b = self.TD[3]
            return stencil, b
    
        # 2. General Flux Handling (Neumann / Mixed / Wall)
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
        
        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]

        # North
        # x-direction velocity acorss the northern face
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        # y-direction velocity acorss the northern face
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        F_N = rho * cp * (self.dy(n, nw) * northern_velocity_x - self.dx(n, nw) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_ee
        
        # South
        # x-direction velocity acorss the southern face
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        # y-direction velocity acorss the southern face
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(sw, s) * southern_velocity_x - self.dx(sw, s) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_ee

        # West
        # x-direction velocity acorss the western face
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        # y-direction velocity acorss the western face
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(nw, sw) * western_velocity_x - self.dx(nw, sw) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_ee

        # --- EAST FACE (The Boundary) ---
        # We need the velocity AT the boundary line (s -> n).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        eastern_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        eastern_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_E = rho * cp * (self.dy(s, n) * eastern_velocity_x - self.dx(s, n) * eastern_velocity_y)
        
        # --- RHS Source (Inlet Case) ---
        # If F_E < 0 (Inflow), we are bringing in enthalpy from outside.
        # We need the temperature of that incoming fluid (T_inf or T_boundary).
        if F_E < 0:
            T_inlet = self.TD[0] # Assuming boundary temp is stored here
            b += (-F_E * T_inlet) / S_ee
        
        # Center (P)
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_ee

        stencil[self.index(i, j)] = D0
        stencil[self.index(i-1, j)] = -D_1
        stencil[self.index(i+1, j)] = -D1
        stencil[self.index(i, j-1)] = -D_3

        return stencil,b  
    
    def build_west(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        # 1. Handle Dirichlet (Fixed Temperature Wall/Inlet)
        if self.boundary[2] == 'D':  # Using self.index 2 for west boundary
            stencil[self.index(i, j)] = 1.0
            b = self.TD[2]
            return stencil, b
        # 2. General Flux Handling (Neumann / Mixed / Wall)
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
        
        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]

        # North
        # x-direction velocity acorss the northern face
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        # y-direction velocity acorss the northern face
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        F_N = rho * cp * (self.dy(ne, n) * northern_velocity_x - self.dx(ne, n) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_ww
        
        # South
        # x-direction velocity acorss the southern face
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        # y-direction velocity acorss the southern face
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(s, se) * southern_velocity_x - self.dx(s, se) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_ww
        
        # East
        # x-direction velocity acorss the eastern face
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        # y-direction velocity acorss the eastern face
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(se, ne) * eastern_velocity_x - self.dx(se, ne) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_ww
        
        # --- WEST FACE (The Boundary) ---
        # We need the velocity AT the boundary line (n -> s).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        western_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        western_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_W = rho * cp * (self.dy(n, s) * western_velocity_x - self.dx(n, s) * western_velocity_y)
        
        # --- RHS Source (Inlet Case) ---
        # If F_W < 0 (Inflow), we are bringing in enthalpy from outside.
        # We need the temperature of that incoming fluid (T_inf or T_boundary).
        if F_W < 0:
            T_inlet = self.TD[0] # Assuming boundary temp is stored here
            b += (-F_W * T_inlet) / S_ww
        
        # Center (P)
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_ww
        
        stencil[self.index(i, j)] = D0
        stencil[self.index(i-1, j)] = -D_1
        stencil[self.index(i+1, j)] = -D1
        stencil[self.index(i, j+1)] = -D3

        return stencil,b
    
    def build_NW(self, i, j):
        stencil = np.zeros(self.n*self.m)
        b = 0.0
        
        # For NW corner, we need to consider both North and West boundary conditions
        if self.boundary[0] == 'D' or self.boundary[2] == 'D':  # If either boundary is Dirichlet
            stencil[self.index(i, j)] = 1.0
            b = self.TD[0] if self.boundary[0] == 'D' else self.TD[2]
            return stencil, b
        # principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        SE = Coordinate2D(self.X[i+1, j+1], self.Y[i+1, j+1])

        # auxiliary node coordinates (Mirrored from NE)
        Se = Coordinate2D((S.x + SE.x)/2, (S.y + SE.y)/2)
        sE = Coordinate2D((E.x + SE.x)/2, (E.y + SE.y)/2)
        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
        
        # --- Consistent se definition (Mirrored from NE's sw) ---
        se = Coordinate2D((sE.x + s.x)/2, (sE.y + s.y)/2)

        # calculate areas (Mirrored from NE)
        S_nw = self.calculate_area(e, se, s, P) # Main CV 
        
        # Get properties at the cell center
        rho = self.rho[i, j]
        cp = self.cp[i, j]
        
        # South
        # x-direction velocity acorss the southern face
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        # y-direction velocity acorss the southern face
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(s, se) * southern_velocity_x - self.dx(s, se) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_nw
        
        # East
        # x-direction velocity acorss the eastern face
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        # y-direction velocity acorss the eastern face
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(se, e) * eastern_velocity_x - self.dx(se, e) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_nw

        # --- WEST FACE (The Boundary) ---
        # We need the velocity AT the boundary line (n -> s).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        western_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        western_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_W = rho * cp * (self.dy(P, s) * western_velocity_x - self.dx(P, s) * western_velocity_y)

        # --- NORTH FACE (The Boundary) ---
        # We need the velocity AT the boundary line (ne -> nw).
        # If No-Slip: u_north = 0. 
        # If Outflow: u_north = extrapolated or specified.
        northern_velocity_x = self.velocity_field[i, j][0] # Or specific boundary array
        northern_velocity_y = self.velocity_field[i, j][1] 
        # Flux
        F_N = rho * cp * (self.dy(e, P) * northern_velocity_x - self.dx(e, P) * northern_velocity_y)

        # RHS
        b_w = 0.0
        b_n = 0.0
        if F_W < 0:
            T_inlet = self.TD[2] # Assuming boundary temp is stored here
            b_w += (-F_W * T_inlet) / S_nw
            
        if F_N < 0:
            T_inlet = self.TD[0] # Assuming boundary temp is stored here
            b_n += (-F_N * T_inlet) / S_nw

        b += b_w + b_n
        # --- D0 (Center) coefficient ---
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_nw


        # Assemble the stencil
        stencil[self.index(i, j)] = D0
        stencil[self.index(i+1, j)] = -D1
        stencil[self.index(i, j+1)] = -D3

        return stencil, b
    
    def build_NE(self, i, j):
        stencil = np.zeros(self.n * self.m)
        b = 0.0
        
        # 1. DIRICHLET OVERRIDE (Fixed Temperature)
        # Check North (0) and East (3)
        if self.boundary[0] == 'D' or self.boundary[3] == 'D': 
            stencil[self.index(i, j)] = 1.0
            # Prioritize North, else East
            b = self.TD[0] if self.boundary[0] == 'D' else self.TD[3]
            return stencil, b

        # 2. GEOMETRY SETUP
        # Principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        S = Coordinate2D(self.X[i+1, j], self.Y[i+1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])     # Changed from E to W
        SW = Coordinate2D(self.X[i+1, j-1], self.Y[i+1, j-1]) # Changed from SE to SW

        # Auxiliary node coordinates
        s = Coordinate2D((S.x + P.x)/2, (S.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        Sw = Coordinate2D((S.x + SW.x)/2, (S.y + SW.y)/2)
        sW = Coordinate2D((W.x + SW.x)/2, (W.y + SW.y)/2)
        
        # sw is the center of the geometric quadrant (Mirrored from NW's se)
        sw = Coordinate2D((Sw.x + w.x)/2, (Sw.y + w.y)/2)

        # Calculate Area
        # Loop CCW: P -> w -> sw -> s
        S_ne = abs(self.calculate_area(P, w, sw, s))
        
        # Get properties
        rho = self.rho[i, j]
        cp = self.cp[i, j]
        
        # --- FLUX CALCULATIONS (CCW Loop) ---
        
        # 1. WEST FACE (Internal Neighbor)
        # Path: w -> sw (Down). Normal: Left.
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(w, sw) * western_velocity_x - self.dx(w, sw) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_ne  # Inflow contributes to neighbor coeff
        
        # 2. SOUTH FACE (Internal Neighbor)
        # Path: sw -> s (Right). Normal: Down.
        southern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i+1, j][0]) / 2
        southern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i+1, j][1]) / 2
        F_S = rho * cp * (self.dy(sw, s) * southern_velocity_x - self.dx(sw, s) * southern_velocity_y)
        D1 = np.maximum(0, -F_S) / S_ne   # Inflow contributes to neighbor coeff

        # 3. EAST FACE (Boundary)
        # Path: s -> P (Up). Normal: Right.
        eastern_velocity_x = self.velocity_field[i, j][0]
        eastern_velocity_y = self.velocity_field[i, j][1]
        F_E = rho * cp * (self.dy(s, P) * eastern_velocity_x - self.dx(s, P) * eastern_velocity_y)

        # 4. NORTH FACE (Boundary)
        # Path: P -> w (Left). Normal: Up.
        northern_velocity_x = self.velocity_field[i, j][0]
        northern_velocity_y = self.velocity_field[i, j][1]
        F_N = rho * cp * (self.dy(P, w) * northern_velocity_x - self.dx(P, w) * northern_velocity_y)

        # --- SOURCE TERMS & RHS ---
        b_e = 0.0
        b_n = 0.0

        # East Inlet Logic (Boundary 3)
        if F_E < 0:
            T_inlet_E = self.TD[3] 
            b_e = (-F_E * T_inlet_E) / S_ne
            
        # North Inlet Logic (Boundary 0)
        if F_N < 0:
            T_inlet_N = self.TD[0] 
            b_n = (-F_N * T_inlet_N) / S_ne

        b += b_e + b_n

        # --- CENTER COEFFICIENT (D0) ---
        # Sum of all Outflows (Positive Fluxes)
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_ne

        # --- ASSEMBLE STENCIL ---
        stencil[self.index(i, j)] = D0
        
        # Negative Neighbors on LHS
        stencil[self.index(i+1, j)] = -D1  # South (i+1)
        stencil[self.index(i, j-1)] = -D_3 # West (j-1)

        return stencil, b
    
    def build_SW(self, i, j):
        stencil = np.zeros(self.n * self.m)
        b = 0.0
        
        # 1. DIRICHLET OVERRIDE
        # Check South (1) and West (2)
        if self.boundary[1] == 'D' or self.boundary[2] == 'D': 
            stencil[self.index(i, j)] = 1.0
            # Prioritize South, else West
            b = self.TD[1] if self.boundary[1] == 'D' else self.TD[2]
            return stencil, b

        # 2. GEOMETRY SETUP
        # Principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        E = Coordinate2D(self.X[i, j+1], self.Y[i, j+1])
        NE = Coordinate2D(self.X[i-1, j+1], self.Y[i-1, j+1])

        # Auxiliary node coordinates
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        e = Coordinate2D((E.x + P.x)/2, (E.y + P.y)/2)
        Ne = Coordinate2D((N.x + NE.x)/2, (N.y + NE.y)/2)
        
        # ne is the center of the geometric quadrant (Mirrored from NW's se)
        # Average of the "North-East" quadrant bounds
        ne = Coordinate2D((Ne.x + e.x)/2, (Ne.y + e.y)/2)

        # Calculate Area
        # Loop CCW: P -> e -> ne -> n
        S_sw = abs(self.calculate_area(P, e, ne, n))
        
        # Get properties
        rho = self.rho[i, j]
        cp = self.cp[i, j]
        
        # --- FLUX CALCULATIONS (CCW Loop) ---
        
        # 1. SOUTH FACE (Boundary)
        # Path: P -> e (Left to Right). Normal: Down.
        southern_velocity_x = self.velocity_field[i, j][0]
        southern_velocity_y = self.velocity_field[i, j][1]
        F_S = rho * cp * (self.dy(P, e) * southern_velocity_x - self.dx(P, e) * southern_velocity_y)
        
        # 2. EAST FACE (Internal Neighbor)
        # Path: e -> ne (Up). Normal: Right.
        eastern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j+1][0]) / 2
        eastern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j+1][1]) / 2
        F_E = rho * cp * (self.dy(e, ne) * eastern_velocity_x - self.dx(e, ne) * eastern_velocity_y)
        D3 = np.maximum(0, -F_E) / S_sw  # Inflow contributes to East neighbor

        # 3. NORTH FACE (Internal Neighbor)
        # Path: ne -> n (Left). Normal: Up.
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        F_N = rho * cp * (self.dy(ne, n) * northern_velocity_x - self.dx(ne, n) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_sw # Inflow contributes to North neighbor

        # 4. WEST FACE (Boundary)
        # Path: n -> P (Down). Normal: Left.
        western_velocity_x = self.velocity_field[i, j][0]
        western_velocity_y = self.velocity_field[i, j][1]
        F_W = rho * cp * (self.dy(n, P) * western_velocity_x - self.dx(n, P) * western_velocity_y)

        # --- SOURCE TERMS & RHS ---
        b_s = 0.0
        b_w = 0.0

        # South Inlet Logic (Boundary 1)
        if F_S < 0:
            T_inlet_S = self.TD[1] 
            b_s = (-F_S * T_inlet_S) / S_sw
            
        # West Inlet Logic (Boundary 2)
        if F_W < 0:
            T_inlet_W = self.TD[2] 
            b_w = (-F_W * T_inlet_W) / S_sw

        b += b_s + b_w

        # --- CENTER COEFFICIENT (D0) ---
        # Sum of all Outflows
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_sw

        # --- ASSEMBLE STENCIL ---
        stencil[self.index(i, j)] = D0
        
        # Negative Neighbors on LHS
        stencil[self.index(i, j+1)] = -D3   # East (j+1)
        stencil[self.index(i-1, j)] = -D_1  # North (i-1)

        return stencil, b
    
    def build_SE(self, i, j):
        stencil = np.zeros(self.n * self.m)
        b = 0.0
        
        # 1. DIRICHLET OVERRIDE
        # Check South (1) and East (3)
        if self.boundary[1] == 'D' or self.boundary[3] == 'D': 
            stencil[self.index(i, j)] = 1.0
            # Prioritize South, else East
            b = self.TD[1] if self.boundary[1] == 'D' else self.TD[3]
            return stencil, b

        # 2. GEOMETRY SETUP
        # Principle node coordinates
        P = Coordinate2D(self.X[i, j], self.Y[i, j])
        N = Coordinate2D(self.X[i-1, j], self.Y[i-1, j])
        W = Coordinate2D(self.X[i, j-1], self.Y[i, j-1])
        NW = Coordinate2D(self.X[i-1, j-1], self.Y[i-1, j-1])

        # Auxiliary node coordinates
        n = Coordinate2D((N.x + P.x)/2, (N.y + P.y)/2)
        w = Coordinate2D((W.x + P.x)/2, (W.y + P.y)/2)
        Nw = Coordinate2D((N.x + NW.x)/2, (N.y + NW.y)/2)
        nW = Coordinate2D((W.x + NW.x)/2, (W.y + NW.y)/2)
        
        # nw is the center of the geometric quadrant
        # Average of the "North-West" quadrant bounds
        nw = Coordinate2D((Nw.x + w.x)/2, (Nw.y + w.y)/2)

        # Calculate Area
        # Loop CCW: P -> n -> nw -> w
        S_se = abs(self.calculate_area(P, n, nw, w))
        
        # Get properties
        rho = self.rho[i, j]
        cp = self.cp[i, j]
        
        # --- FLUX CALCULATIONS (CCW Loop) ---
        
        # 1. EAST FACE (Boundary)
        # Path: P -> n (Up). Normal: Right.
        eastern_velocity_x = self.velocity_field[i, j][0]
        eastern_velocity_y = self.velocity_field[i, j][1]
        F_E = rho * cp * (self.dy(P, n) * eastern_velocity_x - self.dx(P, n) * eastern_velocity_y)

        # 2. NORTH FACE (Internal Neighbor)
        # Path: n -> nw (Left). Normal: Up.
        northern_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i-1, j][0]) / 2
        northern_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i-1, j][1]) / 2
        F_N = rho * cp * (self.dy(n, nw) * northern_velocity_x - self.dx(n, nw) * northern_velocity_y)
        D_1 = np.maximum(0, -F_N) / S_se # Inflow contributes to North neighbor

        # 3. WEST FACE (Internal Neighbor)
        # Path: nw -> w (Down). Normal: Left.
        western_velocity_x = (self.velocity_field[i, j][0] + self.velocity_field[i, j-1][0]) / 2
        western_velocity_y = (self.velocity_field[i, j][1] + self.velocity_field[i, j-1][1]) / 2
        F_W = rho * cp * (self.dy(nw, w) * western_velocity_x - self.dx(nw, w) * western_velocity_y)
        D_3 = np.maximum(0, -F_W) / S_se # Inflow contributes to West neighbor

        # 4. SOUTH FACE (Boundary)
        # Path: w -> P (Right). Normal: Down.
        southern_velocity_x = self.velocity_field[i, j][0]
        southern_velocity_y = self.velocity_field[i, j][1]
        F_S = rho * cp * (self.dy(w, P) * southern_velocity_x - self.dx(w, P) * southern_velocity_y)

        # --- SOURCE TERMS & RHS ---
        b_s = 0.0
        b_e = 0.0

        # South Inlet Logic (Boundary 1)
        if F_S < 0:
            T_inlet_S = self.TD[1] 
            b_s = (-F_S * T_inlet_S) / S_se
            
        # East Inlet Logic (Boundary 3)
        if F_E < 0:
            T_inlet_E = self.TD[3] 
            b_e = (-F_E * T_inlet_E) / S_se

        b += b_s + b_e

        # --- CENTER COEFFICIENT (D0) ---
        # Sum of all Outflows
        D0 = (np.maximum(0, F_E) + 
            np.maximum(0, F_W) + 
            np.maximum(0, F_S) +
            np.maximum(0, F_N)) / S_se

        # --- ASSEMBLE STENCIL ---
        stencil[self.index(i, j)] = D0
        
        # Negative Neighbors on LHS
        stencil[self.index(i-1, j)] = -D_1  # North (i-1)
        stencil[self.index(i, j-1)] = -D_3  # West (j-1)

        return stencil, b
 
class FVMSolver:
    def __init__(self, X, Y, boundary=[], TD=[], q=0, alpha=0, Tinf=0, conductivity=None, velocity_field=None, rho_field=None, cp_field=None):
        self.m, self.n = X.shape
        self.X = X
        self.Y = Y
        self.A = np.zeros((self.n*self.m, self.n*self.m))
        self.B = np.zeros(self.n*self.m)
        self.diffFVM = DiffFVM(X, Y, boundary, TD, q, alpha, Tinf, conductivity)
        self.convFVM = ConvectiveFVM(X, Y, boundary, TD, q, alpha, Tinf, conductivity, velocity_field, rho_field, cp_field)
        
    def source_term(self, x_source=0, y_source=0, q=0, source_type='point', 
                    sigma=0.1, rho_s=981., rho_l=1000., 
                    L_f=334000, flFieldOld=None, flFieldNew=None, dt=0.1):
        """
        Adds a source term to the RHS vector B.
        
        Parameters:
        - x_source, y_source: Coordinates of the source location/center
        - q: Intensity/Amplitude of the source
        - type: 'point' for single node, 'gaussian' for distributed blob
        - sigma: Standard deviation (width) for gaussian type
        """
        
        if source_type is not None:
            if source_type == 'point':
                # 1. Find the nearest grid node to the specified (x_source, y_source)
                # Calculate squared distance to every point in the mesh
                dist_sq = (self.X - x_source)**2 + (self.Y - y_source)**2
                
                # Find index of the minimum distance
                min_idx = np.argmin(dist_sq)
                
                # Convert flattened index back to (i, j) if needed, 
                # or just use min_idx directly as 'k' if your B vector follows the flattened mesh order
                k = min_idx 
                
                # Add source to that single node
                self.B[k] += q

            elif source_type == 'gaussian':
                # 1. Vectorized calculation for the whole grid
                # Calculate squared distance from source center for ALL nodes
                dist_sq = (self.X - x_source)**2 + (self.Y - y_source)**2
                
                # 2. Compute Gaussian distribution
                # S = q * exp( -r^2 / 2*sigma^2 )
                source_field = q * np.exp(-dist_sq / (2 * sigma**2))
                
                # 3. Add to the flattened B vector
                self.B += source_field.flatten()
                
            elif source_type == 'stefan':
                # Stefan source term implementation 
                source_term = (flFieldNew - flFieldOld) / dt
                # Mixed density based on phase field
                rho = rho_s  + flFieldNew * (rho_l - rho_s)
                # Energy is released when water is freezing
                source_term *= rho_l * L_f
                # Add to the flattened B vector
                self.B += source_term.flatten()
            else:
                raise ValueError("Unknown source term type. Use 'point', 'gaussian', or 'stefan'.")
    
    def solve(self):
        for i in range(self.m):
            for j in range(self.n):
                # Set stencil for the node
                k = self.diffFVM.index(i, j)
                a_diff, b_diff = self.diffFVM.set_stencil(i, j)
                a_conv, b_conv = self.convFVM.set_stencil(i, j)
                self.A[k, :] = a_diff + a_conv
                self.B[k] = b_diff + b_conv
        T = np.linalg.solve(self.A, self.B)        
        dimY, dimX = self.m, self.n
        return T.reshape(dimY, dimX)
    
    def implicit_scheme(self, T_initial, t_end, dt):
        implicit_A = np.eye(self.m * self.n) - dt * self.A
        implicit_B = T_initial - dt * self.B
        # Number of time steps
        steps = int(t_end/dt)
        print(f"Total time steps: {steps}")
        # 3D array to store temperature history
        if steps > 1:
            max_size = 1                                  # Maximum 1 time steps
        else:
            max_size = steps
        # Create a dictionary with indexes corresponding to time steps to save
        steps_to_save = {step: idx for idx, step in enumerate(
            np.linspace(1, steps, max_size+1, dtype=int)
        )}
        T_history = np.ndarray((max_size + 1, self.m, self.n))
        T_history[0, :, :] = T_initial.reshape(self.m, self.n)
        # Time-stepping loop
        for step in range(1, steps + 1):
            # Implicit backward Euler
            T_new = np.linalg.solve(implicit_A, implicit_B)
            # Save previous value for next iteration
            T_initial = T_new
            # Update B matrix
            implicit_B = T_initial - dt * self.B
            if (step in steps_to_save):
                # Get the appropriate index in T_history
                idx = steps_to_save.get(step, max_size)
                # Update field evolution
                T_history[idx, :, :] = T_new.reshape(self.m, self.n)
        return T_history
    
    def unsteady_solve(self, T_initial, t_end=1, dt=0.001, theta=0.5, boundries=None):
        """ 
        T_initial: Initial temperature field
        """
        # Copy the initial temperature fieldpass
        T_initial = T_initial.copy().flatten()
        # Build the stencil
        for i in range(self.m):
            for j in range(self.n):
                # Set stencil for the node
                k = self.diffFVM.index(i, j)
                a_diff, b_diff = self.diffFVM.set_stencil(i, j)
                a_conv, b_conv = self.convFVM.set_stencil(i, j)
                self.A[k, :] = a_diff + a_conv
                self.B[k] += b_diff + b_conv
        # Solve using implicit scheme
        self.normalize_matrix(boundries, self.convFVM.rho, self.convFVM.cp)
        T_history = self.implicit_scheme(T_initial, t_end, dt)
        return T_history
    
    def normalize_matrix(self, boundaries, rho_field=None, cp_field=None):
        rho_cp = rho_field * cp_field
        for i, boundary in enumerate(boundaries):
            # N, S, W, E
            if boundary == 'D':
                if i == 0:
                    rho_cp[0, :] = 1.0
                elif i == 1:
                    rho_cp[-1, :] = 1.0
                elif i == 2:  
                    rho_cp[:, 0] = 1.0
                elif i == 3:
                    rho_cp[:, -1] = 1.0
        # Flatten the rho_cp field to match matrix dimensions
        rho_cp_flat = rho_cp.flatten()
        # Normalize each row of A and corresponding B entry
        self.A /= rho_cp_flat[:, None]
        self.B /= rho_cp_flat
                
        

       