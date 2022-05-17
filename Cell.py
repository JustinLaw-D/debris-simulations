# contains class for a single atmospheric layer (Cell)

import numpy as np

class Cell:
    
    def __init__(self, x, lam, alt, dh, tau, del_t=None, sigma=None, v=None, 
                delta=None, alpha=None, P=None, N_0=None):
        '''Constructor for Cell class
    
        Parameter(s):
        x : array with initial S, D, N, C values (live satellites, derelict satellites, lethal debris)
        alt : altitude of the shell centre (km)
        dh : width of the shell (km)
        lam : launch rate of satellites into the shell (1/yr)
        tau : atmospheric drag lifetime

        Keyword Parameter(s):
        del_t : mean satellite lifetime (yr, default 5yr)
        sigma : satellite cross-section (m^2, default 10m^2)
        v : relative collision speed (km/s, default 10km/s)
        delta : ratio of the density of disabling to lethal debris (default 10)
        alpha : fraction of collisions a live satellites fails to avoid (default 0.2)
        P : post-mission disposal probability (default 0.95)
        N_0 : number of lethal debris fragments from a collision (default 100)

        Output(s):
        Cell instance
        '''

        # set default values as needed
        if del_t == None:
            del_t = 5
        if sigma == None:
            sigma = 10
        if v == None:
            v = 10
        if delta == None:
            delta = 10
        if alpha == None:
            alpha = 0.2
        if P == None:
            P = 0.95
        if N_0 == None:
            N_0 = 100

        self.pastx = [x] # list of all past x-values
        self.lam = lam
        self.alt = alt
        self.dh = dh
        self.tau = tau
        self.del_t = del_t
        self.sigma = sigma
        self.v = v
        self.delta = delta
        self.alpha = alpha
        self.P = P
        self.N_0 = N_0

    def dxdt_cell(self, x):
        '''
        Calculates the rate of change of x given the x of this cell, ignoring contributions
        from the cell above

        Parameter(s):
        x : current x-value of this cell

        Output(s):
        dxdt_cell : array of [dSdt_cell, dDdt_cell, dNdt_cell, dCdt_cell]
        '''

        sigma = self.sigma/1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        S, D, N = x[0], x[1], x[2] # get current values

        # run calculation
        n = (N + D/2)/V
        dSdt_cell = self.lam - S/self.del_t - (self.delta + self.alpha)*n*sigma*v*S
        dDdt_cell = (1-self.P)*S/self.del_t + self.delta*n*sigma*v*S - n*sigma*v*D - D/self.tau
        dNdt_cell = n*sigma*v*self.N_0*(self.alpha*S + D) - N/self.tau
        dCdt_cell = (self.delta + self.alpha)*n*sigma*v*S

        return np.array([dSdt_cell, dDdt_cell, dNdt_cell, dCdt_cell])

    def dxdt_out(self, x):
        '''
        Calculates the rate of change of x leaving the cell into the one below

        Parameter(s):
        x : current x-value of this cell

        Output(s):
        dxdt_out : array of [dSdt_out, dDdt_out, dNdt_out, dCdt_out]
        '''
        
        D, N = x[1], x[2] # get current values

        # run calculation
        dSdt_out = 0
        dDdt_out = D/self.tau
        dNdt_out = N/self.tau
        dCdt_out = 0

        return np.array([dSdt_out, dDdt_out, dNdt_out, dCdt_out])