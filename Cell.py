# contains class for a single atmospheric layer (Cell)

import numpy as np
from BreakupModel import *

class Cell:
    
    def __init__(self, S_i, D_i, N_li, lam, alt, dh, tau, del_t=None, sigma=None, m_s=None, v=None, alpha=None, 
                 P=None, N_nli=None, num_L=None, L_min=None, L_max=None, num_chi=None, chi_min=None, chi_max=None):
        '''Constructor for Cell class
    
        Parameter(s):
        S_i : initial number of live satallites
        D_i : initial number of derelict satallites
        N_li : initial number of lethal debris
        alt : altitude of the shell centre (km)
        dh : width of the shell (km)
        lam : launch rate of satellites into the shell (1/yr)
        tau : atmospheric drag lifetime

        Keyword Parameter(s):
        del_t : mean satellite lifetime (yr, default 5yr)
        sigma : satellite cross-section (m^2, default 10m^2)
        m_s : satellite mass (kg, default 250kg)
        v : relative collision speed (km/s, default 10km/s)
        alpha : fraction of collisions a live satellites fails to avoid (default 0.2)
        P : post-mission disposal probability (default 0.95)
        N_nli : initial amount of non-lethal debris (default 10x N_l)
        num_L : number of bins for characteristic length (default 10)
        L_min : minimum characterisic length (m, default 1/1000)
        L_max : maximum characteristic length (m, default 1)
        num_chi : number of bins for log10(A/M) (default 10)
        chi_min : minimum log10(A/M) (m, default -2log(m^2/kg))
        chi_max : maximum log10(A/M) (m, default 2log(m^2/kg))

        Output(s):
        Cell instance

        Notes: By default, it's assumed that there is 10 times more non-lethal debris (assume L_c < 10cm) than 
        letha debris (assume L > 10cm). The debris is then randomly distributed amoungst the bins under these 
        assumptions.
        '''

        # set default values as needed
        if del_t == None:
            del_t = 5
        if sigma == None:
            sigma = 10
        if m_s == None:
            m_s = 250
        if v == None:
            v = 10
        if alpha == None:
            alpha = 0.2
        if P == None:
            P = 0.95
        if N_nli == None:
            N_nli = 10*N_li
        if num_L == None:
            num_L = 10
        if L_min == None:
            L_min = 1/1000
        if L_max == None:
            L_max = 1
        if num_chi == None:
            num_chi = 10
        if chi_min == None:
            chi_min = -2
        if chi_max == None:
            chi_max = 2

        # setup initial values for tracking live satallites, derelict satallites,
        # lethat debris, and non-lethal debris over time
        self.S, self.D, self.N_l, self.N_nl = [S_i], [D_i], [N_li], [N_nli]

        # setup other variables
        self.C = 0
        self.lam = lam
        self.m_s = m_s
        self.alt = alt
        self.dh = dh
        self.tau = tau
        self.del_t = del_t
        self.sigma = sigma
        self.v = v
        self.alpha = alpha
        self.P = P
        self.num_L = num_L
        self.num_chi = num_chi

        # generate bins for log10(L), chi
        self.logL_edges = np.linspace(np.log10(L_min), np.log10(L_max), num=num_L+1)
        self.chi_edges = np.linspace(chi_min, chi_max, num=num_chi+1)

        # generate array for holding the current debris distribution (log10(L) is row, chi is column)
        self.N_bins = np.zeros((num_L, num_chi))

        # generate initial distributions
        lethal_L = np.log10(randL_coll(self.N_l, 1e-1, L_max))
        nlethal_L = np.log10(randL_coll(self.N_nl, L_min, 1e-2))
        for i in range(num_L):
            bin_L = 0
            bin_bot_L, bin_top_L = self.logL_edges[i], self.logL_edges[i+1]
            bin_L += len(lethal_L[bin_bot_L < lethal_L < bin_top_L])
            bin_L += len(nlethal_L[bin_bot_L < nlethal_L < bin_top_L])
            chi_dist = randX_coll(bin_L, chi_min, chi_max, (bin_bot_L + bin_top_L)/2)
            for j in range(num_chi):
                bin_bot_chi, bin_top_chi = self.chi_edges[i], self.chi_edges[i+1]
                self.N_bins[i,j] = len(chi_dist[bin_bot_chi < chi_dist < bin_top_chi])

    def step(self, D_in, dt):
        '''
        TODO: SPLIT THIS INTO A TON OF SEPERATE FUNCTIONS
        calculates the number of collisions and decays from each debris bin, the number
        of decaying derelicts, and updates the numbers for S, D, C based on this

        Parameter(s):
        D_in : number of derelict satallites entering the cell from above in dt
        dt : time step (yr)

        Keyword Parameter(s): None

        Output(s):
        D_out : number of derelict satallites exiting the cell in dt
        N_out : matrix with the number of exiting debris from each bin in dt
        D_dt : number of collisions between derelicts in dt
        C_dt : matrix with the number of collisions from each bin in dt

        Note: Assumes that collisions with debris of L_cm < 10cm cannot be avoided
        '''

        sigma = self.sigma/1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        S, D = self.S[-1], self.D[-1]
        S_change, D_change = 0, 0 # amount S, D change in the time step

        # compute the number of collisions from each debris type
        coll_S = np.zeros(self.N_bins.shape) # collisions with live satallites
        coll_DS = 0
        coll_D = np.zeros(self.N_bins.shape) # collisions with derelict satallites
        decay_N = np.zeros(self.N_bins.shape) # number of debris that decay
        decay_D = 0 # number of derelicts that decay
        coll_DD = 0
        rand = np.random.uniform() # random number to choose if fractional events occur
        for i in (self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            for j in (self.num_chi):
                n = self.N_bins[i,j]/V
                dSCdt = 0
                dDCdt = 0
                if ave_L < 10/100: # collisions cannot be avoided
                    dSCdt = n*sigma*v*S
                else:
                    dSCdt = self.alpha*n*sigma*v*S
                dDCdt = n*sigma*v*D
                nS_col = dSCdt*dt
                nD_col = dDCdt*dt
                # randomly decide if a fractional collision occurs
                if rand > (nS_col - int(nS_col)):
                    coll_S[i,j] = int(nS_col)
                else:
                    coll_S[i,j] = int(nS_col) + 1
                # randomly decide if a fractional collision occurs
                if rand > (nD_col - int(nD_col)):
                    coll_D[i,j] = int(nD_col)
                else:
                    coll_D[i,j] = int(nD_col) + 1
                nN_decay = self.N_bins[i,j]/self.tau*dt # calculate decays
                # randomly decide if a fractional decay occurs
                if rand > (nN_decay - int(nN_decay)):
                    decay_N[i,j] = int(nN_decay)
                else:
                    decay_N[i,j] = int(nN_decay) + 1
        
        # compute collisions involving derelicts
        n = D/V
        nSD_col = self.alpha*n*sigma*v*S*dt
        n /= 2 # avoid double counting
        nDD_col = n*sigma*v*D*dt
        # randomly decide if a fractional collision occurs
        if rand > (nSD_col - int(nSD_col)):
            coll_DS = int(nSD_col)
        else:
            coll_DS = int(nSD_col) + 1
        # randomly decide if a fractional collision occurs
        if rand > (nDD_col - int(nDD_col)):
            coll_DD = int(nDD_col)
        else:
            coll_DD = int(nDD_col) + 1
        decay_D = D/self.tau*dt # calculate decays
        # randomly decide if a fractional decay occurs
        if rand > (decay_D - int(decay_D)):
            decay_D = int(decay_D)
        else:
            decay_D = int(decay_D) + 1

        # compute number of satallites launched and brought down
        sat_up = self.lam*dt
        # randomly decide if a fractional launch occurs
        if rand > (sat_up - int(sat_up)):
            sat_up = int(sat_up)
        else:
            sat_up = int(sat_up) + 1
        sat_down = S/self.del_t*dt
        # randomly decide if a fractional de-orbit occurs
        if rand > (sat_down - int(sat_down)):
            sat_down = int(sat_down)
        else:
            sat_down = int(sat_down) + 1
        sat_down_fail = (1-self.P)*sat_down
        if rand > (sat_down_fail - int(sat_down_fail)):
            sat_down_fail = int(sat_down_fail)
        else:
            sat_down_fail = int(sat_down_fail) + 1

        # sum everything up and update values
        C_dt = coll_S + coll_D
        tot_coll_S, tot_coll_D = np.sum(coll_S) + coll_DS, np.sum(coll_D) + coll_DD
        self.S.append(S + sat_up - sat_down - tot_coll_S)
        self.D.append(D + sat_down_fail + )

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