# contains class for a single atmospheric layer (Cell)

import numpy as np
from BreakupModel import *

class Cell:
    
    def __init__(self, S_i, D_i, N_i, logL_edges, chi_edges, lam, alt, dh, tau_D, tau_N, del_t=None, sigma=None, 
                 m_s=None, v=None, alpha=None, P=None):
        '''Constructor for Cell class
    
        Parameter(s):
        S_i : initial number of live satallites
        D_i : initial number of derelict satallites
        N_i : initial array of number of debris by L and A/M
        logL_edges : bin edges in log10 of characteristic length (log10(m))
        chi_edges : bin edges in log10(A/M) (log10(m^2/kg))
        lam : launch rate of satellites into the shell (1/yr)
        alt : altitude of the shell centre (km)
        dh : width of the shell (km)
        tau_D : atmospheric drag lifetime for derelicts (yr)
        tau_N : array of atmospheric drag lifetimes for debris (yr)

        Keyword Parameter(s):
        del_t : mean satellite lifetime (yr, default 5yr)
        sigma : satellite cross-section (m^2, default 10m^2)
        m_s : satellite mass (kg, default 250kg)
        v : relative collision speed (km/s, default 10km/s)
        alpha : fraction of collisions a live satellites fails to avoid (default 0.2)
        P : post-mission disposal probability (default 0.95)

        Output(s):
        Cell instance

        Notes: By default, it's assumed that non-lethal debris has L_c < 10cm, and lethal debris has L > 10cm.
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

        # setup initial values for tracking live satallites, derelict satallites,
        # lethat debris, and non-lethal debris over time
        self.S, self.D = [S_i], [D_i]
        self.N_bins = N_i
        self.N_l, self.N_nl = [], []

        # setup other variables
        self.C_l = [0] # lethal collisions
        self.C_nl = [0] # non-lethal collisions
        self.lam = lam
        self.m_s = m_s
        self.alt = alt
        self.dh = dh
        self.tau_D = tau_D
        self.tau_N = tau_N
        self.del_t = del_t
        self.sigma = sigma
        self.v = v
        self.alpha = alpha
        self.P = P
        self.logL_edges = logL_edges
        self.chi_edges = chi_edges
        self.num_L = self.N_bins.shape[0]
        self.num_chi = self.N_bins.shape[1]
        self.update_N_vals()
        self.lethal_N = np.full(self.N_bins.shape, False) # whether or not each bin has lethal collisions
        self.update_lethal_N()

    def dxdt_cell(self, D_in):
        '''
        calculates the rate of collisions and decays from each debris bin, the rate
        of decaying derelicts, the rate of launches/deorbits of satallites, and the
        rate of creation of derelicts. also updates collision totals in the cell

        Parameter(s):
        D_in : rate of derelict satallites entering the cell from above (yr^(-1))

        Keyword Parameter(s): None

        Output(s):
        dSdt : rate of change of the number of live satallites in the cell (yr^(-1))
        dDdt : rate of change of the number of derelict satallites in the cell (yr^(-1))
        D_out : rate of derelict satallites exiting the cell (yr^(-1))
        N_out : matrix with the rate of exiting debris from each bin (yr^(-1))
        D_dt : rate of collisions between satallites (yr^(-1))
        C_dt : matrix with the rate of collisions from each bin (yr^(-1))

        Note: Assumes that collisions with debris of L_cm < 10cm cannot be avoided
        '''
        
        S, D = self.S[-1], self.D[-1]

        # compute the rate of collisions from each debris type
        dSdt = np.zeros(self.N_bins.shape, dtype=np.int64) # collisions with live satallites
        dSDdt = 0 # collisions between live and derelict satallites
        dDdt = np.zeros(self.N_bins.shape, dtype=np.int64) # collisions with derelict satallites
        decay_N = np.zeros(self.N_bins.shape, dtype=np.int64) # rate of debris that decay
        decay_D = 0 # rate of derelicts that decay
        dDDdt = 0 # number of collisions between derelict satallites
        lethal_N = np.full(self.N_bins.shape, False) # whether or not each bin has lethal collisions

        # handle debris events
        for i in range(self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            for j in range(self.num_chi):
                ave_chi = (self.chi_edges[j] + self.chi_edges[j+1])/2
                dSdt_loc, dDdt_loc, decay_N_loc = self.N_events(S, D, self.N_bins[i,j], ave_L, self.tau_N[i,j])
                if is_catastrophic(self.m_s, ave_L, ave_chi, self.v) : lethal_N[i,j] = True
                dSdt[i,j] = dSdt_loc
                dDdt[i,j] = dDdt_loc
                decay_N[i,j] = decay_N_loc
        
        # compute collisions involving derelicts
        dSDdt, dDDdt, decay_D = self.D_events(S, D)

        # compute rate of satallites launched and brought down
        sat_up, sat_down, sat_down_fail = self.lam, S/self.del_t, (1-self.P)*S/self.del_t

        # sum everything up
        D_dt = dSDdt + dDDdt
        C_dt = dSdt + dDdt
        dSdt = sat_up - sat_down - np.sum(dSdt) - dSDdt
        dDdt = sat_down_fail - decay_D + D_in - np.sum(dDdt[lethal_N == False]) - dSDdt - 2*dDDdt
        return dSdt, dDdt, decay_D, decay_N, D_dt, C_dt

    def N_events(self, S, D, N, ave_L, tau):
        '''
        calculates the rate of collisions between debris with the given
        ave_L, ave_chi, and tau with stallites in a band, as well as the rate of
        debris that decays out of the band

        Parameter(s):
        S : number of live satellites in the band
        D : number of derelict satellites in the band
        N : number of pieces of debris in the band, of this particular type
        ave_L : average characteristic length of the debris (m)
        tau : atmospheric drag lifetime of the debris in the band (yr)

        Keyword Parameter(s): None
        
        Output(s):
        dSdt : rate of collisions between debris and live satellites
        dDdt : rate of collisions between debris and derelict satellites
        dNdt : rate of debris that decay out of the shell in the time step
        '''
        sigma = self.sigma/1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = N/V # number density of the debris

        dSdt = 0 # rate of collisions between debris and satallites (live/derelict)
        if ave_L < 10/100: # collisions cannot be avoided
            dSdt = n*sigma*v*S
        else: # collisions can be avoided
            dSdt = self.alpha*n*sigma*v*S
        dDdt = n*sigma*v*D
        dNdt = N/tau # calculate decays
        return dSdt, dDdt, dNdt

    def D_events(self, S, D):
        '''
        calculates the rate of collisions between derelicts with stallites in a band, 
        as well as the rate of derelicts that decays out of the band

        Parameter(s):
        S : number of live satellites in the band
        D : number of derelict satellites in the band

        Keyword Parameter(s): None
        
        Output(s):
        dDDdt : rate of collisions between two derelicts ((yr)^-1)
        dSDdt : rate of collisions between a derelelict and live satallite ((yr)^-1)
        dDdt : rate of derelicts that decay out of the shell ((yr)^-1)
        '''
        sigma = 4*self.sigma/1e6 # convert to km^2, and account for increased cross-section
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = D/V # number density of the derelicts

        # rate of collisions between derelicts and satallites (live/derelict)
        dSDdt = n*sigma*v*S # collisions cannot be avoided
        dDDdt = n*sigma*v*D 
        dDdt = D/self.tau_D # calculate decays
        return dSDdt, dDDdt, dDdt

    def update_N_vals(self):
        '''
        updates N_l, N_nl values based on current N values

        Parameter(s): None

        Keyword Parameter(s): None

        Output(s): None
        '''
    
        new_Nl, new_Nnl = 0, 0
        for i in range(self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            if ave_L >= 10 : new_Nl += np.sum(self.N_bins[i, :])
            else : new_Nnl += np.sum(self.N_bins[i, :])

        self.N_l.append(new_Nl)
        self.N_nl.append(new_Nnl)

    def update_lethal_N(self):
        '''
        updates values in lethal_N based on current m_s, v, and bins

        Parameter(s): None

        Keyword Parameter(s): None

        Ouput(s): None
        '''

        for i in range(self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            for j in range(self.num_chi):
                ave_chi = (self.chi_edges[j] + self.chi_edges[j+1])/2
                self.lethal_N[i,j] = is_catastrophic(self.m_s, ave_L, ave_chi, self.v)
