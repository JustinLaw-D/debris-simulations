# contains class for a single atmospheric layer (Cell)

import numpy as np
from BreakupModel import *

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)
Re = 6371 # radius of Earth (km)

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
        self.v_orbit = np.sqrt(G*Me/((self.alt+Re)*1000))/1000 # orbital speed (on average, km/s)
        self.alpha = alpha
        self.P = P
        self.logL_edges = logL_edges
        self.chi_edges = chi_edges
        self.num_L = self.N_bins.shape[0]
        self.num_chi = self.N_bins.shape[1]
        self.update_N_vals()

    def step(self, D_in, dt):
        '''
        calculates the number of collisions and decays from each debris bin, the number
        of decaying derelicts, and updates the numbers for S, D, C based on this

        Parameter(s):
        D_in : number of derelict satallites entering the cell from above in dt
        dt : time step (yr)

        Keyword Parameter(s): None

        Output(s):
        D_out : number of derelict satallites exiting the cell in dt
        N_out : matrix with the number of exiting debris from each bin in dt
        D_dt : number of collisions between satallites in dt
        C_dt : matrix with the number of collisions from each bin in dt

        Note: Assumes that collisions with debris of L_cm < 10cm cannot be avoided
        '''
        
        S, D = self.S[-1], self.D[-1]

        # compute the number of collisions from each debris type
        coll_S = np.zeros(self.N_bins.shape, dtype=np.int64) # collisions with live satallites
        coll_SD = 0 # collisions between live and derelict satallites
        coll_D = np.zeros(self.N_bins.shape, dtype=np.int64) # collisions with derelict satallites
        decay_N = np.zeros(self.N_bins.shape, dtype=np.int64) # number of debris that decay
        decay_D = 0 # number of derelicts that decay
        coll_DD = 0 # number of collisions between 
        lethal_N = np.full(self.N_bins.shape, False) # whether or not each bin has lethal collisions

        # handle debris events
        for i in range(self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            for j in range(self.num_chi):
                ave_chi = (self.chi_edges[j] + self.chi_edges[j+1])/2
                nS_col, nD_col, nN_decay = self.N_events(S, D, self.N_bins[i,j], ave_L, self.tau_N[i,j], dt)
                if is_catastrophic(self.m_s, ave_L, ave_chi, self.v) : lethal_N[i,j] = True
                coll_S[i,j] = nS_col
                coll_D[i,j] = nD_col
                decay_N[i,j] = nN_decay
        
        # compute collisions involving derelicts
        coll_SD, coll_DD, decay_D = self.D_events(S, D, dt)

        # compute number of satallites launched and brought down
        sat_up, sat_down, sat_down_fail = self.S_events(S, dt)

        # sum everything up and update values
        # all satallite-satallite collisions are catastrophic
        self.C_l.append(self.C_l[-1] + coll_SD + coll_DD + np.sum(coll_S[lethal_N]) + np.sum(coll_D[lethal_N]))
        self.C_nl.append(self.C_nl[-1] + np.sum(coll_S[lethal_N == False]) + np.sum(coll_D[lethal_N == False]))
        tot_coll_lS, tot_coll_nlS = np.sum(coll_S[lethal_N]), np.sum(coll_S[lethal_N == False])
        tot_coll_lD = np.sum(coll_D[lethal_N])
        self.S.append(S + sat_up - sat_down - tot_coll_lS - tot_coll_nlS - coll_SD)
        self.D.append(D + sat_down_fail + D_in + tot_coll_nlS - tot_coll_lD - coll_SD - 2*coll_DD - decay_D)
        return decay_D, decay_N, coll_SD + coll_DD, coll_S + coll_D

    def N_events(self, S, D, N, ave_L, tau, dt):
        '''
        calculates the number of collisions between debris with the given
        ave_L, ave_chi, and tau with stallites in a band, as well as the amount of
        debris that decays out of the band, in a given time step dt.

        Parameter(s):
        S : number of live satellites in the band
        D : number of derelict satellites in the band
        N : number of pieces of debris in the band, of this particular type
        ave_L : average characteristic length of the debris (m)
        tau : atmospheric drag lifetime of the debris in the band (yr)
        dt : time period (yr)

        Keyword Parameter(s): None
        
        Output(s):
        nS_col : number of collisions between debris and live satellites in the time step
        nD_col : number of collisions between debris and derelict satellites in the time step
        nN_decay : number of debris that decay out of the shell in the time step
        '''
        sigma = self.sigma/1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = N/V # number density of the debris

        dSdt = 0 # rate of collisions between debris and satallites (live/derelict)
        dDdt = 0
        if ave_L < 10/100: # collisions cannot be avoided
            dSdt = n*sigma*v*S
        else: # collisions can be avoided
            dSdt = self.alpha*n*sigma*v*S
        dDdt = n*sigma*v*D
        nS_col = dSdt*dt # convert rates to number of collisions
        nD_col = dDdt*dt
        # randomly decide if a fractional collision occurs
        nS_col = rand_poisson(nS_col, mx=min(S, N))
        nD_col = rand_poisson(nD_col, mx=min(D, N))
        nN_decay_old = N/tau*dt # calculate decays
        nN_decay = rand_poisson(nN_decay_old, mx=N) # randomly decide if a fractional decay occurs
        return nS_col, nD_col, nN_decay

    def D_events(self, S, D, dt):
        '''
        calculates the number of collisions between derelicts with stallites in a band, 
        as well as the amount of derelicts that decays out of the band, in a given time step dt.

        Parameter(s):
        S : number of live satellites in the band
        D : number of derelict satellites in the band
        dt : time period (yr)

        Keyword Parameter(s): None
        
        Output(s):
        nDD_col : number of collisions between two derelicts in the time step
        nSD_col : number of collisions between a derelelict and live satallite in the time step
        nD_decay : number of derelicts that decay out of the shell in the time step
        '''
        sigma = 4*self.sigma/1e6 # convert to km^2, and account for increased cross-section
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = D/V # number density of the derelicts

        # rate of collisions between derelicts and satallites (live/derelict)
        dSDdt = self.alpha*n*sigma*v*S
        dDDdt = n*sigma*v*D # collisions cannot be avoided
        nSD_col = dSDdt*dt # convert rates to number of collisions
        nDD_col = dDDdt*dt
        # randomly decide if a fractional collision occurs
        nSD_col = rand_poisson(nSD_col, mx=min(S, D))
        nDD_col = rand_poisson(nDD_col, mx=D/2)
        nD_decay = D/self.tau_D*dt # calculate decays
        nD_decay = rand_poisson(nD_decay, mx=D) # randomly decide if a fractional decay occurs
        return nSD_col, nDD_col, nD_decay

    def S_events(self, S, dt):
        '''
        calculates the number satallites launched, that attempt to de-orbit, and that
        fail to de-orbit in dt

        Parameter(s):
        S : number of live satellites in the band
        dt : time period (yr)

        Keyword Parameter(s): None
        
        Output(s):
        sat_up : number of satallites launched in the time step
        sat_down : number of satallites that attempt to de-orbit in the time step
        sat_down_fail : number of satallites that fail a de-orbit attempt in the time step
        '''

        sat_up = rand_poisson(self.lam*dt)
        sat_down = rand_poisson(S/self.del_t*dt, mx=S)
        sat_down_fail = rand_poisson(sat_down*(1-self.P), mx=sat_down)

        return sat_up, sat_down, sat_down_fail

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
