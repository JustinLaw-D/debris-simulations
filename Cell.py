# contains class for a single atmospheric layer (Cell), and satallites (functions more as a struct)

import numpy as np
from BreakupModel import *
G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)
Re = 6371 # radius of Earth (km)

class Cell:
    
    def __init__(self, S_i, N_i, logL_edges, chi_edges, alt, dh, tau_N, N_factor_table, v=None, tau_min=None):
        '''Constructor for Cell class
    
        Parameter(s):
        S_i : list of satellite types with initial values
        N_i : initial array of number of debris by L and A/M
        logL_edges : bin edges in log10 of characteristic length (log10(m))
        chi_edges : bin edges in log10(A/M) (log10(m^2/kg))
        alt : altitude of the shell centre (km)
        dh : width of the shell (km)
        tau_N : array of atmospheric drag lifetimes for debris (yr)
        N_factor_table : same dimention as tau_N, 0s for ignored bins, 1 for non-ignored bins

        Keyword Parameter(s):
        v : relative collision speed (km/s, default 10km/s)
        tau_min : minimum drag lifetime of debris to consider (yr, default 1/10)

        Output(s):
        Cell instance
        '''

        # set default values as needed
        if v == None:
            v = 10
        if tau_min == None:
            tau_min = 1e-1

        # setup initial values for tracking live satallites, derelict satallites,
        # lethat debris, and non-lethal debris over time
        self.satellites = S_i
        self.num_types = len(self.satellites)
        self.N_bins = [N_i]

        # setup other variables
        self.C_l = [0] # lethal collisions
        self.C_nl = [0] # non-lethal collisions
        self.alt = alt
        self.dh = dh
        self.tau_N = tau_N
        self.N_factor_table = N_factor_table
        self.v = v
        self.v_orbit = np.sqrt(G*Me/((Re + alt)*1000))/1000 # orbital velocity in km/s
        self.logL_edges = logL_edges
        self.chi_edges = chi_edges
        self.num_L = self.N_bins[0].shape[0]
        self.num_chi = self.N_bins[0].shape[1]
        self.lethal_N = []
        for i in range(self.num_types):
            self.lethal_N.append(np.full(self.N_bins[0].shape, False)) # whether or not each bin has lethal collisions
        self.update_lethal_N()

    def dxdt_cell(self, time, S_din, D_in):
        '''
        calculates the rate of collisions and decays from each debris bin, the rate
        of decaying/de-orbiting satellites, the rate of launches/deorbit starts of satallites, 
        and the rate of creation of derelicts at the given time

        Parameter(s):
        time : index of the values to use
        S_din : rate of de-orbiting satellites of each type entering the cell from above (yr^(-1))
        D_in : rate of derelict satellites of each type entering the cell from above (yr^(-1))

        Keyword Parameter(s): None

        Output(s):
        dSdt : list of rate of change of the number of live satellites in the cell of each type (yr^(-1))
        dS_ddt : list of rate of change of the number of de-orbiting satellites in the cell of each type (yr^(-1))
        dDdt : list of rate of change of the number of derelict satellites in the cell of each type (yr^(-1))
        S_dout : list of rate of satellites de-orbiting from the cell of each type (yr^(-1))
        D_out : list of rate of satellites decaying from the cell of each type (yr^(-1))
        N_out : matrix with the rate of exiting debris from each bin (yr^(-1))
        D_dt : list of matrices of rates of collisions between satellites (yr^(-1))
        C_dt : list of matrices with the rate of collisions from each bin with each satellite type (yr^(-1))

        Note: Assumes that collisions with debris of L_cm < 10cm cannot be avoided, and
        that the given time input is valid
        '''
        
        N = self.N_bins[time]

        # compute the rate of collisions from each debris type
        dSdt = [] # collisions with live satallites
        dSDdt = np.zeros((self.num_types, self.num_types)) # collisions between live and derelict satallites
        # first index is live satellite type, second is derelict type
        dS_ddt = [] # collisions with de-orbiting satallites
        dS_dDdt = np.zeros((self.num_types, self.num_types)) # collisions between de-orbiting and derelict satallites
        dDdt = [] # collisions with derelict satallites
        dDDdt = np.zeros((self.num_types, self.num_types)) # number of collisions between derelict satallites
        decay_N = np.zeros(N.shape) # rate of debris that decay
        kill_S = [] # rate of satellites being put into de-orbit
        deorbit_S = [] # rate of satellites de-orbiting out of the band
        decay_D = [] # rate of derelicts that decay
        dSdt_tot = [] # total rate of change for live satellites
        dS_ddt_tot = [] # total rate of change for de-orbiting satellites
        dDdt_tot = [] # total rate of change of derelict satellites
        D_dt = []
        C_dt = []
        for i in range(self.num_types):
            dSdt.append(np.zeros(N.shape))
            dS_ddt.append(np.zeros(N.shape))
            dDdt.append(np.zeros(N.shape))
            kill_S.append(0)
            deorbit_S.append(0)
            decay_D.append(0)

        for i in range(self.num_types):

            # get current satellite type values
            S = self.satellites[i].S[time]
            S_d = self.satellites[i].S_d[time]
            D = self.satellites[i].D[time]
            sigma = self.satellites[i].sigma
            alpha = self.satellites[i].alpha

            # handle debris events
            for j in range(self.num_L):
                ave_L = 10**((self.logL_edges[j] + self.logL_edges[j+1])/2) # average L value for these bins
                for k in range(self.num_chi):
                    if self.N_factor_table[j,k] != 0: # only calculate for non-ignored bins
                        dSdt[i][j,k], dS_ddt[i][j,k], dDdt[i][j,k], decay_N[j,k] = self.N_events(S, S_d, D, N[j,k], sigma, alpha, ave_L, self.tau_N[j,k])
        
            # compute colisions involving only satellities
            tot_S_sat_coll = 0 # total collisions destroying live satellites of this type
            tot_Sd_sat_coll = 0 # total collisions destroying de-orbiting satellites of this type
            tot_D_sat_coll = 0 # total collisions destroying derelicts of this type
            for j in range(self.num_types):
                D2 = self.satellites[j].D[time]
                sigma2 = self.satellites[j].sigma
                dSDdt[i,j], dS_dDdt[i,j], dDDdt[i,j] = self.SColl_events(S, S_d, D, sigma, alpha, D2, sigma2)
                if i > j: dDDdt[i,j] = 0 # avoid double counting
                tot_S_sat_coll += dSDdt[i,j]
                tot_Sd_sat_coll += dS_dDdt[i,j]
                if i == j : tot_D_sat_coll += dSDdt[i,j] + dS_dDdt[i,j] + 2*dDDdt[i,j]
                else : tot_D_sat_coll += dSDdt[i,j] + dS_dDdt[i,j] + dDDdt[i,j]

            # compute decay events for satellites
            del_t = self.satellites[i].del_t
            tau_do = self.satellites[i].tau_do
            tau = self.satellites[i].tau_
            kill_S[i], deorbit_S[i], decay_D[i] = S/del_t, S_d/tau_do, D/tau

            # sum everything up
            lam, P = self.satellites[i].lam, self.satellites[i].P
            dSdt_tot.append(lam - kill_S[i] - np.sum(dSdt[i]) - tot_S_sat_coll)
            dS_ddt_tot.append(S_din + P*kill_S[i] - np.sum(dS_ddt[i]) - deorbit_S[i] - tot_Sd_sat_coll)
            dDdt_tot.append(D_in + (1-P)*kill_S[i] - np.sum(dDdt[i]) - decay_D[i] - tot_D_sat_coll
                            + np.sum(dSdt[i][self.lethal_N[i] == False]) + np.sum(dS_ddt[i][self.lethal_N[i] == False]))
            D_dt.append(np.sum(dSDdt) + np.sum(dS_dDdt) + np.sum(dDDdt))
            C_dt.append(dSdt[i] + dS_ddt[i] + dDdt[i])

        return dSdt_tot, dS_ddt_tot, dDdt_tot, deorbit_S, decay_D, decay_N, D_dt, C_dt

    def N_events(self, S, S_d, D, N, sigma, alpha, ave_L, tau):
        '''
        calculates the rate of collisions between debris with the given
        ave_L, ave_chi, and tau with stallites of a particular type in a band, 
        as well as the rate of debris that decays out of the band

        Parameter(s):
        S : number of live satellites in the band of this type
        S_d : number of de-orbiting satellites in the band of this type
        D : number of derelict satellites in the band of this type
        N : number of pieces of debris in the band, of this particular type
        sigma : cross section of the satellites (m^2)
        alpha : fraction of collisions a live satellites fails to avoid
        ave_L : average characteristic length of the debris (m)
        tau : atmospheric drag lifetime of the debris in the band (yr)

        Keyword Parameter(s): None
        
        Output(s):
        dSdt : rate of collisions between debris and live satellites (1/yr)
        dS_ddt : rate of collisions between debris and de-orbiting satellites (1/yr)
        dDdt : rate of collisions between debris and derelict satellites (1/yr)
        dNdt : rate of debris that decay out of the shell in the time step (1/yr)
        '''
        sigma /= 1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = N/V # number density of the debris

        dSdt = 0 # rate of collisions between debris and satallites (live/derelict)
        dS_ddt = 0
        if ave_L < 10/100: # collisions cannot be avoided
            dSdt = n*sigma*v*S
            dS_ddt = n*sigma*v*S_d
        else: # collisions can be avoided
            dSdt = alpha*n*sigma*v*S
            dS_ddt = alpha*n*sigma*v*S_d
        dDdt = n*sigma*v*D
        dNdt = N/tau # calculate decays
        return dSdt, dS_ddt, dDdt, dNdt

    def SColl_events(self, S1, S_d1, D1, sigma1, alpha1, D2, sigma2):
        '''
        calculates the rate of collisions between satellites of two particular types
        in a band

        Parameter(s):
        S1 : number of live satellites of type 1
        S_d1 : number of de-orbiting satellites of type 1
        D1 : number of derelict satellites of type 1
        sigma1 : cross-section of satellites of type 1 (m^2)
        alpha1 : fraction of collisions a live satellites of type 1 fails to avoid
        D2 : number of derelict satellites of type 2
        sigma2 : cross-section of satellites of type 2 (m^2)

        Keyword Parameter(s): None

        Output(s):
        dSDdt : rate of collision between live satellites of type 1 and derelicts of type 2 (1/yr)
        dS_dDdt : rate of collision between de-orbiting satellites of type 1 and derelicts of type 2 (1/yr)
        dDDdt : rate of collision between derelict satellites of type 1 and derelicts of type 2 (1/yr)
        '''

        sigma1 /= 1e6 # convert to km^2
        sigma2 /= 1e6
        sigma = sigma1 + sigma2 + 2*np.sqrt(sigma1*sigma2) # account for increased cross-section
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = D2/V # number density of the derelicts

        # rate of collisions between derelicts and satallites (live/derelict)
        dSDdt = alpha1*n*sigma*v*S1
        dS_dDdt = alpha1*n*sigma*v*S_d1
        dDDdt = n*sigma*v*D1  # collisions cannot be avoided
        return dSDdt, dS_dDdt, dDDdt

    def update_lethal_N(self):
        '''
        updates values in lethal_N based on current m_s, v, and bins

        Parameter(s): None

        Keyword Parameter(s): None

        Ouput(s): None
        '''

        for i in range(self.num_types):
            m_s = self.satellites[i].m_s
            for j in range(self.num_L):
                ave_L = 10**((self.logL_edges[j] + self.logL_edges[j+1])/2) # average L value for these bins
                for k in range(self.num_chi):
                    ave_chi = (self.chi_edges[k] + self.chi_edges[k+1])/2
                    self.lethal_N[i][j,k] = is_catastrophic(m_s, ave_L, 10**ave_chi, self.v)

class Satellite:

    def __init__(self, S_i, S_di, D_i, m, sigma, lam, del_t, tau_do, alpha, P, AM, tau):
        '''
        constructor method for Satellite class

        Parameter(s):
        S_i : initial number of live satellites of this type
        S_di : initial number of de-orbiting satellites of this type
        D_i : initial number of derelict satellites of this type
        m : mass of each satellite (kg)
        sigma : collision cross-section of each satellite (m^2)
        lam : launch rate of the satellites (1/yr)
        del_t : mean satellite lifetime (yr)
        tau_do : mean time for satellite to de-orbit from shell (yr)
        alpha : fraction of collisions a live satellites fails to avoid
        P : post-mission disposal probability
        AM : area-to-mass ratio of the satellite (m^2/kg)
        tau : atmospheric drag lifetime of a satellite (yr)

        Keyword Parameter(s): None

        Output(s): Instance of Satellite class

        Note(s): preforms no validity checks on given values
        '''

        self.S = [S_i]
        self.S_d = [S_di]
        self.D = [D_i]
        self.m = m
        self.sigma = sigma
        self.lam = lam
        self.del_t = del_t
        self.tau_do = tau_do
        self.alpha = alpha
        self.P = P
        self.AM = AM
        self.tau = tau