# contains class for collection of cells representing orbital shells

from Cell import *
import numpy as np
import matplotlib.pyplot as plt
from BreakupModel import *
from copy import deepcopy

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Re = 6371 # radius of Earth (km)
Me = 5.97219e24 # mass of Earth (kg)

class NCell:

    def __init__(self, S, S_d, D, N_l, alts, dh, lam, drag_lifetime, del_t=None, sigma=None, v=None, 
                delta=None, alpha=None, P=None, m_s=None, AM_sat=None, tau_do=None, tau_min=None, 
                L_min=1e-3, L_max=1, num_L=10, chi_min=-2, chi_max=2, num_chi=10, num_dir=100):
        '''
        Constructor for NCell class
    
        Parameter(s):
        S : list of initial number of live satellites in each shell of each type (list of arrays)
        S_d : list of initial number of deorbiting satellites in each shell of each type (list of arrays)
        D : list of initial number of derelict satellites in each shell of each type (list of arrays)
        N_l : initial number of catestrophically lethal debris in each shell (array)
        alts : altitude of the shell's centre (array, km)
        dh : width of the shells (array, km)
        lam : launch rate of satellites of each type into the each shell (list of arrays, 1/yr)
        drag_lifetime : function that computes atmospheric drag lifetime ((km, km, m^2/kg) -> yr)

        Keyword Parameter(s):
        del_t : mean satellite lifetime of each type in each shell (list of lists, yr, default 5yr)
        sigma : satellite cross-section of each type in each shell (list of lists, m^2, default 10m^2)
        v : relative collision speed in each shell (list, km/s, default 10km/s)
        delta : initial ratio of the density of disabling to catestrophic debris in each shell (list, default 10)
        alpha : fraction of collisions a live satellites of each type fails to avoid in each shell (list of lists, default 0.2)
        P : post-mission disposal probability for satellites of each type in each shell (list of lists, default 0.95)
        m_s : mass of the satallites of eachy type in each band (list of lists, kg, default 250kg)
        AM_sat : area-to-mass ratio of the satallites of each type in each shell (list of lists, m^2/kg, default 1/(20*2.2)m^2/kg)
        tau_do : average deorbiting time for satellites of each type in each shell (list of lists, yr, default decay_time/10)
        tau_min : minimum decay lifetimes to consider for debris (list, yr, default 1/10yr)
        L_min : minimum characteristic length to consider (m, default 1mm)
        L_max : maximum characteristic length to consider (m, default 1m)
        num_L : number of debris bins in characteristic length (default 10)
        chi_min : minimum log10(A/M) to consider (log10(m^2/kg), default -3)
        chi_max : maximum log10(A/M) to consider (log10(m^2/kg), default 3)
        num_chi : number of debris bins in log10(A/M) (default 10)
        num_dir : number of random directions to sample in creating probability tables (default 100)

        Output(s):
        NCell instance

        Note: no size checks are done on the arrays, the program will crash if any of the arrays differ in size.
        shells are assumed to be given in order of ascending altitude. if you only want to pass values in the
        keyword argument for certain shells, put None in the list for all other shells. internally, cells have
        padded space in their arrays, use the getter functions to clean those up.
        '''

        # convert Nones to array of Nones
        if del_t is None:
            del_t = [None]*len(S)
        if sigma is None:
            sigma = [None]*len(S)
        if v is None:
            v = [None]*len(S)
        if delta is None:
            delta = [10]*len(S)
        if alpha is None:
            alpha = [None]*len(S)
        if P is None:
            P = [None]*len(S)
        if m_s is None:
            m_s = [None]*len(S)
        if AM_sat is None:
            AM_sat = [None]*len(S)
        if tau_do is None:
            tau_do = [None]*len(S)
        if tau_min is None:
            tau_min = [1/10]*len(S)

        self.alts = alts
        self.dh = dh
        self.num_L = num_L
        self.num_chi = num_chi
        self.time = 0 # index of current time step
        self.t = [0] # list of times traversed
        self.cells = [] # start list of cells
        # generate bins for log10(L), chi
        self.logL_edges = np.linspace(np.log10(L_min), np.log10(L_max), num=num_L+1)
        self.chi_edges = np.linspace(chi_min, chi_max, num=num_chi+1)
        self.probability_tables = list() # list of probability tables for collisions in each bin

        for i in range(0, len(S)): # iterate through shells

            # convert Nones to array of Nones
            if del_t[i] is None:
                del_t[i] = [None]*len(S[i])
            if sigma[i] is None:
                sigma[i] = [None]*len(S[i])
            if alpha[i] is None:
                alpha[i] = [None]*len(S[i])
            if P[i] is None:
                P[i] = [None]*len(S[i])
            if m_s[i] is None:
                m_s[i] = [None]*len(S[i])
            if AM_sat[i] is None:
                AM_sat[i] = [None]*len(S[i])
            if tau_do[i] is None:
                tau_do[i] = [None]*len(S[i])

            sat_list = []

            for j in range(len(S[0])): # iterate through satellite types, and generate object for each
                
                # convert Nones to default values
                if del_t[i][j] is None:
                    del_t[i][j] = 5
                if sigma[i][j] is None:
                    sigma[i][j] = 10
                if alpha[i][j] is None:
                    alpha[i][j] = 0.2
                if P[i][j] is None:
                    P[i][j] = 0.95
                if m_s[i][j] is None:
                    m_s[i][j] = 250
                if AM_sat[i][j] is None:
                    AM_sat[i][j] = 1/(20*2.2)

                # compute atmospheric drag lifetime for satallites in the shell
                tau = drag_lifetime(alts[i] + dh[i]/2, alts[i] - dh[i]/2, AM_sat[i][j])
                if tau_do[i][j] is None:
                    tau_do[i][j] = tau/10
                sat = Satellite(S[i][j], S_d[i][j], D[i][j], m_s[i][j], sigma[i][j], lam[i][j], del_t[i][j],
                                tau_do[i][j], alpha[i][j], P[i][j], AM_sat[i][j], tau)
                sat_list.append(sat)

            # calculate decay paremeters for debris, initial debris values, and use those to make the N_factor_table
            N_initial, tau_N, N_factor_table = np.zeros((num_L, num_chi)), np.zeros((num_L, num_chi)), np.zeros((num_L, num_chi))
            # generate initial distributions
            lethal_L = np.log10(randL_coll(N_l[i], 1e-1, L_max))
            nlethal_L = np.log10(randL_coll(delta[i]*N_l[i], L_min, 1e-1))
            for j in range(num_L):
                bin_L = 0
                bin_bot_L, bin_top_L = self.logL_edges[j], self.logL_edges[j+1]
                ave_L = 10**((bin_bot_L+bin_top_L)/2)
                bin_L += len(lethal_L[(bin_bot_L < lethal_L) & (lethal_L < bin_top_L)])
                bin_L += len(nlethal_L[(bin_bot_L < nlethal_L) & (nlethal_L < bin_top_L)])
                chi_dist = randX_coll(bin_L, chi_min, chi_max, ave_L)
                for k in range(num_chi):
                    bin_bot_chi, bin_top_chi = self.chi_edges[k], self.chi_edges[k+1]
                    ave_chi = (bin_bot_chi + bin_top_chi)/2
                    N_initial[j,k] = len(chi_dist[(bin_bot_chi < chi_dist) & (chi_dist < bin_top_chi)])
                    tau_N[j,k] = drag_lifetime(alts[i] + dh[i]/2, alts[i] - dh[i]/2, 10**ave_chi)
                    N_factor_table[j,k] = int(tau_N[j,k] > tau_min[i])
                    N_initial[j,k] *= N_factor_table[j,k] # only count if you have to

            # initialize cell
            cell = Cell(sat_list, N_initial, self.logL_edges, self.chi_edges, alts[i], dh[i], tau_N, N_factor_table, v=v[i])
            self.cells.append(cell)
            if i == len(S) - 1: self.upper_N = deepcopy(N_initial) # take the debris field above to be initial debris of top

        # compute probability tables
        for i in range(len(S)):
            curr_prob = np.zeros((len(S), self.num_L, self.num_chi))
            self.fill_prob_table(curr_prob, i, num_dir)
            self.probability_tables.append(curr_prob)

    def fill_prob_table(self, curr_prob, cell_index, num_dir):
        '''
        calculates probability table for given cell

        Input(s):
        curr_prob : current probability table (3-d array)
        cell_index : index of the current cell
        num_dir : number of random directions to sample in creating probability tables

        Keyword Input(s): None

        Output(s): None
        '''

        v0 = self.cells[cell_index].v_orbit*1000 # orbital velocity in m/s
        r = self.cells[cell_index].alt # in km
        L_min, L_max = 10**self.logL_edges[0], 10**self.logL_edges[-1]
        chi_min, chi_max = self.chi_edges[0], self.chi_edges[-1]
        theta = np.random.uniform(low=0, high=np.pi, size=num_dir) # random directions
        phi = np.random.uniform(low=0, high=2*np.pi, size=num_dir)
        for i in range(len(self.cells)): # iterate through cells
            curr_cell = self.cells[i]
            alt_min = curr_cell.alt - curr_cell.dh/2 # in km
            alt_max = curr_cell.alt + curr_cell.dh/2
            v_min2 = G*Me*(2/((Re + r)*1000) - 1/((Re + alt_min)*1000)) # minimum velocity squared (m/s)
            v_max2 = G*Me*(2/((Re + r)*1000) - 1/((Re + alt_max)*1000)) # maximum velocity squared (m/s)
            for j in range(self.num_L): # iterate through bins
                bin_bot_L, bin_top_L = self.logL_edges[j], self.logL_edges[j+1]
                ave_L = 10**((bin_bot_L+bin_top_L)/2)
                curr_prob[:, j, :] = L_cdf(10**bin_top_L, L_min, L_max) - L_cdf(10**bin_bot_L, L_min, L_max) # probability of L being in this bin
                for k in range(self.num_chi):
                    bin_bot_chi, bin_top_chi = self.chi_edges[k], self.chi_edges[k+1]
                    ave_chi = (bin_bot_chi+bin_top_chi)/2
                    curr_prob[:, j, k] *= X_cdf(bin_top_chi, chi_min, chi_max, ave_L) - X_cdf(bin_bot_chi, chi_min, chi_max, ave_L)
                    sum = 0
                    for l in range(num_dir): # sample random directions
                        if v_min2 < 0 and v_max2 < 0 : pass
                        elif v_min2 < 0 : sum += curr_prob[i,j,k]*(vprime_cdf(np.sqrt(v_max2), v0, theta[l], phi[l], ave_chi))
                        else : sum += curr_prob[i,j,k]*(vprime_cdf(np.sqrt(v_max2), theta[l], phi[l], v0, ave_chi) - vprime_cdf(np.sqrt(v_min2), v0, theta[l], phi[l], ave_chi))
                    curr_prob[i,j,k] = sum/num_dir
            curr_prob[i,:,:] *= self.cells[cell_index].N_factor_table # only count relevant debris

    def dxdt(self, time, upper):
        '''
        calculates the rates of change of all parameters at the given time

        Parameter(s):
        time : time (index) of the values to be used
        upper : whether or not to have debris come into the top shell (bool)

        Keyword Parameter(s): None

        Output(s):
        dSdt : list of rates of change in S for each cell (1/yr)
        dS_ddt : list of rates of change in S_d for each cell (1/yr)
        dDdt : list of rates of change in D for each cell (1/yr)
        dNdt : list of rates of change in the N matrix for each cell (1/yr)
        dCldt : list of rates of change in C_l for each cell (1/yr)
        dCnldt : list of rates of change in C_nl for each cell (1/yr)

        Note : does not check that the time input is valid
        '''

        top_cell = self.cells[-1]
        num_types = top_cell.num_types
        top_Nin = self.upper_N/top_cell.tau_N # debris going into top cell
        dSdt = [] # array of changes in satallite values
        dS_ddt = [] # array of changes in de-orbiting values
        dDdt = [] # array of changes in derelict values
        dNdt = [] # array of changes in debris values
        sat_coll = []
        N_coll = [] # array of collision values
        for i in range(len(self.cells)):
            dSdt.append([])
            dS_ddt.append([])
            dDdt.append([])
            dNdt.append(np.zeros((self.num_L, self.num_chi)))
            sat_coll.append(np.zeros((num_types, num_types)))
            N_coll.append([])

        # get initial D_in, N_in values
        S_din = np.zeros(num_types)
        D_in = np.zeros(num_types)
        N_in  = np.zeros((self.num_L, self.num_chi))
        if upper : N_in = top_Nin

        # iterate through cells, from top to bottom
        for i in range(-1, (-1)*(len(self.cells) + 1), -1):
            curr_cell = self.cells[i]
            dNdt[i] += N_in
            dSdt[i], dS_ddt[i], dDdt[i], S_din, D_in, N_in, sat_coll[i], N_coll[i] = curr_cell.dxdt_cell(time, S_din, D_in)
            dNdt[i] -= N_in # loses debris decaying outs
            # simulate collisions
            for j in range(num_types): # iterate through satellite types

                m_s1 = curr_cell.satellites[j].m
                for k in range(j+1, num_types): # satellite-satellite collisions
                    m_s2 = curr_cell.satellites[k].m
                    self.sim_colls(dNdt, sat_coll[i][j,k] + sat_coll[i][k,j], m_s1, m_s2, i)
                self.sim_colls(dNdt, sat_coll[i][j,j], m_s1, m_s1, i) # collisions between satellites of same type
                
                for k in range(self.num_L): # satellite-debris collisions
                    ave_L = 10**((self.logL_edges[k] + self.logL_edges[k+1])/2)
                    for l in range(self.num_chi):
                        ave_AM = 10**((self.chi_edges[l] + self.chi_edges[l+1])/2)
                        m_d = find_A(ave_L)/ave_AM
                        self.sim_colls(dNdt, N_coll[i][j][k,l], m_s1, m_d, i)
                    
            # add on debris lost to collisions
            dNdt[i] -= sum(N_coll[i])

        # update values
        dCldt = []
        dCnldt = []
        for i in range(len(self.cells)):
            curr_cell = self.cells[i]
            dCldt.append(np.sum(sat_coll[i]))
            dCnldt.append(0)
            for j in range(num_types):
                lethal_table = curr_cell.lethal_N[j]
                dCldt[i] += np.sum(sum(N_coll[i])[lethal_table==True])
                dCnldt[i] += np.sum(sum(N_coll[i])[lethal_table==False])

        return dSdt, dS_ddt, dDdt, dNdt, dCldt, dCnldt

    def run_sim_euler(self, T, dt=1, upper=True):
        '''
        simulates the evolution of the debris-satallite system for T years using a Euler method

        Parameter(s):
        T : length of the simulation (yr)

        Keyword Parameter(s):
        dt : timestep used by the simulation (yr, default 1yr)
        upper : whether or not to have debris come into the top shell (bool, default True)

        Output(s): None
        '''

        while self.t[self.time] < T:
            dSdt, dS_ddt, dDdt, dNdt, dCldt, dCnldt = self.dxdt(self.time, upper) # get current rates of change
            for i in range(len(self.cells)): # iterate through cells and update values
                curr_cell = self.cells[i]
                curr_cell.N_bins.append(curr_cell.N_bins[self.time] + dNdt[i]*dt)
                curr_cell.C_l.append(curr_cell.C_l[self.time] + dCldt[i]*dt)
                curr_cell.C_nl.append(curr_cell.C_nl[self.time] + dCnldt[i]*dt)
                for j in range(curr_cell.num_types):
                    curr_cell.satellites[j].S.append(curr_cell.satellites[j].S[self.time] + dSdt[i][j]*dt)
                    curr_cell.satellites[j].S_d.append(curr_cell.satellites[j].S_d[self.time] + dS_ddt[i][j]*dt)
                    curr_cell.satellites[j].D.append(curr_cell.satellites[j].D[self.time] + dDdt[i][j]*dt)
            self.t.append(self.t[self.time] + dt) # update time
            self.time += 1

    def run_sim_precor(self, T, dt=1, mindtfactor=1000, maxdt=1, tolerance=1, upper=True):
        '''
        simulates the evolution of the debris-satallite system for T years using predictor-corrector model

        Parameter(s):
        T : length of the simulation (yr)

        Keyword Parameter(s):
        dt : initial timestep used by the simulation (yr, default 1 yr)
        mindtfactor : minimum time step used by the simulation is dt/mindtfactor
        maxdt : maximum time step used by simulation (yr, default 1)
        tolerance : tolerance for adaptive time step
        upper : whether or not to have debris come into the top shell (bool, default True)

        Output(s): None

        Note(s): AB(2) method is used as predictor, Trapezoid method as corrector
        '''
        dt_min = dt/mindtfactor # get minimum possible time step
        # get additional initial value if needed
        if self.time == 0 : self.run_sim_euler(dt_min, dt=dt_min, upper=upper)
        # get previous rate of change values
        dSdt_n, dSddt_n, dDdt_n, dNdt_n, dCldt_n, dCnldt_n = self.dxdt(self.time-1, upper=upper)
        # get current rate of change values
        dSdt_n1, dSddt_n1, dDdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1 = self.dxdt(self.time, upper=upper)
        dt_old = dt_min # set up old time step variable

        while self.t[self.time] < T:
            redo = False
            # step forwards using AB(2) method
            for i in range(len(self.cells)): # iterate through cells and update values
                curr_cell = self.cells[i]

                if len(curr_cell.N_bins) < self.time + 2: # check if we need to lengthen things
                    for j in range(curr_cell.num_types):
                        curr_cell.satellites[j].S.append(0)
                        curr_cell.satellites[j].S_d.append(0)
                        curr_cell.satellites[j].D.append(0)
                    curr_cell.N_bins.append(0)
                    curr_cell.C_l.append(0)
                    curr_cell.C_nl.append(0)

                for j in range(curr_cell.num_types):
                    curr_cell.satellites[j].S[self.time+1] = curr_cell.satellites[j].S[self.time] + 0.5*dt*((2+dt/dt_old)*dSdt_n1[i][j]-(dt/dt_old)*dSdt_n[i][j])
                    curr_cell.satellites[j].S_d[self.time+1] = curr_cell.satellites[j].S_d[self.time] + 0.5*dt*((2+dt/dt_old)*dSddt_n1[i][j]-(dt/dt_old)*dSddt_n[i][j])
                    curr_cell.satellites[j].D[self.time+1] = curr_cell.satellites[j].D[self.time] + 0.5*dt*((2+dt/dt_old)*dDdt_n1[i][j]-(dt/dt_old)*dDdt_n[i][j])
                curr_cell.N_bins[self.time+1] = curr_cell.N_bins[self.time] + 0.5*dt*((2+dt/dt_old)*dNdt_n1[i]-(dt/dt_old)*dNdt_n[i])
                curr_cell.C_l[self.time+1] = curr_cell.C_l[self.time] + 0.5*dt*((2+dt/dt_old)*dCldt_n1[i]-(dt/dt_old)*dCldt_n[i])
                curr_cell.C_nl[self.time+1] = curr_cell.C_nl[self.time] + 0.5*dt*((2+dt/dt_old)*dCnldt_n1[i]-(dt/dt_old)*dCnldt_n[i])
            # get predicted rate of change from AB(2) method prediction
            dSdt_n2, dSddt_n2, dDdt_n2, dNdt_n2, dCldt_n2, dCnldt_n2 = self.dxdt(self.time+1, upper=upper)
            # set up variable for step size checking
            epsilon = 0
            # re-do step using Trapezoid method
            for i in range(len(self.cells)): # iterate through cells and update values
                curr_cell = self.cells[i]
                old_N = curr_cell.N_bins[self.time+1]
                for j in range(curr_cell.num_types):
                    old_S = curr_cell.satellites[j].S[self.time+1] # keep old values
                    old_Sd = curr_cell.satellites[j].S_d[self.time+1]
                    old_D = curr_cell.satellites[j].D[self.time+1]
                    curr_cell.satellites[j].S[self.time+1] = curr_cell.satellites[j].S[self.time] + 0.5*(dSdt_n2[i][j]+dSdt_n1[i][j])*dt
                    if curr_cell.satellites[j].S[self.time] != 0:
                        epsilon = max(np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.satellites[j].S[self.time+1]-old_S)), epsilon)
                    curr_cell.satellites[j].S_d[self.time+1] = curr_cell.satellites[j].S_d[self.time] + 0.5*(dSddt_n2[i][j]+dSddt_n1[i][j])*dt
                    if curr_cell.satellites[j].S_d[self.time] != 0:
                        epsilon = max(np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.satellites[j].S_d[self.time+1]-old_Sd)), epsilon)
                    curr_cell.satellites[j].D[self.time+1] = curr_cell.satellites[j].D[self.time] + 0.5*(dDdt_n2[i][j]+dDdt_n1[i][j])*dt
                    if curr_cell.satellites[j].D[self.time] != 0:
                        epsilon = max(np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.satellites[j].D[self.time+1]-old_D)), epsilon)
                curr_cell.N_bins[self.time+1] = curr_cell.N_bins[self.time] + 0.5*(dNdt_n2[i]+dNdt_n1[i])*dt
                valid_choice = curr_cell.N_bins[self.time] != 0
                if np.any(valid_choice) == True:
                    epsilon_options = np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.N_bins[self.time+1][valid_choice]-old_N[valid_choice]))
                    epsilon = max(np.amax(epsilon_options), epsilon)
                # we don't really care that much about the accuracy of the collision count
                curr_cell.C_l[self.time+1] = curr_cell.C_l[self.time] + 0.5*(dCldt_n2[i]+dCldt_n1[i])*dt
                curr_cell.C_nl[self.time+1] = curr_cell.C_nl[self.time] + 0.5*(dCnldt_n2[i]+dCnldt_n1[i])*dt

            if epsilon > tolerance:
                    redo = True
            # update step size, and check if calculation needs to be redone
            new_dt = min(np.abs(dt_old*(tolerance/epsilon)**(1/3)), maxdt)
            if redo:
                if dt < dt_min:
                    print('WARNING : System may be too stiff to integrate')
                    new_dt = dt_min
                else:
                    dt = new_dt
                    continue

            # update time
            self.t.append(self.t[self.time] + dt)
            self.time += 1
            dt = new_dt
            # update which are the old and new rates of change
            dSdt_n, dSddt_n, dDdt_n, dNdt_n, dCldt_n, dCnldt_n = dSdt_n1, dSddt_n1, dDdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1
            dSdt_n1, dSddt_n1, dDdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1 = self.dxdt(self.time, upper)


    def sim_colls(self, dNdt, rate, m_1, m_2, index):
        '''
        updates dNdt by distributing a rate of collisions between two objects of mass m_1, m_2 in
        the index'th cell
        
        Parameter(s):
        dNdt : current dNdt values (list of matrices, 1/yr)
        rate : rate of collisions to simulate (1/yr)
        m_1 : mass of the first object (kg)
        m_2 : mass of the second object (kg)
        index : index of the cell the collision occurs in

        Keyword Parameter(s): None

        Output(s): None
        '''

        if rate == 0 : return # just skip everything if you can
        v_rel = self.cells[index].v # collision velocity (km/s)
        M = calc_M(m_1, m_2, v_rel) # M factor
        Lmin, Lmax = 10**self.logL_edges[0], 10**self.logL_edges[-1] # min and max characteristic lengths
        N_debris = calc_Ntot_coll(M, Lmin, Lmax)*rate # total rate of debris creation
        prob_table = self.probability_tables[index] # get right probability table
        for i in range(len(self.cells)): # iterate through cells to send debris to
            dNdt[i] += N_debris*prob_table[i, :, :]

    def get_t(self):
        '''
        returns array of times used in the simulation

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        array of t values (yr)
        '''

        return self.t
    
    def get_S(self):
        '''
        returns list of lists for number of live satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of S values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_types):
                to_return[-1].append(cell.satellites[j].S)
        return to_return

    def get_SD(self):
        '''
        returns list of lists for number of de-orbiting satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of D values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_types):
                to_return[-1].append(cell.satellites[j].S_d)
        return to_return

    def get_D(self):
        '''
        returns list of lists for number of derelict satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of D values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_types):
                to_return[-1].append(cell.satellites[j].D)
        return to_return

    def get_N(self):
        '''
        returns arrays for number of debris in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of total N values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            N = []
            for i in range(len(cell.N_bins)):
                N.append(np.sum(cell.N_bins[i]))
            to_return.append(np.array(N))
        return to_return

    def get_C(self):
        '''
        returns arrays for number of collisions in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of total C values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(np.array(cell.C_l) + np.array(cell.C_nl))
        return to_return
    
    def get_Cl(self):
        '''
        returns arrays for number of catastrophic collisions in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of C_l values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.C_l)
        return to_return

    def get_Cnl(self):
        '''
        returns arrays for number of non-catastrophic collisions in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of C_nl values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.C_nl)
        return to_return

    def alt_to_index(self, h):
        '''
        Converts given altitude to cell index

        Parameter(s):
        h : altitude to convert (km)

        Keyword Parameter(s): None

        Output(s):
        index : index corresponding to that altitude, or -1 if none is found
        '''

        for i in range(len(self.cells)):
            alt, dh = self.alts[i], self.dh[i]
            if (alt - dh/2 <= h) and (alt + dh/2 >= h):
                return i
        return -1