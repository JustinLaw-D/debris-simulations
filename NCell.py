# contains class for collection of cells representing orbital shells

from Cell import Cell
import numpy as np
import matplotlib.pyplot as plt
from BreakupModel import *
from copy import deepcopy

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)

class NCell:

    def __init__(self, S, D, N_l, alts, dh, lam, drag_lifetime, del_t=None, sigma=None, v=None, 
                delta=None, alpha=None, P=None, m_s=None, AM_sat=None, L_min=1e-3, L_max=1, num_L=10,
                chi_min=-3, chi_max=3, num_chi=10):
        '''
        Constructor for NCell class
    
        Parameter(s):
        S : initial number of live satellites in each shell (array)
        D : initial number of derelict satellites in each shell (array)
        N_l : initial number of lethal debris in each shell (array)
        alts : altitude of the shell's centre (array, km)
        dh : width of the shells (array, km)
        lam : launch rate of satellites into the each shell (array, 1/yr)
        drag_lifetime : function that computes atmospheric drag lifetime ((km, km, m^2/kg) -> yr)

        Keyword Parameter(s):
        del_t : mean satellite lifetime in each shell (list, yr, default 5yr)
        sigma : satellite cross-section in each shell (list, m^2, default 10m^2)
        v : relative collision speed in each shell (list, km/s, default 10km/s)
        delta : initial ratio of the density of disabling to lethal debris in each shell (list, default 10)
        alpha : fraction of collisions a live satellites fails to avoid in each shell (list, default 0.2)
        P : post-mission disposal probability in each shell (list, default 0.95)
        m_s : mass of the satallites in each band (list, kg, default 250kg)
        AM_sat : area-to-mass ratio of the satallites in each shell (list, m^2/kg, default 1/(20*2.2)m^2/kg)
        L_min : minimum characteristic length to consider (m, default 1mm)
        L_max : maximum characteristic length to consider (m, default 1m)
        num_L : number of debris bins in characteristic length (default 10)
        chi_min : minimum log10(A/M) to consider (log10(m^2/kg), default -3)
        chi_max : maximum log10(A/M) to consider (log10(m^2/kg), default 3)
        num_chi : number of debris bins in log10(A/M) (default 10)

        Output(s):
        NCell instance

        Note: no size checks are done on the arrays, the program will crash if any of the arrays differ in size.
        shells are assumed to be given in order of ascending altitude. if you only want to pass values in the
        keyword argument for certain shells, put None in the list for all other shells. internally, cells have
        padded space in their arrays, use the getter functions to clean those up.
        '''

        # convert Nones to array of Nones
        if del_t == None:
            del_t = [None]*S.size
        if sigma == None:
            sigma = [None]*S.size
        if v == None:
            v = [None]*S.size
        if delta == None:
            delta = [10]*S.size
        if alpha == None:
            alpha = [None]*S.size
        if P == None:
            P = [None]*S.size
        if m_s == None:
            m_s = [None]*S.size
        if AM_sat == None:
            AM_sat = [1/(20*2.2)]*S.size

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

        for i in range(0, S.size):
            # compute atmospheric drag lifetime for satallites in the shell
            tau_D = drag_lifetime(alts[i] + dh[i]/2, alts[i] - dh[i]/2, AM_sat[i])
            # calculate decay paremeters for debris, initial debris values
            N_initial, tau_N = np.zeros((num_L, num_chi), dtype=np.int64), np.zeros((num_L, num_chi))
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

            # initialize cell
            cell = Cell(S[i], D[i], N_initial, self.logL_edges, self.chi_edges, lam[i], alts[i], dh[i], tau_D, 
                        tau_N, del_t=del_t[i], sigma=sigma[i], m_s=m_s[i], v=v[i], alpha=alpha[i], P=P[i])
            self.cells.append(cell)
            if i == S.size - 1: self.upper_N = deepcopy(N_initial) # take the debris field above to be initial debris of top

    def run_sim(self, T, dt=1, upper=True):
        '''
        simulates the evolution of the debris-satallite system for T years

        Parameter(s):
        T : length of the simulation (yr)

        Keyword Parameter(s):
        dt : timestep used by the simulation (yr, default 1yr)
        upper : whether or not to have debris come into the top shell (bool, default True)

        Output(s): None
        '''

        while self.t[self.time] < T:
            change_N = [] # arrays changes in debris values
            for i in range(len(self.cells)):
                change_N.append(np.zeros((self.num_L, self.num_chi),dtype=np.int64))
            
            # get initial D_in, N_in values
            D_in = 0
            N_in  = np.zeros((self.num_L, self.num_chi), dtype=np.int64)
            top_Nin = np.zeros((self.num_L, self.num_chi), dtype=np.int64) # debris going into top cell
            top_cell = self.cells[-1]
            for i in range(self.num_L):
                for j in range(self.num_chi):
                    top_Nin[i,j] = rand_round((self.upper_N[i,j]/top_cell.tau_N[i,j])*dt)
            if upper : N_in = top_Nin

            # iterate through cells, from top to bottom
            for i in range(-1, (-1)*(len(self.cells) + 1), -1):
                curr_cell = self.cells[i]
                change_N[i] += N_in
                m_s = curr_cell.m_s
                D_in, N_in, sat_coll, N_coll = curr_cell.step(D_in, dt)
                change_N[i] -= N_in # loses debris decaying outs
                # simulate collisions
                self.sim_colls(change_N, int(sat_coll), m_s, m_s, i)
                for j in range(self.num_L):
                    ave_L = 10**((self.logL_edges[j] + self.logL_edges[j+1])/2)
                    for k in range(self.num_chi):
                        ave_AM = 10**((self.chi_edges[k] + self.chi_edges[k+1])/2)
                        m_d = find_A(ave_L)/ave_AM
                        self.sim_colls(change_N, int(N_coll[j,k]), m_s, m_d, i)
                        
                # add on debris lost to collisions
                change_N[i] -= N_coll

            # change N values
            for i in range(len(self.cells)):
                curr_cell = self.cells[i]
                curr_cell.N_bins += change_N[i]
                curr_cell.update_N_vals()

            # update time
            self.t.append(self.t[self.time] + dt)
            self.time +=1

    def sim_colls(self, change_N, num, m_1, m_2, index):
        '''
        updates change_N by running num collisions between two objects of mass m_1, m_2 in
        the index'th cell
        
        Parameter(s):
        change_N : current change_N values (list of matrices)
        num : number of collisions to simulate
        m_1 : mass of the first object (kg)
        m_2 : mass of the second object (kg)
        index : index of the cell the collision occurs in

        Keyword Parameter(s): None

        Output(s): None
        '''

        v_rel = self.cells[index].v # collision velocity
        v_orbit = self.cells[index].v_orbit # orbital velocity
        alt = self.alts[index] # altitude of the orbit
        M = calc_M(m_1, m_2, v_rel) # M factor
        Lmin, Lmax = 10**self.logL_edges[0], 10**self.logL_edges[-1] # min and max characteristic lengths
        chi_min, chi_max = self.chi_edges[0], self.chi_edges[-1] # min and max chi values
        N_debris = 0 # total number of debris
        for i in range(num):
            N_debris += calc_Ntot_coll(M, Lmin, Lmax)
        # calculate length distribution
        L_dist = randL_coll(N_debris, Lmin, Lmax)
        L_binned = np.zeros(self.num_L, dtype=np.int64) # those random values binned
        for i in range(self.num_L):
            L_bot, L_top = 10**self.logL_edges[i], 10**self.logL_edges[i+1]
            L_binned[i] = len(L_dist[(L_dist > L_bot) & (L_dist < L_top)])
        for i in range(self.num_L): # iterate through each bin
            L_ave = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2)
            chi_dist = randX_coll(L_binned[i], chi_min, chi_max, L_ave) # get chi distribution for all debris
            chi_binned = np.zeros(self.num_chi, dtype=np.int64) # those random values binned
            for j in range(self.num_chi):
                chi_bot, chi_top = self.chi_edges[j], self.chi_edges[j+1]
                chi_binned[j] = len(chi_dist[(chi_dist > chi_bot) & (chi_dist < chi_top)])
            # calculate velocity distribution, and  update values
            for j in range(self.num_chi):
                chi_ave = (self.chi_edges[i] + self.chi_edges[i+1])/2
                v_dist = randv_coll(chi_binned[j], chi_ave) # get v_kick distribution for all debris
                direction_dist = rand_direction(chi_binned[j]) # get random directions for v_kick for all debris
                for k in range(len(v_dist)): # iterate through the random velocity values
                    v_loc, direction = 10**v_dist[k], direction_dist[:,k]
                    # calculate new orbital velocity
                    v2 = np.sqrt((v_orbit + v_loc*direction[0])**2 + (v_loc*direction[1])**2 + (v_loc*direction[2])**2)
                    alt_new = 1/(2/alt - v2/(G*Me)) # calculate new altitude with vis-viva equation
                    index_new = self.alt_to_index(alt_new) # get index that the debris goes to
                    change_N[index_new][i,j] += 1 # add the debris to the right location

    def run_live_sim(self, dt=1, upper=True, dt_live=1, S=True, D=True, N=True):
        '''
        Runs a live simulation of the evolution of orbital shells for total time T, using the desired
        method, displaying current results, and allowing for the satellite launch rate of some shells
        to be adjusted

        Parameter(s): None
        
        Keyword Parameter(s):
        dt : initial time step used by simulation (yr, default 1)
        upper : wether or not to have debris come into the topmost layer (default True)
        dt_live : time between updates of the graph/oppertunity for user input (yr, default 1)
        S : whether or not to display the number of live satellites (default True)
        D : whether or not to display the number of derelict satellites (default True)
        N : whether or not to display the number of lethal debris (default True)

        Output(s): Nothing

        Note : user input allows you to change which shells are being displayed, change the valus of S, D, N, 
        change the satellite launch rate of any shell, and choose the dt/dt_live for the next step
        '''

        # setup
        display_bools = [] # list representing if each shell is displayed
        for i in range(len(self.cells)): # start by displaying shells with non-zero launch rates
            if self.cells[i].lam != 0:
                display_bools.append(True)
            else:
                display_bools.append(False)
        
        fig, ax = plt.subplots() # setup plots
        
        cont = True
        while cont == True: # start event loop
            ax.clear()
            self.run_sim(self.t[self.time] + dt_live, dt=dt, upper=upper) # step forwards

            # pull well-formatted results
            t_list = self.get_t()
            S_list = self.get_S()
            D_list = self.get_D()
            N_list = self.get_N()

             # display results
            for i in range(len(self.cells)):
                if display_bools[i] == True:
                    if S == True:
                        ax.plot(t_list, S_list[i], label='S'+str(self.alts[i]))
                    if D == True:
                        ax.plot(t_list, D_list[i], label='D'+str(self.alts[i]))
                    if N == True:
                        ax.plot(t_list, N_list[i], label='N'+str(self.alts[i]))
            ax.set_xlabel('time (yr)')
            ax.set_ylabel('log(number)')
            ax.set_yscale('log')
            ax.legend()
            plt.tight_layout()
            plt.show(block=False)

            # take user input
            x = input('Continue running (y/n) : ')
            if x != 'y':
                cont = False
                continue
            x = input('Make any changes (y/n) : ')
            if x != 'y':
                continue
            x = input('Change runtime before next break (y/n) : ')
            if x != 'n':
                dt_live = float(input('Input runtime before next break : '))
            x = input('Change satellite launch rates (y/n) : ')
            while x != 'n':
                h = float(input('Input shell height to change launch rate of : '))
                index = self.alt_to_index(h)
                if index == -1:
                    print('ERROR: Invalid shell height')
                else:
                    lamb = float(input('Input new satellite launch rate : '))
                    self.cells[index].lam = lamb
                    display_bools[index] = True
                x = input('Change more satellite launch rates (y/n) : ')
            x = input('Display live satellites (y/n) : ')
            if x == 'n' : S = False
            else : S = True
            x = input('Display derelict satellites (y/n) : ')
            if x == 'n' : D = False
            else : D = True
            x = input('Display lethal debris (y/n) : ')
            if x == 'n' : N = False
            else : N = True
            x = input('Change which shells are displayed (y/n) : ')
            while x != 'n':
                h = float(input('Input shell height to change display of : '))
                index = self.alt_to_index(h)
                if index == -1:
                    print('ERROR: Invalid shell height')
                else:
                    y = input('Diplay this shell (y/n) : ')
                    if y == 'y' : display_bools[index] = True
                    elif y == 'n' : display_bools[index] = False
                x = input('Change more shell displays (y/n) : ')

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
        returns arrays for number of live satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of S values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.S)
        return to_return

    def get_D(self):
        '''
        returns arrays for number of derelict satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of D values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.D)
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
            to_return.append(np.array(cell.N_l) + np.array(cell.N_nl))
        return to_return
    
    def get_Nl(self):
        '''
        returns arrays for number of lethal debris in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of N_l values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.N_l)
        return to_return

    def get_Nnl(self):
        '''
        returns arrays for number of non-lethal debris in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of N_nl values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append(cell.N_nl)
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