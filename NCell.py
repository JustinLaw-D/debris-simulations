# contains class for collection of cells representing orbital shells

from Cell import *
from ObjectsEvents import *
import numpy as np
from BreakupModel import *
from copy import deepcopy
import time as timer
import os
import shutil
import csv

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Re = 6371 # radius of Earth (km)
Me = 5.97219e24 # mass of Earth (kg)

class NCell:

    def __init__(self, S, S_d, D, N_l, target_alts, alt_edges, lam, drag_lifetime, update_lifetime, events=[], R_i=None, 
                lam_rb=None, up_time=None, del_t=None, expl_rate_L=None, expl_rate_D=None, C_sat=None, sigma_sat=None, 
                expl_rate_R=None, C_rb=None, sigma_rb=None, v=None, delta=None, alphaS=None, alphaD=None, alphaN=None, 
                alphaR=None, P=None, m_s=None, m_rb=None, AM_sat=None, AM_rb=None, tau_do=None, L_min=1e-3, L_max=1, 
                num_L=10, chi_min=-2, chi_max=1.0, num_chi=10, num_dir=100):
        '''
        Constructor for NCell class
    
        Parameter(s):
        S : list of initial number of live satellites in each shell of each type (list of arrays)
        S_d : list of initial number of deorbiting satellites in each shell of each type (list of arrays)
        D : list of initial number of derelict satellites in each shell of each type (list of arrays)
        N_l : initial number of catestrophically lethal debris in each shell (array)
        target_alts : list of target altitude of each satellite type (array, km)
        alt_edges : edges of the altitude bands to be used (array, km)
        lam : launch rate of satellites of each type (array, 1/yr)
        drag_lifetime : function that computes atmospheric drag lifetime ((km, km, m^2/kg, yr) -> yr)
        update_lifetime : function that determines when the drag lifetime needs to be updated ((yr, yr) -> bool)

        Keyword Parameter(s):
        events : the discrete events occuring in the system (list of Event objects, default no events)
        R_i : list of rocket bodies in each shell of each type (list of lists, default no rocket bodies)
        lam_rb : launch rate of rocket bodies of each type into the each shell (list of arrays, 1/yr, default all 0)
        up_time : ascention time of satellites of each type in each shell (list of arrays, yr, default all 1/10yr)
        del_t : mean satellite lifetime of each type in each shell (list of lists, yr, default 5yr)
        expl_rate_L : number of explosions that occur in a 1yr period with a population of 100 live satellites for
                      each type of satellite (list of floats, default all 0)
        expl_rate_D : number of explosions that occur in a 1yr period with a population of 100 derelict satellites
                      for each type of satellite (list of floats, default all 0)
        C_sat : fit constant for explosions of each type of satellite (list of floats, default all 1)
        sigma_sat : satellite cross-section of each type (list, m^2, default 10m^2)
        expl_rate_R : number of explosions that occur in a 1yr period with a population of 100 rocket bodies for
                      each type of rocket body (list of floats, default all 0)
        C_rb : fit constant for explosions of each type of rocket body (list of floats, default all 1)
        sigma_rb : rocket cross-section of each type (list, m^2, default 10m^2)
        v : relative collision speed in each shell (list, km/s, default 10km/s)
        delta : initial ratio of the density of disabling to catestrophic debris in each shell (list, default 10)
        alphaS : fraction of collisions with another live satellite that a live satellites of each type fails to 
                 avoid in each shell (list of lists, default 0)
        alphaD : fraction of collisions with another derelict that a live satellites of each type fails to 
                 avoid in each shell (list of lists, default alphaN)
        alphaN : fraction of collisions with trackable debris that a live satellites of each type fails to 
                 avoid in each shell (list of lists, default 0.2)
        alphaR : fraction of collisions with a rocket body that a live satellites of each type fails to 
                 avoid in each shell (list of lists, default alphaN)
        P : post-mission disposal probability for satellites of each type in each shell (list of lists, default 0.95)
        m_s : mass of the satallites of each type (list, kg, default 250kg)
        m_s : mass of the rocket bodies of each type (list, kg, default 250kg)
        AM_sat : area-to-mass ratio of the satallites of each type (list, m^2/kg, default 1/(20*2.2)m^2/kg)
        AM_rb : area-to-mass ratio of the rocket bodies of each type (list, m^2/kg, default 1/(20*2.2)m^2/kg)
        tau_do : average deorbiting time for satellites of each type in each shell (list of lists, yr, default decay_time/10)
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
        if events is None:
            events = []
        if R_i is None:
            R_i = [[]]*len(S)
        if lam_rb is None:
            lam_rb = [None]*len(S)
        if up_time is None:
            up_time = [None]*len(S)
        if del_t is None:
            del_t = [None]*len(S)
        if v is None:
            v = [None]*len(S)
        if delta is None:
            delta = [10]*len(S)
        if alphaS is None:
            alphaS = [None]*len(S)
        if alphaD is None:
            alphaD = [None]*len(S)
        if alphaN is None:
            alphaN = [None]*len(S)
        if alphaR is None:
            alphaR = [None]*len(S)
        if P is None:
            P = [None]*len(S)
        if tau_do is None:
            tau_do = [None]*len(S)

        self.alts = np.zeros(len(alt_edges)-1)
        self.dh = np.zeros(self.alts.shape)
        for i in range(len(alt_edges)-1):
            self.dh[i] = alt_edges[i+1]-alt_edges[i]
            self.alts[i] = (alt_edges[i]+alt_edges[i+1])/2
        self.num_L = num_L
        self.num_chi = num_chi
        self.drag_lifetime = drag_lifetime
        self.update_lifetime = update_lifetime
        self.time = 0 # index of current time step
        self.lupdate_time = 0 # index of last time drag lifetimes were updated
        self.t = [0] # list of times traversed
        self.cells = [] # start list of cells
        # generate bins for log10(L), chi
        self.logL_edges = np.linspace(np.log10(L_min), np.log10(L_max), num=num_L+1)
        self.chi_edges = np.linspace(chi_min, chi_max, num=num_chi+1)
        self.num_dir = num_dir
        self.sat_coll_probability_tables = list() # list of probability tables for satellite collisions in each bin
        self.rb_coll_probability_tables = list() # list of probability tables for rocket body collisions in each bin
        self.sat_expl_probability_tables = list() # list of probability tables for satellite explosions in each bin
        self.rb_expl_probability_tables = list() # list of probability tables for rocket body explosions in each bin

        for i in range(0, len(S)): # iterate through shells

            # convert Nones to array of Nones
            if lam_rb[i] is None:
                lam_rb[i] = [None]*len(R_i[i])
            if up_time[i] is None:
                up_time[i] = [None]*len(S[i])
            if del_t[i] is None:
                del_t[i] = [None]*len(S[i])
            if expl_rate_L is None:
                expl_rate_L = [None]*len(S[i])
            if expl_rate_D is None:
                expl_rate_D = [None]*len(S[i])
            if C_sat is None:
                C_sat = [None]*len(S[i])
            if sigma_sat is None:
                sigma_sat = [None]*len(S[i])
            if expl_rate_R is None:
                expl_rate_R = [None]*len(R_i[i])
            if C_rb is None:
                C_rb = [None]*len(R_i[i])
            if sigma_rb is None:
                sigma_rb = [None]*len(R_i[i])
            if alphaS[i] is None:
                alphaS[i] = [None]*len(S[i])
            if alphaD[i] is None:
                alphaD[i] = [None]*len(S[i])
            if alphaN[i] is None:
                alphaN[i] = [None]*len(S[i])
            if alphaR[i] is None:
                alphaR[i] = [None]*len(S[i])
            if P[i] is None:
                P[i] = [None]*len(S[i])
            if m_s is None:
                m_s = [None]*len(S[i])
            if m_rb is None:
                m_rb = [None]*len(R_i[i])
            if AM_sat is None:
                AM_sat = [None]*len(S[i])
            if AM_rb is None:
                AM_rb = [None]*len(R_i[i])
            if tau_do[i] is None:
                tau_do[i] = [None]*len(S[i])

            sat_list = []

            for j in range(len(S[0])): # iterate through satellite types, and generate object for each
                
                # convert Nones to default values
                if up_time[i][j] is None:
                    up_time[i][j] = 1/10
                if del_t[i][j] is None:
                    del_t[i][j] = 5
                if expl_rate_L[j] is None:
                    expl_rate_L[j] = 0
                if expl_rate_D[j] is None:
                    expl_rate_D[j] = expl_rate_L[j]
                if C_sat[j] is None:
                    C_sat[j] = 1
                if sigma_sat[j] is None:
                    sigma_sat[j] = 10
                if alphaS[i][j] is None:
                    alphaS[i][j] = 0
                if alphaN[i][j] is None:
                    alphaN[i][j] = 0.2
                if alphaD[i][j] is None:
                    alphaD[i][j] = alphaN[i][j]
                if alphaR[i][j] is None:
                    alphaR[i][j] = alphaN[i][j]
                if P[i][j] is None:
                    P[i][j] = 0.95
                if m_s[j] is None:
                    m_s[j] = 250
                if AM_sat[j] is None:
                    AM_sat[j] = 1/(20*2.2)

                # compute atmospheric drag lifetime for satallites in the shell
                tau = drag_lifetime(self.alts[i] + self.dh[i]/2, self.alts[i] - self.dh[i]/2, AM_sat[j], 0)
                if tau_do[i][j] is None:
                    tau_do[i][j] = tau/10
                sat = Satellite(S[i][j], S_d[i][j], D[i][j], m_s[j], sigma_sat[j], lam[j], del_t[i][j],
                                tau_do[i][j], target_alts[j], up_time[i][j], (alphaS[i][j], alphaD[i][j],
                                alphaN[i][j], alphaR[i][j]), P[i][j], AM_sat[j], tau, C_sat[j], expl_rate_L[j], expl_rate_D[j])
                sat_list.append(sat)

            rb_list = []

            for j in range(len(R_i[0])): # iterate through rocket types, and generate object for each
                
                # convert Nones to default values
                if lam_rb[i][j] is None:
                    lam_rb[i][j] = 0
                if expl_rate_R[j] is None:
                    expl_rate_R[j] = 0
                if C_rb[j] is None:
                    C_rb[j] = 1
                if sigma_rb[j] is None:
                    sigma_rb[j] = 10
                if m_rb[j] is None:
                    m_rb[j] = 250
                if AM_rb[j] is None:
                    AM_rb[j] = 1/(20*2.2)

                # compute atmospheric drag lifetime for rocket bodies in the shell
                tau = drag_lifetime(self.alts[i] + self.dh[i]/2, self.alts[i] - self.dh[i]/2, AM_rb[j], 0)
                rb = RocketBody(R_i[i][j], m_rb[j], sigma_rb[j], lam_rb[i][j], AM_rb[j], tau, C_rb[j], expl_rate_R[j])
                rb_list.append(rb)

            # calculate decay paremeters for debris, initial debris values
            N_initial, tau_N = np.zeros((num_L, num_chi)), np.zeros((num_L, num_chi), dtype=np.double)
            # generate initial distributions
            for j in range(num_L):
                bin_L = 0
                bin_bot_L, bin_top_L = self.logL_edges[j], self.logL_edges[j+1]
                if (10**bin_bot_L < -1) and (bin_top_L > -1):
                    lam_factor = (-1-bin_bot_L)/(bin_top_L-bin_bot_L)
                    bin_L += lam_factor*N_l[i]*delta[i]*(L_cdf(1e-1, L_min, 1e-1, 'expl') - L_cdf(10**bin_bot_L, L_min, 1e-1, 'expl'))
                    bin_L += (1-lam_factor)*N_l[i]*(L_cdf(10**bin_top_L, 1e-1, L_max, 'expl') - L_cdf(10**bin_bot_L, 1e-1, L_max, 'expl'))
                elif bin_bot_L >= -1:
                    bin_L += N_l[i]*(L_cdf(10**bin_top_L, 1e-1, L_max, 'expl') - L_cdf(10**bin_bot_L, 1e-1, L_max, 'expl'))
                else:
                    bin_L += N_l[i]*delta[i]*(L_cdf(10**bin_top_L, L_min, 1e-1, 'expl') - L_cdf(10**bin_bot_L, L_min, 1e-1, 'expl'))
                N_initial[j,0] = bin_L # put everything in the lowest A/M bin
                for k in range(num_chi):
                    bin_bot_chi, bin_top_chi = self.chi_edges[k], self.chi_edges[k+1]
                    ave_chi = (bin_bot_chi + bin_top_chi)/2
                    tau_N[j,k] = drag_lifetime(self.alts[i] + self.dh[i]/2, self.alts[i] - self.dh[i]/2, 10**ave_chi, 0)

            # figure out which events are in this cell
            events_loc = []
            for event in events:
                if (event.alt > self.alts[i] - self.dh[i]/2) and (event.alt <= self.alts[i] + self.dh[i]/2) : events_loc.append(event)

            # initialize cell
            cell = Cell(sat_list, rb_list, N_initial, self.logL_edges, self.chi_edges, events_loc, self.alts[i], self.dh[i], tau_N, v=v[i])
            self.cells.append(cell)
            if i == len(S) - 1: self.upper_N = deepcopy(N_initial) # take the debris field above to be initial debris of top

        self.num_cells = len(self.cells)
        # generate uniformly distributed directions using Fibbonacci spiral
        phi, theta = np.zeros(num_dir), np.zeros(num_dir)
        golden = (1+np.sqrt(5))/2 # golden ratio
        for i in range(num_dir):
            x = (i/golden) % 1
            y = i/num_dir
            phi[i] = 2*np.pi*x
            theta[i] = np.arccos(1-2*y)
        # compute probability tables
        for i in range(self.num_cells):
            curr_sat_coll_prob = np.zeros((self.num_cells, self.num_L, self.num_chi))
            curr_rb_coll_prob = np.zeros((self.num_cells, self.num_L, self.num_chi))
            curr_sat_expl_prob = np.zeros((self.num_cells, self.num_L, self.num_chi))
            curr_rb_expl_prob = np.zeros((self.num_cells, self.num_L, self.num_chi))
            self.fill_prob_table(curr_sat_coll_prob, curr_rb_coll_prob, i, phi, theta, 'coll')
            self.fill_prob_table(curr_sat_expl_prob, curr_rb_expl_prob, i, phi, theta, 'expl')
            self.sat_coll_probability_tables.append(curr_sat_coll_prob)
            self.rb_coll_probability_tables.append(curr_rb_coll_prob)
            self.sat_expl_probability_tables.append(curr_sat_expl_prob)
            self.rb_expl_probability_tables.append(curr_rb_expl_prob)

    def fill_prob_table(self, curr_prob_sat, curr_prob_rb, cell_index, phi, theta, e_typ):
        '''
        calculates probability tables for collisions/explosions given the cell
        they occured in

        Input(s):
        curr_prob_sat : current probability table for satellites (3-d array)
        curr_prob_rb : current probability table for rocket bodies (3-d array)
        cell_index : index of the current cell
        phi : list of phi components of directions
        theta : list of theta components of directions
        e_typ : type of event, either 'coll' (collision) or 'expl' (explosions)

        Keyword Input(s): None

        Output(s): None

        Note(s): behaviour is undefined if an invalid typ is given
        '''

        v0 = self.cells[cell_index].v_orbit*1000 # orbital velocity in m/s
        r = self.cells[cell_index].alt # in km
        L_min, L_max = 10**self.logL_edges[0], 10**self.logL_edges[-1]
        chi_min, chi_max = self.chi_edges[0], self.chi_edges[-1]
        for i in range(self.num_cells): # iterate through cells
            curr_cell = self.cells[i]
            alt_min = curr_cell.alt - curr_cell.dh/2 # in km
            alt_max = curr_cell.alt + curr_cell.dh/2
            v_min2 = G*Me*(2/((Re + r)*1000) - 1/((Re + alt_min)*1000)) # minimum velocity squared (m/s)
            v_max2 = G*Me*(2/((Re + r)*1000) - 1/((Re + alt_max)*1000)) # maximum velocity squared (m/s)
            for j in range(self.num_chi): # handle vprime_cdf
                ave_chi = (self.chi_edges[j]+self.chi_edges[j+1])/2
                sum = 0 # perform monte carlo integration
                if v_min2 < 0 and v_max2 < 0 : pass
                elif v_min2 < 0:
                    for k in range(self.num_dir):
                        sum += vprime_cdf(np.sqrt(v_max2), v0, theta[k], phi[k], ave_chi, e_typ)
                else:
                    for k in range(self.num_dir):
                        sum += vprime_cdf(np.sqrt(v_max2), v0, theta[k], phi[k], ave_chi, e_typ) - vprime_cdf(np.sqrt(v_min2), v0, theta[k], phi[k], ave_chi, e_typ)
                curr_prob_sat[i,:,j] = sum/self.num_dir # save the result
                curr_prob_rb[i,:,j] = sum/self.num_dir

            for j in range(self.num_L): # iterate through bins
                bin_bot_L, bin_top_L = self.logL_edges[j], self.logL_edges[j+1]
                ave_L = 10**((bin_bot_L+bin_top_L)/2)
                curr_prob_sat[i,j,:] *= L_cdf(10**bin_top_L, L_min, L_max, e_typ) - L_cdf(10**bin_bot_L, L_min, L_max, e_typ) # probability of L being in this bin
                curr_prob_rb[i,j,:] *= L_cdf(10**bin_top_L, L_min, L_max, e_typ) - L_cdf(10**bin_bot_L, L_min, L_max, e_typ)
                for k in range(self.num_chi):
                    bin_bot_chi, bin_top_chi = self.chi_edges[k], self.chi_edges[k+1]
                    ave_chi = (bin_bot_chi+bin_top_chi)/2
                    curr_prob_sat[i,j,k] *= X_cdf(bin_top_chi, chi_min, chi_max, ave_L, 'sat') - X_cdf(bin_bot_chi, chi_min, chi_max, ave_L, 'sat')
                    curr_prob_rb[i,j,k] *= X_cdf(bin_top_chi, chi_min, chi_max, ave_L, 'rb') - X_cdf(bin_bot_chi, chi_min, chi_max, ave_L, 'rb')

    def save(self, filepath, name, compress=True, gap=0, force=False):
        '''
        saves the current NCell object to .csv and .npz files

        Input(s):
        filepath : explicit path to folder that the files will be saved in (string)
        name : name of the object, must be a valid unix folder name (string)

        Keyword Input(s):
        compress : whether or not to save the data in a compressed format (default True)
        gap : largest acceptable time gap between saved data points (yr, default 0 i.e. save all data)
        force : whether or not to automatically replace any saved data with the same name (default False)

        Output(s): None

        Note(s): drag_lifetime and events are lost. adherence to the "gap" value is approximate, and may behave
        strangely if the time step is close to the gap size.
        '''

        true_path = filepath + name + '/'
        try:
            os.mkdir(true_path) # make the folder representing the object
        except FileExistsError:
            x = 'y'
            if not force:
                x = input("File with this name already exists. Replace it (y/n): ")
            if x == 'y':
                shutil.rmtree(true_path)
                os.mkdir(true_path)
            else : return

        # write parameters
        csv_file = open(true_path + 'params.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file, dialect='unix')
        csv_writer.writerow([self.num_L, self.num_chi, self.num_cells, self.num_dir])
        csv_file.close()

        # write easy arrays
        t_arr = np.array(self.t)
        filter = np.full(t_arr.shape, False) # build filter based on time steps
        if t_arr.size > 0:
            prev_t = t_arr[0]
            filter[0] = True
            for i in range(1, t_arr.size):
                if t_arr[i] - prev_t >= gap:
                    prev_t = t_arr[i]
                    filter[i] = True
        to_save = {'alts' : self.alts, 'dh' : self.dh, 't' : t_arr[filter], 'logL' : self.logL_edges, 'chi' : self.chi_edges}
        if compress : np.savez_compressed(true_path + "data.npz", **to_save)
        else : np.savez(true_path + "data.npz", **to_save)

        # write probability tables
        sat_coll_tables = dict()
        rb_coll_tables = dict()
        sat_expl_tables = dict()
        rb_expl_tables = dict()
        for i in range(self.num_cells):
            sat_coll_tables[str(i)] = self.sat_coll_probability_tables[i]
            rb_coll_tables[str(i)] = self.rb_coll_probability_tables[i]
            sat_expl_tables[str(i)] = self.sat_expl_probability_tables[i]
            rb_expl_tables[str(i)] = self.rb_expl_probability_tables[i]
        if compress:
            np.savez_compressed(true_path + "sat_coll_tables.npz", **sat_coll_tables)
            np.savez_compressed(true_path + "rb_coll_tables.npz", **rb_coll_tables)
            np.savez_compressed(true_path + "sat_expl_tables.npz", **sat_expl_tables)
            np.savez_compressed(true_path + "rb_expl_tables.npz", **rb_expl_tables)
        else:
            np.savez(true_path + "sat_coll_tables.npz", **sat_coll_tables)
            np.savez(true_path + "rb_coll_tables.npz", **rb_coll_tables)
            np.savez(true_path + "sat_expl_tables.npz", **sat_expl_tables)
            np.savez(true_path + "rb_expl_tables.npz", **rb_expl_tables)

        # save the Cells
        for i in range(self.num_cells):
            cell_path = true_path + "cell" + str(i) + "/"
            os.mkdir(cell_path)
            self.cells[i].save(cell_path, filter, compress=compress)

    def load(filepath):
        '''
        builds an NCell object from saved data

        Input(s):
        filepath : explicit path to folder that the files are saved in (string)

        Keyword Input(s): None

        Output(s):
        atmos : NCell object build from loaded data

        Note(s): atmos will not have events
        '''

        atmos = NCell.__new__(NCell) # empty initialization

        # load parameters
        csv_file = open(filepath + 'params.csv', 'r', newline='')
        csv_reader = csv.reader(csv_file, dialect='unix')
        for row in csv_reader: # there's only one row, this extracts it
            atmos.num_L = int(row[0])
            atmos.num_chi = int(row[1])
            atmos.num_cells = int(row[2])
            atmos.num_dir = int(row[3])
        csv_file.close()

        # load in simple numpy arrays
        array_dict = np.load(filepath + 'data.npz')
        atmos.alts = array_dict['alts']
        atmos.dh = array_dict['dh']
        atmos.t = array_dict['t'].tolist()
        atmos.time = len(atmos.t) - 1 # set time to the end of the data
        atmos.lupdate_time = atmos.time
        atmos.logL_edges = array_dict['logL']
        atmos.chi_edges = array_dict['chi']

        # load in probability tables
        sat_coll_dict = np.load(filepath + "sat_coll_tables.npz")
        rb_coll_dict = np.load(filepath + "rb_coll_tables.npz")
        sat_expl_dict = np.load(filepath + "sat_expl_tables.npz")
        rb_expl_dict = np.load(filepath + "rb_expl_tables.npz")
        atmos.sat_coll_probability_tables = []
        atmos.rb_coll_probability_tables = []
        atmos.sat_expl_probability_tables = []
        atmos.rb_expl_probability_tables = []
        for i in range(atmos.num_cells):
            atmos.sat_coll_probability_tables.append(sat_coll_dict[str(i)])
            atmos.rb_coll_probability_tables.append(rb_coll_dict[str(i)])
            atmos.sat_expl_probability_tables.append(sat_expl_dict[str(i)])
            atmos.rb_expl_probability_tables.append(rb_expl_dict[str(i)])

        # get Cells
        atmos.cells = []
        for i in range(atmos.num_cells):
            cell_path = filepath + "cell" + str(i) + "/"
            atmos.cells.append(Cell.load(cell_path))

        return atmos

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
        dRdt : list of rates of change in R for each cell (1/yr)
        dNdt : list of rates of change in the N matrix for each cell (1/yr)
        dCldt : list of rates of change in C_l for each cell (1/yr)
        dCnldt : list of rates of change in C_nl for each cell (1/yr)

        Note : does not check that the time input is valid
        '''

        top_cell = self.cells[-1]
        num_sat_types = top_cell.num_sat_types
        num_rb_types = top_cell.num_rb_types
        top_Nin = self.upper_N/top_cell.tau_N # debris going into top cell
        dSdt = np.zeros((self.num_cells, num_sat_types)) # array of changes in satallite values
        dS_ddt = np.zeros((self.num_cells, num_sat_types)) # array of changes in de-orbiting values
        dDdt = np.zeros((self.num_cells, num_sat_types)) # array of changes in derelict values
        dRdt = np.zeros((self.num_cells, num_rb_types)) # array of changes in rocket body values
        dNdt =  np.zeros((self.num_cells, self.num_L, self.num_chi)) # array of changes in debris values
        sat_coll =  np.zeros((self.num_cells, num_sat_types, num_sat_types)) # array of satellite-satellite collisions
        RS_coll = np.zeros((self.num_cells, num_sat_types, num_rb_types)) # array of rocket-satellite collisions
        R_coll = np.zeros((self.num_cells, num_rb_types, num_rb_types)) # array of rocket-rocket collisions
        NS_coll = np.zeros((self.num_cells, num_sat_types, self.num_L, self.num_chi)) # array of collision values for satellites
        NR_coll = np.zeros((self.num_cells, num_rb_types, self.num_L, self.num_chi)) # array of collision values for rockets
        NS_expl = np.zeros((self.num_cells, num_sat_types)) # array of explosion values for satellites
        NR_expl = np.zeros((self.num_cells, num_rb_types)) # array of explosion values for rockets

        # get initial D_in, N_in values
        S_in = np.zeros((self.num_cells+1, num_sat_types))
        for i in range(num_sat_types):
            S_in[0,i] = top_cell.satellites[i].lam
        S_din = np.zeros((self.num_cells+1, num_sat_types))
        D_in = np.zeros((self.num_cells+1, num_sat_types))
        R_in = np.zeros((self.num_cells+1, num_rb_types))
        N_in  = np.zeros((self.num_cells+1, self.num_L, self.num_chi))
        if upper : N_in[-1,:,:] = top_Nin

        # iterate through cells, from top to bottom
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            x = timer.time()
            dSdt[i,:], dS_ddt[i,:], dDdt[i,:], dRdt[i,:], S_in[i+1,:], S_din[i,:], D_in[i,:], R_in[i,:], N_in[i,:,:], sat_coll[i,:,:], RS_coll[i,:,:], R_coll[i,:,:], NS_coll[i,:,:,:], NR_coll[i,:,:,:], NS_expl[i,:], NR_expl[i,:] = curr_cell.dxdt_cell(time)
            # simulate collisions and explosions
            for j in range(num_sat_types): # iterate through satellite types

                m_s1 = curr_cell.satellites[j].m
                C = curr_cell.satellites[j].C
                for k in range(j+1, num_sat_types): # satellite-satellite collisions
                    m_s2 = curr_cell.satellites[k].m
                    self.sim_colls(dNdt, sat_coll[i,j,k] + sat_coll[i,k,j], m_s1, m_s2, i, 'sat')
                self.sim_colls(dNdt, sat_coll[i,j,j], m_s1, m_s1, i, 'sat') # collisions between satellites of same type

                for k in range(num_rb_types): # satellite-rb collisions
                    m_rb2 = curr_cell.rockets[k].m
                    self.sim_colls_satrb(dNdt, RS_coll[i,j,k], m_s1, i, 'sat')
                    self.sim_colls_satrb(dNdt, RS_coll[i,j,k], m_rb2, i, 'rb')

                ave_time = 0
                for k in range(self.num_L): # satellite-debris collisions
                    ave_L = 10**((self.logL_edges[k] + self.logL_edges[k+1])/2)
                    for l in range(self.num_chi):
                        ave_AM = 10**((self.chi_edges[l] + self.chi_edges[l+1])/2)
                        m_d = find_A(ave_L)/ave_AM
                        print(NS_coll[i,j,k,l])
                        x = timer.time()
                        self.sim_colls(dNdt, NS_coll[i,j,k,l], m_s1, m_d, i, 'sat')
                        ave_time += timer.time()-x
                #print("{:e}".format(ave_time/(self.num_L*self.num_chi)))

                self.sim_expl(dNdt, NS_expl[i,j], C, i, 'sat')

            for j in range(num_rb_types): # iterate through rocket body types

                m_rb1 = curr_cell.rockets[j].m
                C = curr_cell.rockets[j].C
                for k in range(j+1, num_rb_types): # rocket-rocket collisions
                    m_rb2 = curr_cell.rockets[k].m
                    self.sim_colls(dNdt, R_coll[i,j,k] + R_coll[i,k,j], m_rb1, m_rb2, i, 'rb')
                self.sim_colls(dNdt, R_coll[i,j,j], m_rb1, m_rb1, i, 'rb') # collisions between rockets of same type

                for k in range(self.num_L): # rocket-debris collisions
                    ave_L = 10**((self.logL_edges[k] + self.logL_edges[k+1])/2)
                    for l in range(self.num_chi):
                        ave_AM = 10**((self.chi_edges[l] + self.chi_edges[l+1])/2)
                        m_d = find_A(ave_L)/ave_AM
                        self.sim_colls(dNdt, NR_coll[i,j,k,l], m_rb1, m_d, i, 'rb')

                self.sim_expl(dNdt, NR_expl[i,j], C, i, 'rb')
                    
            # add on debris lost to collisions
            if num_sat_types != 0:
                dNdt[i] -= np.sum(NS_coll[i,:,:,:], axis=0)
            if num_rb_types != 0:
                dNdt[i] -= np.sum(NR_coll[i,:,:,:], axis=0)

        # go through cells from bottom to top to correct values
        for i in range(self.num_cells):
            dSdt[i] += S_in[i,:] - S_in[i+1,:]
            dS_ddt[i] += S_din[i+1,:] - S_din[i,:]
            dDdt[i] += D_in[i+1,:] - D_in[i,:]
            dRdt[i] += R_in[i+1,:] - R_in[i,:]
            dNdt[i] += N_in[i+1,:] - N_in[i,:]

        # update values
        dCldt = np.zeros(self.num_cells)
        dCnldt = np.zeros(self.num_cells)
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            dCldt[i] += np.sum(sat_coll[i,:,:]) + np.sum(RS_coll[i,:,:]) + np.sum(R_coll[i,:,:])
            for j in range(num_sat_types):
                lethal_table = curr_cell.lethal_sat_N[j]
                dCldt[i] += np.sum(NS_coll[i,j,:,:][lethal_table==True])
                dCnldt[i] += np.sum(NS_coll[i,j,:,:][lethal_table==False])
            for j in range(num_rb_types):
                lethal_table = curr_cell.lethal_rb_N[j]
                dCldt[i] += np.sum(NR_coll[i,j,:,:][lethal_table==True])
                dCnldt[i] += np.sum(NR_coll[i,j,:,:][lethal_table==False])

        return dSdt, dS_ddt, dDdt, dRdt, dNdt, dCldt, dCnldt

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

        self.sim_events() # run initial discrete events

        while self.t[self.time] < T:
            if self.update_lifetime(self.t[self.time], self.t[self.lupdate_time]):
                    self.update_lifetimes(self.t[self.time])
                    self.lupdate_time = self.time
            dSdt, dS_ddt, dDdt, dRdt, dNdt, dCldt, dCnldt = self.dxdt(self.time, upper) # get current rates of change

            for i in range(self.num_cells): # iterate through cells and update values
                curr_cell = self.cells[i]
                curr_cell.N_bins.append(curr_cell.N_bins[self.time] + dNdt[i]*dt)
                curr_cell.C_l.append(curr_cell.C_l[self.time] + dCldt[i]*dt)
                curr_cell.C_nl.append(curr_cell.C_nl[self.time] + dCnldt[i]*dt)
                for j in range(curr_cell.num_sat_types):
                    curr_cell.satellites[j].S.append(curr_cell.satellites[j].S[self.time] + dSdt[i][j]*dt)
                    curr_cell.satellites[j].S_d.append(curr_cell.satellites[j].S_d[self.time] + dS_ddt[i][j]*dt)
                    curr_cell.satellites[j].D.append(curr_cell.satellites[j].D[self.time] + dDdt[i][j]*dt)
                for j in range(curr_cell.num_rb_types):
                    curr_cell.rockets[j].num.append(curr_cell.rockets[j].num[self.time] + dRdt[i][j]*dt)
            self.t.append(self.t[self.time] + dt) # update time
            self.time += 1
            self.sim_events() # run discrete events

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
        warning_given = False # whether or not a warning has been given yet
        # get additional initial value if needed
        if self.time == 0 : self.run_sim_euler(dt_min, dt=dt_min, upper=upper)
        # get previous rate of change values
        self.update_lifetimes(self.t[self.time-1])
        dSdt_n, dSddt_n, dDdt_n, dRdt_n, dNdt_n, dCldt_n, dCnldt_n = self.dxdt(self.time-1, upper=upper)
        # get current rate of change values
        self.update_lifetimes(self.t[self.time])
        self.lupdate_time = self.time
        dSdt_n1, dSddt_n1, dDdt_n1, dRdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1 = self.dxdt(self.time, upper=upper)
        dt_old = dt_min # set up old time step variable
        updated, redo = False, False

        while self.t[self.time] < T:
            if updated and redo:
                self.update_lifetimes(self.t[self.time])
            elif updated:
                self.lupdate_time = self.time
            redo = False
            updated = False
            # step forwards using AB(2) method
            for i in range(self.num_cells): # iterate through cells and update values
                curr_cell = self.cells[i]

                if len(curr_cell.N_bins) < self.time + 2: # check if we need to lengthen things
                    for j in range(curr_cell.num_sat_types):
                        curr_cell.satellites[j].S.append(0)
                        curr_cell.satellites[j].S_d.append(0)
                        curr_cell.satellites[j].D.append(0)
                    for j in range(curr_cell.num_rb_types):
                        curr_cell.rockets[j].num.append(0)
                    curr_cell.N_bins.append(0)
                    curr_cell.C_l.append(0)
                    curr_cell.C_nl.append(0)

                for j in range(curr_cell.num_sat_types):
                    curr_cell.satellites[j].S[self.time+1] = curr_cell.satellites[j].S[self.time] + 0.5*dt*((2+dt/dt_old)*dSdt_n1[i][j]-(dt/dt_old)*dSdt_n[i][j])
                    curr_cell.satellites[j].S_d[self.time+1] = curr_cell.satellites[j].S_d[self.time] + 0.5*dt*((2+dt/dt_old)*dSddt_n1[i][j]-(dt/dt_old)*dSddt_n[i][j])
                    curr_cell.satellites[j].D[self.time+1] = curr_cell.satellites[j].D[self.time] + 0.5*dt*((2+dt/dt_old)*dDdt_n1[i][j]-(dt/dt_old)*dDdt_n[i][j])
                for j in range(curr_cell.num_rb_types):
                        curr_cell.rockets[j].num[self.time+1] = curr_cell.rockets[j].num[self.time] + 0.5*dt*((2+dt/dt_old)*dRdt_n1[i][j]-(dt/dt_old)*dRdt_n[i][j])
                curr_cell.N_bins[self.time+1] = curr_cell.N_bins[self.time] + 0.5*dt*((2+dt/dt_old)*dNdt_n1[i]-(dt/dt_old)*dNdt_n[i])
                curr_cell.C_l[self.time+1] = curr_cell.C_l[self.time] + 0.5*dt*((2+dt/dt_old)*dCldt_n1[i]-(dt/dt_old)*dCldt_n[i])
                curr_cell.C_nl[self.time+1] = curr_cell.C_nl[self.time] + 0.5*dt*((2+dt/dt_old)*dCnldt_n1[i]-(dt/dt_old)*dCnldt_n[i])
            # get predicted rate of change from AB(2) method prediction
            if self.update_lifetime(self.t[self.time] + dt, self.t[self.lupdate_time]):
                    self.update_lifetimes(self.t[self.time] + dt)
                    updated = True
            dSdt_n2, dSddt_n2, dDdt_n2, dRdt_n2, dNdt_n2, dCldt_n2, dCnldt_n2 = self.dxdt(self.time+1, upper=upper)
            # set up variable for step size checking
            epsilon = 0
            # re-do step using Trapezoid method
            for i in range(self.num_cells): # iterate through cells and update values
                curr_cell = self.cells[i]
                old_N = curr_cell.N_bins[self.time+1]
                for j in range(curr_cell.num_sat_types):
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
                for j in range(curr_cell.num_rb_types):
                    old_R = curr_cell.rockets[j].num[self.time+1]
                    curr_cell.rockets[j].num[self.time+1] = curr_cell.rockets[j].num[self.time] + 0.5*(dRdt_n2[i][j] + dRdt_n1[i][j])*dt
                    if curr_cell.rockets[j].num[self.time] != 0:
                        epsilon = max(np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.rockets[j].num[self.time+1]-old_R)), epsilon)
                curr_cell.N_bins[self.time+1] = curr_cell.N_bins[self.time] + 0.5*(dNdt_n2[i]+dNdt_n1[i])*dt
                valid_choice = curr_cell.N_bins[self.time] != 0
                if np.any(valid_choice) == True:
                    epsilon_options = np.abs((1/3)*(dt/(dt+dt_old))*(curr_cell.N_bins[self.time+1][valid_choice]-old_N[valid_choice]))
                    epsilon = max(np.amax(epsilon_options), epsilon)
                # we don't really care that much about the accuracy of the collision count
                curr_cell.C_l[self.time+1] = curr_cell.C_l[self.time] + 0.5*(dCldt_n2[i]+dCldt_n1[i])*dt
                curr_cell.C_nl[self.time+1] = curr_cell.C_nl[self.time] + 0.5*(dCnldt_n2[i]+dCnldt_n1[i])*dt

            # update step size, and check if calculation needs to be redone
            if epsilon > tolerance:
                redo = True
            new_dt = min(np.abs(dt*(tolerance/epsilon)**(1/3)), maxdt)
            if redo:
                if dt <= dt_min:
                    if not warning_given:
                        print('WARNING : System may be too stiff to integrate')
                        warning_given = True
                    redo=False
                    new_dt = dt_min
                else:
                    dt = new_dt
                    continue

            # update time
            self.t.append(self.t[self.time] + dt)
            self.time += 1
            dt_old = dt
            dt = new_dt
            # run events
            self.sim_events()
            # update which are the old and new rates of change
            dSdt_n, dSddt_n, dDdt_n, dRdt_n, dNdt_n, dCldt_n, dCnldt_n = dSdt_n1, dSddt_n1, dDdt_n1, dRdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1
            dSdt_n1, dSddt_n1, dDdt_n1, dRdt_n1, dNdt_n1, dCldt_n1, dCnldt_n1 = self.dxdt(self.time, upper)


    def sim_colls(self, dNdt, rate, m_1, m_2, index, typ):
        '''
        updates dNdt by distributing a rate of collisions between two objects of mass m_1, m_2 in
        the index'th cell
        
        Parameter(s):
        dNdt : current dNdt values (list of matrices, 1/yr)
        rate : rate of collisions to simulate (1/yr)
        m_1 : mass of the first object (kg)
        m_2 : mass of the second object (kg)
        index : index of the cell the collision occurs in
        typ : object type of the main (first) object, either 'sat' (satellite) or 'rb' (rocket body)

        Keyword Parameter(s): None

        Output(s): None
        '''

        if rate == 0 : return # just skip everything if you can
        v_rel = self.cells[index].v # collision velocity (km/s)
        M = calc_M(m_1, m_2, v_rel) # M factor
        Lmin, Lmax = 10**self.logL_edges[0], 10**self.logL_edges[-1] # min and max characteristic lengths
        N_debris = calc_Ntot(M, Lmin, Lmax, 'coll')*rate # total rate of debris creation
        if typ == 'sat':
            prob_table = self.sat_coll_probability_tables[index] # get right probability table
        elif typ == 'rb':
            prob_table = self.rb_coll_probability_tables[index]
        for i in range(self.num_cells): # iterate through cells to send debris to
            dNdt[i,:,:] += N_debris*prob_table[i,:,:]

    def sim_colls_satrb(self, dNdt, rate, m, index, typ):
        '''
        version of sim_coll used for the satellite-rocket body collisions workaround, where
        each object is simulated as having its own catastrophic collision
        
        Parameter(s):
        dNdt : current dNdt values (list of matrices, 1/yr)
        rate : rate of collisions to simulate (1/yr)
        m : mass of the object (kg)
        index : index of the cell the collision occurs in
        typ : object type in the collision, either 'sat' (satellite) or 'rb' (rocket body)

        Keyword Parameter(s): None

        Output(s): None
        '''

        if rate == 0 : return # just skip everything if you can
        Lmin, Lmax = 10**self.logL_edges[0], 10**self.logL_edges[-1] # min and max characteristic lengths
        N_debris = calc_Ntot(m, Lmin, Lmax, 'coll')*rate # total rate of debris creation
        if typ == 'sat':
            prob_table = self.sat_coll_probability_tables[index] # get right probability table
        elif typ == 'rb':
            prob_table = self.rb_coll_probability_tables[index]
        for i in range(self.num_cells): # iterate through cells to send debris to
            dNdt[i,:,:] += N_debris*prob_table[i,:,:]

    def sim_expl(self, dNdt, rate, C, index, typ):
        '''
        updates dNdt by distributing a rate of explosions for an object with constant C in
        the index'th cell
        
        Parameter(s):
        dNdt : current dNdt values (list of matrices, 1/yr)
        rate : rate of explosions to simulate (1/yr)
        C : fit constant for the explosion
        index : index of the cell the collision occurs in
        typ : object type of the main object, either 'sat' (satellite) or 'rb' (rocket body)

        Keyword Parameter(s): None

        Output(s): None
        '''

        if rate == 0 : return # just skip everything if you can
        Lmin, Lmax = 10**self.logL_edges[0], 10**self.logL_edges[-1] # min and max characteristic lengths
        N_debris = calc_Ntot(0, Lmin, Lmax, 'expl', C=C)*rate # total rate of debris creation
        if typ == 'sat':
            prob_table = self.sat_expl_probability_tables[index] # get right probability table
        elif typ == 'rb':
            prob_table = self.rb_expl_probability_tables[index]
        for i in range(self.num_cells): # iterate through cells to send debris to
            dNdt[i,:,:] += N_debris*prob_table[i, :, :]

    def sim_events(self):
        '''
        simulates discrete events at the current time

        Input(s): None

        Keyword Input(s): None

        Output(s): None
        '''

        dN = np.zeros((self.num_cells, self.num_L, self.num_chi)) # debris change matrix

        for i in range(self.num_cells):

            curr_cell = self.cells[i]
            dS, dS_d, dD = np.zeros(curr_cell.num_sat_types), np.zeros(curr_cell.num_sat_types), np.zeros(curr_cell.num_sat_types)
            dR = np.zeros(curr_cell.num_rb_types)
            dN_loc = np.zeros((self.num_L, self.num_chi)) # debris change from non-collision sources
            coll_list = []
            expl_list = []
            S, S_d, D, R = self.get_curr_S(self.time)[i], self.get_curr_SD(self.time)[i], self.get_curr_D(self.time)[i], self.get_curr_R(self.time)[i]
            N = curr_cell.N_bins[self.time]

            for event in curr_cell.event_list: # iterate through possible events

                if event.time is not None: # events at specific times
                    while event.time != [] and event.time[0] <= self.t[self.time]:
                        dS_temp, dS_d_temp, dD_temp, dR_temp, dN_loc_temp, coll_temp, expl_temp = event.run_event(S, S_d, D, R, N, self.logL_edges, self.chi_edges)
                        event.time.pop(0)
                        dS += dS_temp
                        dS_d += dS_d_temp
                        dD += dD_temp
                        dR += dR_temp
                        dN_loc += dN_loc_temp
                        coll_list.extend(coll_temp)
                        expl_list.extend(expl_temp)

                if event.freq is not None: # events occuring at specific frequencies
                    if self.t[self.time] - event.last_event <= event.freq:
                        dS_temp, dS_d_temp, dD_temp, dR_temp, dN_loc_temp, coll_temp, expl_temp = event.run_event(S, S_d, D, R, N, self.logL_edges, self.chi_edges)
                        dS += dS_temp
                        dS_d += dS_d_temp
                        dD += dD_temp
                        dR += dR_temp
                        dN_loc += dN_loc_temp
                        coll_list.extend(coll_temp)
                        expl_list.extend(expl_temp)

                # update values
                for j in range(curr_cell.num_sat_types):
                    curr_cell.satellites[j].S[self.time] += dS[j]
                    curr_cell.satellites[j].S_d[self.time] += dS_d[j]
                    curr_cell.satellites[j].D[self.time] += dD[j]
                for j in range(curr_cell.num_rb_types):
                    curr_cell.rockets[j].num[self.time] += dR[j]
                curr_cell.N_bins[self.time] += dN_loc

                # handle collisions and explosions
                self.parse_coll(dN, coll_list, i)
                self.parse_expl(dN, expl_list, i)

        # update with debris from collisions/explosions
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            curr_cell.N_bins[self.time] += dN[i,:,:]

    def parse_coll(self, dN, coll_list, i):
        '''
        parses and runs discrete collision events, storing the debris generated in dN

        Input(s):
        dN : 3d matrix of changes in debris for each bin and cell
        coll_list : list of collisions occuring in the current cell in the form [(kg, kg, typ, #)],
                    i.e. [(m1, m2, typ, number of collisions)]. typ can be one of 'sat' (satellite-satellite),
                    'sr' (satellite-rocket, where satellite is m1), or 'rb' (rocket-rocket)
        i : index of the current cell
        '''

        for coll in coll_list: # iterate through list
            m1, m2, typ, num = coll # unpack the list
            if typ == 'sat' or typ == 'rb':
                self.sim_colls(dN, num, m1, m2, i, typ)
            elif typ == 'sr':
                self.sim_colls_satrb(dN, num, m1, i, 'sat')
                self.sim_colls_satrb(dN, num, m2, i, 'rb')
    
    def parse_expl(self, dN, expl_list, i):
        '''
        parses and runs discrete explosion events, storing the debris generated in dN

        Input(s):
        dN : 3d matrix of changes in debris for each bin and cell
        expl_list : list of explosions occuring in the current cell in the form [(C, typ, #)], where
                    C is the relevant fit constant and typ is the type of body exploding ('sat' or 'rb)
        i : index of the current cell
        '''

        for expl in expl_list: # iterate through list
            C, typ, num = expl # unpack the list
            self.sim_expl(dN, num, C, i, typ)

    def update_lifetimes(self, t):
        '''
        updates all drag lifetimes in the system, using drag_lifetime function

        Input(s):
        t : time to call drag_lifetime at (yr)

        Keyword Input(s): None

        Output(s): None
        '''

        for i in range(self.num_cells): # iterate through cells
            curr_cell = self.cells[i]
            alt = curr_cell.alt
            dh = curr_cell.dh
            for j in range(curr_cell.num_sat_types): # handle satellites
                AM = curr_cell.satellites[j].AM
                curr_cell.satellites[j].tau = self.drag_lifetime(alt + dh/2, alt - dh/2, AM, t)
            for j in range(curr_cell.num_rb_types): # handle rockets
                AM = curr_cell.rockets[j].AM
                curr_cell.rockets[j].tau = self.drag_lifetime(alt + dh/2, alt - dh/2, AM, t)
            for j in range(self.num_chi): # handle debris
                bin_bot_chi, bin_top_chi = self.chi_edges[j], self.chi_edges[j+1]
                ave_chi = (bin_bot_chi + bin_top_chi)/2
                curr_cell.tau_N[:,j] = self.drag_lifetime(alt + dh/2, alt - dh/2, 10**ave_chi, t)

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
        returns list of lists of lists for number of live satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of S values for each cell of each type, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_sat_types):
                to_return[-1].append(cell.satellites[j].S)
        return to_return

    def get_curr_S(self, time):
        '''
        returns array for the number of live satellites of each type in the each shell
        at the given time

        Paramter(s):
        time : time index to pull values from

        Keyword Parameter(s): None

        Returns:
        S : (list of) array of S values at the current time
        '''

        to_return = []
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            to_return.append(np.zeros(curr_cell.num_sat_types))
            for j in range(curr_cell.num_sat_types):
                to_return[i][j] = curr_cell.satellites[j].S[time]
        return to_return

    def get_SD(self):
        '''
        returns list of lists of lists for number of de-orbiting satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of S_d values for each cell of each type, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_sat_types):
                to_return[-1].append(cell.satellites[j].S_d)
        return to_return

    def get_curr_SD(self, time):
        '''
        returns array for the number of de-orbiting satellites of each type in the each shell
        at the given time

        Paramter(s):
        time : time index to pull values from

        Keyword Parameter(s): None

        Returns:
        S_d : (list of) array of S_d values at the current time
        '''

        to_return = []
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            to_return.append(np.zeros(curr_cell.num_sat_types))
            for j in range(curr_cell.num_sat_types):
                to_return[i][j] = curr_cell.satellites[j].S_d[time]
        return to_return

    def get_D(self):
        '''
        returns list of lists of lists for number of derelict satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of D values for each cell of each type, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_sat_types):
                to_return[-1].append(cell.satellites[j].D)
        return to_return

    def get_curr_D(self, time):
        '''
        returns array for the number of derelict satellites of each type in the each shell
        at the given time

        Paramter(s):
        time : time index to pull values from

        Keyword Parameter(s): None

        Returns:
        D : (list of) array of D values at the current time
        '''

        to_return = []
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            to_return.append(np.zeros(curr_cell.num_sat_types))
            for j in range(curr_cell.num_sat_types):
                to_return[i][j] = curr_cell.satellites[j].D[time]
        return to_return

    def get_R(self):
        '''
        returns list of lists of lists for number of rocket bodies in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of R values for each cell of each type, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_return.append([])
            for j in range(cell.num_rb_types):
                to_return[-1].append(cell.rockets[j].num)
        return to_return

    def get_curr_R(self, time):
        '''
        returns array for the number of rocket bodies of each type in the each shell
        at the given time

        Paramter(s):
        time : time index to pull values from

        Keyword Parameter(s): None

        Returns:
        R : (list of) array of R values at the current time
        '''

        to_return = []
        for i in range(self.num_cells):
            curr_cell = self.cells[i]
            to_return.append(np.zeros(curr_cell.num_rb_types))
            for j in range(curr_cell.num_rb_types):
                to_return[i][j] = curr_cell.rockets[j].num[time]
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

        for i in range(self.num_cells):
            alt, dh = self.alts[i], self.dh[i]
            if (alt - dh/2 <= h) and (alt + dh/2 >= h):
                return i
        return -1
