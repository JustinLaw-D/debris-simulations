# contains class for a single atmospheric layer (Cell), satallites (functions more as a struct), and
# discrete events (Event)

import numpy as np
from BreakupModel import *
from ObjectsEvents import *
import os
import csv
G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)
Re = 6371 # radius of Earth (km)

class Cell:
    
    def __init__(self, S_i, R_i, N_i, logL_edges, chi_edges, event_list, alt, dh, tau_N, N_factor_table, v=None):
        '''Constructor for Cell class
    
        Parameter(s):
        S_i : list of satellite types with initial values
        R_i : list of rocket body types with initial values
        N_i : initial array of number of debris by L and A/M
        logL_edges : bin edges in log10 of characteristic length (log10(m))
        chi_edges : bin edges in log10(A/M) (log10(m^2/kg))
        event_list : list of discrete events that occur in the cell
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

        # setup initial values for tracking live satallites, derelict satallites,
        # lethat debris, and non-lethal debris over time
        self.satellites = S_i
        self.rockets = R_i
        self.num_sat_types = len(self.satellites)
        self.num_rb_types = len(self.rockets)
        self.N_bins = [N_i]

        # setup other variables
        self.C_l = [0] # lethal collisions
        self.C_nl = [0] # non-lethal collisions
        self.event_list = event_list
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
        self.lethal_sat_N = []
        self.lethal_rb_N = []
        for i in range(self.num_sat_types):
            self.lethal_sat_N.append(np.full(self.N_bins[0].shape, False)) # whether or not each bin has lethal collisions
        for i in range(self.num_rb_types):
            self.lethal_rb_N.append(np.full(self.N_bins[0].shape, False)) 
        self.update_lethal_N()
        self.ascending = [] # list of which satellite types are ascending
        for sat in self.satellites:
            if sat.target_alt < self.alt - self.dh/2 : self.ascending.append(True)
            else : self.ascending.append(False)

    def save(self, filepath):
        '''
        saves the current Cell object to .csv and .npz files

        Input(s):
        filepath : explicit path to folder that the files will be saved in (string)

        Keyword Input(s): None

        Output(s): None

        Note(s): event_list is lost
        '''

        # save parameters
        csv_file = open(filepath + 'params.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file, dialec='unix')
        csv_writer.write_row([self.num_sat_types, self.num_rb_types, self.alt, self.dh, self.v, self.v_orbit, self.num_L,
                              self.num_chi])
        csv_file.close()

        # write easy arrays
        Cl_array, Cnl_array = np.array(self.C_l), np.array(self.C_nl)
        to_save = {'C_l' : Cl_array, 'C_nl' : Cnl_array, 'N_factor' : self.N_factor_table, 'tau_N' : self.tau_N, 
                   'ascending' : self.ascending, 'logL' : self.logL_edges, 'chi' : self.chi_edges}
        np.savez(filepath + "data.npz", **to_save)

        # write N_bins values
        N_dict = dict()
        for i in range(len(self.N_bins)):
            dict[str(i)] = self.N_bins[i]
        np.savez(filepath + "N_bins.npz", **N_dict)

        # write lethal table values
        lethal_dict = dict()
        for i in range(self.num_sat_types):
            lethal_dict["sat" + str(i)] = self.lethal_sat_N[i]
        for i in range(self.num_rb_types):
            lethal_dict["rb" + str(i)] = self.lethal_rb_N[i]
        np.savez(filepath + "lethal_tables.npz", **lethal_dict)

        # write satellites and rockets
        for i in range(self.num_sat_types):
            sat_path = filepath + 'Satellite' + str(i) + '/'
            os.mkdir(sat_path)
            self.satellites[i].save(sat_path)
        for i in range(self.num_rb_types):
            rb_path = filepath + 'RocketBody' + str(i) + '/'
            os.mkdir(rb_path)
            self.rockets[i].save(rb_path)

    def load(filepath):
        '''
        builds a Cell object from saved data

        Input(s):
        filepath : explicit path to folder that the files are saved in (string)

        Keyword Input(s): None

        Output(s):
        cell : Cell object build from loaded data

        Note(s): cell will not have events
        '''

        cell = Cell.__new__(Cell) # create blank Cell

        # load parameters
        csv_file = open(filepath + 'params.csv', 'r', newline='')
        csv_reader = csv.reader(csv_file, dialec='unix')
        for row in csv_reader: # there's only one row, this extracts it
            cell.num_sat_types = int(row[0])
            cell.num_rb_types = int(row[1])
            cell.alt = float(row[2])
            cell.dh = float(row[3])
            cell.v = float(row[4])
            cell.v_orbit = float(row[5])
            cell.num_L = int(row[6])
            cell.num_chi = int(row[7])
        csv_file.close()

        # load basic arrays
        array_dict = np.load(filepath + "data.npz")
        cell.C_l = array_dict['C_l'].to_list()
        cell.C_nl = array_dict['C_nl'].to_list()
        cell.N_factor_table = array_dict['N_factor']
        cell.tau_N = array_dict['tau_N']
        cell.ascending = array_dict['ascending']
        cell.logL_edges = array_dict['logL']
        cell.chi_edges = array_dict['chi']

        # load N_bins values
        cell.N_bins = []
        bins_dict = np.load(filepath + "N_bins.npz")
        i = 0
        while True:
            try:
                N_bins = bins_dict[str(i)]
                cell.N_bins.append(N_bins)
            except KeyError:
                break
            i += 1
        
        # load lethal table values
        lethal_dict = np.load(filepath + "lethal_tables.npz")
        cell.lethal_sat_N = []
        cell.lethal_rb_N = []
        for i in range(cell.num_sat_types):
            cell.lethal_sat_N.append(lethal_dict["sat" + str(i)])
        for i in range(cell.num_rb_types):
            cell.lethal_rb_N.append(lethal_dict["rb" + str(i)])

        # load satellites and rockets
        cell.satellites = []
        cell.rockets = []
        for i in range(cell.num_sat_types):
            sat_path = filepath + 'Satellite' + str(i) + '/'
            cell.satellites.append(Satellite.load(sat_path))
        for i in range(cell.num_rb_types):
            rb_path = filepath + 'RocketBody' + str(i) + '/'
            cell.rockets.append(RocketBody.load(rb_path))

        cell.event_list = []
        return cell

    def dxdt_cell(self, time, S_in, S_din, D_in, R_in):
        '''
        calculates the rate of collisions and decays from each debris bin, the rate
        of decaying/de-orbiting satellites, the rate of launches/deorbit starts of satallites, 
        and the rate of creation of derelicts at the given time

        Parameter(s):
        time : index of the values to use
        S_in : rate of ascending live satellites of each type entering the cell from below (yr^(-1))
        S_din : rate of de-orbiting satellites of each type entering the cell from above (yr^(-1))
        D_in : rate of derelict satellites of each type entering the cell from above (yr^(-1))
        R_in : rate of rocket bodies of each type entering the cell from above (yr^(-1))

        Keyword Parameter(s): None

        Output(s):
        dSdt : list of rate of change of the number of live satellites in the cell of each type (yr^(-1))
        dS_ddt : list of rate of change of the number of de-orbiting satellites in the cell of each type (yr^(-1))
        dDdt : list of rate of change of the number of derelict satellites in the cell of each type (yr^(-1))
        dRdt : list of rate of change of number of rocket bodies in the cell of each type (yr^(-1))
        S_out : list of rate of satellites ascending from the cell of each type (yr^(-1))
        S_dout : list of rate of satellites de-orbiting from the cell of each type (yr^(-1))
        D_out : list of rate of satellites decaying from the cell of each type (yr^(-1))
        R_out : list of rate of rocket bodies decaying from the cell of each type (yr^(-1))
        N_out : matrix with the rate of exiting debris from each bin (yr^(-1))
        D_dt : matrix with total rate of collisions between satellites (yr^(-1))
        RD_dt : matrix with total rate of collisions between satellites and rocket bodies (yr^(-1))
        R_dt : matrix with total rate of collisions between rocket bodies (yr^(-1))
        CS_dt : list of matrices with the rate of collisions from each bin with each satellite type (yr^(-1))
        CR_dt : list of matrices with the rate of collisions from each bin with each rocket body type (yr^(-1))
        expl_S : list of rate of explosions for satellites of each type (yr^(-1))
        expl_R : list of rate of explosions for rocket bodies of each type (yr^(-1))

        Note: Assumes that collisions with debris of L_cm < 10cm cannot be avoided, and
        that the given time input is valid
        '''
        
        N = self.N_bins[time]

        # compute the rate of collisions from each debris type
        dSdt = [] # collisions with live satallites
        dSDdt = np.zeros((self.num_sat_types, self.num_sat_types)) # collisions between live and derelict satallites
        dSRdt = np.zeros((self.num_sat_types, self.num_rb_types)) # collisions between live satellites and rocket bodies
        # first index is live satellite type, second is derelict/rocket type
        dS_ddt = [] # collisions with de-orbiting satallites
        dS_dDdt = np.zeros((self.num_sat_types, self.num_sat_types)) # collisions between de-orbiting and derelict satallites
        dS_dRdt = np.zeros((self.num_sat_types, self.num_rb_types)) # collisions between de-orbiting satellites and rocket bodies
        dDdt = [] # collisions with derelict satallites
        dDDdt = np.zeros((self.num_sat_types, self.num_sat_types)) # number of collisions between derelict satallites
        dDRdt = np.zeros((self.num_sat_types, self.num_rb_types)) # collisions between derelict satellites and rocket bodies
        dRdt = [] # collisions with rocket bodies
        dRRdt = np.zeros((self.num_rb_types, self.num_rb_types)) # collisions between rocket bodies
        decay_N = np.zeros(N.shape) # rate of debris that decay
        ascend_S = np.zeros(self.num_sat_types) # rate of satellites ascending into a higher orbit
        kill_S = np.zeros(self.num_sat_types) # rate of satellites being put into de-orbit
        deorbit_S = np.zeros(self.num_sat_types) # rate of satellites de-orbiting out of the band
        decay_D = np.zeros(self.num_sat_types) # rate of derelicts that decay
        decay_R = np.zeros(self.num_rb_types) # rate of rocket bodies that decay
        dSdt_tot = np.zeros(self.num_sat_types) # total rate of change for live satellites
        dS_ddt_tot = np.zeros(self.num_sat_types) # total rate of change for de-orbiting satellites
        dDdt_tot = np.zeros(self.num_sat_types) # total rate of change of derelict satellites
        dRdt_tot = np.zeros(self.num_rb_types) # total rate of change for rocket bodies
        CS_dt = []
        CR_dt = []
        expl_S = np.zeros(self.num_sat_types)
        expl_Sd = np.zeros(self.num_sat_types)
        expl_D = np.zeros(self.num_sat_types)
        expl_R = np.zeros(self.num_rb_types)
        for i in range(self.num_sat_types):
            dSdt.append(np.zeros(N.shape))
            dS_ddt.append(np.zeros(N.shape))
            dDdt.append(np.zeros(N.shape))
        for i in range(self.num_rb_types):
            dRdt.append(np.zeros(N.shape))

        for i in range(self.num_sat_types):

            # get current satellite type values
            S = self.satellites[i].S[time]
            S_d = self.satellites[i].S_d[time]
            D = self.satellites[i].D[time]
            sigma = self.satellites[i].sigma
            alpha = self.satellites[i].alpha
            expl_rate_L = self.satellites[i].expl_rate_L
            expl_rate_D = self.satellites[i].expl_rate_D

            # handle debris events with satellites
            for j in range(self.num_L):
                ave_L = 10**((self.logL_edges[j] + self.logL_edges[j+1])/2) # average L value for these bins
                for k in range(self.num_chi):
                    if self.N_factor_table[j,k] != 0: # only calculate for non-ignored bins
                        dSdt[i][j,k], dS_ddt[i][j,k], dDdt[i][j,k] = self.N_sat_events(S, S_d, D, N[j,k], sigma, alpha, ave_L)
        
            # compute collisions involving only satellities
            tot_S_sat_coll = 0 # total collisions destroying live satellites of this type
            tot_Sd_sat_coll = 0 # total collisions destroying de-orbiting satellites of this type
            tot_D_sat_coll = 0 # total collisions destroying derelicts of this type
            for j in range(self.num_sat_types):
                D2 = self.satellites[j].D[time]
                sigma2 = self.satellites[j].sigma
                dSDdt[i,j], dS_dDdt[i,j], dDDdt[i,j] = self.SColl_events(S, S_d, D, sigma, alpha, D2, sigma2)
                if i > j: dDDdt[i,j] = 0 # avoid double counting
                tot_S_sat_coll += dSDdt[i,j]
                tot_Sd_sat_coll += dS_dDdt[i,j]
                if i == j : tot_D_sat_coll += dSDdt[j,i] + dS_dDdt[j,i] + 2*dDDdt[i,j]
                else : tot_D_sat_coll += dSDdt[j,i] + dS_dDdt[j,i] + dDDdt[i,j]

            # compute collisions between satellites and rocket bodies
            for j in range(self.num_rb_types):
                R = self.rockets[j].num[time]
                sigma2 = self.rockets[j].sigma
                dSRdt[i,j], dS_dRdt[i,j], dDRdt[i,j] = self.SRColl_events(S, S_d, D, sigma, alpha, R, sigma2)
                tot_S_sat_coll += dSRdt[i,j]
                tot_Sd_sat_coll += dS_dDdt[i,j]
                tot_D_sat_coll += dRdt[i,j]

            # compute explosions for satellites
            expl_S[i] = expl_rate_L*S/100
            expl_Sd[i] = expl_rate_L*S_d/100
            expl_D[i] = expl_rate_D*D/100

            # compute decay/ascend events for satellites
            up_time = self.satellites[i].up_time
            del_t = self.satellites[i].del_t
            tau_do = self.satellites[i].tau_do
            tau = self.satellites[i].tau
            kill_S[i], deorbit_S[i], decay_D[i] = S/del_t, S_d/tau_do, D/tau
            if self.ascending[i] : ascend_S[i] = S/up_time

            # sum everything up
            P = self.satellites[i].P
            dSdt_tot[i] = S_in[i] - kill_S[i] - np.sum(dSdt[i]) - tot_S_sat_coll - expl_S[i]
            dS_ddt_tot[i] = S_din[i] + P*kill_S[i] - np.sum(dS_ddt[i]) - deorbit_S[i] - tot_Sd_sat_coll - expl_Sd[i]
            dDdt_tot[i] = D_in[i] + (1-P)*kill_S[i] - np.sum(dDdt[i][self.lethal_sat_N[i] == True]) - decay_D[i] - tot_D_sat_coll + np.sum(dSdt[i][self.lethal_sat_N[i] == False]) + np.sum(dS_ddt[i][self.lethal_sat_N[i] == False]) - expl_D[i]
            CS_dt.append(dSdt[i] + dS_ddt[i] + dDdt[i])

        for i in range(self.num_rb_types): # handle rocket body only events

            # get current rocket body values
            R = self.rockets[i].num[time]
            sigma = self.rockets[i].sigma
            lam = self.rockets[i].lam
            expl_rate = self.rockets[i].expl_rate

            # handle rocket-debris collisions
            for j in range(self.num_L):
                for k in range(self.num_chi):
                    if self.N_factor_table[j,k] != 0: # only calculate for non-ignored bins
                        dRdt[i][j,k] = self.N_rb_events(R, N[j,k], sigma)

            # handle rocket-rocket collisions
            tot_R_coll = 0 # total collisions destroying rocket bodies of this type
            for j in range(i, self.num_rb_types): # to avoid double counting, start at i
                R2 = self.rockets[j].num[time]
                sigma2 = self.rockets[j].sigma
                dRRdt[i,j] = self.RColl_events(R, sigma, R2, sigma2)
                if i != j : tot_R_coll += dRRdt[i,j]
                else : tot_R_coll += 2*dRRdt[i,j]

            # handle rocket explosions
            expl_R[i] = expl_rate*R/100

            # sum everything up
            dRdt_tot[i] = lam + R_in - np.sum(dRdt[i][self.lethal_rb_N[i] == True]) - tot_R_coll - expl_R[i]
            CR_dt.append(dRdt[i])

        # calculate rates of decay for debris
        for i in range(self.num_L):
            for j in range(self.num_chi):
                if self.N_factor_table[j,k] != 0: # only calculate for non-ignored bins
                    decay_N[i][j] = N[i,j]/self.tau_N[i,j]

        D_dt = dSDdt + dS_dDdt + dDDdt
        RD_dt = dSRdt + dS_dRdt + dDRdt
        R_dt = dRRdt
        expl_S_tot = expl_S + expl_Sd + expl_D
        return dSdt_tot, dS_ddt_tot, dDdt_tot, dRdt_tot, ascend_S, deorbit_S, decay_D, decay_R, decay_N, D_dt, RD_dt, R_dt, CS_dt, CR_dt, expl_S_tot, expl_R

    def N_sat_events(self, S, S_d, D, N, sigma, alpha, ave_L):
        '''
        calculates the rate of collisions between debris with the given
        ave_L, ave_chi, and tau with stallites of a particular type in a band

        Parameter(s):
        S : number of live satellites in the band of this type
        S_d : number of de-orbiting satellites in the band of this type
        D : number of derelict satellites in the band of this type
        N : number of pieces of debris in the band, of this particular type
        sigma : cross section of the satellites (m^2)
        alpha : fraction of collisions a live satellites fails to avoid
        ave_L : average characteristic length of the debris (m)

        Keyword Parameter(s): None
        
        Output(s):
        dSdt : rate of collisions between debris and live satellites (1/yr)
        dS_ddt : rate of collisions between debris and de-orbiting satellites (1/yr)
        dDdt : rate of collisions between debris and derelict satellites (1/yr)
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
        return dSdt, dS_ddt, dDdt

    def N_rb_events(self, R, N, sigma):
        '''
        calculates the rate of collisions between debris with the given
        ave_L, ave_chi, and tau with stallites of a particular type in a band

        Parameter(s):
        R : number of rocket bodies in the band of this type
        N : number of pieces of debris in the band, of this particular type
        sigma : cross section of the rocket bodies (m^2)
        ave_L : average characteristic length of the debris (m)

        Keyword Parameter(s): None
        
        Output(s):
        dRdt : rate of collisions between debris and rocket bodies (1/yr)
        '''
        sigma /= 1e6 # convert to km^2
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = N/V # number density of the debris

        return n*sigma*v*R

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

    def SRColl_events(self, S1, S_d1, D1, sigma1, alpha1, R, sigma2):
        '''
        calculates the rate of collisions between satellites and rocket bodies of two particular types
        in a band

        Parameter(s):
        S1 : number of live satellites of a particular type
        S_d1 : number of de-orbiting satellites of a particular type
        D1 : number of derelict satellites of a particular type
        sigma1 : cross-section of satellites (m^2)
        alpha1 : fraction of collisions a live satellites fails to avoid
        R : number of rocket bodies of the particular type
        sigma2 : cross-section of rocket bodies (m^2)

        Keyword Parameter(s): None

        Output(s):
        dSRdt : rate of collision between live satellites of type 1 and rocket bodies (1/yr)
        dS_dRdt : rate of collision between de-orbiting satellites of type 1 and rocket bodies (1/yr)
        dDDdt : rate of collision between derelict satellites of type 1 and rocket bodies (1/yr)
        '''

        sigma1 /= 1e6 # convert to km^2
        sigma2 /= 1e6
        sigma = sigma1 + sigma2 + 2*np.sqrt(sigma1*sigma2) # account for increased cross-section
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = R/V # number density of the rockets

        # rate of collisions between satellites and rockets
        dSRdt = alpha1*n*sigma*v*S1
        dS_dRdt = alpha1*n*sigma*v*S_d1
        dDRdt = n*sigma*v*D1  # collisions cannot be avoided
        return dSRdt, dS_dRdt, dDRdt

    def RColl_events(self, R1, sigma1, R2, sigma2):
        '''
        calculates the rate of collisions between rocket bodies of two types in a band

        Parameter(s):
        R1 : number of rocket bodies of a type 1 in the band
        sigma1 : cross-section of rocket bodies of type 1 (m^2)
        R2 : number of rocket bodies of a type 2 in the band
        sigma2 : cross-section of rocket bodies of type 2 (m^2)

        Keyword Parameter(s): None

        Output(s):
        dRRdt : rate of collision between the two types of rocket bodies (1/yr)
        '''

        sigma1 /= 1e6 # convert to km^2
        sigma2 /= 1e6
        sigma = sigma1 + sigma2 + 2*np.sqrt(sigma1*sigma2) # account for increased cross-section
        v = self.v*365.25*24*60*60 # convert to km/yr
        V = 4*np.pi*(6371 + self.alt)**2*self.dh # volume of the shell
        n = R2/V # number density of the rockets of the second type
        return n*sigma*v*R1

    def update_lethal_N(self):
        '''
        updates values in lethal_N based on current mass, v, and bins

        Parameter(s): None

        Keyword Parameter(s): None

        Ouput(s): None
        '''

        for i in range(self.num_L):
            ave_L = 10**((self.logL_edges[i] + self.logL_edges[i+1])/2) # average L value for these bins
            for j in range(self.num_chi):
                ave_chi = (self.chi_edges[j] + self.chi_edges[j+1])/2
                for k in range(self.num_sat_types):
                    self.lethal_sat_N[k][i,j] = is_catastrophic(self.satellites[k].m, ave_L, 10**ave_chi, self.v)
                for k in range(self.num_rb_types):
                    self.lethal_rb_N[k][i,j] = is_catastrophic(self.rockets[k].m, ave_L, 10**ave_chi, self.v)
