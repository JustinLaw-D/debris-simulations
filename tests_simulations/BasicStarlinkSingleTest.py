# test of starlink system, over a range of alpha and launch rate parameters
# neglecting outside contributions to the system besides the current debris
# environment

import sys
sys.path.append('./../')
sys.path.append('./../catalog_data/')

from copy import deepcopy
from NCell import NCell
from AtmosphericDecayModels import drag_lifetime, need_update
from data_utilities import *
import numpy as np
from multiprocessing import Pool
from time import time

def to_run(atmosphere, lamfac=None, alpha=None):
    directory = "BasicStarlinkSingleData/"
    name = str(lamfac) + "x" + str(alpha)
    T_loc = 0.1
    while T_loc <= T:
        atmosphere.run_sim_precor(T_loc, dt=0.1, mindtfactor=10000)
        print(name + " done to T = " + str(T_loc) + " at " + str(time())) 
        atmosphere.save(directory, name, gap=0.01, force=True)
        print("Saved at " + str(time()))
        T_loc += 0.1
    return True

def gen_atmosphere(args):
    S_i, D_i, N_i, alt_edges, lam, drag_lifetime, update_lifetime, m_s, R_i, expl_rate_D, expl_rate_R, alpha = args
    return NCell(S_i, D_i, N_i, alt_edges, lam, drag_lifetime, update_lifetime, m_s=m_s, R_i=R_i, expl_rate_D=expl_rate_D, expl_rate_R=expl_rate_R, alphaN=alpha)

if __name__ == '__main__':
    STARLINK_NAME = "./../catalog_data/starlink_data.json"
    DEBRIS_NAME = "./../catalog_data/debris_data.json"
    SATELLITE_NAME = "./../catalog_data/satellite_data.json"
    ROCKET_NAME = "./../catalog_data/rocketbody_data.json"
    lower_alts = [332.5, 337.5, 342.5, 347.5] # lower altitude bands of starlink satellites
    upper_alts = [535, 545, 555, 565, 575]
    max_alt = 585 # maximum altitude the satellites appear at
    min_alt = 270 # minimum altitude the satellites appear at
    alt_edges = np.array([min_alt] + lower_alts + upper_alts + [max_alt])
    S_i = []
    D_i = []
    R_i = []
    N_i = []
    expl_rate_D = np.full(4, 1) # assume 1 explosion per year per 100 objects
    expl_rate_R = np.array([1])
    for i in range(len(alt_edges)-1): # initialize value lists (v0.9, 4 types each of v1.0, v1.5, 7 types of v2.0, 1 type for derelicts)
        S_i.append(np.zeros(4))
        D_i.append(np.zeros(4))
        R_i.append(np.zeros(1))
        N_i.append(0)
    print("Getting Data at " + str(time()))
    S_data = get_starlink(STARLINK_NAME, alt_edges)
    D_data = get_objects(SATELLITE_NAME, alt_edges)
    N_data = get_objects(DEBRIS_NAME, alt_edges)
    R_data = get_objects(ROCKET_NAME, alt_edges)
    print("Parsing Data at " + str(time()))
    for i in range(len(alt_edges)-1): # go through and actually fill the values, right now there's no v2.0 launched
        S_loc = S_data[i]
        S_i[i][0] += S_loc['v0.9']
        S_i[i][1] += S_loc['v1.0']
        S_i[i][2] += S_loc['v1.5']
        D_i[i][0] += D_data[i]/3
        D_i[i][1] += D_data[i]/3
        D_i[i][2] += D_data[i]/3
        R_i[i][0] += R_data[i]
        N_i[i] += int(N_data[i])
    target_num = [0, 2493, 2478, 2547, 0, 1584, 1584, 520, 720, 0] # target number of starlink satellites in each orbit
    lam_new = np.zeros(len(alt_edges)-1) # launch rates of v2.0 for each altitude
    for i in range(len(lam_new)):
        lam_new[i] = max((target_num[i] - np.sum(S_i[i]))/5.5, 0) # happens over 5.5 years (approximately)

    m_s = [227, 260, 295, 1250]

    def drag_lifetime_loc(hmax, hmin, a_over_m, t):
        m0 = int(t*12) % 144
        return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)

    print("Generating atmosphere at " + str(time()))
    lam = []
    for i in range(len(alt_edges)-1):
        lam.append([0,0,0,lam_new[i]])
    print(lam)
    print(S_i)
    print(D_i)
    alpha_options = []
    alpha = [0]*(len(alt_edges)-1)
    for k in range(len(alpha)):
        alpha[k] = np.full(4, 0.2)
    alpha_options.append([deepcopy(S_i), deepcopy(D_i), deepcopy(N_i), deepcopy(alt_edges),
                          deepcopy(lam), drag_lifetime_loc, need_update, deepcopy(m_s), deepcopy(R_i), deepcopy(expl_rate_D),
                          deepcopy(expl_rate_R), alpha])
    atmosphere = gen_atmosphere(alpha_options[0])
    print("Generation done at " + str(time()))
    print(atmosphere.get_N())

    T = 50 # how long each simulation runs for

    # launch the simulations
    print("Starting Simulation at " + str(time()))
    to_run(atmosphere, lamfac=1, alpha=0.2)
    print("Done Simulating at " + str(time()))
