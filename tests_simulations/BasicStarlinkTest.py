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

def to_run(atmosphere, lamfac=None, alpha=None):
    directory = "BasicStarlinkData/"
    name = str(lamfac) + "x" + str(alpha)
    atmosphere.run_sim_precor(T, mindtfactor=10000)
    atmosphere.save(directory, name, gap=0.01)
    return True

def gen_atmosphere(args):
    S_i, SD_i, D_i, N_i, target_alts, alt_edges, lam, drag_lifetime, update_lifetime, m_s, alpha = args
    return NCell(S_i, SD_i, D_i, N_i, target_alts, alt_edges, lam, drag_lifetime, update_lifetime, m_s=m_s, alphaN=alpha)

if __name__ == '__main__':
    STARLINK_NAME = "./../catalog_data/starlink_data.json"
    DEBRIS_NAME = "./../catalog_data/debris_data.json"
    lower_alts = [332.5, 337.5, 342.5, 347.5] # lower altitude bands of starlink satellites
    upper_alts = [535, 545, 555, 565, 575]
    max_alt = 585 # maximum altitude the satellites appear at
    min_alt = 270 # minimum altitude the satellites appear at
    alt_edges = np.array([min_alt] + lower_alts + upper_alts + [max_alt])
    target_alts = [550, 540, 550, 560, 570, 540, 550, 560, 570, 335.9,
                340.8, 345.6, 540, 550, 560, 570]
    S_i = []
    SD_i = []
    D_i = []
    N_i = []
    for i in range(len(alt_edges)-1): # initialize value lists (v0.9, 4 types each of v1.0, v1.5, 7 types of v2.0)
        S_i.append(np.zeros(16))
        SD_i.append(np.zeros(16))
        D_i.append(np.zeros(16))
        N_i.append(0)
    for i in range(len(alt_edges)-1): # go through and actually fill the values, right now there's no v2.0 launched
        S_loc = get_starlink(STARLINK_NAME, alt_edges[i], alt_edges[i+1])
        S_i[i][0] += S_loc['v0.9']
        if alt_edges[i] < 540 :
            S_i[i][1:5] += S_loc['v1.0']/4
            S_i[i][5:9] += S_loc['v1.5']/4
        elif alt_edges[i] < 550 :
            S_i[i][2:5] += S_loc['v1.0']/3
            S_i[i][6:9] += S_loc['v1.5']/3
        elif alt_edges[i] < 560 :
            S_i[i][3:5] += S_loc['v1.0']/2
            S_i[i][7:9] += S_loc['v1.5']/2
        else:
            S_i[i][4] += S_loc['v1.0']
            S_i[i][8] += S_loc['v1.5']
        N_i[i] += get_debris(DEBRIS_NAME, alt_edges[i], alt_edges[i+1])
    target_num = [2493, 2478, 2547, 0, 1584, 1584, 520, 720] # target number of satellites in each orbit
    lam_start = [0]*9 # none of the old versions of satellites are launched
    lam_new = np.zeros(7) # launch rates of v2.0 for each target altitude
    j = 0
    for i in range(len(lam_new)):
        if i < 3 : j = i+1
        else : j = i+2
        lam_new[i] = max((target_num[i] - np.sum(S_i[j]))/5.5, 0) # happens over 5.5 years (approximately)

    m_s = [227, 260, 260, 260, 260, 295, 295, 295, 295, 
        1250, 1250, 1250, 1250, 1250, 1250, 1250]

    lam_factors = np.linspace(1, 2, 10) # range of launch rate factors to consider
    alphas = np.linspace(0.01, 0.2, 20) # range of alphas to consider

    def drag_lifetime_loc(hmax, hmin, a_over_m, t):
        m0 = int(t*12) % 144
        return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)

    atmospheres = []
    with Pool(processes=20) as pool:
        print("Generating atmospheres")
        for i in range(len(lam_factors)):
            lam = lam_start + (lam_new*lam_factors[i]).tolist()
            alpha_options = []
            for j in range(len(alphas)):
                alpha = [0]*(len(alt_edges)-1)
                for k in range(len(alpha)):
                    alpha[k] = np.full(16, alphas[j])
                alpha_options.append([deepcopy(S_i), deepcopy(SD_i), deepcopy(D_i), deepcopy(N_i), deepcopy(target_alts), deepcopy(alt_edges),
                                      deepcopy(lam), drag_lifetime_loc, need_update, deepcopy(m_s), alpha])

            print("Generating atmospheres for lam_factor = " + str(lam_factors[i]))
            atmospheres.extend(list(pool.map(gen_atmosphere, iter(alpha_options))))
    print("Generation done")

    T = 50 # how long each simulation runs for

    # launch the simulations
    with Pool(processes=10) as pool:
        print("Starting Simulations")
        for i in range(len(lam_factors)):
            for j in range(len(alphas)):
                lamfac = lam_factors[i]
                alpha = alphas[j]
                print("Starting simulation " + str(i) + ", " + str(j))
                pool.apply_async(to_run, (atmospheres[i][j],), {'lamfac' : lamfac, 'alpha' : alpha})
        pool.close()
        pool.join()
        print("Done Simulating")
