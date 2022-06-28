import sys
sys.path.append('./../')
sys.path.append('./../catalog_data')

from NCell import NCell
from AtmosphericDecayModels import drag_lifetime, need_update
from data_utilities import *
import numpy as np

R = 6371 # radius of earth in km
dh = 50 # height of band (km)
alt_bins = np.arange(600-dh/2, 900+dh/2+1, dh)
N_data = get_objects('./../catalog_data/debris_data.json', alt_bins)
N_i = []
S_i=[]
S_di=[]
D_i=[]
for i in range(len(alt_bins)-1):
    S_i.append([0])
    S_di.append([0])
    D_i.append([0])
    N_i.append(int(N_data[i]))
target_alts = [500]
lam = [0]
T = 100
def drag_lifetime_loc(hmax, hmin, a_over_m, t):
    m0 = (t*12) % 144
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)
atmosphere = NCell(S_i, S_di, D_i, N_i, target_alts, alt_bins, lam, drag_lifetime_loc, need_update, chi_max=1.5)

atmosphere.run_sim_precor(T, upper=False)
atmosphere.save('./', "test_save_NDebrisFlow", gap=0.05, force=True)
