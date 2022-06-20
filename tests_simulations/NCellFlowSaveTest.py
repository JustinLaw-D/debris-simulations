import sys
sys.path.append('./../')

from NCell import NCell
from AtmosphericDecayModels import drag_lifetime 
import numpy as np
R = 6371 # radius of earth in km
dh = 50 # height of band (km)
alts = np.arange(600, 910, dh)
V = 4*np.pi*dh*(R+alts)**2 # volume of band
dhs = np.zeros(len(alts))
dhs.fill(dh)
S_i = [[0]]*len(alts)
S_di = [[0]]*len(alts)
D_i = [[0]]*len(alts)
N_i = np.zeros(len(alts), dtype=np.int64)
N_i[-1] = int(2.5e-8*V[-1])
target_alts = [500]
lam = [0]
tau_min = [1e-2]*len(alts)
T = 30
def drag_lifetime_loc(hmax, hmin, a_over_m):
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1)
atmosphere = NCell(S_i, S_di, D_i, N_i, target_alts, alts, dhs, lam, drag_lifetime_loc, tau_min=tau_min)

atmosphere.run_sim_euler(T, dt=0.01, upper=False)
atmosphere.save('./', "test_save", gap=1, force=True)
