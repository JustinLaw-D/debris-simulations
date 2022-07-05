import sys
sys.path.append('./../')

from NCell import NCell
from AtmosphericDecayModels import drag_lifetime, need_update
import numpy as np
R = 6371 # radius of earth in km
dh = 50 # height of band (km)
alts = np.arange(600, 900+dh/2, dh)
alt_edges = np.arange(575, 925+1, dh)
V = 4*np.pi*dh*(R+alts)**2 # volume of band
S_i = [[0,0,0]]*len(alts)
D_i = [[0,0,0]]*len(alts)
S_di = [[0,0,0]]*len(alts)
N_i = np.zeros(len(alts), np.int64)
for i in range(len(alts)):
    N_i[i] = int(2.5e-8*V[i])
lam = [1000, 1000, 1000]
T = 60
def drag_lifetime_loc(hmax, hmin, a_over_m, t):
    m0 = (t*12) % 144
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)
atmosphere = NCell(S_i, S_di, D_i, N_i, [600, 750, 900],  alt_edges, lam, drag_lifetime_loc, need_update)

atmosphere.run_sim_precor(T, mindtfactor=10000, upper=True)
atmosphere.save("./", "NCellLaunchData", gap=0.1, force=True)
