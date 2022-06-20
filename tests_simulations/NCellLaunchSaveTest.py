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
S_i = [[0,0,0]]*len(alts)
D_i = [[0,0,0]]*len(alts)
S_di = [[0,0,0]]*len(alts)
N_i = np.zeros(len(alts), np.int64)
for i in range(len(alts)):
    N_i[i] = int(2.5e-8*V[i])
lam = [500, 500, 50]
T = 60
def drag_lifetime_loc(hmax, hmin, a_over_m):
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1)
atmosphere = NCell(S_i, S_di, D_i, N_i, [600, 750, 900],  alts, dhs, lam, drag_lifetime_loc)

atmosphere.run_sim_euler(T, dt=1/5000)
atmosphere.save("./", "NCellLaunchData", gap=0.5, force=True)
