import sys
sys.path.append('./../')

from NCell import NCell
from AtmosphericDecayModels import drag_lifetime, need_update
import numpy as np
R = 6371 # radius of earth in km
dh = 50 # height of band (km)
alts = np.arange(600, 910, dh)
V = 4*np.pi*dh*(R+alts)**2 # volume of band
dhs = np.zeros(len(alts))
dhs.fill(dh)
S_i=[]
S_di=[]
D_i=[]
tau_do=[]
for i in range(len(alts)):
    S_i.append([0])
    S_di.append([0])
    D_i.append([0])
    tau_do.append([10])
N_i = np.zeros(len(alts), dtype=np.int64)
S_di[-1][0] = 100
target_alts = [500]
lam = [0]
T = 30
def drag_lifetime_loc(hmax, hmin, a_over_m, t):
    m0 = (t*12) % 144
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)
atmosphere = NCell(S_i, S_di, D_i, N_i, target_alts, alts, dhs, lam, drag_lifetime_loc, need_update, chi_max=1.5, tau_do=tau_do)

atmosphere.run_sim_precor(T, upper=False)
atmosphere.save('./', "test_save_NDeorbitFlow", gap=0.05, force=True)
