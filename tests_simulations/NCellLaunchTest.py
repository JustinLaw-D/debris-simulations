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
lam = [500, 500, 100]
T = 60
def drag_lifetime_loc(hmax, hmin, a_over_m):
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1)
atmosphere = NCell(S_i, S_di, D_i, N_i, [600, 750, 900],  alts, dhs, lam, drag_lifetime_loc)

atmosphere.run_sim_euler(T, dt=1/5000)
t = atmosphere.get_t()
S0, S1, S2 = atmosphere.get_S()[0], atmosphere.get_S()[3], atmosphere.get_S()[-1]
D0, D1, D2 = atmosphere.get_D()[0], atmosphere.get_D()[3], atmosphere.get_D()[-1]
N0, N1, N2 = atmosphere.get_N()[0], atmosphere.get_N()[3], atmosphere.get_N()[-1]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, S0, label='S0')
ax1.plot(t, S1, label='S1')
ax1.plot(t, S2, label='S2')
ax1.plot(t, D0, label='D0')
ax1.plot(t, D1, label='D1')
ax1.plot(t, D2, label='D2')
ax1.plot(t, N0, label='N0')
ax1.plot(t, N1, label='N1')
ax1.plot(t, N2, label='N2')
ax1.set_ylim(100, 1e10)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
