# Test of stable single-cell system, using drag lifetime from JASON model

import sys
sys.path.append('./../')

from NCell import NCell
from ObjectsEvents import Event, ExplEvent
from AtmosphericDecayModels import drag_lifetime, need_update
import numpy as np

class MyEvent(Event):

    def run_event(self, S, S_d, D, R, N, logL_edges, chi_edges):
        dN = np.zeros((len(logL_edges)-1, len(chi_edges)-1))
        dN[3,3] = 1e8
        return 0, 0, 0, 0, dN, [],[]

R = 6371 # radius of earth in km
alt = 600 # altitude of Starlink satellites (km)
dh = 25 # height of band (km)
V = 4*np.pi*dh*(R+alt)**2 # volume of band
S_i = [0]
S_di = [0]
D_i = [0]
N_i = int(2.5e-8*V)
lam = 2000
T = 50
events = [MyEvent(alt, time=[1,10]), ExplEvent(600, [(1,'rb',1)],freq=10)]
def drag_lifetime_loc(hmax, hmin, a_over_m, t):
    m0 = int(t*12) % 144
    return drag_lifetime(hmax, hmin, 0, 0, a_over_m=a_over_m, dt=100/(60*60*24*365.25), maxdt=0.1, m0=m0)
atmosphere = NCell([S_i], [S_di], [D_i], [N_i], [alt], [alt], [dh], [lam], drag_lifetime_loc, need_update, tau_do=[[2]], events=events)
atmosphere.run_sim_precor(T, dt=1, mindtfactor=1000, tolerance=2)
t = atmosphere.get_t()
S = atmosphere.get_S()[0][0]
S_d = atmosphere.get_SD()[0][0]
D = atmosphere.get_D()[0][0]
N = atmosphere.get_N()[0]
C = atmosphere.get_C()[0]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, S, label='S')
ax1.plot(t, S_d, label='S_d')
ax1.plot(t, D, label='D')
ax1.plot(t, N, label='N')
ax1.set_ylim(100, 1e9)
ax1.set_xlim(0,T)
ax1.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('log(collisions)')  # we already handled the x-label with ax1
ax2.plot(t, C, label='C', color='k')
ax2.set_ylim(1e-3, 1e8)
ax2.set_yscale('log')
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
