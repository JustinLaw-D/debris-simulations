import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np
R = 6371 # radius of earth in km
alt1, alt2 = 600, 625 # altitude of Starlink satellites (km)
dh = 25 # height of bands (km)
V1, V2 = 4*np.pi*dh*(R+alt1)**2, 4*np.pi*dh*(R+alt2)**2 # volume of bands
S_i = [[0],[0]]
S_di = [[0],[0]]
D_i = [[0],[1e2]]
N_i1, N_i2 = int(0), int(0)
lam = 0
T = 50
def drag_lifetime(_a, _b, _c, _d):
    return 5
def update_lifetime(_a, _b):
    return False
atmosphere = NCell(S_i, S_di, D_i, [N_i1, N_i2], [alt2], [alt1, alt2], [dh, dh], [lam], drag_lifetime, update_lifetime)

atmosphere.run_sim_euler(T, dt=0.01, upper=False)
t = atmosphere.get_t()
D0, D1 = atmosphere.get_D()[0][0], atmosphere.get_D()[1][0]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, D0, label='D0')
ax1.plot(t, D1, label='D1')
ax1.set_ylim(1, 100)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
