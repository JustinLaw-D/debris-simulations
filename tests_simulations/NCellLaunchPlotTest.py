import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./NCellLaunchData/')
t = atmosphere.get_t()
T = t[-1]
S0, S1, S2 = atmosphere.get_S()[0][0], atmosphere.get_S()[3][1], atmosphere.get_S()[-1][2]
D0, D1, D2 = atmosphere.get_D()[0][0], atmosphere.get_D()[3][1], atmosphere.get_D()[-1][2]
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
ax1.set_ylim(1, 1e6)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
