import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./test_save_2CellNTypeLive/')
t = atmosphere.get_t()
T = t[-1]
S00, S10 = atmosphere.get_S()[0][0], atmosphere.get_S()[1][0]
S01, S11 = atmosphere.get_S()[0][1], atmosphere.get_S()[1][1]
S02, S12 = atmosphere.get_S()[0][2], atmosphere.get_S()[1][2]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, S00, label='S00')
ax1.plot(t, S10, label='S10')
ax1.plot(t, S01, label='S01')
ax1.plot(t, S11, label='S11')
ax1.plot(t, S02, label='S02')
ax1.plot(t, S12, label='S12')
ax1.set_ylim(1, 1e3)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
