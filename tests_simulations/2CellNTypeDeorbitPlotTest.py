import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./test_save_2CellNTypeDeorbit/')
t = atmosphere.get_t()
T = t[-1]
SD00, SD10 = atmosphere.get_SD()[0][0], atmosphere.get_SD()[1][0]
SD01, SD11 = atmosphere.get_SD()[0][1], atmosphere.get_SD()[1][1]
SD02, SD12 = atmosphere.get_SD()[0][2], atmosphere.get_SD()[1][2]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, SD00, label='SD00')
ax1.plot(t, SD10, label='SD10')
ax1.plot(t, SD01, label='SD01')
ax1.plot(t, SD11, label='SD11')
ax1.plot(t, SD02, label='SD02')
ax1.plot(t, SD12, label='SD12')
ax1.set_ylim(1, 200)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
