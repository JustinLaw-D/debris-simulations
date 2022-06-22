import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./test_save_2CellNTypeRocket/')
t = atmosphere.get_t()
T = t[-1]
R00, R10 = atmosphere.get_R()[0][0], atmosphere.get_R()[1][0]
R01, R11 = atmosphere.get_R()[0][1], atmosphere.get_R()[1][1]
R02, R12 = atmosphere.get_R()[0][2], atmosphere.get_R()[1][2]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, R00, label='R00')
ax1.plot(t, R10, label='R10')
ax1.plot(t, R01, label='R01')
ax1.plot(t, R11, label='R11')
ax1.plot(t, R02, label='R02')
ax1.plot(t, R12, label='R12')
ax1.set_ylim(1, 200)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
