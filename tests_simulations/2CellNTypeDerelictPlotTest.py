import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./test_save_2CellNTypeDerelict/')
t = atmosphere.get_t()
T = t[-1]
D00, D10 = atmosphere.get_D()[0][0], atmosphere.get_D()[1][0]
D01, D11 = atmosphere.get_D()[0][1], atmosphere.get_D()[1][1]
D02, D12 = atmosphere.get_D()[0][2], atmosphere.get_D()[1][2]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
ax1.plot(t, D00, label='D00')
ax1.plot(t, D10, label='D10')
ax1.plot(t, D01, label='D01')
ax1.plot(t, D11, label='D11')
ax1.plot(t, D02, label='D02')
ax1.plot(t, D12, label='D12')
ax1.set_ylim(1, 200)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
