import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np
 
atmosphere = NCell.load("./test_save_NRocketFlow/")
t = atmosphere.get_t()
T = t[-1]
R = atmosphere.get_R()

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
for i in range(0, len(R)):
    ax1.plot(t, R[i][0], label='R'+str(i))
ax1.set_ylim(1e-2, 1e2)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
