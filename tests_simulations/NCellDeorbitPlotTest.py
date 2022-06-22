import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np
 
atmosphere = NCell.load("./test_save_NDeorbitFlow/")
t = atmosphere.get_t()
T = t[-1]
S_d = atmosphere.get_SD()

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
for i in range(0, len(S_d)):
    ax1.plot(t, S_d[i][0], label='SD'+str(i))
ax1.set_ylim(1, 1e4)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
