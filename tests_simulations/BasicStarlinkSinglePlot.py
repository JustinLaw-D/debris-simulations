import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np

atmosphere = NCell.load('./BasicStarlinkSingleFixedData/1x0.2/')
t = atmosphere.get_t()
T = t[-1]
S = []
D = []
S_obj = atmosphere.get_S()
D_obj = atmosphere.get_D()
N = atmosphere.get_N()
for i in range(len(S_obj)):
    S_loc = []
    D_loc = []
    for j in range(len(t)):
        sum_S_loc = 0
        sum_D_loc = 0
        for k in range(4):
            sum_S_loc += S_obj[i][k][j]
            sum_D_loc += D_obj[i][k][j]
        S_loc.append(sum_S_loc)
        D_loc.append(sum_D_loc)
    S.append(S_loc)
    D.append(D_loc)

import matplotlib.pyplot as plt

for i in range(4):
    print(S[i][-1])

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (yr)')
ax1.set_ylabel('log(number)')
ax1.set_yscale('log')
for i in range(len(S)):
    ax1.plot(t, S[i], label='S'+str(i))
    ax1.plot(t, N[i], label='N'+str(i))
ax1.set_ylim(1, 1e5)
ax1.set_xlim(0,T)
ax1.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
