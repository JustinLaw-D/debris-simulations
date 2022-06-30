import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np
 
atmosphere = NCell.load("./test_save_NDebrisFlow/")
t = atmosphere.get_t()
T = t[-1]
N = atmosphere.get_N()
dh = 50 # height of band (km)
alt_bins = np.arange(600-dh/2, 900+dh/2+1, dh)

import matplotlib.pyplot as plt

t_last = 0
x = list()
for i in range(len(N)):
    x += [(alt_bins[i] + alt_bins[i+1])/2]*int(N[i][0])
plt.hist(x, bins=alt_bins)
plt.savefig('./../../interesting_plots/debris_histograms/t0.png')

count = 1
for i in range(len(t)):
    if t[i] - t_last >= 1:
        plt.clf()
        t_last = t[i]
        x = []
        for j in range(len(N)):
            x += [(alt_bins[j] + alt_bins[j+1])/2]*int(N[j][i])
        plt.hist(x, bins=alt_bins)
        plt.savefig('./../../interesting_plots/debris_histograms/t' + str(count) + '.png')
        count += 1

