import sys
sys.path.append('./../')

from NCell import NCell
import numpy as np
R = 6371 # radius of earth in km
alt1, alt2 = 600, 625 # altitude of Starlink satellites (km)
dh = 25 # height of bands (km)
V1, V2 = 4*np.pi*dh*(R+alt1)**2, 4*np.pi*dh*(R+alt2)**2 # volume of bands
S_i = [[0,0,0],[0,0,0]]
S_di = [[0,0,0],[0,0,0]]
D_i = [[0,0,0],[50,1e2,2e2]]
N_i1, N_i2 = int(0), int(0)
lam = [0,0,0]
target_alts = [alt2,alt2,alt2]
T = 50
def drag_lifetime(_a, _b, _c, _d):
    return 5
def update_lifetime(_a, _b):
    return False
atmosphere = NCell(S_i, S_di, D_i, [N_i1, N_i2], target_alts, [alt1, alt2], [dh, dh], lam, drag_lifetime, update_lifetime)

atmosphere.run_sim_precor(T, upper=False)
atmosphere.save('./', 'test_save_2CellNTypeDerelict', gap=0.01, compress=True)
