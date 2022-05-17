from NCell import NCell
from AtmosphericDecayModels import drag_lifetime 
import numpy as np
R = 6371 # radius of earth in km
dh = 50 # height of band (km)

def get_atmosphere():
    alts = np.arange(600, 910, dh)
    V = 4*np.pi*dh*(R+alts)**2 # volume of band
    dhs = np.zeros(len(alts))
    dhs.fill(dh)
    S_i = np.zeros(len(alts))
    D_i = np.zeros(len(alts))
    N_i = np.zeros(len(alts))
    for i in range(len(alts)):
        N_i[i] = 2.5e-8*V[i]
    lam = np.zeros(len(alts))
    lam[0] = 2000
    lam[3] = 2000
    lam[-1] = 2000
    T = 250
    def drag_lifetime_loc(hmax, hmin):
        return drag_lifetime(hmax, hmin, 0, 0, a_over_m=(1/(20*2.2)), dt=100/(60*60*24*365.25), maxdt=0.1)
    return NCell(S_i, D_i, N_i, alts, dhs, lam, drag_lifetime_loc)
