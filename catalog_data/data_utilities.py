# contains various functions for processing data pulled from SpaceTrack

import numpy as np
import json

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Re = 6371 # radius of Earth (km)
Me = 5.97219e24 # mass of Earth (kg)

def get_debris(filename, alt_min, alt_max):
    '''
    gets the amount of trackable debris in an altitude range

    Input(s):
    filename : path to the file, either absolute or relative to the current directory
    alt_min : minimum altitude (km)
    alt_max : maximum altitude (km)

    Keyword Input(s): None

    Output(s):
    num : number of trackable debris in the given range
    '''

    num = 0 # counter for number of debris in the bin

    with open(filename, 'r') as file:
        data = json.load(file)
        for debris in data: # loop through the data
            try: 
                P = float(debris['PERIOD'])*60 # get relevant values
                r_max = float(debris['APOGEE']) + Re
                r_min = float(debris['PERIGEE']) + Re
            except (ValueError, TypeError): continue # ignore data that can't be used
            ecc = (r_max - r_min)/(r_max + r_min) # eccentricity of orbit
            a = np.cbrt(((P**2)*G*Me)/(4*np.pi**2)) # semi-major axis (m)
            alt_ave = a*(1+0.5*ecc**2)/1000 - Re # average orbital altitude (km)
            if alt_ave > alt_min and alt_ave <= alt_max : num += 1
    return num