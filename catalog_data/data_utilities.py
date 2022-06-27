# contains various functions for processing data pulled from SpaceTrack

from datetime import date
from scipy.optimize import brentq
import numpy as np
import json

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Re = 6371 # radius of Earth (km)
Me = 5.97219e24 # mass of Earth (kg)


def get_objects(filename, bin_edges, num=1000, tol=1e-3):
    '''
    gets the amount of objects (as stored in file) in each altitude bin

    Input(s):
    filename : path to the file, either absolute or relative to the current directory
    bind_edges : altitude bin edges (km)

    Keyword Input(s):
    num : number of samples to take in eccentric anomoly (default 1000)
    tol : absolute tolerance in numerical solver for eccentric anomoly (default 1e-3)

    Output(s):
    bin_num : number of objects in each bin range
    '''

    bin_num = np.zeros(len(bin_edges) - 1) # counter for number of objects in the bin
    ma = np.linspace(0, np.pi, num) # range of mean anomolies to sample
    eca = np.zeros(ma.shape) # range of eccentric anomolies to sample

    with open(filename, 'r') as file:
        data = json.load(file)
        for debris in data: # loop through the data
            try: 
                P = float(debris['PERIOD'])*60 # get relevant values
                r_max = float(debris['APOGEE'])
                r_min = float(debris['PERIGEE'])
            except (ValueError, TypeError): continue # ignore data that can't be used
            ecc = (r_max - r_min)/(r_max + r_min + 2*Re) # eccentricity of orbit
            if r_min > bin_edges[-1] or r_max < bin_edges[0] : continue # skip if you can
            for i in range(len(ma)):
                to_solve = lambda E : E - ecc*np.sin(E) - ma[i]
                eca[i] = brentq(to_solve, 0, np.pi, xtol=tol)
            a = np.cbrt(((P**2)*G*Me)/(4*np.pi**2))/1000 # semi-major axis (km)
            alt_samp = a*(1 - ecc*np.cos(eca)) - Re # list of altitude samples
            j = 0
            bin_num_loc = np.zeros(bin_num.shape)
            for i in range(len(bin_num)): # bin the objects
                while j < num and bin_edges[i] < alt_samp[j] and bin_edges[i+1] >= alt_samp[j]:
                    bin_num_loc[i] += 1
                    j += 1
            bin_num += bin_num_loc/len(alt_samp)

    return bin_num

def get_starlink(filename, bin_edges, num=1000, tol=1e-3):
    '''
    gets the amount of starlink satellites of each type in each altitude bin

    Input(s):
    filename : path to the file, either absolute or relative to the current directory
    bind_edges : altitude bin edges (km)

    Keyword Input(s):
    num : number of samples to take in eccentric anomoly (default 1000)
    tol : absolute tolerance in numerical solver for eccentric anomoly (default 1e-3)

    Output(s):
    bin_num : list of dictionaries of number of satellites of each type in each altitude range

    Note : no Starlink satellites in orbit are currently v2.0
    '''

    bin_num = [] # counter for number of Starlink satellites in the bin
    for i in range(len(bin_edges) - 1): bin_num.append({'v0.9' : 0, 'v1.0' : 0, 'v1.5' : 0, 'v2.0' : 0})
    ma = np.linspace(0, np.pi, num) # range of mean anomolies to sample
    eca = np.zeros(ma.shape) # range of eccentric anomolies to sample

    with open(filename, 'r') as file:
        v10_date = date(2019, 11, 11) # starting launch dates for version 1.x types
        v15_date = date(2021, 1, 24)
        data = json.load(file)
        for debris in data: # loop through the data
            try: 
                P = float(debris['PERIOD'])*60 # get relevant values
                r_max = float(debris['APOGEE'])
                r_min = float(debris['PERIGEE'])
                launch_lst = debris['LAUNCH'].split(sep='-')
                launch_date = date(int(launch_lst[0]), int(launch_lst[1]), int(launch_lst[2]))
            except (ValueError, TypeError, IndexError): continue # ignore data that can't be used
            ecc = (r_max - r_min)/(r_max + r_min + 2*Re) # eccentricity of orbit
            if r_min > bin_edges[-1] or r_max < bin_edges[0] : continue # skip if you can
            for i in range(len(ma)):
                to_solve = lambda E : E - ecc*np.sin(E) - ma[i]
                eca[i] = brentq(to_solve, 0, np.pi, xtol=tol)
            a = np.cbrt(((P**2)*G*Me)/(4*np.pi**2))/1000 # semi-major axis (km)
            alt_samp = a*(1 - ecc*np.cos(eca)) - Re # list of altitude samples
            j = 0
            for i in range(len(bin_num)): # bin the satellite
                temp_dict = {'v0.9' : 0, 'v1.0' : 0, 'v1.5' : 0, 'v2.0' : 0}
                while j < num and bin_edges[i] < alt_samp[j] and bin_edges[i+1] >= alt_samp[j]:
                    if launch_date < v10_date : temp_dict['v0.9'] += 1
                    elif launch_date < v15_date : temp_dict['v1.0'] += 1
                    else : temp_dict['v1.5'] += 1
                    j += 1
                for key in bin_num[i].keys():
                    bin_num[i][key] += temp_dict[key]/len(alt_samp)

    return bin_num
