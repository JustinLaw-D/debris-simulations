# contains models for atmospheric density, drag lifetime

import numpy as np

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)
Re = 6371 # radius of Earth (km)

altitudes = list(range(100, 901, 20)) # altitudes used by the model (km)
# density for low activity (kg/m^3)
rho_low = [5.31e-7,2.18e-8,3.12e-9,9.17e-10,3.45e-10,1.47e-10,6.96e-11,3.54e-11,1.88e-11,1.03e-11,5.86e-12,3.40e-12,2.02e-12,1.22e-12,7.46e-13,4.63e-13,2.92e-13,1.87e-13,1.21e-13,8.04e-14,5.44e-14,3.77e-14,2.68e-14,1.96e-14,1.47e-14,1.14e-14,9.10e-15,7.41e-15,6.16e-15,5.22e-15,4.50e-15,3.93e-15,3.48e-15,3.10e-15,2.79e-15,2.53e-15,2.30e-15,2.11e-15,1.94e-15,1.78e-15,1.65e-15] 
# density for high activity (kg/m^3)
rho_high = [5.44e-7,2.45e-8,4.32e-9,1.54e-9,7.40e-10,4.10e-10,2.46e-10,1.56e-10,1.04e-10,7.12e-11,5.00e-11,3.59e-11,2.61e-11,1.93e-11,1.44e-11,1.09e-11,8.32e-12,6.40e-12,4.96e-12,3.87e-12,3.04e-12,2.40e-12,1.91e-12,1.52e-12,1.22e-12,9.82e-13,7.93e-13,6.43e-13,5.22e-13,4.25e-13,3.47e-13,2.84e-13,2.34e-13,1.92e-13,1.59e-13,1.32e-13,1.10e-13,9.21e-14,7.72e-14,6.50e-14,5.49e-14]

# convert altitudes to m
rho_lowl = np.log10(rho_low)
rho_highl = np.log10(rho_high)
altitudesl = np.log10(altitudes) + 3

def density(alt, t, phase, ave_phase):
    '''
    Calculates the atmospheric density at a given altitude via interpolation

    Parameter(s):
    alt : altitude (km)
    t : time since arbitrary start point (yr)
    phase : phase of the solar cycle at t = 0 (rad)
    ave_phase : average value over all phases (bool)

    Keyword Parameter(s): None

    Output(s):
    rho : atmospheric density at the given altitude and time (kg/m^3)
    '''
    index = int((alt - 100)/20) # calculate index for the altitude
    if index < 0: index = 0
    elif index > len(altitudes) - 2: index = len(altitudes) - 2
    try: 
        altl = np.log10(alt) + 3
    except:
        altl = 0
    
    # interpolate for low and high solar activity
    low_estimate = 10**(rho_lowl[index] + (rho_lowl[index+1]-rho_lowl[index])/(altitudesl[index+1]-altitudesl[index])*(altl-altitudesl[index]))
    high_estimate = 10**(rho_highl[index] + (rho_highl[index+1]-rho_highl[index])/(altitudesl[index+1]-altitudesl[index])*(altl-altitudesl[index]))

    # calculate weighting factor between low and high activity
    if not ave_phase: w = np.sin(t*2*np.pi/22+phase)**2
    else: w = ((11-np.sin(2*(11+np.pi*t/11))/2)/2 - (0-np.sin(2*(0+np.pi*t/11))/2)/2)/11

    rho = low_estimate*(1-w) + high_estimate*w

    return rho

def dadt(alt, t, phase, a_over_m, CD, ave_phase):
    '''
    Calculates the rate of change in the altitude of a circular orbit

    Parameter(s):
    alt : altitude of the orbit (km)
    t : time passed since the start of the solar cycle (yr)
    phase : initial phase of the solar cycle
    a_over_m : area-to-mass ratio of the object (m^2/kg)
    CD : drag coefficient of the object
    ave_phase : average value over all phases (bool)

    Keyword Parameter(s): None

    Outputs:
    dadt value (km/yr)
    '''
    return -(CD*density(alt, t, phase, ave_phase)*a_over_m*np.sqrt(G*Me*(alt + Re)*1e3))*60*60*24*365.25*1e-3

def drag_lifetime(alt_i, alt_f, diameter, rho_m, a_over_m=None, CD=2.2, dt=1/365.25, phase=0, mindt=0, maxdt=None, dtfactor=1/100, tmax=np.inf, ave_phase=False):
    '''
    Estimates the drag lifetime of an object at altitude alt_i to degrade to altitude alt_f

    Parameter(s):
    alt_i : initial altitude of the object (km)
    alt_f : desired final altitude of the object (km)
    diameter : diameter of the object, ignored if a_over_m != None (m)
    rho_m : density of the object, ignored if a_over_m != None (kg/m^3)

    Keyword Parameter(s):
    a_over_m : area-to-mass ratio of the object (m^2/kg, default None)
    CD : drag coefficient of the object (default 2.2)
    dt : initial time step of the integration (yr, default 1/365.25)
    phase : initial phase in the solar cycle, ignored if ave_phase=True (yr, default 0)
    mindt : minimum time step for integration (yr, default 0)
    maxdt : maximum time step of the integration (yr, default None)
    dtfactor : fraction of altitude/rate of change to take as dt (default 1/100)
    tmax : maximum time to search to (yr, default infinite)
    ave_phase : average value over all phases (default False)

    Output(s):
    tau : drag lifetime, possibly infinite (yr)

    Notes : if no a_over_m is given, the object is assumed to be a uniform sphere with the given
    density and diameter.
    '''

    # initialize variables
    time = 0
    alt = alt_i
    if a_over_m == None : a_over_m = 3/(2*rho_m*diameter)

    # integrate using predictor-corrector method
    while alt > alt_f:
        dadt0 = dadt(alt, time, phase, a_over_m, CD, ave_phase)
        alt1 = alt + dadt0*dt
        dadt1 = dadt(alt1, time, phase, a_over_m, CD, ave_phase)
        ave_dadt = (dadt0 + dadt1)/2
        alt += ave_dadt*dt
        time += dt
        dt = -(alt/ave_dadt)*dtfactor
        if dt < mindt:
            print('WARNING: Problem is possibly too stiff for integrator.')
            dt = mindt
        elif maxdt != None:
            dt = min(dt, maxdt)
        if tmax is not None: # give up?
            if time > tmax : return np.inf

    return time

def shell_ave_lifetime(alt, dh, a_to_m, CD, ave_phases=True, start_alts=None):
    '''
    Calculates the lifetime of a object in circular orbit in a shell of altitude alt
    and thickness dh, averaging over starting altitude and a range of object parameters

    Parameter(s):
    alt : altitude of the centre of the shell (km)
    dh : thickness of the shell (km)
    a_to_m : array of area-to-mass values to average over (m^2/kg)
    CD : array of drag coefficients to average over

    Keyword Parameter(s):
    ave_phases : weather or not to average over the initial solar phase (default False)
    start_alts : array of starting altitudes to average over (km, default None)

    Note: the altitudes in start_alts are assumed to be larger than alt - dh/2, behaviour
    is undefined if this assumption is not met.
    '''

    if start_alts == None: # standard starting altitudes to average over
        alts0 = np.linspace(alt - dh/2, alt, 10)
        alts1 = np.linspace(alt, alt + dh/2, 20)
        start_alts = np.concatenate(alts0, alts1)

    total = 0 # sum of all drag coefficients calculated
    num = len(start_alts)*len(a_to_m)*len(CD) # total number of drag coefficients calculated
    for start in start_alts:
        for am in a_to_m:
            for C in CD:
                total += drag_lifetime(start, alt-dh/2, 0, 0, a_over_m=am, CD=C, ave_phase=ave_phases)
    return total/num