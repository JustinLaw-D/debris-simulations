# contains models for atmospheric density, drag lifetime

import os
import numpy as np

G = 6.67430e-11 # gravitational constant (N*m^2/kg^2)
Me = 5.97219e24 # mass of Earth (kg)
Re = 6371 # radius of Earth (km)

filepath, _ = os.path.split(__file__) # path to current folder
filepath += '/'

# read density model
atmfile=open(filepath + "atmosphere_data/cira-2012.dat","r")
header=atmfile.readline()
zmodel=[]
denmodelL=[]
denmodelM=[]
denmodelHL=[]
for line in atmfile:
  alt,low,med,highL,_=line.split()
  zmodel.append(float(alt))
  denmodelL.append(float(low))
  denmodelM.append(float(med))
  denmodelHL.append(float(highL))

atmfile.close()

zmodel=np.array(zmodel)*1000 # convert to m
denmodelL=np.array(denmodelL)
denmodelM=np.array(denmodelM)
denmodelHL=np.array(denmodelHL)

logdenL = np.log10(denmodelL)
logdenM = np.log10(denmodelM)
logdenHL = np.log10(denmodelHL)
logz = np.log10(zmodel)

# read solar cycle template (using F10.7 as the solar activity index)
f107file = open(filepath + "atmosphere_data/solar_cycle_table36_cira2012.dat","r")
header=f107file.readline()
f107_mo=[]
for line in f107file:
   mo,_,f107,_,_,_,_,_,_,_,_,_,_=line.split()
   f107_mo.append(float(f107))
f107file.close()
f107_mo=np.array(f107_mo) 

def density(alt,t,mo0,setF107=None):
    '''
    Calculates the atmospheric density at a given altitude via interpolation

    Parameter(s):
    alt : altitude (km)
    t : time since arbitrary start point (yr)
    m0 : starting month in the solar cycle (int)

    Keyword Parameter(s):
    setF107 : if not None, value taken for solar flux regardless of current time
              (None or 10^(-22)W/m^2, default None)

    Output(s):
    rho : atmospheric density at the given altitude and time (kg/m^3)
    '''

    i=int((alt-100)/20) # calculate index for altitude
    if i > len(zmodel)-2: i=len(zmodel)-2
    if i < 0: i=0

    try:
       logalt = np.log10(alt) + 3 # convert to m
    except:
       logalt = 0.
 
    mo_frac = t*12 + mo0

    mo = mo_frac % 144

    moID = int(mo)

    if setF107==None: # get flux value
       moID1 = moID+1
       if moID1>143:moID1=0
       F107 = f107_mo[moID] + (f107_mo[moID1]-f107_mo[moID])*(mo-moID)
    else: F107 = setF107

    if F107 <= 65: # interpolate to get density value
       rho = 10.**(  logdenL[i]+(logdenL[i+1]-logdenL[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )

    elif F107 <= 140:
      d0 = 10.**(  logdenL[i]+(logdenL[i+1]-logdenL[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
      d1 = 10.**(  logdenM[i]+(logdenM[i+1]-logdenM[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
      rho = d0 + (d1-d0)*(F107-65.)/75.

    elif F107 <= 250:
      d0 = 10.**(  logdenM[i]+(logdenM[i+1]-logdenM[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
      d1 = 10.**(  logdenHL[i]+(logdenHL[i+1]-logdenHL[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
      rho = d0 + (d1-d0)*(F107-140.)/110.

    else:
      rho = 10.**(  logdenHL[i]+(logdenHL[i+1]-logdenHL[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )

    return rho

def dadt(alt, t, m0, a_over_m, CD, setF107=None):
    '''
    Calculates the rate of change in the altitude of a circular orbit

    Parameter(s):
    alt : altitude of the orbit (km)
    t : time passed since the start of the solar cycle (yr)
    m0 : starting month in the solar cycle (int)
    a_over_m : area-to-mass ratio of the object (m^2/kg)
    CD : drag coefficient of the object

    Keyword Parameter(s):
    setF107 : if not None, value taken for solar flux regardless of current time
              (None or 10^(-22)W/m^2, default None)

    Outputs:
    dadt value (km/yr)
    '''
    return -(CD*density(alt, t, m0, setF107=setF107)*a_over_m*np.sqrt(G*Me*(alt + Re)*1e3))*60*60*24*365.25*1e-3

def drag_lifetime(alt_i, alt_f, diameter, rho_m, a_over_m=None, CD=2.2, dt=1/365.25, m0=0, mindt=0, maxdt=None, dtfactor=1/100, tmax=np.inf, setF107=None):
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
    m0 : starting month in the solar cycle (int, default 0)
    mindt : minimum time step for integration (yr, default 0)
    maxdt : maximum time step of the integration (yr, default None)
    dtfactor : fraction of altitude/rate of change to take as dt (default 1/100)
    tmax : maximum time to search to (yr, default infinite)
    setF107 : if not None, value taken for solar flux regardless of current time
              (None or 10^(-22)W/m^2, default None)

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
        dadt0 = dadt(alt, time, m0, a_over_m, CD, setF107=setF107)
        alt1 = alt + dadt0*dt
        dadt1 = dadt(alt1, time + dt, m0, a_over_m, CD, setF107=setF107)
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

def need_update(t_curr, t_last):
  '''
  outputs whether or not the atmospheric drag lifetime needs to be updated

  Input(s):
  t_curr : current time since "start" of simulation (yr)
  t_last : time the lifetime was last updated (yr, since start of simulation)

  Keyword Input(s): None

  Output(s):
  update : True is the value needs to be updated, False otherwise

  Note(s): default function, call for an update once per month
  '''

  return t_curr - t_last > 1/12
