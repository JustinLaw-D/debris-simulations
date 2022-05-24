# implementation of the NASA standard breakup model

from scipy.special import erf, erfinv
import numpy as np

def rand_round(num):
    '''
    randomly rounds a real number either up or down to the nearest integer

    Parameter(s):
    num : a real number

    Keyword Parameter(s): None

    Output(s):
    int_num : number rounded to an integer
    '''

    if num == 0 : return num # nothing to do in this case
    sign_fac = num/num # factor to account for the sign of the number
    rand = np.random.uniform() # get random number between 0 and 1
    if rand > abs(num - int(num)): # randomly round
        return int(num)
    else:
        return int(num) + sign_fac

def is_catastrophic(m_s, L, AM, v):
    '''
    determines if a collision between debris and a satallite is catastrophic

    Parameter(s):
    m_s : mass of the satallite (kg)
    L : characteristic length of the debris (m)
    AM : area to mass ratio of the debris (m^2/kg)
    v : relative velocity of the objects (km/s)

    Keyword Parameter(s): None

    Output(s):
    cat : True if the collision is catastrophic, False otherwise
    '''
    
    v *= 1000 # convert to m/s
    m_d = find_A(L)/AM # mass of the debris (on average)
    k_d = 0.5*m_d*(v**2) # relative kinetic energy of the debris
    dec_fact = (k_d/m_s)/1000 # factor for making the decision (J/g)
    if dec_fact >= 40 : return True
    else : return False

def calc_M(m_s, m_d, v):
    '''
    calculates the M factor used for L distribution calculation
    
    Parameter(s):
    m_s : satellite mass (kg)
    m_d : mass of the debris (kg)
    v : collision velocity (m/s)

    Keyword parameter(s): None

    Output(s):
    M : value of M parameter (variable units)
    cat : whether or not the collision was catestrophic (boolean)
    '''

    E_p = (0.5*m_d*(v**2)/m_s)/1000 # E_p in J/g

    if E_p >= 40 : return m_s + m_d # catestrophic collision
    else : return m_d*(v/1000) # non-catestrophic collision

def calc_Ntot_coll(M, Lmin, Lmax):
    '''
    calculates the total number of debris produced with characteristic length
    between Lmin and Lmax, for a collision

    Parameter(s):
    M : fit parameter given by calc_M (variable units)
    Lmin : minimum characteristic length (m)
    Lmax : maximum characteristic length (m)

    Keyword Parameter(s): None

    Output(s):
    N : total number of fragments of size > Lmin

    Notes:
    Model output is a continuous value, and is simply truncated
    '''

    return int(0.1*(M**0.75)*(Lmin**(-1.71)) - 0.1*(M**0.75)*(Lmax**(-1.71)))

def find_A(L):
    '''
    calculates the average cross-sectional area of an object with a given characteristic length

    Parameter(s):
    L : characteristic length (m)

    Keyword Parameter(s): None

    Output(s):
    A : average cross-sectional area (m^2)
    '''

    if L < 0.00167 : return 0.540424*(L**2)
    else : return 0.556945*(L**2.0047077)

def randL_coll(num, L_min, L_max):
    '''
    generates num random characteristic lengths for debris from a collision

    Parameter(s):
    num : number of random lengths to generate
    L_min : minimum characteristic length to consider (m)
    L_max : maximum characteristic length to consider (m)

    Keyword Parameter(s): None

    Output(s):
    L : array of random characteristic lengths (m)
    '''

    lam_min, lam_max = np.log10(L_min), np.log10(L_max)
    beta = -1.71
    P = np.random.uniform(size=num) # get random P values
    lam = np.log10(10**(beta*lam_min) - P*(10**(beta*lam_min) - 10**(beta*lam_max)))/beta
    return 10**lam

def randX_coll(num, x_min, x_max, L):
    '''
    generates num random log10(A/M) values for debris from a collision

    Parameter(s):
    num : number of random values to generate
    x_min : minimum log10(A/M) value to consider (log10(m^2/kg))
    x_max : maximum log10(A/M) value to consider (log10(m^2/kg))
    L : characteristic length of the debris (m)

    Keyword Parameter(s): None

    Output(s):
    x : array of random log10(A/M) values (log10(m^2/kg))
    '''

    if L >= 11/100 : return _randX_coll_11(num, x_min, x_max, L)
    elif L <= 8/100 : return _randX_coll_8(num, x_min, x_max, L)
    else:
        comp = 10*(np.log10(L) + 1.05)
        if np.random.uniform() > comp : return _randX_coll_11(num, x_min, x_max, L)
        else : return _randX_coll_8(num, x_min, x_max, L)

def _randX_coll_8(num, x_min, x_max, L):
    '''
    generates num random log10(A/M) values for debris from a collision, assuming that
    the characteristic length of the debris is less than 8cm

    Parameter(s):
    num : number of random values to generate
    x_min : minimum log10(A/M) value to consider (log10(m^2/kg))
    x_max : maximum log10(A/M) value to consider (log10(m^2/kg))
    L : characteristic length of the debris (m)

    Keyword Parameter(s): None

    Output(s):
    x : array of random log10(A/M) values (log10(m^2/kg))
    '''

    lam = np.log10(L)

    # define functions for determining normal distribution parameters
    def mu_soc(lambda_c):
        if lambda_c <= -1.75 : return -0.3
        elif lambda_c < -1.25 : return -0.3 - 1.4*(lambda_c + 1.75)
        else : return -1

    def sigma_soc(lambda_c):
        if lambda_c <= -3.5 : return 0.2
        else : return 0.2 + 0.1333*(lambda_c + 3.5)
    
    mu = mu_soc(lam) # calculate parameters
    sigma = sigma_soc(lam)
    C = 1/(erf((x_max-mu)/(np.sqrt(2)*sigma)) - erf((x_min-mu)/(np.sqrt(2)*sigma))) # normalization factor
    P = np.random.uniform(size=num) # get random P values
    # use these to generate random x-values
    x = sigma*np.sqrt(2)*erfinv(P/C + erf((x_min - mu)/(sigma*np.sqrt(2)))) + mu
    return x

def _randX_coll_11(num, x_min, x_max, L):
    '''
    generates num random log10(A/M) values for debris from a collision, assuming that
    the characteristic length of the debris is greater than 11cm

    Parameter(s):
    num : number of random values to generate
    x_min : minimum log10(A/M) value to consider (log10(m^2/kg))
    x_max : maximum log10(A/M) value to consider (log10(m^2/kg))
    L : characteristic length of the debris (m)

    Keyword Parameter(s): None

    Output(s):
    x : array of random log10(A/M) values (log10(m^2/kg))
    '''

    lam = np.log10(L)

    # define functions for determining normal distribution parameters
    def alpha_sc(lambda_c):
            if lambda_c <= -1.95 : return 0
            elif lambda_c < 0.55 : return 0.3 + 0.4*(lambda_c + 1.2)
            else : return 1

    def mu1_sc(lambda_c):
        if lambda_c <= -1.1 : return -0.6
        elif lambda_c < 0 : return -0.6 - 0.318*(lambda_c + 1.1)
        else : return -0.95

    def sigma1_sc(lambda_c):
        if lambda_c <= -1.3 : return 0.1
        elif lambda_c < -0.3 : return 0.1 + 0.2*(lambda_c + 1.3)
        else : return 0.3

    def mu2_sc(lambda_c):
        if lambda_c <= -0.7 : return -1.2
        elif lambda_c < -0.1 : return -1.2 - 1.333*(lambda_c + 0.7)
        else : return -2

    def sigma2_sc(lambda_c):
        if lambda_c <= -0.5 : return 0.5
        elif lambda_c < -0.3 : return 0.5 - (lambda_c + 0.5)
        else : return 0.3
    
    mu1 = mu1_sc(lam) # calculate parameters
    sigma1 = sigma1_sc(lam)
    mu2 = mu2_sc(lam)
    sigma2 = sigma2_sc(lam)
    alpha = alpha_sc(lam)
    # compute normalization factor
    top = alpha*erf((x_max-mu1)/(np.sqrt(2)*sigma1)) + (1-alpha)*erf((x_max-mu2)/(np.sqrt(2)*sigma2))
    bot = alpha*erf((x_min-mu1)/(np.sqrt(2)*sigma1)) + (1-alpha)*erf((x_min-mu2)/(np.sqrt(2)*sigma2))
    C = 1/(top - bot)
    x_table = np.linspace(x_min, x_max, num=1000) # table of x values
    # corresponding table of P values
    P_table = C*(alpha*erf((x_table-mu1)/(np.sqrt(2)*sigma1)) + (1-alpha)*erf((x_table-mu2)/(np.sqrt(2)*sigma2)) - bot)
    P = np.random.uniform(size=num) # get random P values
    # use these to generate random x-values
    x = np.zeros(P.shape)
    for i in range(len(P)):
        index = np.abs(P_table - P[i]).argmin() # find location of closest value on the table
        x[i] = x_table[index] # use this to find the corresponding x-value
    return x

def randv_coll(num, x):
    '''
    generates num random log10(Delta v) values for debris from a collision

    Parameter(s):
    num : number of random values to generate
    x : log10(A/M) value of the debris (log10(m^2/kg))

    Keyword Parameter(s): None

    Output(s):
    v : array of random log10(Delta v) values (log10(m/s))
    '''

    mu = 0.9*x + 2.9 # calculate normal distribution parameters
    sigma_fac = 0.4*np.sqrt(2)
    C = 1/2 # calculate normalization factor
    P = np.random.uniform(size=num) # get random P values
    # use these to generate random v-values
    v = sigma_fac*erfinv(P/C - 1) + mu
    return v

def rand_direction(num):
    '''
    generates random directions in 3-d space

    Parameter(s):
    num : number of random values to generate

    Keyword Parameter(s): None

    Output(s)
    u : array of 3-d unit vectors in cartesian coordinates

    Notes: Output u is a matrix, where the first row a list of x-coordinates,
    the second y-coordinates, and the third z-coordinates
    '''

    theta = np.random.uniform(0.0, np.pi, size=num) # generate inclinations
    phi = np.random.uniform(0.0, 2*np.pi, size=num) # generate azimuthal angles
    to_return = np.zeros((3, num)) # first row is x, second is y, third is z
    to_return[0, :] = np.cos(phi)*np.sin(theta)
    to_return[1, :] = np.sin(phi)*np.sin(theta)
    to_return[2, :] = np.cos(theta)
    return to_return