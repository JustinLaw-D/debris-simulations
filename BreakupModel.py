# implementation of the NASA standard breakup model
# TODO: FIX NORMALIZATION IN EVERYTHING

from imp import load_module
from scipy.stats import norm
from scipy.special import erf, erfinv
import numpy as np

def calc_M(m_s, m_d, v):
    '''
    calculates the M factor used for L distribution calculation, and
    determines if a collision is catestrophic or not
    
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

    if E_p >= 40 : return m_s + m_d, True # catestrophic collision
    else : return m_d*(v/1000), False # non-catestrophic collision

def calc_Ntot_coll(M, Lmin):
    '''
    calculates the total number of debris produced with characteristic length
    greater than Lmin, for a collision

    Parameter(s):
    M : fit parameter given by calc_M (variable units)
    Lmin : minimum characteristic length (m)

    Keyword Parameter(s): None

    Output(s):
    N : total number of fragments of size > Lmin

    Notes:
    Model output is a continuous value, and is simply truncated
    '''

    return int(0.1*(M**0.75)*(Lmin**(-1.71)))

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

def randL_coll(num, M, L_min, L_max):
    '''
    generates num random characteristic lengths for debris from a collision

    Parameter(s):
    num : number of random lengths to generate
    M : fit parameter given by calc_M (variable units)
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
    M : fit parameter given by calc_M (variable units)
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
    M : fit parameter given by calc_M (variable units)
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

    P = np.random.uniform(size=num) # get random P values
    # use these to generate random x-values
    x = sigma_soc(lam)*np.sqrt(2)*erfinv(P + erf((x_min - mu_soc(lam)/(sigma_soc(lam)*np.sqrt(2))))) + mu_soc(lam)
    return x


class AMcoll_dist(rv_continuous):
    '''probability distribution for the A/M of fragments from a collision, given a characteristic length.
    all inputs in log_10 of standard SI units.'''

    def _pdf11(self, chi, lambda_c):
        
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

        norm_one, norm_two = norm.pdf(chi, mu1_sc(lambda_c), sigma1_sc(lambda_c)), norm.pdf(chi, mu2_sc(lambda_c), sigma2_sc(lambda_c))
        return alpha_sc(lambda_c)*norm_one + (1-alpha_sc(lambda_c))*norm_two

    def _pdf8(self, chi, lambda_c):

        def mu_soc(lambda_c):
            if lambda_c <= -1.75 : return -0.3
            elif lambda_c < -1.25 : return -0.3 - 1.4*(lambda_c + 1.75)
            else : return -1

        def sigma_soc(lambda_c):
            if lambda_c <= -3.5 : return 0.2
            else : return 0.2 + 0.1333*(lambda_c + 3.5)

        return norm.pdf(chi, mu_soc(lambda_c), sigma_soc(lambda_c))

    def _pdf(self, chi, lambda_c):

        if lambda_c >= 11/100 : return self._pdf11(chi, lambda_c)
        elif lambda_c <= 8/100 : return self._pdf8(chi, lambda_c)
        else:
            comp = 10*(lambda_c + 1.05)
            if random.uniform(0,1) > comp : return self._pdf11(chi, lambda_c)
            else : return self._pdf8(chi, lambda_c)

    def pdf(self, chi, lambda_c):
        '''
        probability distribution function for the log of the A/M ratio of debris

        Parameter(s):
        chi : log10 of A/M (log10(m^2/kg))
        lambda_c : log10 of the characteristic length (log10(m))

        Keyword Parameter(s): None

        Output(s):
        PDF : value of the pdf at chi, given lambda_c
        '''

        return self._pdf(chi, lambda_c)
