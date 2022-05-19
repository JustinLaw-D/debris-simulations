# implementation of the NASA standard breakup model

from scipy.stats import rv_continuous, norm
import random
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
    Lmin : minimu characteristic length (m)

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

class Lcol_dist(rv_continuous):
    '''
    Probability distribution of characteristic length for spacecraft collision with debris, 
    all inputs are in standard SI units
    '''

    def __init__(self, momtype=1, a=1/1000, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        super().__init__(momtype, a, b, xtol, badvalue, name, longname, shapes, extradoc, seed)

    def cdf(self, L, M):
        '''
        cummulative distribution function for the charactersitic length of debris from a collision

        Parameter(s):
        L : characteristic length (m)
        M : fit parameter given by calc_M (variable units)

        Keyword Parameter(s): None

        Output(s):
        CDF : value of CDF at L
        '''

        if M <= 0 or L < self.a: return 0
        else: return self._cdf(L, M)
    
    def _cdf(self, L, M):
        max_val = 0.1*(M**0.75)*(self.a**(-1.71)) # maximum value of the power law distribution
        return (1- 0.1*(M**0.75)*(L**(-1.71)))/max_val

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
            elif lambda_c < -0.1 : return -1.2 - 1.333(lambda_c + 0.7)
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

class dvcoll_dist(rv_continuous):
    '''generates a random change in velocity for debris from a collision, based on the A/M
    ratio'''

    def _pdf(self, v, chi):
        mu = 0.9*chi + 2.9
        sigma = 0.4
        return norm.pdf(v, mu, sigma)

    def pdf(self, v, chi):
        '''
        probability distribution function for the log of the ejection velocity of debris

        Parameter(s):
        v : log10 of the ejection velocity (log10(m/s))
        chi : log10 of A/M (log10(m^2/kg))

        Keyword Parameter(s): None

        Output(s):
        PDF : value of the pdf at v, given chi
        '''

        return self._pdf(v, chi)