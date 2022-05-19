# implementation of the NASA standard breakup model

from scipy.stats import rv_continuous, norm
import random
import numpy as np

class Lcol_dist(rv_continuous):
    '''Probability distribution of characteristic length for spacecraft collision with debris, 
    all inputs are in standard SI units'''

    def __init__(self, momtype=1, a=1/1000, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        super().__init__(momtype, a, b, xtol, badvalue, name, longname, shapes, extradoc, seed)

    def cdf(self, L, M):
        if M <= 0 or L < self.a: return 0
        else: return self._cdf(L, M)
    
    def _cdf(self, L, M):
        max_val = 0.1*(M**0.75)*(self.a**(-1.71)) # maximum value of the power law distribution
        return (1- 0.1*(M**0.75)*(L**(-1.71)))/max_val

class AMcoll_dist(rv_continuous):
    '''probability distribution for the A/M of fragments, given a characteristic length.
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

class dvcoll_dist(rv_continuous):
    '''generates a random change is velocity for debris, based on the A/M
    ratio'''

    def _pdf(self, v, chi):
        mu = 0.9*chi + 2.9
        sigma = 0.4
        return norm.pdf(v, mu, sigma)
