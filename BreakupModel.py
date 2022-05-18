# implementation of the NASA standard breakup model

from scipy.stats import rv_continuous

class Lcol_dist(rv_continuous):
    # probability distribution of characteristic length for spacecraft collision with debris
    def __init__(self, momtype=1, a=1, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        super().__init__(momtype, a, b, xtol, badvalue, name, longname, shapes, extradoc, seed)

    def cdf(self, L, M):
        if M <= 0 or L < self.a: return 0
        else: return self._cdf(L, M)
    
    def _cdf(self, L, M):
        max_val = 0.1*(M**0.75)*(self.a**(-1.71)) # maximum value of the power law distribution
        return (1- 0.1*(M**0.75)*(L**(-1.71)))/max_val