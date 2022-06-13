# classes for object (Satellite, RocketBody) and discrete events

class Satellite:

    def __init__(self, S_i, S_di, D_i, m, sigma, lam, del_t, tau_do, alpha, P, AM, tau):
        '''
        constructor method for Satellite class

        Parameter(s):
        S_i : initial number of live satellites of this type
        S_di : initial number of de-orbiting satellites of this type
        D_i : initial number of derelict satellites of this type
        m : mass of each satellite (kg)
        sigma : collision cross-section of each satellite (m^2)
        lam : launch rate of the satellites (1/yr)
        del_t : mean satellite lifetime (yr)
        tau_do : mean time for satellite to de-orbit from shell (yr)
        alpha : fraction of collisions a live satellites fails to avoid
        P : post-mission disposal probability
        AM : area-to-mass ratio of the satellite (m^2/kg)
        tau : atmospheric drag lifetime of a satellite (yr)

        Keyword Parameter(s): None

        Output(s): Instance of Satellite class

        Note(s): preforms no validity checks on given values
        '''

        self.S = [S_i]
        self.S_d = [S_di]
        self.D = [D_i]
        self.m = m
        self.sigma = sigma
        self.lam = lam
        self.del_t = del_t
        self.tau_do = tau_do
        self.alpha = alpha
        self.P = P
        self.AM = AM
        self.tau = tau

class RocketBody:

    def __init__(self, num, m, sigma, lam, AM, tau):
        '''
        constructor method for RocketBody class

        Parameter(s):
        num : initial number of rocket bodies of this type
        m : mass of each rocket body (kg)
        sigma : collision cross-section of each rocket body (m^2)
        lam : launch rate of the rocket bodies (1/yr)
        AM : area-to-mass ratio of a rocket body (m^2/kg)
        tau : atmospheric drag lifetime of a rocket body (yr)

        Keyword Parameter(s): None

        Output(s): Instance of Satellite class

        Note(s): preforms no validity checks on given values
        '''

        self.num = [num]
        self.m = m
        self.sigma = sigma
        self.lam = lam
        self.AM = AM
        self.tau = tau


class Event:
    
    def __init__(self, alt, time=None, freq=None):
        '''
        constructor for general event class

        Paremeter(s):
        alt : altitude of the event (km)

        Keyword Parameter(s):
        time : list of times that the event occurs (yr, default None)
        freq : frequency the event occurs at (1/yr, default None)

        Output(s): instance of Event

        Note(s): time and freq cannot both be None
        '''

        if time is None and freq is None:
            print('Invlid Event : No occurance time specified')

        self.time = time
        self.last_event = 0 # time of the last event (yr)
        self.freq = freq
        self.alt = alt

    def run_event(self, S, S_d, D, N, logL_edges, chi_edges):
        '''
        function representing the discrete event occuring

        Input(s):
        S : number of live satellites of each type in the current cell (list of floats)
        S_d : number of de-orbiting satellites of each type in the current cell (list of floats)
        D : number of derelict satellites of each type in the current cell (list of floats)
        N : binned amount of debris in current cell (2d array)
        logL_edges : logL edge values for the bins (log10(m))
        chi_edges : chi edge values for the bins (log10(m^2/kg))

        Keyword Input(s): None

        Output(s):
        dS : change in the number of live satellites of each type in the current cell (list of floats)
        dS_d : change in the number of de-orbiting satellites of each type in the current cell (list of floats)
        dD : change in the number of derelict satellites of each type in the cell (list of floats)
        dN : change in the number of debris in the curren cell, not including debris
             produced by collisions
        coll : list of collisions occuring in the current cell in the form [(kg, kg, #)],
               i.e. [(m1, m2, number of collisions)]
        expl : list of explosions occuring in the current cell in the form [(C, typ, #)], where
               C is the relevant fit constant and typ is the type of body exploding ('sat' or 'rb)

        Note(s): this function is meant to be overwritten, and in the default form just returns
                 zero
        '''

        return 0, 0, 0, 0, 0, 0

# class for handling basic explosions
class ExplEvent(Event):
    
    def __init__(self, alt, expl_list, time=None, freq=None):
        '''
        constructor for a basic explosions event class

        Paremeter(s):
        alt : altitude of the event (km)
        expl_list : list of explosions occuring in the current cell on an event in the form [(C, typ, #)], 
                    where C is the relevant fit constant and typ is the type of body exploding ('sat' or 'rb)

        Keyword Parameter(s):
        time : list of times that the event occurs (yr, default None)
        freq : frequency the event occurs at (1/yr, default None)

        Output(s): instance of Event

        Note(s): time and freq cannot both be None
        '''

        super().__init__(alt, time=time, freq=freq)
        self.expl_list = expl_list

    def run_event(self, S, S_d, D, N, logL_edges, chi_edges):
        '''
        function representing the discrete event occuring

        Input(s):
        S : number of live satellites of each type in the current cell (list of floats)
        S_d : number of de-orbiting satellites of each type in the current cell (list of floats)
        D : number of derelict satellites of each type in the current cell (list of floats)
        N : binned amount of debris in current cell (2d array)
        logL_edges : logL edge values for the bins (log10(m))
        chi_edges : chi edge values for the bins (log10(m^2/kg))

        Keyword Input(s): None

        Output(s):
        dS : change in the number of live satellites of each type in the current cell (list of floats)
        dS_d : change in the number of de-orbiting satellites of each type in the current cell (list of floats)
        dD : change in the number of derelict satellites of each type in the cell (list of floats)
        dN : change in the number of debris in the curren cell, not including debris
             produced by collisions
        coll : list of collisions occuring in the current cell in the form [(kg, kg, #)],
               i.e. [(m1, m2, number of collisions)]
        expl : list of explosions occuring in the current cell in the form [(C, typ, #)], where
               C is the relevant fit constant and typ is the type of body exploding ('sat' or 'rb)

        Note(s): this function is meant to be overwritten, and in the default form just returns
                 zero
        '''

        return 0, 0, 0, 0, 0, self.expl_list

# class for handling basic collisions
class CollEvent(Event):
    
    def __init__(self, alt, coll_list, time=None, freq=None):
        '''
        constructor for a basic collisions event class

        Paremeter(s):
        alt : altitude of the event (km)
        coll_list : list of collisions occuring on an event in the current cell 
                    in the form [(kg, kg, #)], i.e. [(m1, m2, number of collisions)]

        Keyword Parameter(s):
        time : list of times that the event occurs (yr, default None)
        freq : frequency the event occurs at (1/yr, default None)

        Output(s): instance of Event

        Note(s): time and freq cannot both be None
        '''

        super().__init__(alt, time=time, freq=freq)
        self.coll_list = coll_list

    def run_event(self, S, S_d, D, N, logL_edges, chi_edges):
        '''
        function representing the discrete event occuring

        Input(s):
        dS : change in the number of live satellites of each type in the current cell (list of floats)
        dS_d : change in the number of de-orbiting satellites of each type in the current cell (list of floats)
        dD : change in the number of derelict satellites of each type in the cell (list of floats)
        N : binned amount of debris in current cell (2d array)
        logL_edges : logL edge values for the bins (log10(m))
        chi_edges : chi edge values for the bins (log10(m^2/kg))

        Keyword Input(s): None

        Output(s):
        dS : change in the number of live satellites in the current cell
        dS_d : change in the number of de-orbiting satellites in the current cell
        dD : change in the number of derelict satellites in the cell
        dN : change in the number of debris in the curren cell, not including debris
             produced by collisions
        coll : list of collisions occuring in the current cell in the form [(kg, kg, #)],
               i.e. [(m1, m2, number of collisions)]
        expl : list of explosions occuring in the current cell in the form [(C, typ, #)], where
               C is the relevant fit constant and typ is the type of body exploding ('sat' or 'rb)

        Note(s): this function is meant to be overwritten, and in the default form just returns
                 zero
        '''

        return 0, 0, 0, 0, self.coll_list, 0