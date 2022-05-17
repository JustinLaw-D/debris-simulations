# contains class for collection of cells representing orbital shells

from Cell import Cell
import numpy as np
import matplotlib.pyplot as plt

class NCell:

    def __init__(self, S, D, N, alts, dh, lam, drag_lifetime, del_t=None, sigma=None, v=None, 
                delta=None, alpha=None, P=None, N_0=None):
        '''
        Constructor for NCell class
    
        Parameter(s):
        S : initial number of live satellites in each shell (array)
        D : initial number of derelict satellites in each shell (array)
        N : initial number of lethal debris in each shell (array)
        alts : altitude of the shell's centre (array, km)
        dh : width of the shells (array, km)
        lam : launch rate of satellites into the each shell (array, 1/yr)
        drag_lifetime : function that computes atmospheric drag lifetime ((km, km) -> yr)

        Keyword Parameter(s):
        del_t : mean satellite lifetime in each shell (list, yr, default 5yr)
        sigma : satellite cross-section in each shell (list, m^2, default 10m^2)
        v : relative collision speed in each shell (list, km/s, default 10km/s)
        delta : ratio of the density of disabling to lethal debris in each shell (list, default 10)
        alpha : fraction of collisions a live satellites fails to avoid in each shell (list, default 0.2)
        P : post-mission disposal probability in each shell (list, default 0.95)
        N_0 : number of lethal debris fragments from a collision in each shell (list, default 100)

        Output(s):
        NCell instance

        Note: no size checks are done on the arrays, the program will crash if any of the arrays differ in size.
        shells are assumed to be given in order of ascending altitude. if you only want to pass values in the
        keyword argument for certain shells, put None in the list for all other shells. internally, cells have
        padded space in their arrays, use the getter functions to clean those up.
        '''

        # convert Nones to array of Nones
        if del_t == None:
            del_t = [None]*S.size
        if sigma == None:
            sigma = [None]*S.size
        if v == None:
            v = [None]*S.size
        if delta == None:
            delta = [None]*S.size
        if alpha == None:
            alpha = [None]*S.size
        if P == None:
            P = [None]*S.size
        if N_0 == None:
            N_0 = [None]*S.size

        self.alts = alts
        self.dh = dh
        self.time = 0 # index of current time step
        self.t = [0] # list of times traversed
        self.cells = [] # start list of cells
        for i in range(0, S.size):
            tau = drag_lifetime(alts[i] + dh[i]/2, alts[i] - dh[i]/2) # compute atmospheric drag lifetime for the shell
            cell = Cell(np.array([S[i], D[i], N[i], 0]), lam[i], alts[i], dh[i], tau, del_t=del_t[i], sigma=sigma[i], 
                        v=v[i], delta=delta[i], alpha=alpha[i], P=P[i], N_0=N_0[i]) # create cell
            self.cells.append(cell)

    def run_live_sim(self, method, dt=1, upper=True, dt_live=1, S=True, D=True, N=True):
        '''
        Runs a live simulation of the evolution of orbital shells for total time T, using the desired
        method, displaying current results, and allowing for the satellite launch rate of some shells
        to be adjusted

        Parameter(s): None
        
        Keyword Parameter(s):
        dt : initial time step used by simulation (yr, default 1)
        upper : wether or not to have debris come into the topmost layer (default True)
        dt_live : time between updates of the graph/oppertunity for user input (yr, default 1)
        S : whether or not to display the number of live satellites (default True)
        D : whether or not to display the number of derelict satellites (default True)
        N : whether or not to display the number of lethal debris (default True)

        Output(s): Nothing

        Note : user input allows you to change which shells are being displayed, change the valus of S, D, N, 
        change the satellite launch rate of any shell, and choose the dt/dt_live for the next step
        '''

        # setup
        display_bools = [] # list representing if each shell is displayed
        for i in range(len(self.cells)): # start by displaying shells with non-zero launch rates
            if self.cells[i].lam != 0:
                display_bools.append(True)
            else:
                display_bools.append(False)
        
        fig, ax = plt.subplots() # setup plots
        
        cont = True
        while cont == True: # start event loop
            ax.clear()
            self.run_sim_precor(self.t[self.time] + dt_live, dt=dt, upper=upper) # step forwards

            # pull well-formatted results
            t_list = self.get_t()
            S_list = self.get_S()
            D_list = self.get_D()
            N_list = self.get_N()

             # display results
            for i in range(len(self.cells)):
                if display_bools[i] == True:
                    if S == True:
                        ax.plot(t_list, S_list[i], label='S'+str(self.alts[i]))
                    if D == True:
                        ax.plot(t_list, D_list[i], label='D'+str(self.alts[i]))
                    if N == True:
                        ax.plot(t_list, N_list[i], label='N'+str(self.alts[i]))
            ax.set_xlabel('time (yr)')
            ax.set_ylabel('log(number)')
            ax.set_yscale('log')
            ax.legend()
            plt.tight_layout()
            plt.show(block=False)

            # take user input
            x = input('Continue running (y/n) : ')
            if x != 'y':
                cont = False
                continue
            x = input('Make any changes (y/n) : ')
            if x != 'y':
                continue
            x = input('Change runtime before next break (y/n) : ')
            if x != 'n':
                dt_live = float(input('Input runtime before next break : '))
            x = input('Change satellite launch rates (y/n) : ')
            while x != 'n':
                h = float(input('Input shell height to change launch rate of : '))
                index = self.alt_to_index(h)
                if index == -1:
                    print('ERROR: Invalid shell height')
                else:
                    lamb = float(input('Input new satellite launch rate : '))
                    self.cells[index].lam = lamb
                    display_bools[index] = True
                x = input('Change more satellite launch rates (y/n) : ')
            x = input('Display live satellites (y/n) : ')
            if x == 'n' : S = False
            else : S = True
            x = input('Display derelict satellites (y/n) : ')
            if x == 'n' : D = False
            else : D = True
            x = input('Display lethal debris (y/n) : ')
            if x == 'n' : N = False
            else : N = True
            x = input('Change which shells are displayed (y/n) : ')
            while x != 'n':
                h = float(input('Input shell height to change display of : '))
                index = self.alt_to_index(h)
                if index == -1:
                    print('ERROR: Invalid shell height')
                else:
                    y = input('Diplay this shell (y/n) : ')
                    if y == 'y' : display_bools[index] = True
                    elif y == 'n' : display_bools[index] = False
                x = input('Change more shell displays (y/n) : ')

    def run_sim_euler(self, T, dt=1, upper=True):
        '''
        Simulates evolution of orbital shells for total time T, using Euler method

        Parameter(s):
        T : total length of the simulation (yr)

        Keyword Parameter(s):
        dt : time step used by the simulation (yr, default 1)
        upper : wether or not to have debris come into the topmost layer (default True)

        Returns: None
        '''

        top_cell = self.cells[-1] # setup handling top cell
        V_top = 4*np.pi*(6371 + top_cell.alt)**2*top_cell.dh # volume of the top shell
        n_0 = (top_cell.pastx[0][2] + top_cell.pastx[0][1]/2)/V_top
        dNdt_in_top = 0 # rate of debris entering the top shell
        if upper:
            dNdt_in_top = n_0*V_top/top_cell.tau
        dxdt_in_top = np.array([0,0,dNdt_in_top,0])

        while self.t[self.time] < T:
            dxdt_in = dxdt_in_top
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                dxdt_cell = curr_cell.dxdt_cell(curr_cell.pastx[self.time]) + dxdt_in
                curr_cell.pastx.append(curr_cell.pastx[self.time] + dxdt_cell*dt)
                dxdt_in = curr_cell.dxdt_out(curr_cell.pastx[self.time])
        
            self.time += 1 # update times
            if self.time >= len(self.t):
                self.t.append(self.t[self.time-1] + dt)
            else:
                self.t[self.time] = self.t[self.time-1] + dt

    def run_sim_precor(self, T, dt=1/100, mindtfactor=1000, maxdt=1, tolerance=1, upper=True):
        '''
        Simulates evolution of orbital shells for total time T, using predictor-corrector method
        with adaptive timestep

        Parameter(s):
        T : total length of the simulation (yr)

        Keyword Parameter(s):
        dt : time step used by the simulation (yr, default 1/100)
        mindtfactor : minimum time step used by the simulation is dt/mindtfactor
        maxdt : maximum time step used by simulation (yr, default 1)
        tolerance : tolerance for adaptive time step
        upper : wether or not to have debris come into the topmost layer (default True)

        Returns: None
        '''

        orgdt = dt
        if self.time == 0: # generate two time points if they don't exist yet
            self.run_sim_euler(dt/100, dt=dt/100, upper=upper)
        top_cell = self.cells[-1] # setup handling top cell
        V_top = 4*np.pi*(6371 + top_cell.alt)**2*top_cell.dh # volume of the top shell
        n_0 = (top_cell.pastx[0][2] + top_cell.pastx[0][1]/2)/V_top
        dNdt_in_top = 0 # rate of debris entering the top shell
        if upper:
            dNdt_in_top = n_0*V_top/top_cell.tau
        dxdt_in_top = np.array([0,0,dNdt_in_top,0])

        while self.t[self.time] < T:
            retry = False
            dxdt_in_n = dxdt_in_top # calculate (x_n)'
            dxdt_n = []
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                dxdt_n.insert(0, curr_cell.dxdt_cell(curr_cell.pastx[self.time-1]) + dxdt_in_n)
                dxdt_in_n = curr_cell.dxdt_out(curr_cell.pastx[self.time-1])

            dxdt_in_n1 = dxdt_in_top # calculate (x_{n+1})'
            dxdt_n1 = []
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                dxdt_n1.insert(0, curr_cell.dxdt_cell(curr_cell.pastx[self.time]) + dxdt_in_n1)
                dxdt_in_n1 = curr_cell.dxdt_out(curr_cell.pastx[self.time])

            # use these to approximate x_{n+2} with AB(2) method
            x_n2_temp = []
            dt_n = self.t[self.time] - self.t[self.time-1] # get previous time step
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                x_n2_temp.insert(0, curr_cell.pastx[self.time] + 0.5*dt*((2+dt/dt_n)*dxdt_n1[i]-dt/dt_n*dxdt_n[i]))

            # approximate (x_{n+2})' using the AB(2) prediction
            dxdt_in_n2 = dxdt_in_top # calculate (x_{n+1})'
            dxdt_n2 = []
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                dxdt_n2.insert(0, curr_cell.dxdt_cell(x_n2_temp[i]) + dxdt_in_n2)
                dxdt_in_n2 = curr_cell.dxdt_out(x_n2_temp[i])

            # use this to calculate better x_{n+2} with trapezoid method
            x_n2 = []
            for i in range(-1, -(len(self.cells) + 1), -1): # iterate through all cells
                curr_cell = self.cells[i]
                x_n2.insert(0, curr_cell.pastx[self.time] + 0.5*dt*(dxdt_n2[i] + dxdt_n1[i]))

            # check that the error isn't too large on any of the shells
            t_new_options = []
            for i in range(len(self.cells)):
                error = np.amax(np.absolute((1/3)*(dt/(dt + dt_n))*(x_n2[i] - x_n2_temp[i])))
                if error > tolerance : retry = True
                t_new_options.append(dt*(abs(tolerance/error))**(1/3))
            
            new_dt = min(t_new_options)
            if retry:
                if new_dt < orgdt/mindtfactor: 
                    print("WARNING: Problem potentially too stiff for integrator")
                    dt = orgdt/mindtfactor
                else:
                    dt = new_dt 
                    continue
            else : dt = min(new_dt, maxdt)
            for i in range(len(self.cells)):
                curr_cell = self.cells[i]
                curr_cell.pastx.append(x_n2[i])
        
            self.time += 1 # update times
            if self.time >= len(self.t):
                self.t.append(self.t[self.time-1] + dt)
            else:
                self.t[self.time] = self.t[self.time-1] + dt

    def get_t(self):
        '''
        returns array of times used in the simulation

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        array of t values (yr)
        '''

        return self.t
    
    def get_S(self):
        '''
        returns arrays for number of live satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of S values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_add = []
            for x in cell.pastx:
                to_add.append(x[0])
            to_return.append(to_add)
        return to_return

    def get_D(self):
        '''
        returns arrays for number of derelict satellites in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of D values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_add = []
            for x in cell.pastx:
                to_add.append(x[1])
            to_return.append(to_add)
        return to_return

    def get_N(self):
        '''
        returns arrays for number of lethal debris in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of N values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_add = []
            for x in cell.pastx:
                to_add.append(x[2])
            to_return.append(to_add)
        return to_return

    def get_C(self):
        '''
        returns arrays for total number of collisions in each shell

        Parameter(s): None

        Keyword Parameter(s): None

        Returns:
        list of array of C values for each cell, in order of ascending altitude
        '''

        to_return = []
        for cell in self.cells:
            to_add = []
            for x in cell.pastx:
                to_add.append(x[3])
            to_return.append(to_add)
        return to_return

    def alt_to_index(self, h):
        '''
        Converts given altitude to cell index

        Parameter(s):
        h : altitude to convert (km)

        Keyword Parameter(s): None

        Output(s):
        index : index corresponding to that altitude, or -1 if none is found
        '''

        for i in range(len(self.cells)):
            alt, dh = self.alts[i], self.dh[i]
            if (alt - dh/2 <= h) and (alt + dh/2 >= h):
                return i
        return -1