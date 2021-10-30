
""" @file controls.py
"""

from scipy.linalg import solve_continuous_are as care
import scipy.integrate as scint
import numpy as np
import copy

###############################
## Control Policies
###############################
class ZeroPol:

    def __init__(self, du):
        self.du = du
        self.u = np.zeros((du))

    def evaluate(self, time, state):
        return self.u

class LinearFeedback: # Continuous Infinite-Horizon Linear Quadratic Regulator

    def __init__(self, A, B, Q, R):
        self.P = care(A, B, Q, R)
        self.K = np.linalg.solve(np.linalg.inv(R), np.dot(B.T, self.P))
        self.A = A
        self.B = B

        self.R = R
        self.Q = Q

    def evaluate(self, time, state):
        return -np.dot(self.K, state)

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

    def get_R(self):
        return self.R

class LinearFeedbackConstTracker:

    """ Class for linear quadratic tracker for constant state tracking
    """

    def __init__(self, A, B, C, D, Q, R, const, g=None):

        """ LinearFeedbackConstTracker constructor

        Input:
        - A:            linear time-invariant state matrix
        - B:            linear time-invariant input matrix
        - Q:            control weighting matrix for state
        - R:            control weighting matrix for control inputs
        - const:        constant state to track
        - g:            state to track in error dynamics system

        """

        ##DEBUG
        #print("\n\n\n\n\n")
        #print("A = ")
        #print(A)
        #print("B = ")
        #print(B)
        #print("Q = ")
        #print(Q)
        #print("R = ")
        #print(R)

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.Q = Q
        self.R = R

        self.P = care(A, B, Q, R)
        self.K = np.linalg.solve(R, np.dot(B.T, self.P))
        self.Bt = copy.deepcopy(B.T)
        self.RBt = np.dot(np.linalg.inv(R), self.Bt)
        self.BRBt = np.dot(B, self.RBt)
        if g is None:
            self.g = np.dot(A, const)
        else:
            self.g = copy.deepcopy(g)
        # print("g = ", self.g)

        self.p = -np.linalg.solve(A.T - np.dot(self.P, self.BRBt), np.dot(self.P, self.g))
        self.R = copy.deepcopy(R)
        self.const = copy.deepcopy(const)

        # Closed loop is
        # \dot{x} = A_cl x + g_cl
        self.Acl = A - np.dot(B, self.K)

        self.g_cl = np.dot(B, np.dot(self.K, const)) - np.dot(B, np.dot(np.linalg.inv(R),
            np.dot(B.T, self.p)))

        self.tracking = None

        # steady-state
        self.xss = np.dot(self.BRBt, self.p) - self.g
        self.xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), self.xss)
        # self.xss = self.xss - self.const

        # steady-state optimal control
        # self.uss = self.evaluate(0, self.xss)
        self.uss = None

    def track(self, time, jj):
        self.tracking = jj

    def evaluate(self, time, state):
        # print("TIME: ", time, " STATE: ", state.T)

        return -np.dot(self.RBt, np.dot(self.P, state - self.const) + self.p)

    def get_closed_loop_A(self):
        return self.Acl

    def get_closed_loop_g(self):
        return self.g_cl

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

    def get_R(self):
        return self.R

    def get_p(self):
        return self.p

    def get_uss(self):
        # return self.uss
        return -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

    def get_xss(self):
        return self.xss

    def set_const(self, const):
        self.g = np.dot(self.A, const)

        self.p = -np.linalg.solve(self.A.T - np.dot(self.P, self.BRBt), np.dot(self.P, self.g))
        self.const = copy.deepcopy(const)

        # Closed loop is
        # \dot{x} = A_cl x + g_cl
        self.Acl = self.A - np.dot(self.B, self.K)

        self.g_cl = np.dot(self.B, np.dot(self.K, self.const)) - np.dot(self.B,
                np.dot(np.linalg.inv(self.R), np.dot(self.B.T, self.p)))

        self.tracking = None

        # steady-state
        self.xss = np.dot(self.BRBt, self.p) - self.g
        self.xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), self.xss)


class LinearFeedbackAugmented(LinearFeedbackConstTracker):

    """ Class representing the linear quadtratic tracker for tracking non-constant states
    """

    def __init__(self, A, B, C, D, Q, R, Fcl, g):

        """ LinearFeedbackConstTracker constructor

        Input:
        - A:            linear time-invariant state matrix of the agent/target doing the tracking
        - B:            linear time-invariant input matrix of the agent/target doing the tracking
        - Q:            control weighting matrix for state
        - R:            control weighting matrix for control inputs
        - Fcl:          closed-loop state matrix of system being tracked
        - g:            state to track in error dynamics system

        """

        # Top
        self.pre_augmented_A = A
        self.pre_augmented_B = B
        self.pre_augmented_C = C
        self.D = D
        self.pre_augmented_Q = Q

        self.nstates = A.shape[0] + Fcl.shape[0]
        const = np.zeros((self.nstates))

        Aconcat, Bconcat, Cconcat, Qconcat, gconcat = self.augment(A, B, C, Q, R, Fcl, g)

        super(LinearFeedbackAugmented, self).__init__(Aconcat, Bconcat, Cconcat, D, Qconcat, R,
                const, g=gconcat)

    def augment(self, A, B, C, Q, R, Fcl, g):

        """ Computes augmented matrices
        """

        # compute augmented matrices
        nstates = A.shape[0] + Fcl.shape[0]
        Aconcat = np.zeros((nstates, nstates))
        Aconcat[:A.shape[0], :A.shape[0]] = copy.deepcopy(A)
        Aconcat[A.shape[0]:, A.shape[0]:] = copy.deepcopy(Fcl)

        ncontrol = B.shape[1]
        Bconcat = np.zeros((nstates, ncontrol))
        Bconcat[:A.shape[0], :] = copy.deepcopy(B)

        # Caugmented = np.eye(nstates) # pxn
        # Cconcat = np.hstack((C, -np.eye(Fcl.shape[0])))
        Cconcat = np.hstack((C, -np.eye(A.shape[0])))
        # Qconcat = np.dot(Cconcat.T, np.dot(Q, Cconcat))
        Qconcat = np.zeros((nstates, nstates))
        Qconcat[:A.shape[0], :A.shape[0]] = copy.deepcopy(Q) # top left
        Qconcat[A.shape[0]:, A.shape[0]:] = copy.deepcopy(Q) # bottom right
        Qconcat[:A.shape[0], A.shape[0]:] = -copy.deepcopy(Q) # bottom left
        Qconcat[A.shape[0]:, :A.shape[0]] = -copy.deepcopy(Q) # top right

        gconcat = np.zeros((nstates))
        gconcat[A.shape[0]:] = copy.deepcopy(g)

        return Aconcat, Bconcat, Cconcat, Qconcat, gconcat

    # Recreate the augmented matrices for new tracking assignments
    def track(self, time, jj, Fcl, g): # systems.py line 64: precompute AUG LQ Tracker Policy

        """ Checks if tracker needs to be updated due to assignment change
        Input:
        - time:             float
        - jj:               integer global object ID
        - Fcl:              np.array closed-loop state matrix of the tracked system
        - g:                np.array feedforward term of the tracked system
        Output:
        """

        if time == 0: # initial assignment
            self.tracking = jj
            Fcl = copy.deepcopy(Fcl)
            self.nstates = self.pre_augmented_A.shape[0] + Fcl.shape[0]
            const = np.zeros((self.nstates))

            g = copy.deepcopy(g)
            self.A, self.B, self.C, self.Q, self.g = self.augment(self.pre_augmented_A,
                    self.pre_augmented_B, self.pre_augmented_C, self.pre_augmented_Q, self.R, Fcl,
                    g)
            self.P = care(self.A, self.B, self.Q, self.R)

            self.p = -np.linalg.solve(self.A.T - np.dot(self.P, self.BRBt), np.dot(self.P, self.g))

            # steady-state
            self.xss = np.dot(self.BRBt, self.p) - self.g
            self.xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), self.xss)

            # steady-state optimal control
            self.uss = super(LinearFeedbackAugmented, self).evaluate(0, self.xss) # subtracts const inside this function

        # TEST
        if time > 0 and self.tracking != jj: # recompute P for LQ TRACKER if assignment changes

            self.tracking = jj
            Fcl = copy.deepcopy(Fcl)
            self.nstates = self.pre_augmented_A.shape[0] + Fcl.shape[0]
            const = np.zeros((self.nstates))

            g = copy.deepcopy(g)
            self.A, self.B, self.C, self.Q, self.g = self.augment(self.pre_augmented_A,
                    self.pre_augmented_B, self.pre_augmented_C, self.pre_augmented_Q, self.R, Fcl,
                    g)
            self.P = care(self.A, self.B, self.Q, self.R)

            self.Bt = copy.deepcopy(self.B.T)
            self.RBt = np.dot(np.linalg.inv(self.R), self.Bt)
            self.BRBt = np.dot(self.B, self.RBt)
            self.p = -np.linalg.solve(self.A.T - np.dot(self.P, self.BRBt), np.dot(self.P, self.g))

            # steady-state
            self.xss = np.dot(self.BRBt, self.p) - self.g
            self.xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), self.xss)

            # steady-state optimal control
            self.uss = super(LinearFeedbackAugmented, self).evaluate(time, self.xss)

        else:
            return

    def evaluate(self, time, state1, state2, feedforward=0):

        """ Computes control input
        """

        # print("state = ", state)

        aug_state = np.hstack((copy.deepcopy(state1), copy.deepcopy(state2)))
        control = super(LinearFeedbackAugmented, self).evaluate(time, aug_state)
        return control

    # for the aug system, cost-to-go is coupled with state2 and it's dynamics
    # thus, must re-compute P and p
    def aug_cost_to_go(self, time, state1, state2, Fcl, g):

        """ Computes cost_to_go for a agent/target to track a single agent/target
        """

        aug_state = np.hstack((copy.deepcopy(state1), copy.deepcopy(state2)))

        Aconcat, Bconcat, Cconcat, Qconcat, gconcat = self.augment(self.pre_augmented_A,
                self.pre_augmented_B, self.pre_augmented_C, self.pre_augmented_Q, self.R, Fcl, g)

        P = care(Aconcat, Bconcat, Qconcat, self.R)

        Bt = copy.deepcopy(Bconcat.T)
        RBt = np.dot(np.linalg.inv(self.R), Bt)
        BRBt = np.dot(Bconcat, RBt)
        p = -np.linalg.solve(Aconcat.T - np.dot(P, BRBt), np.dot(P, gconcat))

        # steady-state
        xss = np.dot(BRBt, p) - gconcat
        xss = np.dot(np.linalg.inv(Aconcat - np.dot(BRBt.T, P)), xss)

        # steady-state optimal control
        self.uss = super(LinearFeedbackAugmented, self).evaluate(time, xss)

        cost_to_go = np.dot(aug_state, np.dot(P, aug_state)) + 2*np.dot(p.T, aug_state) -\
            np.dot(xss, np.dot(P, xss)) + 2*np.dot(p.T, xss)

        return cost_to_go

# IN DEVELOPMENT
class MinimumTimeIntercept(): # augmented proportional navigation - min time intercept with target accel

    """
    see bryson and ho (1975) 288 ch.9 eqn. 9.4.30
    """

    def __init__(self, time_final, dx):
        self.time_final = time_final
        self.dim_position = dx
        self.cp = 1
        self.ce = 3

    def get_final_time(self, state2):
        pass

    def track(self):
        pass

    def evaluate(self, time, state1, state2, feedforward=0):

        t = time
        tf = self.time_final

        ce = self.ce
        cp = self.cp

        # pursuer position
        rp = state1[:3]
        # pursuer velocity
        vp = state1[3:6]

        # evader position
        re = state2[:3]
        # evader velocity
        ve = state2[3:6]

        # proportional navigation guidance law
        u = -3/((1-ce/cp)*(tf-t)**2) * (rp-re + (vp-ve)*(tf-t) )

        return u

    def cost_to_go(self, time, state1, state2):
        # y1 = state1[:self.dim_position] - state2[:self.dim_position] # relative position
        # y2 = state1[self.dim_position:] - state2[self.dim_position:] # relative velocity
        # return  y1 + y2*(time_final-time) # miss distance
        time_to_go = self.time_final - time


class NonlinearController():

    """ Class representing generic nonlinear tracker for tracking non-constant states
    """

    def __init__(self):

        """ Nonelinear constructor

        Input:

        """

        self.tracking = None

    def track(self, time, jj):
        self.tracking = jj

    def evaluate(self, time, state1, state2):

        """ Computes control input
        """

        # Nonlinear Predictive Controllers - Ping Lu

        h = 0.5
        Q1 = 1.0
        Q2 = 1.0
        Q3 = 1.0
        g = 9.81
        R = 1000
        omega = 0.15
        V = 150

        dx = state1[0] - state2[0]
        dy = state1[1] - state2[1]
        dpsi = state1[2] - state2[2]

        ddotx = V * np.cos(state1[2]) - V * np.cos(state2[2])
        ddoty = V * np.sin(state1[2]) - V * np.sin(state2[2])

        psi = state1[2]

        # eq. 34
        P = 0.25 * g * (Q1 * np.sin(psi)**2 + Q2 * np.cos(psi)**2) + \
            g * Q3 / V**2


        # eq. 33
        term1 = 1/2/h**2 *(-Q1 * (dx + h * ddotx) * np.sin(psi) + \
                           Q2 * (dy + h * ddoty) * np.cos(psi))
        term2 = -0.25 * R * omega**2 * (Q1 * np.sin(psi) * np.cos(omega * time) - \
                                        Q2 * np.cos(psi) * np.sin(omega * time))
        term3 = Q3 / (h*V) * (dpsi - h * omega)

        control = (term1 + term2 + term3) / -P

        # print("control = ", control)

        # eq. 6
        # bounded control
        sigma = np.arctan(control)
        if sigma > (80/360)*2*np.pi:
            control = np.tan((80/360)*2*np.pi)
        elif sigma < (-80/360)*2*np.pi:
            control = np.tan((-80/360)*2*np.pi)

        return np.array([control])


    def sim_combined_dynamics(self, time, state1, state2):

        R = 1000
        g = 9.81
        V = 150

        tan_bank_angle_ref = V**2 / (R*g)


        def dynamics(t, x):

            out = np.zeros((6))
            out[0] = V * np.cos(x[2])
            out[1] = V * np.sin(x[2])
            out[2] = (g / V) * self.evaluate(t, x[:3], x[3:])
            out[3] = V * np.cos(x[5])
            out[4] = V * np.sin(x[5])
            out[5] = (g / V) * tan_bank_angle_ref
            return out
        
        h = 0.5
        final_time = 40
        all_time = np.arange(0, final_time, h)
        integrate_time = all_time[all_time > time-1e-10]
        print("integrate time", integrate_time[:4], state1, state2)
        states = np.concatenate((state1, state2))
        print("states = ", states)
        res = scint.solve_ivp(dynamics, (integrate_time[0], integrate_time[-1]), states,
                              t_eval=integrate_time, method='BDF', rtol=1e-6, atol=1e-8)
        return res, integrate_time
        
    def cost_to_go(self, time, state1, state2):

        """ Computes cost_to_go for a agent/target to track a single agent/target
        """
        # cost = np.linalg.norm(state1 - state2);
        
        # Brute force cost-to-go computer

        # these should be consistent with the main function, this is a hack for now!

        # print("time = ", time)

        # simulate the system into the future
        res, integrate_time = self.sim_combined_dynamics(time, state1, state2)

        # evaluate the controls at all the times
        controls = np.array([self.evaluate(integrate_time[ii], res.y[:3, ii], res.y[3:, ii]) \
                             for ii in range(integrate_time.shape[0])])
        r = 0.0 # this is confusing is it 1 or zero?
        cost_u = 0.5*r*np.cumsum(controls * controls)[:-1]
        
        # evaluate all of the residuals
        Q = 1.0    
        cost = 0.5*np.cumsum([Q*np.linalg.norm(res.y[:3, ii+1] - res.y[3:, ii+1])**2
                              for ii in range(integrate_time.shape[0]-1)])
        
        # sum up teh costs
        total_cost = cost + cost_u
        # return time[:-1], total_cost
        return total_cost[-1]

        # return cost

class NonlinearControllerTarget():

    """ Class representing generic nonlinear tracker for tracking non-constant states
    """

    def __init__(self):

        """ Nonelinear constructor

        Input:

        """

    def evaluate(self, time, state1):

        """ Computes control input
        """
        V = 150
        R = 1000
        g = 9.81
        tan_bank_angle_ref = V**2 / (R*g)
        # print("control = ", control)
        return np.array([tan_bank_angle_ref])

    def cost_to_go(self, time, state1, state2):
        assert 1 == 0
        """ Computes cost_to_go for a agent/target to track a single agent/target
        """


