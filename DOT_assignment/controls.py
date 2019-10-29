
""" @file controls.py
"""

from scipy.linalg import solve_continuous_are as care
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

    def __init__(self, A, B, Q, R, const, g=None):

        """ LinearFeedbackConstTracker constructor

        Input:
        - A:            linear time-invariant state matrix
        - B:            linear time-invariant input matrix
        - Q:            control weighting matrix for state
        - R:            control weighting matrix for control inputs
        - const:        constant state to track
        - g:            state to track in error dynamics system

        """

        self.A = A
        self.B = B
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

        self.g_cl = np.dot(B, np.dot(self.K, const)) - np.dot(B, np.dot(np.linalg.inv(R), np.dot(B.T, self.p)))

        self.tracking = None

        # steady-state
        self.xss = np.dot(self.BRBt, self.p) - self.g
        self.xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), self.xss)
        # self.xss = self.xss - self.const

        # steady-state optimal control
        # self.uss = self.evaluate(0, self.xss)
        self.uss = -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

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
        return self.uss
        # return -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

    def get_xss(self):
        return self.xss

    def set_const(self, time, target_id, const):
        self.tracking = target_id

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

        self.uss = -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

    def cost_to_go(self, time, state1, const_state):

        """ Computes cost_to_go for a agent/target to track a single agent/target
        """

        error_state = state1 - const_state

        g = np.dot(self.A, const_state)

        p = -np.linalg.solve(self.A.T - np.dot(self.P, self.BRBt), np.dot(self.P, g))

        g_cl = np.dot(self.B, np.dot(self.K, const_state)) - np.dot(self.B, np.dot(np.linalg.inv(self.R),
            np.dot(self.B.T, p)))

        self.tracking = None

        # steady-state
        xss = np.dot(self.BRBt, p) - g
        xss = np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), xss)

        # steady-state optimal control
        # self.uss = self.evaluate(0, self.xss)
        uss = -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

        cost_to_go = np.dot(error_state, np.dot(self.P, error_state)) + 2*np.dot(p.T, error_state) -\
            np.dot(xss, np.dot(self.P, xss)) + 2*np.dot(p.T, xss)

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


