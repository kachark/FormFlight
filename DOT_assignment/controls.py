
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
        return self.uss
        # return -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

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

    def __init__(self, A, B, Q, R, Fcl, g):

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
        self.pre_augmented_Q = Q

        self.nstates = A.shape[0] + Fcl.shape[0]
        const = np.zeros((self.nstates))

        Aconcat, Bconcat, Qconcat, gconcat = self.augment(A, B, Q, R, Fcl, g)

        super(LinearFeedbackAugmented, self).__init__(Aconcat, Bconcat, Qconcat, R, const, g=gconcat)

    def augment(self, A, B, Q, R, Fcl, g):

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

        Qconcat = np.zeros((nstates, nstates))
        Qconcat[:A.shape[0], :A.shape[0]] = copy.deepcopy(Q)
        Qconcat[A.shape[0]:, A.shape[0]:] = copy.deepcopy(Q)
        Qconcat[:A.shape[0], A.shape[0]:] = -copy.deepcopy(Q)
        Qconcat[A.shape[0]:, :A.shape[0]] = -copy.deepcopy(Q)

        gconcat = np.zeros((nstates))
        gconcat[A.shape[0]:] = copy.deepcopy(g)

        return Aconcat, Bconcat, Qconcat, gconcat

    # Recreate the augmented matrices for new tracking assignments
    def track(self, time, jj, Fcl, g): # systems.py line 64: precompute AUG LQ Tracker Policy

        """ Checks if tracker needs to be updated due to assignment change
        """

        if time == 0: # initial assignment
            self.tracking = jj
            Fcl = copy.deepcopy(Fcl)
            self.nstates = self.pre_augmented_A.shape[0] + Fcl.shape[0]
            const = np.zeros((self.nstates))

            g = copy.deepcopy(g)
            self.A, self.B, self.Q, self.g = self.augment(self.pre_augmented_A, self.pre_augmented_B,
                                                          self.pre_augmented_Q, self.R, Fcl, g)
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
            self.A, self.B, self.Q, self.g = self.augment(self.pre_augmented_A, self.pre_augmented_B,
                                                          self.pre_augmented_Q, self.R, Fcl, g)
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

        Aconcat, Bconcat, Qconcat, gconcat = self.augment(self.pre_augmented_A, self.pre_augmented_B,
                                                          self.pre_augmented_Q, self.R, Fcl, g)

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


