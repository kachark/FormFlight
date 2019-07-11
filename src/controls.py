
from scipy.linalg import solve_continuous_are as care
import numpy as np
import copy

################################
## Agent Policies
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

class LinearFeedbackTracking(LinearFeedback):

    def __init__(self, A, B, Q, R):
        super(LinearFeedbackTracking, self).__init__(A, B, Q, R)

        self.P = care(A, B, Q, R)
        self.K = np.linalg.solve(R, np.dot(B.T, self.P))
        self.Bt = copy.deepcopy(B.T)
        self.RBt = np.dot(np.linalg.inv(R), self.Bt)
        self.BRBt = np.dot(B, self.RBt)

        self.R = copy.deepcopy(R)

        # Closed loop is
        # \dot{x} = A_cl x + g_cl
        self.Acl = A - np.dot(B, self.K)

        self.tracking = None

        # steady-state
        self.xss = None

        # steady-state optimal control
        # self.uss = self.evaluate(0, self.xss)
        self.uss = None


    def evaluate(self, time, state1, state2, feedforward=0):
        # print("state = ", state)
        diff = state1 - state2
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, diff))
        agent_pol = -np.dot(self.K, diff) + feedforward
        return agent_pol

    def cost_to_go(self, time, state1, state2):
        diff = state1 - state2
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

    def get_uss(self, xss, ud_ss):
        self.uss = -np.dot(self.K, xss) + ud_ss
        return self.uss
        # return -np.dot(self.RBt, np.dot(self.P, self.xss) + self.p)

    def get_xss(self, ud_ss):
        self.xss = -np.dot(np.linalg.inv(self.A - np.dot(self.BRBt.T, self.P)), np.dot(self.B, ud_ss))
        return self.xss

##################################3
# NEW
class LinearFeedbackConstTracker:
    def __init__(self, A, B, Q, R, const, g=None):

        print("\n\n\n\n\n")
        print("A = ")
        print(A)
        print("B = ")
        print(B)
        print("Q = ")
        print(Q)
        print("R = ")
        print(R)

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
        print("g = ", self.g)

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
        self.xss = self.xss - self.const

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
    
class LinearFeedbackAugmented(LinearFeedbackConstTracker):

    def __init__(self, A, B, Q, R, Fcl, g):

        # Top
        self.pre_augmented_A = A
        self.pre_augmented_B = B
        self.pre_augmented_Q = Q

        self.nstates = A.shape[0] + Fcl.shape[0]
        const = np.zeros((self.nstates))

        Aconcat, Bconcat, Qconcat, gconcat = self.augment(A, B, Q, R, Fcl, g)

        super(LinearFeedbackAugmented, self).__init__(Aconcat, Bconcat, Qconcat, R, const, g=gconcat)

    def augment(self, A, B, Q, R, Fcl, g):
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

        if self.tracking != jj: # recompute P for LQ TRACKER if assignment changes
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
        # print("state = ", state)

        aug_state = np.hstack((copy.deepcopy(state1), copy.deepcopy(state2)))
        control = super(LinearFeedbackAugmented, self).evaluate(time, aug_state)
        return control

    # for the aug system, cost-to-go is coupled with state2 and it's dynamics
    # thus, must re-compute P and p
    def aug_cost_to_go(self, time, state1, state2, Fcl, g):
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

##################################3

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, C, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, C, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0] # np array
        self.P = care(A, B, Q, R)
        self.K = -np.linalg.solve(R, np.dot(B.T, self.P))
        self.Acl = A + np.matmul(B, self.K)
        # self.r = np.matmul(B, np.matmul(-self.K, self.offset))
        RB = np.matmul(np.linalg.inv(R), B.T)
        ur = np.matmul(RB, np.linalg.inv(self.Acl.T))
        ur = np.matmul(ur, np.matmul(self.P, np.matmul(A, offset)))
        Kr = np.matmul(-self.K, offset)
        self.r = np.matmul(B, Kr - ur)

        # self.r = -np.matmul(B, ur)

        # rho = -np.matmul(A, offset)
        # self.r = -np.matmul(B, ur) + rho

    def evaluate(self, time, state1):

        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        diff -= self.offset
        agent_pol = np.dot(self.K, diff)
        # agent_pol = np.dot(-self.K, diff) # ltidyn_cl
        return agent_pol

    def cost_to_go(self, time, state1):
        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

class MinimumTimeIntercept():

    def __init__(self, time_final, dx):
        self.time_final = time_final
        self.dim_position = dx

    def evaluate(self, time, state1, state2):
        pass

    def cost_to_go(self, time, state1, state2):
        y1 = state1[:self.dim_position] - state2[:self.dim_position] # relative position
        y2 = state1[self.dim_position:] - state2[self.dim_position:] # relative velocity
        return  y1 + y2*(time_final-time) # miss distance

class LinearFeedbackIntegralTracking(LinearFeedback):

    def __init__(self, A, B, Q, R):
        super(LinearFeedbackIntegralTracking, self).__init__(A, B, Q, R)

    def evaluate(self, time, state1, state2, feedforward=0):
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1[:4] - s2
        agent_pol = np.dot(self.K, diff) + feedforward
        return agent_pol

    def cost_to_go(self, time, state1):
        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go    
