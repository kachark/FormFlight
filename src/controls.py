
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

    def __init__(self, A, B, C, Q, R):
        self.P = care(A, B, Q, R)
        self.K = -np.linalg.solve(np.linalg.inv(R), np.dot(B.T, self.P))
        # self.K = np.linalg.solve(R, np.dot(B.T, self.P)) # ltidyn_cl
        self.A = A
        self.B = B
        self.C = C

        self.R = R
        self.Q = Q

    def evaluate(self, time, state):
        return np.dot(-self.K, state)

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

    def get_R(self):
        return self.R

class LinearFeedbackTracking(LinearFeedback):

    def __init__(self, A, B, C, Q, R):
        super(LinearFeedbackTracking, self).__init__(A, B, C, Q, R)
        self.Paug = None
        self.T = None
        self.B1 = None
        self.R1 = None
        # self.Q1 = np.concatenate((Q, -Q), axis=1)
        # lower = np.concatenate((-Q, Q), axis=1)
        # self.Q1 = np.concatenate((self.Q1, lower), axis=0)
        # self.Q1 = np.eye(8)

        # self.Q[2,2] = 0
        # self.Q[3,3] = 0
        self.C1 = np.concatenate((np.eye(A.shape[0]), -np.eye(A.shape[0])), axis=1)
        self.Q1 = np.matmul(self.C1.T, np.matmul(self.Q, self.C1))
        self.assigned_to = None

    def augment(self, Fcl, r):
        T = np.concatenate((self.A, np.zeros((self.A.shape[0], Fcl.shape[1]))), axis=1)
        lower = np.concatenate((np.zeros((Fcl.shape[0], self.A.shape[1])), Fcl), axis=1)
        T = np.concatenate((T, lower), axis=0)

        B1 = np.concatenate((self.B, np.zeros((Fcl.shape[0], self.B.shape[1]))), axis=0)

        r = r.reshape((4,1))
        R1 = np.concatenate((np.zeros((self.A.shape[0], 1)), r), axis=0)
        R1 = np.concatenate((r, r), axis=0)

        return T, B1, R1

    def track(self, t, jj, Fcl, r): # systems.py line 64: precompute AUG LQ Tracker Policy

        if t == 0: # initial assignment
            self.assigned_to = jj
            self.assigned_to = jj
            T = copy.deepcopy(Fcl)
            r = copy.deepcopy(r)
            self.T, self.B1, self.R1 = self.augment(T, r)
            self.Paug = care(self.T, self.B1, self.Q1, self.R)

        if self.assigned_to != jj: # recompute P for LQ TRACKER if assignment changes
            self.assigned_to = jj
            T = copy.deepcopy(Fcl)
            r = copy.deepcopy(r)
            self.T, self.B1, self.R1 = self.augment(T, r)
            self.Paug = care(self.T, self.B1, self.Q1, self.R)
        else:
            return

    def evaluate(self, time, state1, state2, feedforward=0): # ORIGINAL CONTROL LAW
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        agent_pol = np.dot(self.K, diff) + feedforward
        return agent_pol

    def evaluate2(self, time, state1, state2): # AUGMENTED LQ TRACKER
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        # X = np.concatenate((s1, s2), axis=0) # augmented state
        X = np.hstack((s1, s2))
        X = X.reshape((8,1))
        K1 = np.matmul(-np.linalg.inv(self.R), np.matmul(self.B1.T, self.Paug))
        agent_pol = np.matmul(K1, X)
        return agent_pol

    def evaluate3(self, time, state1, state2): # NON-AUGMENTED LQ TRACKER (NO FEEDFORWARD)
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        PBRB = np.matmul(self.B.T, self.P)
        PBRB = np.matmul(np.linalg.inv(self.R), PBRB)
        PBRB = np.matmul(self.B, PBRB)
        p_inner = self.A - PBRB
        p_inner = p_inner.T
        p_lhs = -np.matmul(self.C.T, np.matmul(self.Q, s2))
        ppp = np.linalg.solve(p_inner, p_lhs)

        RB = np.matmul(np.linalg.inv(self.R), self.B.T)
        agent_pol = np.matmul(self.K, s1) + np.matmul(RB, ppp)
        return agent_pol

        # s1 = copy.deepcopy(state1)
        # s2 = copy.deepcopy(state2)
        # PBRB = np.matmul(np.linalg.inv(self.R), self.B.T)
        # PBRB = np.matmul(self.B, PBRB)
        # PBRB = np.matmul(self.P, PBRB)
        # p_inner = self.A.T - PBRB
        # ppp = np.linalg.solve(p_inner, s2)

        # RB = np.matmul(np.linalg.inv(self.R), self.B.T)
        # agent_pol = np.matmul(self.K, s1) + np.matmul(-RB, ppp)
        # return agent_pol

    # deprecate
    def evaluate4(self, time, state1, state2, feedforward): # NON-AUGMENTED LQ TRACKER (FEEDFORWARD)
        # s1 = copy.deepcopy(state1)
        # s2 = copy.deepcopy(state2)
        # ud = copy.deepcopy(feedforward)
        # PBRB = np.matmul(np.linalg.inv(self.R), self.B.T)
        # PBRB = np.matmul(self.B, PBRB)
        # PBRB = np.matmul(self.P, PBRB)
        # p_inner = self.A.T - PBRB

        # ff = 2*np.matmul(self.P, np.matmul(self.B, ud))
        # ppp = np.linalg.solve(p_inner, 2*s2 - ff)

        # RB = np.matmul(np.linalg.inv(self.R), self.B.T)
        # agent_pol = ud + (np.matmul(self.K, s1) + np.matmul(-RB, 0.5*ppp))
        # return agent_pol

        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        PBRB = np.matmul(self.B.T, self.P)
        PBRB = np.matmul(np.linalg.inv(self.R), PBRB)
        PBRB = np.matmul(self.B, PBRB)
        p_inner = self.A - PBRB
        p_inner = p_inner.T
        p_lhs = -np.matmul(self.C.T, np.matmul(self.Q, s2))
        ppp = np.linalg.solve(p_inner, p_lhs)

        RB = np.matmul(np.linalg.inv(self.R), self.B.T)
        agent_pol = np.matmul(self.K, s1) + np.matmul(RB, ppp) + feedforward
        return agent_pol
        
    def cost_to_go(self, time, state1, state2):
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

    def cost_to_go2(self, time, state1, state2, Fcl, r):
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        X = np.concatenate((s1, s2), axis=0) # augmented state
        # X = np.hstack((s1, s2))
        T, B1, R1 = self.augment(Fcl, r)
        Paug = care(T, B1, self.Q1, self.R)
        cost_to_go = np.matmul(X.T, np.matmul(Paug, X))
        return cost_to_go

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, C, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, C, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0] # np array
        self.P = care(A, B, Q, R)
        self.K = -np.linalg.solve(R, np.dot(B.T, self.P))
        self.Acl = A + np.matmul(B, self.K)
        self.r = np.matmul(B, np.matmul(-self.K, self.offset))

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
