
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
        self.K = -np.linalg.solve(R, np.dot(B.T, self.P))
        # self.K = np.linalg.solve(R, np.dot(B.T, self.P)) # ltidyn_cl
        self.B = B

        self.R = R
        self.Q = Q

    def evaluate(self, time, state):
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, state))
        # return np.dot(self.K, state)
        return np.dot(-self.K, state)

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

    def get_R(self):
        return self.R

class LinearFeedbackTracking(LinearFeedback):

    def __init__(self, A, B, Q, R):
        super(LinearFeedbackTracking, self).__init__(A, B, Q, R)

    def evaluate(self, time, state1, state2, feedforward=0):
        # print("state = ", state)
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, diff))
        agent_pol = np.dot(self.K, diff) + feedforward
        return agent_pol

    def cost_to_go(self, time, state1, state2):
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0] # np array

    def evaluate(self, time, state1):

        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        # diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
        # diff[:self.dim_offset] -= self.offset
        diff -= self.offset
        # print("state = ", state1, diff)
        agent_pol = np.dot(self.K, diff)
        # agent_pol = np.dot(-self.K, diff) # ltidyn_cl
        return agent_pol

    def cost_to_go(self, time, state1):
        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go    

