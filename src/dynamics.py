
import numpy as np
import copy

################################
## LINEAR Agent Dynamics
###############################    
class LTIDyn:

    def __init__(self, A, B):
        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)

    def rhs(self, t, x, u):
        x = copy.deepcopy(x)
        u = copy.deepcopy(u)
        return np.dot(self.A, x) + np.dot(self.B, u)

class LTIDyn_closedloop(LTIDyn): # equivalent to LTIDyn

    def  __init__(self, A, B, K):
        super(LTIDyn_closedloop, self).__init__(A, B)
        self.A = A
        self.B = B
        self.Acl = self.A - np.dot(self.B, K)
        self.K = K

    def rhs(self, t, x, xt, ud):
        s1 = copy.deepcopy(x)
        s2 = copy.deepcopy(xt)
        su = copy.deepcopy(ud)
        ucl = np.dot(self.K, s2) + su
        utr = np.dot(self.K, s2)
        return np.dot(self.Acl, s1) + np.dot(self.B, utr)

class LTI_Tracking_Dyn(LTIDyn):

    def __init__(self, agent_dyn, target_dyn, Fcl, r):
        super(LTI_Tracking_Dyn, self).__init__(agent_dyn.A, agent_dyn.B)
        self.agent_dyn = agent_dyn
        self.target_dyn = target_dyn
        self.A = agent_dyn.A
        self.B = agent_dyn.B
        self.Fcl = Fcl
        self.r = r
        self.T, self.B1, self.R1 = self.augment()

    def augment(self):
        T = np.concatenate((self.A, np.zeros((self.A.shape[0], self.Fcl.shape[1]))), axis=1)
        lower = np.concatenate((np.zeros((self.Fcl.shape[0], self.A.shape[1])), self.Fcl), axis=1)
        T = np.concatenate((T, lower), axis=0)


        B1 = np.concatenate((self.B, np.zeros((self.target_dyn.B.shape[0], self.B.shape[1]))), axis=0)
        self.r = self.r.reshape((4,1))
        R1 = np.concatenate((np.zeros((self.A.shape[0], 1)), self.r), axis=0)
        # R1 = np.concatenate((self.r, np.zeros((self.A.shape[0], 1))), axis=0)

        return T, B1, R1

    def separate(self, X):
        dxagent = self.A.shape[0]
        dxtarget = self.Fcl.shape[0]
        return X[:dxagent], X[dxagent:dxagent+dxtarget]

    def rhs(self, t, x, xt, u):
        s1 = copy.deepcopy(x)
        s2 = copy.deepcopy(xt)

        # X = np.concatenate((s1, s2), axis=1) # augmented state
        X = np.hstack((s1, s2))
        X = X.reshape(X.shape[0], 1)
        dXdt = np.matmul(self.T, X) + np.matmul(self.B1, u)
        # dXdt = np.matmul(self.T, X) + np.matmul(self.B1, u) + self.R1
        dxagent, dxtarget = self.separate(dXdt)
        return dxagent, dxtarget
        # return np.matmul(self.T, X) + np.matmul(self.B1, u) + self.R1







