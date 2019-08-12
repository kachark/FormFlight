
""" @file dynamics.py
"""

import numpy as np
import copy

################################
## LINEAR Agent Dynamics
################################
class LTIDyn:

    """ Class representing linear time-invariant dynamics
    """

    def __init__(self, A, B):

        """ LTIDyn constructor

        Input:
        - A:            state matrix
        - B:            control input matrix

        """

        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)

    def rhs(self, t, x, u):

        """ Computes the right-hand side of the LTI system

        Input:
        - t:            time
        - x:            state vector
        - u:            control input vector

        Output:
        - \dot{x} = Ax + Bu

        """

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

