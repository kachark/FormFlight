
""" @file dynamics.py
"""

import numpy as np
import copy

################################
## Linear Dynamics
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

