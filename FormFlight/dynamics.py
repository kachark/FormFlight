
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

    def __init__(self, A, B, C, D):

        """ LTIDyn constructor

        Input:
        - A:            state matrix
        - B:            control input matrix

        """

        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)
        self.C = copy.deepcopy(C)
        self.D = copy.deepcopy(D)

    def rhs(self, t, x, u):

        """ Computes the right-hand side of the LTI system

        Input:
        - t:            float time
        - x:            np.array state vector
        - u:            np.array control input vector

        Output:
        - \dot{x} = Ax + Bu

        """

        x = copy.deepcopy(x)
        u = copy.deepcopy(u)
        xdot = np.dot(self.A, x) + np.dot(self.B, u)

        return xdot

    def output(self, t, x, u):

        """ Computes the observed state (output vector) of the LTI system

        Input:
        - t:            float time
        - x:            np.array state vector
        - u:            np.array control input vector

        Output:
        - y = Cx + Du
        """

        x = copy.deepcopy(x)
        u = copy.deepcopy(u)
        y = np.dot(self.C, x) + np.dot(self.D, u)

        return y


class NonlinearDyn:

    """ Class representing nonlinear dynamics
    """

    def __init__(self, f):

        """ NonlinearDyn constructor

        Input:
        - f: function pointer with signature f(t, x)

        """

        self.f = copy.deepcopy(f)

    def rhs(self, t, x, u):

        """ Computes the right-hand side of the LTI system

        Input:
        - t:            float time
        - x:            np.array state vector

        Output:
        - \dot{x} = f(t, x)

        """

        x = copy.deepcopy(x)
        u = copy.deepcopy(u)
        return self.f(t, x, u)


