
import numpy as np

def double_integrator_2D():

    dx = 4
    du = 2
    A = np.array([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float_)

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float_)

    C = np.eye(dx)

    D = 0

    return A, B, C, D, dx, du

