

import copy
import numpy as np

class State:

    def __init__(self, x):

        self.x = x

    def add_state(self, xi):

        xi = copy.deepcopy(xi)
        self.x = np.append(self.x, xi)

