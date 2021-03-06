
""" @file agents.py
"""

import copy


class Point:

    """ Class representing a general point in statespace
    """

    def __init__(self, dx, du, statespace, dim):

        """ Agent class constructor

        Input:
        - dx:                   state size
        - du:                   control input size
        - statespace:           dict containing descriptions of the components of an agent/target state
        - dim:                  int representing the dimension the Agent acts within (2D/3D)

        """

        self.dx = dx
        self.du = du
        self.statespace = statespace
        self.dim = dim

    def state_size(self):
        return self.dx

    def get_statespace(self):
        return self.statespace

#################################
### Agents
################################
class Agent(Point):

    """ Class representing a member of agent or target swarm
    """

    def __init__(self, dx, du, statespace, dim, dyn, pol):

        """ Agent class constructor

        Input:
        - dx:                   state size
        - du:                   control input size
        - statespace:           dict containing descriptions of the components of an agent/target state
        - dim:                  int representing the dimension the Agent acts within (2D/3D)
        - dyn:                  dynamics model
        - pol:                  control policy

        Output:

        """

        self.dyn = copy.deepcopy(dyn)
        self.pol = copy.deepcopy(pol)
        super(Agent, self).__init__(dx, du, statespace, dim)

    def get_pol(self):
        return self.pol

    def update_pol(self, pol):
        self.pol = copy.deepcopy(pol)

    def get_dim(self):
        return self.dim

class TrackingAgent(Agent):

    def  __init__(self, dx, du, statespace, dim, dyn, pol):
        super(TrackingAgent, self).__init__(dx, du, statespace, dim, dyn, pol)

    def rhs(self, t, x, ref_signal):
        u = self.pol(t, x, ref_signal)
        return self.dyn.rhs(t, x, u)

