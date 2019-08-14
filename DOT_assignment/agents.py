
""" @file agents.py
"""

import copy

#################################
### Agents
################################
class Agent:

    """ Class representing a member of agent or target swarm
    """

    def __init__(self, dx, statespace, dim, dyn, pol):

        """ Agent class constructor

        Input:
        - dx:                   state size
        - statespace:           dict containing descriptions of the components of an agent/target state
        - dim:                  int representing the dimension the Agent acts within (2D/3D)
        - dyn:                  dynamics model
        - pol:                  control policy

        Output:

        """

        self.dx = dx
        self.statespace = statespace
        self.dim = dim
        self.dyn = copy.deepcopy(dyn)
        self.pol = copy.deepcopy(pol)

    def get_pol(self):
        return self.pol

    def update_pol(self, pol):
        self.pol = copy.deepcopy(pol)

    def state_size(self):
        return self.dx

    def get_statespace(self):
        return self.statespace

    def get_dim(self):
        return self.dim

class TrackingAgent(Agent):

    def  __init__(self, dx, statespace, dim, dyn, pol):
        super(TrackingAgent, self).__init__(dx, statespace, dim, dyn, pol)

    def rhs(self, t, x, ref_signal):
        u = self.pol(t, x, ref_signal)
        return self.dyn.rhs(t, x, u)

