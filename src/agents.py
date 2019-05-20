
import copy

#################################
### Agents
################################
class Agent:

    def __init__(self, dx, dyn, pol):
        self.dx = dx
        self.dyn = copy.deepcopy(dyn)
        self.pol = copy.deepcopy(pol)

    def get_pol(self):
        return self.pol
    
    def update_pol(self, pol):
        self.pol = copy.deepcopy(pol)

    def state_size(self):
        return self.dx

class TrackingAgent(Agent):

    def  __init__(self, dx, dyn, pol):
        super(TrackingAgent, self).__init__(dx, dyn, pol)

    def rhs(self, t, x, ref_signal):
        u = self.pol(t, x, ref_signal)
        return self.dyn.rhs(t, x, u)

