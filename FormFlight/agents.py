
""" @file agents.py
"""

import copy

#################################
### Objects
################################

class Object():

    def __init__(self, object_type):
        self.type = object_type
        self.ID = None

class Point(Object):

    def __init__(self, point_type):

        """ Agent class constructor

        Input:

        Output:

        """

        super(Point, self).__init__(point_type)

        self.info = None
        self.dx = None
        self.statespace = None
        self.dim = None

class Agent(Object):

    """ Class representing a dynamic object
    """

    def __init__(self, agent_type):

        """ Agent class constructor

        Input:

        Output:

        """

        super(Agent, self).__init__(agent_type)

        self.info = None
        self.dx = None
        self.du = None
        self.statespace = None
        self.dim = None
        self.dyn = None
        self.pol = None

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

    def get_ID(self):
        return self.ID

    def get_type(self):
        return self.type

class MultiAgentSystem():

    def __init__(self, name, agent_list):

        self.name = name
        self.agent_list = agent_list # don't deepcopy - need reference to original
        self.nagents = len(self.agent_list)
        self.ID = None

        self.formation = None

        self.decision_maker_type = None
        self.decision_maker = None
        self.decision_epoch = None
        self.current_decision = None

    def get_agent_list(self):
        return self.agent_list

    def add_agent(self, agent):
        self.agent_list.append(agent)
        self.nagents = len(self.agent_list)

    def remove_agent(self, ID):
        for agent in self.agent_list:
            if agent.ID == ID:
                self.agent_list.remove(agent)
                break

        self.nagents = len(self.agent_list)

    def compute_decision(self):
        pass

