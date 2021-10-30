
import numpy as np
import copy
import ot

################################
## Assignments
###############################
class Assignment:

    """ Assignment parent class
    """

    def __init__(self, nref, ntarget):

        """ Assignmnent constructor

        Input:
        Output:

        """

        self.nref = nref
        self.ntarget = ntarget

    def assignment(self, t, ref_states, target_states):
        pass

class AssignmentLexical(Assignment):

    """ Assignment class representing lexical assignment policy
    """

    def assignment(self, t, ref_states, target_states):

        """ Returns assignment list based on indices of targets
        """

        inds_out = np.array(range(len(target_states)))
        return inds_out, None

class AssignmentEMD(Assignment):

    """ Class representing Earth-Movers Distance (EMD) assignment policy
    """

    def assignment(self, t, ref_states, target_states):

        """ Returns EMD assignment list

        ref_states and target_states are lists of tuples
        each tuple is (state <np.array>, agent/target <Agent>).

        For the nearest neighbor EMD assignment, the information
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.

        Input:
        - t:                    time
        - ref_states:           list of tuples constructed as (state at time <np.array>, agent <Agent class>)
        - target_states:        list of tuples constructed as (state at time <np.array>, target <Agent class>

        Output:
        - assignment:           numpy array which maps each column (agent index) to an integer representing target index
        - cost:                 discrete optimal transport cost

        """

        n = len(ref_states) +  len(target_states)

        ## Assume first two states are the positions
        nagents = len(ref_states)
        ntargets = len(target_states)

        dim_state = ref_states[0][0].shape[0]


        dim = ref_states[0][1].get_dim()
        xs = np.zeros((nagents, dim))
        xt = np.zeros((ntargets, dim))

        for ii, state in enumerate(ref_states):
            # agent state components (differs per dynamic model)
            ref_state_statespace = ref_states[ii][1].get_statespace()
            dim_pos = ref_state_statespace['position']
            dim_vel = ref_state_statespace['velocity']
            xs[ii, 0] = state[0][dim_pos[0]]
            xs[ii, 1] = state[0][dim_pos[1]]
            if dim == 3:
                xs[ii, 2] = state[0][dim_pos[2]]

        for jj, target in enumerate(target_states):
            # target state components (differs per dynamic model)
            target_state_statespace = target_states[jj][1].get_statespace()
            dim_pos = target_state_statespace['position']
            dim_vel = target_state_statespace['velocity']
            xt[jj, 0] = target[0][dim_pos[0]]
            xt[jj, 1] = target[0][dim_pos[1]]
            if dim == 3:
                xt[jj, 2] = target[0][dim_pos[2]]

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        M = ot.dist(xs, xt)

        M /= M.max()

        G0, log = ot.emd(a, b, M, log=True)
        cost = log['cost']

        thresh = 1/n
        G0[G0>thresh] = 1

        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

class AssignmentCustom(Assignment):

    """ Class representing dynamics-based assignment policy
    """

    def assignment(self, t, ref_states, target_states):

        """
        ref_states and target_states are lists of tuples
        each tuple is (state <np.array>, agent/target <Agent>).

        For the nearest neighbor EMD assignment, the information
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.

        Input:
        - t:                    time
        - ref_states:           list of tuples constructed as (state at time <np.array>, agent <Agent class>)
        - target_states:        list of tuples constructed as (state at time <np.array>, target <Agent class>

        Output:
        - assignment:           numpy array which maps each column (agent index) to an integer representing target index
        - cost:                 discrete optimal transport cost

        """

        ref_states = copy.deepcopy(ref_states)
        target_states = copy.deepcopy(target_states)

        n = len(ref_states) + len(target_states)

        ## Assume first two states are the positions
        nagents = len(ref_states)
        ntargets = len(target_states)

        dim_state = ref_states[0][0].shape[0]

        M = np.zeros((nagents, ntargets))
        for ii, agent in enumerate(ref_states):
            for jj, target in enumerate(target_states):
                Cij = np.linalg.norm(agent[0] - target[0])
                M[ii, jj] = Cij

        # M /= M.max() # I dont divide b
        M = M/M.max()

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        G0, log = ot.emd(a, b, M, log=True)

        cost = log['cost']
        thresh = 1/n
        G0[G0>thresh] = 1

        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

