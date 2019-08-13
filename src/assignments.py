
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

        # TODO fix to select correct position components for linearized quadcopter
        # dim_pos = int(dim_state / 2)

        # TODO update
        # xs = np.zeros((nagents, dim_pos))
        # xt = np.zeros((ntargets, dim_pos))
        dim = ref_states[0][1].get_dim() # same for all agents/targets in the same sim
        xs = np.zeros((nagents, dim))
        xt = np.zeros((ntargets, dim))

        # TODO fix to select correct position components for linearized quadcopter
        # for ii, state in enumerate(ref_states):
        #     # print(ii, state[0])
        #     xs[ii, :] = state[0][:dim_pos]

        # for jj, target in enumerate(target_states):
        #     xt[jj, :] = target[0][:dim_pos]

        for ii, state in enumerate(ref_states):
            # print(ii, state[0])
            # TEST
            # agent state components (differs per dynamic model)
            ref_state_statespace = ref_states[ii][1].get_statespace()
            dim_pos = ref_state_statespace['position']
            dim_vel = ref_state_statespace['velocity']
            xs[ii, 0] = state[0][dim_pos[0]]
            xs[ii, 1] = state[0][dim_pos[1]]
            if dim == 3:
                xs[ii, 2] = state[0][dim_pos[2]]

        for jj, target in enumerate(target_states):
            # TEST
            # agent state components (differs per dynamic model)
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
        # print("GO: ", G0)
        cost = log['cost']

        # thresh = 4e-1 # 2v2 case
        # thresh = 4e-2 # 2v2 case
        thresh = 1/n
        G0[G0>thresh] = 1

        # print(log)
        # exit(1)
        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

class AssignmentDyn(Assignment):

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

                # AUGMENTED LQ TRACKER COST-TO-GO )
                Fcl = target[1].pol.get_closed_loop_A()
                g = target[1].pol.get_closed_loop_g()
                M[ii, jj] = agent[1].pol.aug_cost_to_go(t, agent[0], target[0], Fcl, g)


        # if M[0,0] >= 894:
        #     import ipdb; ipdb.set_trace()

        # M /= M.max() # I dont divide b
        M = M/M.max()

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        G0, log = ot.emd(a, b, M, log=True)

        # if t >= 0.088:
        #     import ipdb; ipdb.set_trace()

        cost = log['cost']
        # thresh = 4e-1 # 2v2
        # thresh = 4e-2
        thresh = 1/n
        G0[G0>thresh] = 1

        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

