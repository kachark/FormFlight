
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

    def assignment(self, t, object_list, target_object_list):

        """ Returns EMD assignment list

        object_list and target_object_list are lists of tuples
        each tuple is (state <np.array>, agent/target/etc. <Agent>).

        For the nearest neighbor EMD assignment, the information
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.

        Input:
        - t:                         time
        - object_list:               list of tuples constructed as (state at time <np.array>, agent <Agent class>)
        - target_object_list:        list of tuples constructed as (state at time <np.array>, target <Agent class>

        Output:
        - assignment:           numpy array which maps each column (agent index) to an integer representing target index
        - cost:                 discrete optimal transport cost

        """

        ## Assume first two states are the positions
        nagents = len(object_list)
        ntargets = len(target_object_list)
        n = nagents + ntargets

        # create a numpy array which represents the state distribution of all objects
        dim = object_list[0][1].get_dim() # same for all agents/targets in the same sim
        xs = np.zeros((nagents, dim))
        xt = np.zeros((ntargets, dim))

        for ii in range(nagents):
            object_state = object_list[ii][0]
            object_i = object_list[ii][1]

            object_statespace = object_i.get_statespace()
            dim_pos = object_statespace['position']
            dim_vel = object_statespace['velocity']
            assert dim_vel == None
            xs[ii, 0] = object_state[dim_pos[0]]
            xs[ii, 1] = object_state[dim_pos[1]]
            if dim == 3:
                xs[ii, 2] = object_state[dim_pos[2]]

        for jj in range(ntargets):
            target_state = target_object_list[jj][0]
            target = target_object_list[jj][1]

            target_statespace = target.get_statespace()
            dim_pos = target_statespace['position']
            dim_vel = target_statespace['velocity']
            assert dim_vel == None
            xt[jj, 0] = target_state[dim_pos[0]]
            xt[jj, 1] = target_state[dim_pos[1]]
            if dim == 3:
                xt[jj, 2] = target_state[dim_pos[2]]

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        M = ot.dist(xs, xt)

        M /= M.max()

        G0, log = ot.emd(a, b, M, log=True)
        # print("GO: ", G0)
        cost = log['cost']

        # thresh = 4e-1 # 2v2 case
        # thresh = 4e-2 # 2v2 case
        # thresh = 1/n
        # G0[G0>thresh] = 1
        thresh = 1/nagents # seems to work for n < m case and n=m
        G0[G0>=thresh] = 1

        # print(log)
        # exit(1)

        # TODO ok should not be simply index assignment. take the index assignment and translate
        # that to assignment of global object indices
        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds) # BUG breaks if ntargets > nagents, need robust way
        assignment = np.array([int(a) for a in assignment]) # convert to integers
        ID_assignment = self.translate(assignment, object_list, target_object_list) # translate this
                                                # assignment to assignment of global IDs
        return ID_assignment, cost

    def translate(self, assignment, object_list, target_object_list):

        """ Translates the assignment made between two distributions and returns the equivalent
        assignment between global IDs of the objects whose states comprise the two distributions
        Input:

        Output:

        """

        ID_assignment = [None]*assignment.shape[0]

        for ii, jj in enumerate(assignment):
            object_i = object_list[ii][1]
            try:
                target = target_object_list[jj][1]
            except IndexError:
                import ipdb; ipdb.set_trace()
            ID_assignment[ii] = (object_i.ID, target.ID)

        return ID_assignment

