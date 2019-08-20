
""" @file system.py
"""

from decimal import Decimal
from time import time, process_time
import scipy.integrate as scint
import numpy as np

from DOT_assignment import assignments

################################
## Big Systems
################################
class System:

    """ System parent class
    """

    def __init__(self, agents):

        """ System constructor
        """

        self.agents = agents
        self.nagents = len(agents)
        self.sizes = np.array([a.state_size() for a in self.agents])
        self.size_inds = np.cumsum(self.sizes)

class OneVOne(System):

    """ System representing scenario consisting of one agent to one target engagements

    Assumes equal agent and target swarm sizes

    """

    def __init__(self, agents, targets, pol, assignment_epoch):

        """ Constructor for OneVOne System
        """

        super(OneVOne, self).__init__(agents)
        self.targets = targets
        self.tsizes = np.array([a.state_size() for a in self.agents])
        self.tsize_inds = self.size_inds[-1] + np.cumsum(self.sizes)
        self.ntargets = len(self.targets)
        self.apol = pol

        self.assignment_epoch = assignment_epoch
        self.optimal_assignment = None

        self.costs = []

        self.current_assignment = None #(tuple of assignment, cost)

    def compute_assignments(self, t, x0, collisions):

        """ Compute assignments between agent and target swarms

        Does not perfrom assignments for agents that are collided with targets

        """

        agents = [None] * self.nagents
        ind = 0
        for ii in range(self.nagents):
            for c in collisions: # if this agent predicted to collide, skip assignment
                if ii == c[0]:
                    continue

            agents[ii] = (x0[ind:self.size_inds[ii]], self.agents[ii])
            ind = self.size_inds[ii]

        targets = [None] * self.ntargets
        for ii in range(self.ntargets):
            for c in collisions: # if this target predicted to collide, don't use in assignment
                if ii == c[1]:
                    continue

            targets[ii] = (x0[ind:self.tsize_inds[ii]], self.targets[ii])
            ind = self.tsize_inds[ii]

        # perform assignment
        assignments, cost = self.apol.assignment(t, agents, targets)

        return assignments, cost

    def compute_optimal_assignments(self, t, x0, collisions):

        """ Compute optimal assignments (DYN) between agent and target swarms

        Does not perfrom assignments for agents that are collided with targets

        """

        # Get dyn assignment at initial conditions (optimal asst that we use to compare against)
        if t == 0:
            agents = [None] * self.nagents
            ind = 0
            for ii in range(self.nagents):
                for c in collisions: # if this agent predicted to collide, skip assignment
                    if ii == c[0]:
                        continue

                agents[ii] = (x0[ind:self.size_inds[ii]], self.agents[ii])
                ind = self.size_inds[ii]

            targets = [None] * self.ntargets
            for ii in range(self.ntargets):
                for c in collisions: # if this target predicted to collided, don't use in assignment
                    if ii == c[1]:
                        continue

                targets[ii] = (x0[ind:self.tsize_inds[ii]], self.targets[ii])
                ind = self.tsize_inds[ii]

            opt_asst_pol = assignments.AssignmentDyn(self.nagents, self.ntargets)
            self.optimal_assignment, _ = opt_asst_pol.assignment(t, agents, targets)
        else:
            return

    def pre_process(self, t0, x0, collisions):

        """ System pre-processor

        Perform functions prior to starting engine loop
        """

        # compute optimal assignment
        # useful to have optimal assignment to compare against in case running EMD simulation alone
        self.compute_optimal_assignments(t0, x0, collisions)

    def update(self, t0, x0, collisions, dt, tick):

        """ Computes assignments at assignment epoch and advances dynamics per engine tick

        Input:
        - t0:           start time of integration
        - x0:           agent, target, target terminal states at start time of integration
        - dt:           engine time step size

        Output:
        return tout, yout, assign_out, diagnostics
        - tout:         time integrated over between [t0, t0+dt]
        - yout:         agent states, target states between [t0, t0+dt]
        - assign_out:   index assignments between agent_i and target_j
        - diagnostics:  diagnostics recorded between [t0, t0+dt]

        """

        # print("Warning: Assumes that Each Target is Assigned To")
        # print("Dont forget to fix this (easy fix)")

        # measure assignment execution time
        start_assign_time = process_time()

        if t0 == 0:
            assignment, cost = self.compute_assignments(t0, x0, collisions)
            self.current_assignment = assignment
        if t0 > 0 and self.apol.__class__.__name__ != 'AssignmentDyn':
            if tick % self.assignment_epoch == 0:
                print("------> ASSIGNMENT AT: ", t0)
                assignment, cost = self.compute_assignments(t0, x0, collisions)
            else:
                assignment = self.current_assignment
        else:
            assignment = self.current_assignment

        # after assignment done
        # pre-compute tracking control policy
        if t0 == 0:
            for ii, agent in enumerate(self.agents):
                jj = assignment[ii]
                agent.pol.track(t0, jj, self.targets[jj].pol.get_closed_loop_A(), self.targets[jj].pol.get_closed_loop_g())
        if t0 > 0 and self.apol.__class__.__name__ != 'AssignmentDyn':
            for ii, agent in enumerate(self.agents):
                jj = assignment[ii]
                agent.pol.track(t0, jj, self.targets[jj].pol.get_closed_loop_A(), self.targets[jj].pol.get_closed_loop_g())

        # measure assignment execution time
        elapsed_assign_time = process_time() - start_assign_time

        # record cost-to-go (# TODO is this true anymore?)
        # self.costs.append(cost)

        # if cost is not None:
        #     print("TIME: ", t0, "ASST TYPE: ", self.apol.__class__.__name__)
            # print("TIME: ", t0, "COST: ", cost, "ASST: ", assignment)

        print("TIME: ", t0, "ASST TYPE: ", self.apol.__class__.__name__)

        def dyn(t, x):

            dxdt = np.zeros(x.shape)
            for ii, agent in enumerate(self.agents):
                jj = assignment[ii]

                # Target Indices
                tind_end = self.tsize_inds[jj]
                tind_start = self.tsize_inds[jj] - self.tsizes[jj]
                xtarget = x[tind_start:tind_end]

                # Target Control
                tu = self.targets[jj].pol.evaluate(t, xtarget)

                # Agent Indices
                ind_end = self.size_inds[ii]
                ind_start = self.size_inds[ii] - self.sizes[ii]
                xagent = x[ind_start:ind_end]

                # Agent Control
                # print(agent.pol)
                u = agent.pol.evaluate(t, xagent, xtarget, feedforward=tu)

                if not bool(collisions):
                    dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, u)
                    dxdt[tind_start:tind_end] = self.targets[jj].dyn.rhs(t, xtarget, tu)
                else: # don't propogate dynamics
                    for c in collisions: # collisions = set of tuples
                        if ii == c[0] or jj == c[1]:
                            # break # or continue?
                            continue

            return dxdt


        tspan = (t0, t0+dt)

        # measure dynamics execution time
        start_dynamics_time = process_time()

        bunch = scint.solve_ivp(dyn, tspan, x0, method='BDF', rtol=1e-6, atol=1e-8)

        # measure dynamics execution time
        elapsed_dynamics_time = process_time() - start_dynamics_time

        tout = bunch.t
        yout = bunch.y.T
        assign_out = np.tile(assignment, (tout.shape[0], 1))

        # **** system diagnostics
        assign_comp_cost = np.tile(elapsed_assign_time, (tout.shape[0], 1))
        assign_comp_cost[1:tout.shape[0]] = 0

        dynamics_comp_cost = np.tile(elapsed_dynamics_time, (tout.shape[0], 1))
        dynamics_comp_cost[1:tout.shape[0]] = 0

        diagnostics = [assign_comp_cost, dynamics_comp_cost]
        # **** system diagnostics

        return tout, yout, assign_out, diagnostics

        # print(tout, yout)
        # exit(1)

