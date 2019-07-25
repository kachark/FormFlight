
from decimal import Decimal
import scipy.integrate as scint
import numpy as np
from time import time, process_time

from dynamics import *
from assignments import *

################################
## Big Systems
################################
class System:

    def __init__(self, agents):

        self.agents = agents
        self.nagents = len(agents)
        self.sizes = np.array([a.state_size() for a in self.agents])
        self.size_inds = np.cumsum(self.sizes)

class OneVOne(System):

    def __init__(self, agents, targets, pol):
        super(OneVOne, self).__init__(agents)
        self.targets = targets
        self.tsizes = np.array([a.state_size() for a in self.agents])
        self.tsize_inds = self.size_inds[-1] + np.cumsum(self.sizes)
        self.ntargets = len(self.targets)
        self.apol = pol

        # every 10 ticks, perform assignment
        self.assignment_epoch = 10
        self.optimal_assignment = None

        self.costs = []

        self.current_assignment = None #(tuple of assignment, cost)

    def compute_assignments(self, t, x0, collisions):

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

        # perform assignment
        assignments, cost = self.apol.assignment(t, agents, targets)

        # TODO at t=0 we don't want to have 2 assignments skewing diagnostics
        # # TEST
        # # Get dyn assignment at initial conditions (optimal asst that we use to compare against)
        # if t == 0:
        #     opt_asst_pol = AssignmentDyn(self.nagents, self.ntargets)
        #     self.optimal_assignment, _ = opt_asst_pol.assignment(t, agents, targets)

        return assignments, cost

    # TODO at t=0 we don't want to have 2 assignments skewing diagnostics
    # TEST
    def compute_optimal_assignments(self, t, x0, collisions):

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

            opt_asst_pol = AssignmentDyn(self.nagents, self.ntargets)
            self.optimal_assignment, _ = opt_asst_pol.assignment(t, agents, targets)
        else:
            return

    # TODO at t=0 we don't want to have 2 assignments skewing diagnostics
    def pre_process(self, t0, x0, collisions):
        # compute optimal assignment
        self.compute_optimal_assignments(t0, x0, collisions)

    def update(self, t0, x0, collisions, dt, tick):

        # print("Warning: Assumes that Each Target is Assigned To")
        # print("Dont forget to fix this (easy fix)")

        # TODO measure assignment execution time
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
            assignment = self.optimal_assignment

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

        # TODO measure assignment execution time
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

                # if not collisions:
                # dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, xd_c, tu) #ltidyn_cl

                # NON-AUGMENTED DYNAMICS SOLVED HERE
                # dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, u)
                # dxdt[tind_start:tind_end] = self.targets[jj].dyn.rhs(t, xtarget, tu)

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

        # TODO measure dynamics execution time
        start_dynamics_time = process_time()

        bunch = scint.solve_ivp(dyn, tspan, x0, method='BDF', rtol=1e-6, atol=1e-8)

        # TODO measure dynamics execution time
        elapsed_dynamics_time = process_time() - start_dynamics_time

        tout = bunch.t
        yout = bunch.y.T
        assign_out = np.tile(assignment, (tout.shape[0], 1))
        # return tout, yout, assign_out

        # TODO measure computation cost of assignment
        # TODO should not repeat costs that aren't happening
        assign_comp_cost = np.tile(elapsed_assign_time, (tout.shape[0], 1))
        assign_comp_cost[1:tout.shape[0]] = 0

        dynamics_comp_cost = np.tile(elapsed_dynamics_time, (tout.shape[0], 1))
        dynamics_comp_cost[1:tout.shape[0]] = 0

        diagnostics = [assign_comp_cost, dynamics_comp_cost]

        return tout, yout, assign_out, diagnostics

        # print(tout, yout)
        # exit(1)

