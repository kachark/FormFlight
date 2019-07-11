
import scipy.integrate as scint
import numpy as np

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

        # Get dyn assignment at initial conditions (optimal asst that we use to compare against)
        if t == 0:
            opt_asst_pol = AssignmentDyn(self.nagents, self.ntargets)
            self.optimal_assignment, _ = opt_asst_pol.assignment(t, agents, targets)

        return assignments, cost

    def update(self, t0, x0, collisions, dt):

        # print("Warning: Assumes that Each Target is Assigned To")
        # print("Dont forget to fix this (easy fix)")
        assignment, cost = self.compute_assignments(t0, x0, collisions)

        # after assignment done
        # pre-compute tracking control policy
        for ii, agent in enumerate(self.agents):
            jj = assignment[ii]
            agent.pol.track(t0, jj, self.targets[jj].pol.get_closed_loop_A(), self.targets[jj].pol.get_closed_loop_g())

        # record cost-to-go
        self.costs.append(cost)

        if cost is not None:
            # print(t0, cost, assignment)
            print("TIME: ", t0, "COST: ", cost, "ASST: ", assignment)

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
        bunch = scint.solve_ivp(dyn, tspan, x0, method='BDF', rtol=1e-6, atol=1e-8)
        tout = bunch.t
        yout = bunch.y.T
        assign_out = np.tile(assignment, (tout.shape[0], 1))
        return tout, yout, assign_out

        # print(tout, yout)
        # exit(1)

