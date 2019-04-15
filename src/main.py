import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.integrate as scint
from scipy.linalg import solve_continuous_are as care
import pandas as pd
import copy

import ot

from post_process import *


################################
## Assignments
###############################
class Assignment:

    def __init__(self, nref, ntarget):
        self.nref = nref
        self.ntarget = ntarget

    def assignment(self, t, ref_states, target_states):
        pass
    
class AssignmentLexical(Assignment):

    def assignment(self, t, ref_states, target_states):
        inds_out = np.array(range(len(target_states)))
        return inds_out, None

class AssignmentEMD(Assignment):

    def assignment(self, t, ref_states, target_states):
        """
        ref_states and target_states are lists of tuples
        each tuple is (state, Agent). 

        For the nearest neighbor EMD assignment, the information 
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.
        """
        
        ## Assume first two states are the positions
        nagents = len(ref_states)
        ntargets = len(target_states)
        
        dim_state = ref_states[0][0].shape[0]
        dim_pos = int(dim_state / 2)

        xs = np.zeros((nagents, dim_pos))
        xt = np.zeros((ntargets, dim_pos))
        
        for ii, state in enumerate(ref_states):
            # print(ii, state[0])
            xs[ii, :] = state[0][:dim_pos]

        for jj, target in enumerate(target_states):
            xt[jj, :] = target[0][:dim_pos]

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets
        
        M = ot.dist(xs, xt)
        M /= M.max()
        
        G0, log = ot.emd(a, b, M, log=True)
        cost = log['cost']

        # thresh = 4e-1 # 2v2 case
        thresh = 4e-2 # 2v2 case
        G0[G0>thresh] = 1

        # print(log)
        # exit(1)
        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

class AssignmentDyn(Assignment):

    def assignment(self, t, ref_states, target_states):
        """
        ref_states and target_states are lists of tuples
        each tuple is (state, Agent). 

        For the nearest neighbor EMD assignment, the information 
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.
        """
        
        ## Assume first two states are the positions
        nagents = len(ref_states)
        ntargets = len(target_states)
        
        dim_state = ref_states[0][0].shape[0]

        M = np.zeros((nagents, ntargets))
        for ii, agent in enumerate(ref_states):
            for jj, target in enumerate(target_states):
                M[ii, jj] = agent[1].pol.cost_to_go(t, agent[0], target[0])

        # print(M)
        M /= M.max() # I dont divide b
        
        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        G0, log = ot.emd(a, b, M, log=True)
        # print(G0)
        cost = log['cost']        
        # thresh = 4e-1 # 2v2
        thresh = 4e-2
        G0[G0>thresh] = 1
        
        inds = np.arange(ntargets)
        assignment = np.dot(G0, inds)
        assignment = np.array([int(a) for a in assignment])
        return assignment, cost

################################
## Agent Policies
###############################
class ZeroPol:

    def __init__(self, du):
        self.du = du
        self.u = np.zeros((du))

    def evaluate(self, time, state):
        return self.u
    
class LinearFeedback:

    def __init__(self, A, B, Q, R):
        self.P = care(A, B, Q, R)
        self.K = -np.linalg.solve(R, np.dot(B.T, self.P))

    def evaluate(self, time, state):
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, state))
        return np.dot(self.K, state)

    def get_P(self):
        return self.P    

class LinearFeedbackTracking(LinearFeedback):

    def __init__(self, A, B, Q, R):
        super(LinearFeedbackTracking, self).__init__(A, B, Q, R)

    def evaluate(self, time, state1, state2, feedforward=0):
        # print("state = ", state)
        diff = state1 - state2
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, diff))
        agent_pol = np.dot(self.K, diff) + feedforward
        return agent_pol

    def cost_to_go(self, time, state1, state2):
        diff = state1 - state2
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0] # np array

    def evaluate(self, time, state1):

        # diff = state1
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = diff[:self.dim_offset] - self.offset
        # print("state = ", state1, diff)
        agent_pol = np.dot(self.K, diff)
        return agent_pol

    def cost_to_go(self, time, state1):
        # diff = state1
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = diff[:self.dim_offset] - self.offset
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go    

################################
## Agent Dynamics
###############################    
class LTIDyn:

    def __init__(self, A, B):
        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)

    def rhs(self, t, x, u):
        return np.dot(self.A, x) + np.dot(self.B, u)


################################
## Agents
###############################        
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

    def  __init__(self, dx, pol, dyn):
        super(TrackingAgent, self).__init__(dx, pol, dyn)

    def rhs(self, t, x, ref_signal):
        u = self.pol(t, x, ref_signal)
        return self.dyn.rhs(t, x, u)


################################
## Big Systems
############################### 
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
        
        self.current_assignment = None #(tuple of assignment, cost)

    def compute_assignments(self, t, x0):

        agents = [None] * self.nagents
        ind = 0
        for ii in range(self.nagents):
            agents[ii] = (x0[ind:self.size_inds[ii]], self.agents[ii])
            ind = self.size_inds[ii]

        targets = [None] * self.ntargets
        for ii in range(self.ntargets):
            targets[ii] = (x0[ind:self.tsize_inds[ii]], self.targets[ii])
            ind = self.tsize_inds[ii]

        assignments, cost = self.apol.assignment(t, agents, targets)
        return assignments, cost
        
    def update(self, t0, x0, dt):

        # print("Warning: Assumes that Each Target is Assigned To")
        # print("Dont forget to fix this (easy fix)")
        assignment, cost = self.compute_assignments(t0, x0)

        if cost is not None:
            print(t0, cost, assignment)
        
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

                # print(x[ind_start:ind_end])
                # print("u = ", u)
                dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, u)
                dxdt[tind_start:tind_end] = self.targets[jj].dyn.rhs(t, xtarget, tu)
                
            return dxdt
            

        tspan = (t0, t0+dt)
        bunch = scint.solve_ivp(dyn, tspan, x0, method='BDF')
        tout = bunch.t
        yout = bunch.y.T
        assign_out = np.tile(assignment, (tout.shape[0], 1))
        return tout, yout, assign_out

        # print(tout, yout)
        # exit(1)

################################
## Game Engine
###############################         
class Engine:

    def __init__(self, dt=0.1, maxtime=10):
        self.dt = dt
        self.maxtime = maxtime
        self.df = None

    def log(self, newdf):
        if self.df is None:
            self.df = newdf
        else:
            self.df = pd.concat([self.df, newdf.iloc[1:,:]], ignore_index=True)
            
    def run(self, x0, system):

        current_state = copy.deepcopy(x0)
        running = True
        time = 0
        while running:
            # print("Time: {0:3.2E}".format(time))
            thist, state_hist, assign_hist = system.update(time, current_state, self.dt)

            newdf = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                                            state_hist,
                                            assign_hist)))
            
            self.log(newdf)

            time = time + self.dt
            if time > self.maxtime:
                running = False

            current_state = state_hist[-1, :]

def run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltargets, apol, nagents, ntargets):

    # nagents = 2
    # ntargets = 2
    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, ltidyn, poltarget) for ii, poltarget in enumerate(poltargets)]
    
    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=0.1, maxtime=10)
    eng.run(x0, sys)
    # print(eng.df.tail(10))
    
    post_process_identical_2d_doubleint(eng.df, poltrack, Q, R, nagents)    

            
if __name__ == "__main__":

    dx = 4
    du = 2
    A = np.array([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])    


    Q = np.eye(dx)
    R = np.eye(du)

    ltidyn = LTIDyn(A, B)
    poltrack = LinearFeedbackTracking(A, B, Q, R)
    # poltarget = LinearFeedback(A, B, Q, R)


    # ntargets = 2
    # cities = [np.array([-10, 0]), np.array([10, 0])]
    # cities = [np.array([-10, 0]), np.array([10, 0]), np.array([0, -10]), np.array([0, 10])]
    # ntargets = 3
    # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
    ntargets = 4
    cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
    poltarget = [LinearFeedbackOffset(A, B, Q, R, c) for c in cities]
    # poltarget = [LinearFeedbackTracking(A, B, Q, R) for c in cities]

    # poltarget = [LinearFeedback(A, B, Q, R) for ii in range(ntargets)]

    # poltarget = [ZeroPol(du) for ii in range(ntargets)]
    # poltarget = ZeroPol(du)

    # 1v1
    # x0 = np.array([5, 5, 0, 0,
    #                -10, -5, 5, 5])
    # apol = AssignmentLexical(1, 1)
    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, [poltarget[0]], apol, 1, 1)


    # 2 v 2 Lexical 
    # apol = AssignmentLexical(2, 2)    
    # x0 = np.array([5, 5, 0, 0, 10, 10, -3, 8,
    #                -10, -5, 5, 5, 20, 10, -3, -8])

    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 2, 2)

    # 2 v 2 EMD
    # apol = AssignmentEMD(2, 2)
    # x0 = np.random.rand(16) * 20 - 10
    # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
    #                -10, -5, 5, 5, 20, 10, -3, -8])

    # 3 v 3 EMD
    # apol = AssignmentEMD(3, 3)
    # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
    #                -10, -5, 5, 5])
    # x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
    #                -110, -15, 15, 15])
    # x0 = np.hstack((x0, x02))

    # 4 v 4 EMD
    apol = AssignmentEMD(4, 4)
    x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
                   -10, -5, 5, 5, 20, 10, -3, -8])
    x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
                   -110, -15, 15, 15, 120, 110, -13, -18])
    x0 = np.hstack((x0, x02))



    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 2, 2)
    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)
    run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 4, 4)


    # 2 v 2 Dyn
    # apol = AssignmentDyn(2, 2)
    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 2, 2)

    # 3 v 3 Dyn
    # apol = AssignmentDyn(3, 3)
    # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)

    # 4 v 4 Dyn
    apol = AssignmentDyn(4, 4)
    run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 4, 4)


    plt.show()
