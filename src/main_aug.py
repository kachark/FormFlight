import numpy as np
import matplotlib.pyplot as plt
import functools
import scipy.integrate as scint
from scipy.linalg import solve_continuous_are as care
import pandas as pd
import copy

try:
    import ot
except ModuleNotFoundError:
    print("OT not found")

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

class AssignmentFixed(Assignment):

    def __init__(self, nref, ntarget, assignment):
        super(AssignmentFixed, self).__init__(nref, ntarget)
        self.assign = assignment
    
    def assignment(self, t, ref_states, target_states):
        return self.assign, None

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
            xt[ii, :] = target[0][:dim_pos]

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets
        
        M = ot.dist(xs, xt)
        M /= M.max()
        
        G0, log = ot.emd(a, b, M, log=True)
        cost = log['cost']
        
        thresh = 1/nagents -1e-10
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

        # print("In AssignmentDyn")
        # print(dim_state)
        # print(nagents)
        # print(ntargets)
        
        M = np.zeros((nagents, ntargets))
        for ii, agent in enumerate(ref_states):
            for jj, target in enumerate(target_states):
                # print(agent[0])
                # print(target[0])
                M[ii, jj] = agent[1].pol.cost_to_go(t, agent[0], target[0])

        M /= M.max() # I dont divide b
        # print(M)

        # print("\n")
        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        G0, log = ot.emd(a, b, M, log=True)
        # print(G0)
        # print(log)
        
        cost = log['cost']
        thresh = 1/nagents - 1e-10
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


class LinearFeedbackConstTracker:
    def __init__(self, A, B, Q, R, const, g=None):

        print("\n\n\n\n\n")
        print("A = ")
        print(A)
        print("B = ")
        print(B)
        print("Q = ")
        print(Q)
        print("R = ")
        print(R)        
        
        self.P = care(A, B, Q, R)
        self.K = np.linalg.solve(R, np.dot(B.T, self.P))
        self.Bt = copy.deepcopy(B.T)
        self.RBt = np.dot(np.linalg.inv(R), self.Bt)        
        self.BRBt = np.dot(B, self.RBt)
        if g is None:
            self.g = np.dot(A, const)
        else:
            self.g = copy.deepcopy(g)
        print("g = ", self.g)
            
        self.p = -np.linalg.solve(A.T - np.dot(self.P, self.BRBt), np.dot(self.P, self.g))
        self.R = copy.deepcopy(R)
        self.const = copy.deepcopy(const)
        
        # Closed loop is
        # \dot{x} = A_cl x + g_cl
        self.Acl = A - np.dot(B, self.K)

        self.g_cl = np.dot(B, np.dot(self.K, const)) - np.dot(B, np.dot(np.linalg.inv(R), np.dot(B.T, self.p)))

    def evaluate(self, time, state):
        # print("TIME: ", time, " STATE: ", state.T)

        return -np.dot(self.RBt, np.dot(self.P, state - self.const) + self.p)

    def get_closed_loop_A(self):
        return self.Acl

    def get_closed_loop_g(self):
        return self.g_cl
    
    def get_P(self):
        return self.P
    
class LinearFeedbackAugmented(LinearFeedbackConstTracker):

    def __init__(self, A, B, Q, R, Acl, g):
        
        nstates = A.shape[0] + Acl.shape[0]
        Aconcat = np.zeros((nstates, nstates))
        Aconcat[:A.shape[0],:A.shape[0]] = copy.deepcopy(A)
        Aconcat[A.shape[0]:, A.shape[0]:] = copy.deepcopy(Acl)

        ncontrol = B.shape[1]
        Bconcat = np.zeros((nstates, ncontrol))
        Bconcat[:A.shape[0], :] = copy.deepcopy(B)

        Qconcat = np.zeros((nstates, nstates))
        Qconcat[:A.shape[0], :A.shape[0]] = copy.deepcopy(Q)
        Qconcat[A.shape[0]:, A.shape[0]:] = copy.deepcopy(Q)
        Qconcat[:A.shape[0], A.shape[0]:] = -copy.deepcopy(Q)
        Qconcat[A.shape[0]:, :A.shape[0]] = -copy.deepcopy(Q)        

        const = np.zeros((nstates))
        
        gconcat = np.zeros((nstates))
        gconcat[A.shape[0]:] = copy.deepcopy(g)
        
        super(LinearFeedbackAugmented, self).__init__(Aconcat, Bconcat, Qconcat, R, const, g=gconcat)

    def evaluate(self, time, state1, state2, feedforward=0):
        # print("state = ", state)

        aug_state = np.hstack((copy.deepcopy(state1), copy.deepcopy(state2)))
        control = super(LinearFeedbackAugmented, self).evaluate(time, aug_state)
        return control

    # def cost_to_go(self, time, state1, state2):
    #     diff = state1 - state2
    #     cost_to_go = np.dot(diff, np.dot(self.P, diff))
    #     return cost_to_go    

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0]

    def evaluate(self, time, state1):

        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = diff[:self.dim_offset] - self.offset
        agent_pol = np.dot(self.K, diff)
        return agent_pol

    def cost_to_go(self, time, state1):
        diff = state1
        diff[:self.dim_offset] = diff[:self.dim_offset] - self.offset        
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go    

# def gen_augmented_dynamics()
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

        # print(agents)
        # print(targets)
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
            
        # print("\n\n\n")
        tspan = (t0, t0+dt)
        t_eval = np.linspace(t0, t0+dt, 10)
        bunch = scint.solve_ivp(dyn, tspan, x0, t_eval=t_eval, method='BDF')# , rtol=1e-8, atol=1e-8)
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
            self.df = copy.deepcopy(newdf)
        else:
            self.df = pd.concat([self.df, newdf.iloc[1:,:]], ignore_index=True)

    def run(self, x0, system):

        current_state = copy.deepcopy(x0)
        
        # self.df = pd.DataFrame(x0)
        # print(self.df)
        # exit(1)
        running = True
        time = 0
        while running:
            # print("Time: {0:3.2E}".format(time))
            # print(current_state)
            thist, state_hist, assign_hist = system.update(time, current_state, self.dt)

            newdf = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                                            state_hist,
                                            assign_hist)))
            
            self.log(newdf)
            # print("newdf = ")
            # print(newdf)
            time = time + self.dt
            if time > self.maxtime:
                running = False

            current_state = state_hist[-1, :]

def run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltargets,
                            apol, nagents, ntargets, dt=0.01, maxtime=10):

    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, ltidyn, poltarget) for ii, poltarget in enumerate(poltargets)]
    
    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=dt, maxtime=maxtime)
    # eng = Engine(dt=0.001, maxtime=maxtime)    
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

    Q2 = copy.deepcopy(Q)
    Q2[2,2] = 0.0
    Q2[3,3] = 0.0

    Q3 = copy.deepcopy(Q)
    Q3[0, 0] = 100
    Q3[1, 1] = 100
    Q3[2,2] = 0.0
    Q3[3,3] = 0.0
    
    ntargets = 3
    city = np.array([-10, 0, 0, 0])
    poltarget = LinearFeedbackConstTracker(A, B, Q2, R, city)
    
    ltidyn = LTIDyn(A, B)
    Acl = poltarget.get_closed_loop_A()
    gcl = poltarget.get_closed_loop_g()
    print("gcl = ", gcl)
    poltrack = LinearFeedbackAugmented(A, B, Q3, R, Acl, gcl) # new_controller
    # poltrack = LinearFeedbackTracking(A, B, Q3, R) # old controller
    
    # poltarget = [ZeroPol(du) for ii in range(ntargets)]
    # poltarget = ZeroPol(du)

    #1v1
    x0 = np.array([5, 5, 0, 0,
                   20, -10, 5, 5])
    apol = AssignmentLexical(1, 1)
    run_identical_doubleint(dx, du, x0, ltidyn, poltrack, [poltarget], apol, 1, 1, maxtime=15)

    plt.show()
