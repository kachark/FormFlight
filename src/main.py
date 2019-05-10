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

        n = len(ref_states) +  len(target_states)

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

        # if t >= 0.45:
        #     import ipdb; ipdb.set_trace()

        M /= M.max()

        G0, log = ot.emd(a, b, M, log=True)
        print("M: ", M, "GO: ", G0)
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

    def assignment(self, t, ref_states, target_states):
        """
        ref_states and target_states are lists of tuples
        each tuple is (state, Agent).

        For the nearest neighbor EMD assignment, the information
        about the Agent is unnecessary. However, for other distances
        or other costs, this information should be extracted
        from the agents.
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
                M[ii, jj] = agent[1].pol.cost_to_go(t, agent[0], target[0])


        # if M[0,0] >= 894:
        #     import ipdb; ipdb.set_trace()

        if t >= 0.05:
            print("---- M: ", M)
        # M /= M.max() # I dont divide b
        M = M/M.max()

        a = np.ones((nagents,)) / nagents
        b = np.ones((ntargets,)) / ntargets

        G0, log = ot.emd(a, b, M, log=True)
        if t >= 0.45:
            print("---- M/Mmax: ", M)
            print("---- G0: ", G0)

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
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        # print("TIME: ", time, " CONTROL: ", np.dot(self.K, diff))
        agent_pol = np.dot(self.K, diff) + feedforward
        return agent_pol

    def cost_to_go(self, time, state1, state2):
        s1 = copy.deepcopy(state1)
        s2 = copy.deepcopy(state2)
        diff = s1 - s2
        cost_to_go = np.dot(diff, np.dot(self.P, diff))
        return cost_to_go

class LinearFeedbackOffset(LinearFeedback):

    def __init__(self, A, B, Q, R, offset):
        super(LinearFeedbackOffset, self).__init__(A, B, Q, R)
        self.offset = copy.deepcopy(offset)
        self.dim_offset = self.offset.shape[0] # np array

    def evaluate(self, time, state1):

        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        # diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
        diff[:self.dim_offset] -= self.offset
        # print("state = ", state1, diff)
        agent_pol = np.dot(self.K, diff)
        return agent_pol

    def cost_to_go(self, time, state1):
        s1 = copy.deepcopy(state1)
        diff = copy.deepcopy(state1)
        diff[:self.dim_offset] = s1[:self.dim_offset] - self.offset
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
        x = copy.deepcopy(x)
        u = copy.deepcopy(u)
        return np.dot(self.A, x) + np.dot(self.B, u)

class LTIDyn_closedloop(LTIDyn):

    def  __init__(self, A, B, K):
        super(LTIDyn_closedloop, self).__init__(A, B)
        self.A = A
        self.B = B
        self.Acl = self.A + np.dot(self.B, K)
        self.Bcl = np.dot(self.B, K)
        self.K = K

    # def rhs(self, t, x):
    def rhs(self, t, x, xt, tu):
        s1 = copy.deepcopy(x)
        s2 = copy.deepcopy(xt)
        diff = s1-s2
        # return np.dot(self.Acl, s1) - np.dot(self.Bcl, s2) # no feedforward
        return np.dot(self.Acl, diff) # error system
        # return np.dot(self.Acl, s1) - np.dot(self.Bcl, s2) + np.dot(self.B, tu) # w/ feedforward


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

        self.costs = []

        self.current_assignment = None #(tuple of assignment, cost)

    def compute_assignments(self, t, x0, collisions):

        agents = [None] * self.nagents
        ind = 0
        for ii in range(self.nagents):
            # for c in collisions: # if this agent predicted to collide, skip
            #     if ii == c[0]:
            #         continue

            agents[ii] = (x0[ind:self.size_inds[ii]], self.agents[ii])
            ind = self.size_inds[ii]

        targets = [None] * self.ntargets
        for ii in range(self.ntargets):
            # for c in collisions: # if this target predicted to collided, skip
            #     if ii == c[1]:
            #         continue

            targets[ii] = (x0[ind:self.tsize_inds[ii]], self.targets[ii])
            ind = self.tsize_inds[ii]

        assignments, cost = self.apol.assignment(t, agents, targets)
        return assignments, cost

    def update(self, t0, x0, collisions, dt):

        # print("Warning: Assumes that Each Target is Assigned To")
        # print("Dont forget to fix this (easy fix)")
        # assignment, cost = self.compute_assignments(t0, x0)
        assignment, cost = self.compute_assignments(t0, x0, collisions)

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

                # print(x[ind_start:ind_end])
                # print("u = ", u)

                # if not collisions:
                # dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, xtarget, tu) #ltidyn_cl
                dxdt[ind_start:ind_end] = agent.dyn.rhs(t, xagent, u) #ltidyn
                dxdt[tind_start:tind_end] = self.targets[jj].dyn.rhs(t, xtarget, tu)
                # else:
                    # for c in collisions:
                    #     if ii == c[0] or jj == c[1]:
                    #         break
                        

            return dxdt


        tspan = (t0, t0+dt)
        bunch = scint.solve_ivp(dyn, tspan, x0, method='BDF', rtol=1e-6, atol=1e-8)
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

    def __init__(self, dt=0.1, maxtime=10, collision_tol=0.25):
        self.dt = dt
        self.maxtime = maxtime
        self.df = None
        self.collision_tol = collision_tol

    def log(self, newdf):
        if self.df is None:
            self.df = newdf
        else:
            self.df = pd.concat([self.df, newdf.iloc[1:,:]], ignore_index=True)

    # Physics
    # 2d and 3d
    def apriori_collisions(self, current_state, nagents, ntargets, time):

        tstart = time
        tfinal = time + self.dt

        updated_state = copy.deepcopy(current_state)

        # implement a-prior (continuous) collision detection
        # use bounding circles/spheres around each particle
            # easy to calculate distances for
            # more complicated shapes - use gilbert-johnson-keerthi algorithm (GJK)

        # for now consider all agent-target pairs - can be optimized
        collided = set() # tuple(i, j)
        bounding_radius_agent = self.collision_tol
        bounding_radius_target = self.collision_tol
        for i in range(nagents):
            y_agent = updated_state[i*4:(i+1)*4] # time history of agent i
            y_agent_final = y_agent[:2] + np.array([y_agent[2], y_agent[3]])*self.dt
            # print(y_agent)

            # check each agent against each target
            for j in range(ntargets):
                y_target = updated_state[(j+ntargets)*4:(j+ntargets+1)*4]
                y_target_final = y_target[:2] + np.array([y_target[2], y_target[3]])*self.dt

                # agent/target current and future positions
                a0 = y_agent[:2]
                af = y_agent_final[:2]
                t0 = y_target[:2]
                tf = y_target_final[:2]
                del_a = af - a0
                del_t = tf - t0

                # ax = del_a[0] - del_t[0]
                # ay = del_a[1] - del_t[1]
                # bx = a0[0] - t0[0]
                # by = a0[1] - t0[1]

                # a = ax**2 + ay**2
                # b = 2 * (ax*bx + ay*by)
                # c = (bx**2 + by**2) - (bounding_radius_agent+bounding_radius_target)**2


                a = np.linalg.norm(del_t-del_a)**2
                b = 2*np.dot((t0-a0), (del_t-del_a))
                c = np.linalg.norm(t0-a0)**2 - (bounding_radius_target+bounding_radius_agent)**2

                coeff = [a, b, c]

                t_sol = np.roots(coeff)
                t_collisions = t_sol[np.isreal(t_sol)] # get real valued times
                # print(t_collisions)
                # print(t_collision[np.isreal(t_collision)])
                for t in t_collisions[np.isreal(t_collisions)]:
                    if 0 < t < 1:
                        # print("COLLISION DETECTED ", "(", i, ", ", j, ") ", t)
                        # print("       ", a0, " t0: ", t0)
                        collided.add((i,j))

                # if t_collisions.size != 0:
                #     if 0 <= np.amin(t_collisions[np.isreal(t_collisions)]) <= 1:
                #         collided.append((i,j))

        print(collided)
        return collided

    def run(self, x0, system):

        current_state = copy.deepcopy(x0)
        running = True
        time = 0
        while running:
            # print("Time: {0:3.2E}".format(time))
            collisions = self.apriori_collisions(current_state, system.nagents, system.ntargets, time)

            # thist, state_hist, assign_hist = system.update(time, current_state, self.dt)
            thist, state_hist, assign_hist = system.update(time, current_state, collisions, self.dt)


            newdf = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                                            state_hist,
                                            assign_hist)))

            self.log(newdf)

            time = time + self.dt
            if time > self.maxtime:
                running = False

            current_state = state_hist[-1, :]

# def run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltargets, apol, nagents, ntargets):
def run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets):

    # nagents = 2
    # ntargets = 2
    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    # targets = [Agent(dx, ltidyn, poltarget) for ii, poltarget in enumerate(poltargets)]
    targets = [Agent(dx, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=0.01, maxtime=4, collision_tol=1e-3)
    eng.run(x0, sys)
    # print(eng.df.tail(10))

    # TEST
    # filtered_yout, collisions, switches = post_process_identical_2d_doubleint(eng.df, poltrack, Q, R, nagents)
    # post_process_identical_2d_doubleint(eng.df, poltrack, poltargets, Q, R, nagents, ntargets, sys.costs)
    post_process_identical_3d_doubleint(eng.df, poltrack, poltargets, Q, R, nagents, ntargets, sys.costs)

    # return filtered_yout, collisions, switches
    return eng.df.iloc[:, 1:].to_numpy()


if __name__ == "__main__": 

    dim = 3

    if dim == 3:

        # 3D CASES
        dx = 6
        du = 3
        A = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

        B = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])


        Q = np.eye(dx)
        R = np.eye(du)

        # Agent dynamics
        ltidyn = LTIDyn(A, B)
        dyn_target = LTIDyn(A, B)
        # Agent control law
        poltrack = LinearFeedbackTracking(A, B, Q, R) # same controls shared by A's
        # poltarget = LinearFeedback(A, B, Q, R)

        # Target control law
        ntargets = 2
        cities = [np.array([10, 0, 0]), np.array([5, -5, 0])]
        poltarget = [LinearFeedbackOffset(A, B, Q, R, c) for c in cities] # list of T controls

        # 2 v 2 EMD
        # apol = AssignmentEMD(2, 2)
        apol = AssignmentDyn(2, 2)

        x0 = np.array([-10, -9, -6, -8, -6, -8,
                       6, -2, -10, 7, -10, 7,
                       7, 1, 1, -8, 1, -8,
                       -2, -1, 0, 8, 0, 8])

        yout_dyn = run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltarget, apol, 2, 2)

    if dim == 2:

        # 2D CASES
        dx = 4
        du = 2
        A = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=np.float_)

        B = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 1.0]], dtype=np.float_)


        Q = np.eye(dx)
        R = np.eye(du)

        # Agent dynamics
        ltidyn = LTIDyn(A, B)
        dyn_target = LTIDyn(A, B)
        # Agent control law
        poltrack = LinearFeedbackTracking(A, B, Q, R)
        # poltarget = LinearFeedback(A, B, Q, R)

        # NEW
        # Agent Closed-Loop Dynamics
        # ltidyn_cl = LTIDyn_closedloop(A, B, poltrack.K)
        # ltidyn_cl = LTIDyn(A, B)


        # Target control law

        ntargets = 2
        cities = [np.array([10, 0]), np.array([5, -5])]


        # ntargets = 3
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
        # cities = [np.array([-10,0], np.float_), np.array([10,0], dtype=np.float_), np.array([5,-5], dtype=np.float_)] # ntargets = 4
        # ntargets = 4
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
        # ntargets = 8
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]

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
        x0 = np.array([-10, -9, -6, -8, 6, -2, -10, 7,
                       7, 1, 1, -8, -2, -1, 0, 8])

        # 3 v 3 EMD
        # apol = AssignmentEMD(3, 3)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5])
        # x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
        #                -110, -15, 15, 15])
        # x0 = np.hstack((x0, x02))

        # x0 = np.array([-9.49472109, -9.01609684, -6.30746324, -8.61933317, -4.85049153,  8.27163463,
        #     -0.84300976, -7.39576421,  6.19783331, -1.93060319, -9.5113471,   7.13662085,
        #     -4.51410362,  4.18211928, -2.88455314,  5.88618124,  6.89237722,  0.76295034,
        #      1.18173033, -7.54980037, -2.44716163, -1.42505342,  0.22417293,  7.8352514 ], dtype=np.float_)

        # 4 v 4 EMD
        # apol = AssignmentEMD(4, 4)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5, 20, 10, -3, -8])
        # x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
        #                -110, -15, 15, 15, 120, 110, -13, -18])
        # x0 = np.hstack((x0, x02))

        # 8 v 8 EMD
        # apol = AssignmentEMD(8, 8)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5, 20, 10, -3, -8,
        #                25, 5, 10, -155, 70, 230, -13, 13,
        #                -92, -12, 33, 66, 123, 110, -13, -18])
        # x02 = np.array([15, 15, 11, -11, 110, 110, -13, 13,
        #                 -110, -15, 15, 15, 120, 130, -13, -18,
        #                 25, 35, 18, -17, 150, 100, -13, 13,
        #                -140, -45, 15, 15, 220, 140, -13, -18])
        # x0 = np.hstack((x0, x02))



        # yout = run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltarget, apol, 2, 2)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 4, 4)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 8, 8)


        # 2 v 2 Dyn
        apol = AssignmentDyn(2, 2)
        yout_dyn = run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltarget, apol, 2, 2)

        # 3 v 3 Dyn
        # apol = AssignmentDyn(3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)

        # 4 v 4 Dyn
        # apol = AssignmentDyn(4, 4)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 4, 4)

        # 8 v 8 Dyn
        # apol = AssignmentDyn(8, 8)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 8, 8)


        
        # superimposed trajectories from emd and dyn simulations
        # plt.figure()
        # for zz in range(2):
        #     y_agent = yout[:, zz*4:(zz+1)*4]
        #     plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
        #     plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

        # for zz in range(2):
        #     y_agent = yout_dyn[:, zz*4:(zz+1)*4]
        #     plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
        #     plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')


    plt.show()

    print("done!")
