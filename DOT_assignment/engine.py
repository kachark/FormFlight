
""" @file engine.py
"""

import copy
import pandas as pd
import numpy as np

###############################
## Game Engine
###############################
class Engine:

    def __init__(self, dim, dt=0.1, maxtime=10, collisions=False, collision_tol=0.25):

        """ Engine constructor

        Input:
        - dim:          simulation dimension (2D/3D)
        - dt:           engine tick size
        - maxtime:      simulation time
        - collisions:   collisions on/off
        - collision_tol:collitions tolerance

        Output:

        """

        self.dim = dim
        self.dt = dt
        self.maxtime = maxtime
        self.df = None
        self.diagnostics = None
        self.collisions = collisions
        self.collision_tol = collision_tol

    def log(self, newdf):

        """ logs simulation updates after each tick
        """

        if self.df is None:
            self.df = newdf
        else:
            self.df = pd.concat([self.df, newdf.iloc[1:,:]], ignore_index=True)

    def log_diagnostics(self, diag_df):

        """ logs simulation diagnostic updates after each tick
        """

        if self.diagnostics is None:
            self.diagnostics = diag_df
        else:
            # self.diagnostics = pd.concat([self.diagnostics, diag_df.iloc[1:,:]], ignore_index=True)
            # replace last element of self.diagnostics with new value
            self.diagnostics.iloc[-1, :] = diag_df.iloc[0, :]
            self.diagnostics = pd.concat([self.diagnostics, diag_df.iloc[1:,:]], ignore_index=True)

    def apriori_collisions(self, current_state, agents, targets, time):

        """ computes apriori collisions between agents and targets in 2D/3D

        # implements apriori (continuous) collision detection
        # use bounding circles/spheres around each particle
            # easy to calculate distances for circles
            # more complicated shapes - use gilbert-johnson-keerthi algorithm (GJK)

        """

        nagents = len(agents)
        ntargets = len(targets)

        # assumes agent and targets share the same state shape
        dim = self.dim

        tstart = time
        tfinal = time + self.dt

        updated_state = copy.deepcopy(current_state)

        # circle/sphere around every agent/target, if they touch = collision
        bounding_radius_agent = self.collision_tol/2
        bounding_radius_target = self.collision_tol/2

        # for now consider all agent-target pairs - can be optimized
        collided = set() # tuple(i, j)

        for i in range(nagents):
            # agent state components (differs per dynamic model)
            y_agent_statespace = agents[i].get_statespace()
            agent_dim_pos = y_agent_statespace['position']
            agent_dim_vel = y_agent_statespace['velocity']
            dx = agents[i].state_size()

            # full agent state
            y_agent = updated_state[i*dx:(i+1)*dx] # agent i

            # agent current and projected future position
            y_agent_current_pos = y_agent[agent_dim_pos]
            y_agent_final_pos = y_agent_current_pos + y_agent[agent_dim_vel]*self.dt

            # check each agent against each target
            for j in range(ntargets):
                y_target_statespace = targets[j].get_statespace()
                target_dim_pos = y_target_statespace['position']
                target_dim_vel = y_target_statespace['velocity']
                dx = targets[j].state_size()

                # full target state
                y_target = updated_state[(j+ntargets)*dx:(j+ntargets+1)*dx]

                # target current and projected future position
                y_target_current_pos = y_target[target_dim_pos]
                y_target_final_pos = y_target_current_pos + y_target[target_dim_vel]*self.dt

                # agent/target current and future positions
                a0 = y_agent_current_pos
                af = y_agent_final_pos
                t0 = y_target_current_pos
                tf = y_target_final_pos
                del_a = af - a0
                del_t = tf - t0

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
                        print("COLLISION DETECTED ", "(", i, ", ", j, ") ", t)
                        print("       ", a0, " t0: ", t0)
                        collided.add((i,j))

                        # TODO update agent/target state to show projected collision location
                        # update agent to be at location of collision
                        updated_state[i*dx:(i+1)*dx][agent_dim_pos] = y_agent_final_pos
                        # updated_state[i*dx:(i+1)*dx][agent_dim_vel] = np.zeros((3))

                        # update target to be at location of collision
                        updated_state[(j+ntargets)*dx:(j+ntargets+1)*dx][target_dim_pos] = y_target_final_pos
                        # updated_state[(j+ntargets)*dx:(j+ntargets+1)*dx][target_dim_vel] = np.zeros((3))


                # if t_collisions.size != 0:
                #     if 0 <= np.amin(t_collisions[np.isreal(t_collisions)]) <= 1:
                #         collided.append((i,j))

        print("COLLISIONS: ", collided)
        # return collided

        # TODO return the collision location and set as the final location of that agent/target
        return collided, updated_state

    def run(self, x0, system):

        """ Main simulation loop

        Input:
        - x0:           initial agent, target, target terminal states
        - system:       System which encapsulates the agent-target engagement

        """

        current_state = copy.deepcopy(x0)
        running = True
        time = 0

        # SYSTEM PREPROCESSOR
        if self.collisions:
            # collisions = self.apriori_collisions(current_state, system.agents, system.targets, time)
            collisions, updated_state = self.apriori_collisions(current_state, system.agents, system.targets, time)
        else:
            # collisions = set()
            collisions, updated_state = set()

        # system.pre_process(time, current_state, collisions)
        system.pre_process(time, updated_state, collisions)

        # RUN THE SYSTEM
        for time in np.arange(0.0, self.maxtime, self.dt):

            tick = time / self.dt

            # print("Time: {0:3.2E}".format(time))
            if self.collisions:
                # collisions = self.apriori_collisions(current_state, system.agents, system.targets, time)
                collisions, updated_state = self.apriori_collisions(current_state, system.agents, system.targets, time)

            else:
                # collisions = set()
                collisions, updated_state = set()

            # thist, state_hist, assign_hist, diagnostics = system.update(time, current_state, collisions, self.dt, tick)
            thist, state_hist, assign_hist, diagnostics = system.update(time, updated_state, collisions, self.dt, tick)

            newdf = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                                            state_hist,
                                            assign_hist)))

            assign_comp_cost = diagnostics[0]
            dynamics_comp_cost = diagnostics[1]
            diag_df = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                assign_comp_cost,
                dynamics_comp_cost)))

            self.log(newdf)
            self.log_diagnostics(diag_df)

            if time > self.maxtime:
                running = False

            current_state = state_hist[-1, :]

