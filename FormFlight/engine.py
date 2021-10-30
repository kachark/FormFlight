
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

    def apriori_collisions(self, time, world_state, world):

        """ computes apriori collisions between agents and targets in 2D/3D

        # implements apriori (continuous) collision detection
        # use bounding circles/spheres around each particle
            # easy to calculate distances for circles
            # more complicated shapes - use gilbert-johnson-keerthi algorithm (GJK)

        """

        # NOTE only checks collisions between a multiagent system named 'Agent_MAS' and another
        # named 'Target_MAS', ie. 1 group vs 1 group collision detection

        agent_mas = world.get_multi_object('Agent_MAS')
        agents = agent_mas.agent_list
        target_mas = world.get_multi_object('Target_MAS')
        targets = target_mas.agent_list

        nagents = len(agents)
        ntargets = len(targets)

        # assumes agent and targets share the same state shape
        dim = self.dim

        tstart = time
        tfinal = time + self.dt

        updated_state = copy.deepcopy(world_state)

        # circle/sphere around every agent/target, if they touch = collision
        bounding_radius_agent = self.collision_tol/2
        bounding_radius_target = self.collision_tol/2

        # for now consider all agent-target pairs - can be optimized
        collided = set() # tuple(i, j)

        # TODO need to update to the new API
        # every dynamic object should be checked against every other dynamic + static object
        dynamic_object_IDs = world.dynamic_object_IDs
        for global_ID in dynamic_object_IDs:
            object_i = world.objects[global_ID]
            obj_i_start_ind, obj_i_end_ind = world.get_object_world_state_index(global_ID)
            object_i_state = updated_state[obj_i_start_ind:obj_i_end_ind]
            object_i_statespace = object_i.get_statespace()

            object_i_dim_pos = object_i_statespace['position']
            object_i_dim_vel = object_i_statespace['velocity']

            object_i_current_pos = object_i_state[object_i_dim_pos]
            object_i_final_pos = object_i_current_pos + object_i_state[object_i_dim_vel]*self.dt

            # # TODO use this approach when can recall static obj state indices
            # for object_j in world.objects:
            #     object_j_ID = object_j.ID
            #     if global_ID == object_j_ID:
            #         continue


            for global_ID_j in world.dynamic_object_IDs:
                if global_ID == global_ID_j:
                    continue

                object_j = world.objects[global_ID_j]
                obj_j_start_ind, obj_j_end_ind = world.get_object_world_state_index(global_ID_j)
                object_j_state = updated_state[obj_j_start_ind:obj_j_end_ind]
                object_j_statespace = object_j.get_statespace()

                object_j_dim_pos = object_j_statespace['position']
                object_j_dim_vel = object_j_statespace['velocity']

                # target current and projected future position
                object_j_current_pos = object_j_state[object_j_dim_pos]
                object_j_final_pos = object_j_current_pos + object_j_state[object_j_dim_vel]*self.dt

                # agent/target current and future positions
                a0 = object_i_current_pos
                af = object_i_final_pos
                t0 = object_j_current_pos
                tf = object_j_final_pos
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
                        print("COLLISION DETECTED ", "(", global_ID, ", ", global_ID_j, ") ", t)
                        print("       ", a0, " t0: ", t0)
                        collided.add((global_ID, global_ID_j))

                        # TODO update agent/target state to show projected collision location
                        # update agent to be at location of collision
                        updated_state[obj_i_start_ind:obj_i_end_ind][object_i_dim_pos] = \
                                object_i_final_pos
                        # updated_state[i*dx:(i+1)*dx][agent_dim_vel] = np.zeros((3))

                        # update target to be at location of collision
                        updated_state[obj_j_start_ind:obj_j_end_ind][object_j_dim_pos] = \
                                object_j_final_pos
                        # updated_state[(j+ntargets)*dx:(j+ntargets+1)*dx][target_dim_vel] = np.zeros((3))

        print("COLLISIONS: ", collided)
        # return collided

        # TODO return the collision location and set as the final location of that agent/target
        return collided, updated_state

    def run(self, x0, system):

        """ Calls System pre-processor and contains main simulation loop

        Input:
        - x0:           initial agent, target, target terminal states
        - system:       System which encapsulates the agent-target engagement

        """

        current_state = copy.deepcopy(x0)
        running = True
        time = 0

        # SYSTEM PREPROCESSOR
        # check initial time collision condition
        if self.collisions:
            collisions, updated_state = self.apriori_collisions(time, current_state, system.world)
        else:
            collisions = set()
            updated_state = current_state

        system.pre_process(time, updated_state, collisions)

        # RUN THE SYSTEM
        for time in np.arange(0.0, self.maxtime, self.dt):

            tick = time / self.dt

            # print("Time: {0:3.2E}".format(time))
            if self.collisions:
                collisions, updated_state = self.apriori_collisions(time, current_state,
                        system.world)
            else:
                collisions = set()
                updated_state = current_state

            thist, state_hist, assign_hist, diagnostics = system.update(time, self.dt, tick,
                    updated_state, collisions)

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

