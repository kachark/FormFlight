
import pandas as pd
import numpy as np
import copy

################################
## Game Engine
###############################
class Engine:

    def __init__(self, dim, dt=0.1, maxtime=10, collisions=False, collision_tol=0.25):
        self.dim = dim
        self.dt = dt
        self.maxtime = maxtime
        self.df = None
        self.collisions = collisions
        self.collision_tol = collision_tol

    def log(self, newdf):
        if self.df is None:
            self.df = newdf
        else:
            self.df = pd.concat([self.df, newdf.iloc[1:,:]], ignore_index=True)

    # TODO UPDATE TO HANDLE LINEARIZED QC - DIM_POS
    # Physics
    # 2d and 3d
    # def apriori_collisions(self, current_state, nagents, ntargets, time):
    def apriori_collisions(self, current_state, agents, targets, time):

        nagents = len(agents)
        ntargets = len(targets)

        # assumes agent and targets share the same state shape
        dim = self.dim

        # TODO dx dependent on agent/target dynamic models
        # if self.dim == 3:
        #     dx = 6
        # if self.dim == 2:
        #     dx = 4

        tstart = time
        tfinal = time + self.dt

        updated_state = copy.deepcopy(current_state)

        # implement a-prior (continuous) collision detection
        # use bounding circles/spheres around each particle
            # easy to calculate distances for circles
            # more complicated shapes - use gilbert-johnson-keerthi algorithm (GJK)

        # for now consider all agent-target pairs - can be optimized
        collided = set() # tuple(i, j)
        bounding_radius_agent = self.collision_tol
        bounding_radius_target = self.collision_tol
        for i in range(nagents):
            # agent state components (differs per dynamic model)
            y_agent_statespace = agents[i].get_statespace()
            agent_dim_pos = y_agent_statespace['position']
            agent_dim_vel = y_agent_statespace['velocity']
            dx = agents[i].state_size()

            y_agent = updated_state[i*dx:(i+1)*dx] # time history of agent i

            # TODO UPDATE TO HANDLE LINEARIZED QC - DIM_POS, DIM_VEL
            if dim == 2:
                # y_agent_final = y_agent[:dim] + np.array([y_agent[2], y_agent[3]])*self.dt
                x = agent_dim_pos[0]
                y = agent_dim_pos[1]
                xdot = agent_dim_vel[0]
                ydot = agent_dim_vel[1]
                y_agent_current = np.array([y_agent[x], y_agent[y]])
                y_agent_final = y_agent_current + np.array([y_agent[xdot], y_agent[ydot]])*self.dt
            if dim == 3:
                # y_agent_final = y_agent[:dim] + np.array([y_agent[3], y_agent[4], y_agent[5]])*self.dt
                x = agent_dim_pos[0]
                y = agent_dim_pos[1]
                z = agent_dim_pos[2]
                xdot = agent_dim_vel[0]
                ydot = agent_dim_vel[1]
                zdot = agent_dim_vel[2]
                y_agent_current = np.array([y_agent[x], y_agent[y], y_agent[z]])
                y_agent_final = y_agent_current + np.array([y_agent[xdot], y_agent[ydot], y_agent[zdot]])*self.dt
            # print(y_agent)

            # check each agent against each target
            for j in range(ntargets):
                y_target_statespace = targets[j].get_statespace()
                target_dim_pos = y_target_statespace['position']
                target_dim_vel = y_target_statespace['velocity']
                dx = targets[j].state_size()

                y_target = updated_state[(j+ntargets)*dx:(j+ntargets+1)*dx]

                if dim == 2:
                    # y_target_final = y_target[:dim] + np.array([y_target[2], y_target[3]])*self.dt # final position components
                    x = target_dim_pos[0]
                    y = target_dim_pos[1]
                    xdot = target_dim_vel[0]
                    ydot = target_dim_vel[1]
                    y_target_current = np.array([y_target[x], y_target[y]])
                    y_target_final = y_target_current + np.array([y_target[xdot], y_target[ydot]])*self.dt
                if dim == 3:
                    # y_target_final = y_target[:dim] + np.array([y_target[3], y_target[4], y_target[5]])*self.dt # final position components
                    x = target_dim_pos[0]
                    y = target_dim_pos[1]
                    z = target_dim_pos[2]
                    xdot = target_dim_vel[0]
                    ydot = target_dim_vel[1]
                    zdot = target_dim_vel[2]
                    y_target_current = np.array([y_target[x], y_target[y], y_target[z]])
                    y_target_final = y_target_current + np.array([y_target[xdot], y_target[ydot], y_target[zdot]])*self.dt

                # agent/target current and future positions

                # TODO
                # a0 = y_agent[:dim]
                # af = y_agent_final[:dim]
                # t0 = y_target[:dim]
                # tf = y_target_final[:dim]
                # del_a = af - a0
                # del_t = tf - t0
                a0 = y_agent_current
                af = y_agent_final
                t0 = y_target_current
                tf = y_target_final
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
                        print("COLLISION DETECTED ", "(", i, ", ", j, ") ", t)
                        print("       ", a0, " t0: ", t0)
                        collided.add((i,j))

                # if t_collisions.size != 0:
                #     if 0 <= np.amin(t_collisions[np.isreal(t_collisions)]) <= 1:
                #         collided.append((i,j))

        print("COLLISIONS: ", collided)
        return collided

    def run(self, x0, system):

        current_state = copy.deepcopy(x0)
        running = True
        time = 0

        for time in np.arange(0.0, self.maxtime, self.dt):

            tick = time / self.dt

            # print("Time: {0:3.2E}".format(time))
            if self.collisions:
                # collisions = self.apriori_collisions(current_state, system.nagents, system.ntargets, time)

                # TEST
                # TODO UPDATE TO HANDLE LINEARIZED QC - statespace
                collisions = self.apriori_collisions(current_state, system.agents, system.targets, time)

            else:
                collisions = set()

            # thist, state_hist, assign_hist = system.update(time, current_state, self.dt)
            thist, state_hist, assign_hist = system.update(time, current_state, collisions, self.dt, tick)


            newdf = pd.DataFrame(np.hstack((thist[:, np.newaxis],
                                            state_hist,
                                            assign_hist)))

            self.log(newdf)

            if time > self.maxtime:
                running = False

            current_state = state_hist[-1, :]

