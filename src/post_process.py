import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import pandas as pd
import copy

def post_process_identical_2d_doubleint(df, poltrack, Q, R, nagents):

    du = 2

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()
    # print(yout)

    # list of n lists (each for an agent) with collision times and targets collided with
    pos_tolerance = 1
    filtered_yout, collisions, assignment_switches = filter_collisions(df, poltrack, Q, R, nagents, pos_tolerance)

    # exit(1)

    P = poltrack.get_P()

    # plt.figure()
    # plt.plot(yout[:, 0], yout[:, 1], '-r')
    # plt.plot(yout[0, 0], yout[0, 1], 'rs')
    # plt.plot(yout[:, 4], yout[:, 5], '--g')
    # plt.plot(yout[0, 4], yout[0, 5], 'gx')


    # fig, axs = plt.subplots(3, 1, figsize=(10, 5))
    # axs[0].plot(tout, yout[:, 0] - yout[:, 4])
    # axs[0].set_ylabel('x')
    # axs[1].plot(tout, yout[:, 1] - yout[:, 5])
    # axs[1].set_ylabel('y')
    # axs[2].plot(yout[:, 0], yout[:,1])
    # axs[2].set_xlabel('x')
    # axs[2].set_ylabel('y')

    # assignments = yout[:, nagents*2*4:].astype(np.int32)
    assignments = filtered_yout[:, nagents*2*4:].astype(np.int32)


    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    for zz in range(nagents):
        y_agent = yout[:, zz*4:(zz+1)*4]

        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*4:(assignments[ii][zz]+nagents+1)*4]
            controls[ii, :] = poltrack.evaluate(tout[ii], y_agent[ii, :], y_target)

        y_target = yout[0, (assignments[0][zz]+nagents)*4:(assignments[0][zz]+nagents+1)*4]
        xp[0, zz] = np.dot(y_agent[0, :]- y_target, np.dot(P, y_agent[0, :] - y_target))

        stage_cost[0, zz] = np.dot(y_agent[0, :] - y_target,
                                   np.dot(Q, y_agent[0, :] - y_target)) + \
                            np.dot(controls[ii, :], np.dot(R, controls[ii, :]))
        for ii in range(1, yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*4:(assignments[ii][zz]+nagents+1)*4]

            xp[ii, zz] = np.dot(y_agent[ii, :] - y_target, np.dot(P, y_agent[ii, :] - y_target))
            stage_cost[ii, zz] = np.dot(y_agent[ii, :] - y_target, np.dot(Q, y_agent[ii, :] - y_target)) + \
                                        np.dot(controls[ii, :], np.dot(R, controls[ii, :]))

        for ii in range(tout.shape[0]):
            final_cost[ii, zz] = np.trapz(stage_cost[:ii, zz], x=tout[:ii])


    optcost = np.sum(xp[0, :])
    # final_cost = np.sum(final_cost, axis=1)
    fig, axs = plt.subplots(1,1)
    plt.plot(tout, optcost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')

    plt.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
    plt.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')

    if nagents == 2:
        plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (1)')    
        plt.plot(tout, final_cost[:, 1], '--c', label='Cum. Stage Cost (2)')        
        plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (1)')
        plt.plot(tout, xp[:, 1], '--r', label='Cost-to-go (assuming no switch) (2)')
    plt.legend()

    if nagents > 1:

        plt.figure()
        for ii in range(nagents):
            plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))

        plt.title("Assignments")
        plt.legend()



    if nagents == 2:
        plt.figure()
        for zz in range(nagents):
            y_agent = yout[:, zz*4:(zz+1)*4]
            plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            plt.plot(y_target[0, 0], y_target[0, 1], 'bs')
            plt.plot(y_target[:, 0], y_target[:, 1], '-b')

    if nagents == 3:
        # plt.figure()
        fig, ax = plt.subplots()
        for zz in range(nagents):
            # y_agent = yout[:, zz*4:(zz+1)*4] # time history of agent i
            y_agent = filtered_yout[:, zz*4:(zz+1)*4] # time history of agent i
            ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            ax.plot(y_agent[:, 0], y_agent[:, 1], '-r')

            # collision circles
            collision_time = collisions[zz][0][0]
            patches = []
            c1 = Circle( (y_agent[collision_time, 0], y_agent[collision_time, 1]), 2*pos_tolerance, color='r', fill=False)
            patches.append(c1)

            # assignment switch circles
            for switch_ind in assignment_switches[zz]:
                ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 1, color='m', fill=True)
                patches.append(ci)

            p = PatchCollection(patches)
            ax.add_collection(p)

            # y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            y_target = filtered_yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
            ax.plot(y_target[:, 0], y_target[:, 1], '-b')

        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")


    if nagents == 4:
        plt.figure()
        for zz in range(nagents):
            y_agent = yout[:, zz*4:(zz+1)*4]
            plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            plt.plot(y_target[0, 0], y_target[0, 1], 'bs')
            plt.plot(y_target[:, 0], y_target[:, 1], '-b')

        plt.figure()

    if nagents == 8:
        plt.figure()
        for zz in range(nagents):
            y_agent = yout[:, zz*4:(zz+1)*4]
            plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            plt.plot(y_target[0, 0], y_target[0, 1], 'bs')
            plt.plot(y_target[:, 0], y_target[:, 1], '-b')

        # for i, t in enumerate(tout):
        #     agents = []
        #     targets = []
        #
        #     for zz in range(nagents): # plot animated trajectories with assignments
        #         agents[zz] = yout[:, zz*4:(zz+1)*4]
        #         targets[zz] = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
        #
        #     plt.plot()

    # TEST
    return filtered_yout, collisions, assignment_switches


def filter_collisions(df, poltrack, Q, R, nagents, pos_tolerance):

    # Output:
    # agent and target states modified to account for collisions
    # collisions (time index, agent i, target j)
    # assignment switch indices per agent

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()

    filtered_yout = copy.deepcopy(yout)

    agents = [None]*nagents
    targets = [None]*nagents

    collisions = []
    switches = []

    for zz in range(nagents):
        y_agent = filtered_yout[:, zz*4:(zz+1)*4] # time history of agent
        agents[zz] = y_agent
        y_agent_assignments = filtered_yout[:, (nagents+nagents)*4 + zz]

        agent_collision = []

        # compare agent i against all target positions
        for tt in range(nagents):
            y_target = filtered_yout[:, (tt+nagents)*4:(tt+nagents+1)*4]
            targets[tt] = y_target

            # element-wise comparison
            comparison = np.isclose(y_agent, y_target, atol=pos_tolerance)
            # identify collision
            for ind, c in enumerate(comparison):
                if c[:2].all(): # x,y both Truefi
                    agent_collision.append((ind, zz, tt))

        # identify indices where assignment switches
        assignment_switch_ind = np.where(y_agent_assignments[:-1] != y_agent_assignments[1:])[0]

        switches.append(assignment_switch_ind)
        collisions.append(agent_collision)

    # every state after collision = state before collision
    for agent_i, c in enumerate(collisions):
        collision_start = c[0][0]
        # filter agents
        filtered_yout[collision_start: , agent_i*4:(agent_i+1)*4] = filtered_yout[collision_start-1, agent_i*4:(agent_i+1)*4]
        # filter targets
        target_j = c[0][2]
        filtered_yout[collision_start: , (target_j+nagents)*4:(target_j+nagents+1)*4] = filtered_yout[collision_start-1, (target_j+nagents)*4:(target_j+nagents+1)*4]
        # filter assignments - DYN may have terminal reassignments
        # filtered_yout[collision_start: , (nagents+nagents)*4 + agent_i] = filtered_yout[collision_start-1, (nagents+nagents)*4 + agent_i]

    return filtered_yout, collisions, switches


    # find collisions
    # if collision found, save indices (agent i, time ind)


