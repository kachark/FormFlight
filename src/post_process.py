import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from matplotlib.collections import PatchCollection
import pandas as pd
import copy

# def post_process_identical_2d_doubleint(df, poltrack, Q, R, nagents):
def post_process_identical_2d_doubleint(df, poltrack, poltargets, Q, R, nagents, ntargets, ot_costs):

    du = 2

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()
    # print(yout)

    # list of n lists (each for an agent) with collision times and targets collided with
    pos_tolerance = 1

    yout = copy.deepcopy(yout)
    assignment_switches = find_switches(tout, yout, nagents, nagents, 4, 4)

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
    assignments = yout[:, nagents*2*4:].astype(np.int32)


    # PLOT COSTS
    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    for zz in range(nagents):
        y_agent = yout[:, zz*4:(zz+1)*4]

        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*4:(assignments[ii][zz]+nagents+1)*4]
            controls[ii, :] = poltrack.evaluate(tout[ii], y_agent[ii, :], y_target)

        y_target = yout[0, (assignments[0][zz]+nagents)*4:(assignments[0][zz]+nagents+1)*4]
        xp[0, zz] = np.dot(y_agent[0, :]- y_target, np.dot(P, y_agent[0, :] - y_target))

        stage_cost[0, zz] = np.dot(y_agent[0, :] - y_target,
                                   np.dot(Q, y_agent[0, :] - y_target)) + \
                            np.dot(controls[ii, :], np.dot(R, controls[ii, :]))

        # stage_cost[0, zz] = np.dot(y_agent[0, :] - y_target,
        #                            np.dot(Q, y_agent[0, :] - y_target))

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
    axs.plot(tout, optcost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')

    axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
    axs.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')

    # f123 = plt.figure()
    # tt = np.linspace(0,4,4/0.01 + 1) % run with dt=0.01, maxtime=4
    # plt.plot(tt,costs)
    # plt.plot(tout, stage_cost)
    # plt.title('Optimal Transport Cost')
    # plt.xlabel('time')
    # plt.ylabel('cost')
    # plt.legend()

    if nagents == 2:
        plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (0)')    
        plt.plot(tout, final_cost[:, 1], '--c', label='Cum. Stage Cost (1)')        
        plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (0)')
        plt.plot(tout, xp[:, 1], '--r', label='Cost-to-go (assuming no switch) (1)')
    # plt.legend()


    # PLOT ASSIGNMENTS
    if nagents > 1:

        plt.figure()
        for ii in range(nagents):
            plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))

        plt.title("Assignments")
        plt.legend()


    # PLOT EXTRA INFO
    if nagents > 1:
        plt.figure()
        for ii in range(nagents):
            # plt.plot(tout, stage_cost[:, ii], label='A{0}'.format(ii))
            plt.plot(tout, xp[:, ii], label='Cost-to-go A{0}'.format(ii))
            # plt.plot(tout, final_cost[:, ii], label='Cum. Stage Cost A{0}'.format(ii))    

        plt.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')
        # plt.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
        # for switch_ind in assignment_switches[ii]:
        #     plt.axvline(tout[switch_ind], linewidth=0.4)
        #     y1 = np.sum(final_cost, axis=1)[switch_ind]
        #     y2 = np.sum(final_cost, axis=1)[switch_ind-30]
        #     x1 = tout[switch_ind]
        #     x2 = tout[switch_ind-30]
        #     print("SLOPE BEFORE SWITCH: ", (y1-y2)/(x1-x2))
        #     plt.plot((x1, x2), (y1, y2))
        #     y1 = np.sum(final_cost, axis=1)[switch_ind]
        #     y2 = np.sum(final_cost, axis=1)[switch_ind+30]
        #     x1 = tout[switch_ind]
        #     x2 = tout[switch_ind+30]
        #     print("SLOPE AFTER SWITCH: ", (y1-y2)/(x1-x2))
        #     plt.plot((x1, x2), (y1, y2))
        # plt.title("Agent Cumulative Stage Costs")
        # plt.xlabel('time')
        plt.legend()


    # PLOT TRAJECTORIES

    if nagents == 2:
        # plt.figure()
        fig, ax = plt.subplots()

        for zz in range(nagents):

            # agent state over time
            y_agent = yout[:, zz*4:(zz+1)*4]

            # plot agent trajectory with text
            ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            ax.plot(y_agent[:, 0], y_agent[:, 1], '-r') # returns line2d object
            ax.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

            # plot location of assignment switches
            patches = []
            for switch_ind in assignment_switches[zz]:
                ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='m', fill=True)
                patches.append(ci)

            p = PatchCollection(patches)
            ax.add_collection(p)

            # plot target trajectory
            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
            ax.plot(y_target[:, 0], y_target[:, 1], '-b')
            ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if nagents == 3:
        # plt.figure()
        fig, ax = plt.subplots()
        for zz in range(nagents):
            # y_agent = yout[:, zz*4:(zz+1)*4] # time history of agent i
            y_agent = yout[:, zz*4:(zz+1)*4] # time history of agent i
            ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
            ax.plot(y_agent[:, 0], y_agent[:, 1], '-r')

            # collision circles
            # collision_time = collisions[zz][0][0]
            patches = []
            # c1 = Circle( (y_agent[collision_time, 0], y_agent[collision_time, 1]), 2*pos_tolerance, color='r', fill=False)
            # patches.append(c1)

            # assignment switch circles
            for switch_ind in assignment_switches[zz]:
                ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 1, color='m', fill=True)
                patches.append(ci)

            p = PatchCollection(patches)
            ax.add_collection(p)

            # y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
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
    # return filtered_yout, collisions, assignment_switches

def post_process_identical_3d_doubleint(df, poltrack, poltargets, Q, R, nagents, ntargets, costs):

    du = 3

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()
    # print(yout)

    yout = copy.deepcopy(yout)
    assignment_switches = find_switches(tout, yout, nagents, nagents, 6, 6)

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
    assignments = yout[:, nagents*2*6:].astype(np.int32)


    # PLOT COSTS
    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    for zz in range(nagents):
        y_agent = yout[:, zz*6:(zz+1)*6]

        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*6:(assignments[ii][zz]+nagents+1)*6]
            controls[ii, :] = poltrack.evaluate(tout[ii], y_agent[ii, :], y_target)

        y_target = yout[0, (assignments[0][zz]+nagents)*6:(assignments[0][zz]+nagents+1)*6]
        xp[0, zz] = np.dot(y_agent[0, :]- y_target, np.dot(P, y_agent[0, :] - y_target))

        stage_cost[0, zz] = np.dot(y_agent[0, :] - y_target,
                                   np.dot(Q, y_agent[0, :] - y_target)) + \
                            np.dot(controls[ii, :], np.dot(R, controls[ii, :]))

        # stage_cost[0, zz] = np.dot(y_agent[0, :] - y_target,
        #                            np.dot(Q, y_agent[0, :] - y_target))

        for ii in range(1, yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*6:(assignments[ii][zz]+nagents+1)*6]

            xp[ii, zz] = np.dot(y_agent[ii, :] - y_target, np.dot(P, y_agent[ii, :] - y_target))
            stage_cost[ii, zz] = np.dot(y_agent[ii, :] - y_target, np.dot(Q, y_agent[ii, :] - y_target)) + \
                                        np.dot(controls[ii, :], np.dot(R, controls[ii, :]))


        for ii in range(tout.shape[0]):
            final_cost[ii, zz] = np.trapz(stage_cost[:ii, zz], x=tout[:ii])


    optcost = np.sum(xp[0, :])
    # final_cost = np.sum(final_cost, axis=1)
    fig, axs = plt.subplots(1,1)
    axs.plot(tout, optcost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')

    axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
    axs.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')

    # f123 = plt.figure()
    # tt = np.linspace(0,4,4/0.01 + 1) % run with dt=0.01, maxtime=4
    # plt.plot(tt,costs)
    # plt.plot(tout, stage_cost)
    # plt.title('Optimal Transport Cost')
    # plt.xlabel('time')
    # plt.ylabel('cost')
    # plt.legend()

    if nagents == 2:
        plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (0)')    
        plt.plot(tout, final_cost[:, 1], '--c', label='Cum. Stage Cost (1)')        
        plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (0)')
        plt.plot(tout, xp[:, 1], '--r', label='Cost-to-go (assuming no switch) (1)')
    # plt.legend()


    # PLOT ASSIGNMENTS
    if nagents > 1:

        plt.figure()
        for ii in range(nagents):
            plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))

        plt.title("Assignments")
        plt.legend()


    # PLOT EXTRA INFO
    if nagents > 1:
        plt.figure()
        for ii in range(nagents):
            # plt.plot(tout, stage_cost[:, ii], label='A{0}'.format(ii))
            plt.plot(tout, xp[:, ii], label='Cost-to-go A{0}'.format(ii))
            # plt.plot(tout, final_cost[:, ii], label='Cum. Stage Cost A{0}'.format(ii))    

        plt.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')
        # plt.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
        # for switch_ind in assignment_switches[ii]:
        #     plt.axvline(tout[switch_ind], linewidth=0.4)
        #     y1 = np.sum(final_cost, axis=1)[switch_ind]
        #     y2 = np.sum(final_cost, axis=1)[switch_ind-30]
        #     x1 = tout[switch_ind]
        #     x2 = tout[switch_ind-30]
        #     print("SLOPE BEFORE SWITCH: ", (y1-y2)/(x1-x2))
        #     plt.plot((x1, x2), (y1, y2))
        #     y1 = np.sum(final_cost, axis=1)[switch_ind]
        #     y2 = np.sum(final_cost, axis=1)[switch_ind+30]
        #     x1 = tout[switch_ind]
        #     x2 = tout[switch_ind+30]
        #     print("SLOPE AFTER SWITCH: ", (y1-y2)/(x1-x2))
        #     plt.plot((x1, x2), (y1, y2))
        # plt.title("Agent Cumulative Stage Costs")
        # plt.xlabel('time')
        plt.legend()


    # PLOT TRAJECTORIES

    if nagents == 2:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # destination location
        for zz in range(ntargets):
            pt = poltargets[zz]
            offset = pt.offset
            ax.scatter3D(offset[0], offset[1], offset[2], color='k')

        # agent/target trajectories
        for zz in range(nagents):

            # agent state over time
            y_agent = yout[:, zz*6:(zz+1)*6]

            # plot agent trajectory with text
            ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
            ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r') # returns line2d object
            # ax.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

            # plot location of assignment switches
            # patches = []
            for switch_ind in assignment_switches[zz]:
                ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m')
                # ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='m', fill=True)
                # patches.append(ci)

            # p = PatchCollection(patches)
            # ax.add_collection(p)

            # plot target trajectory
            y_target = yout[:, (zz+nagents)*6:(zz+nagents+1)*6]
            ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
            ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
            # ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")



def find_switches(tout, yout, nagents, ntargets, agent_config_size, target_config_size):

    # Output:
    # assignment switch indices per agent

    switches = []
    yout = copy.deepcopy(yout)
    tout = copy.deepcopy(tout)

    for zz in range(nagents):
        y_agent_assignments = yout[:, (nagents+ntargets)*agent_config_size + zz]

        # identify indices where assignment switches
        assignment_switch_ind = np.where(y_agent_assignments[:-1] != y_agent_assignments[1:])[0]

        switches.append(assignment_switch_ind)

    return switches

def index_at_time(tout, time):
     
    """
    input: time history, time point
    output: index in data of time point
    """

    return (np.abs(tout-time)).argmin()



