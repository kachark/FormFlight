import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from matplotlib.collections import PatchCollection
import pandas as pd
import copy
from controls import *

def post_process_identical_2d_doubleint(df, poltrack, poltargets, nagents, ntargets, ot_costs, polagents, opt_asst):

    dx = 4
    du = 2

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()
    # print(yout)

    yout = copy.deepcopy(yout)
    assignment_switches = find_switches(tout, yout, nagents, nagents, 4, 4)

    print("INITIAL CONDITION: ", yout[0])

    # assignments = yout[:, nagents*2*4:].astype(np.int32)
    assignments = yout[:, nagents*2*dx:].astype(np.int32)

    # PLOT COSTS
    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    optimal_cost = np.zeros((1, nagents))

    xss = np.zeros((yout.shape[0], nagents*2*dx))
    for zz in range(nagents):
        y_agent = yout[:, zz*dx:(zz+1)*dx]

        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # AUGMENTED TRACKER
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)

            # NEW
            # Get agent policy in correct tracking state for P, Q, p at time ii
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            controls[ii, :] = polagents[zz].evaluate(tout[ii], y_agent[ii, :], y_target, controls_targ)

        # COSTS

        # post-process for t=0
        y_target = yout[0, (assignments[0][zz]+nagents)*dx:(assignments[0][zz]+nagents+1)*dx]

        # Get agent policy in correct tracking state for P, Q, p at t=0
        asst_0 = assignments[0] # assignments at time ii
        sigma_i = asst_0[zz] # target assigned-to agent zz
        Acl_0 = poltargets[sigma_i].get_closed_loop_A()
        gcl_0 = poltargets[sigma_i].get_closed_loop_g()
        polagents[zz].track(0, sigma_i, Acl_0, gcl_0)

        R = polagents[zz].get_R()
        Q_0 = polagents[zz].get_Q()
        P_0 = polagents[zz].get_P()
        p_0 = polagents[zz].get_p()

        uss_0 = polagents[zz].get_uss()
        Xss_0 = polagents[zz].get_xss()
        X_0 = np.hstack((y_agent[0, :], y_target))
        xp[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        stage_cost[0, zz] = np.dot(X_0, np.dot(Q_0, X_0)) + np.dot(controls[0, :], np.dot(R, controls[0, :])) -\
            (np.dot(Xss_0, np.dot(Q_0, Xss_0)) + np.dot(uss_0, np.dot(R, uss_0)))

        # optimal cost (ie. DYN)
        opt_asst_y_target = yout[0, (opt_asst[zz]+nagents)*dx:(opt_asst[zz]+nagents+1)*dx]
        X_0 = np.hstack((y_agent[0, :], opt_asst_y_target))

        # Get agent policy in correct tracking state for P, Q, p at t=0
        optasst_0 = opt_asst # assignments at time ii
        optasst_sigma_i = asst_0[zz] # target assigned-to agent zz
        optasst_Acl_0 = poltargets[optasst_sigma_i].get_closed_loop_A()
        optasst_gcl_0 = poltargets[optasst_sigma_i].get_closed_loop_g()
        polagents[zz].track(0, optasst_sigma_i, optasst_Acl_0, optasst_gcl_0)

        R = polagents[zz].get_R()
        optasst_Q_0 = polagents[zz].get_Q()
        optasst_P_0 = polagents[zz].get_P()
        optasst_p_0 = polagents[zz].get_p()

        optasst_uss_0 = polagents[zz].get_uss()
        optasst_Xss_0 = polagents[zz].get_xss()
        optimal_cost[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        # continue post-processing for rest of time points
        for ii in range(1, yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # TEST
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)
            X = np.hstack((y_agent[ii, :], y_target))

            # Get agent policy in correct tracking state for P, Q, p at time ii
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            R = polagents[zz].get_R()
            Q = polagents[zz].get_Q()
            P = polagents[zz].get_P()
            p = polagents[zz].get_p()

            # STEADY-STATE TERMS
            uss = polagents[zz].get_uss()
            Xss = polagents[zz].get_xss()

            # STAGE COST
            stage_cost[ii, zz] = np.dot(X, np.dot(Q, X)) + np.dot(controls[ii, :], np.dot(R, controls[ii, :])) -\
                (np.dot(Xss, np.dot(Q, Xss)) + np.dot(uss, np.dot(R, uss)))

            # COST-TO-GO
            xp[ii, zz] = np.dot(X, np.dot(P, X)) + 2*np.dot(p, X) -\
                (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        for ii in range(tout.shape[0]):
            final_cost[ii, zz] = np.trapz(stage_cost[:ii, zz], x=tout[:ii])

    # optcost = np.sum(xp[0, :])
    optcost = np.sum(optimal_cost[0, :])

    # final_cost = np.sum(final_cost, axis=1)
    fig, axs = plt.subplots(1,1)
    axs.plot(tout, optcost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')

    axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost')
    axs.plot(tout, np.sum(xp, axis=1), '-r', label='Cost-to-go')

    if nagents == 1:
        plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (0)')    
        plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (0)')

    if nagents == 2:
        plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (0)')    
        plt.plot(tout, final_cost[:, 1], '--c', label='Cum. Stage Cost (1)')        
        plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (0)')
        plt.plot(tout, xp[:, 1], '--r', label='Cost-to-go (assuming no switch) (1)')
    # plt.legend()


    print("POLICY: ", poltrack.__class__.__name__)
    print("FINAL TIME: ", tout[-1])
    print("initial optimal cost: ", optcost)
    print("initial incurred cost: ", final_cost[0])
    print("final cost-to-go value: ", np.sum(xp, axis=1)[-1])
    print("final incurred cost value: ", np.sum(final_cost, axis=1)[-1]) # last stage cost
    print("initial optimal cost - final incurred cost value = ", optcost - np.sum(final_cost, axis=1)[-1])
    print("INITIAL CONDITIONS")
    print(yout[0, :])
    print("FINAL STATE")
    print(yout[-1, :])
    print("OFFSET")
    for pt in poltargets:
        print(pt.const)

    print("FINAL CONTROL")
    print(controls[-1, :])

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
    if nagents:
        # plt.figure()
        fig, ax = plt.subplots()

        for zz in range(ntargets):
            pt = poltargets[zz]
            offset = pt.const
            ax.plot(offset[0], offset[1], 'ks')
            ax.text(offset[0], offset[1], 'C{0}'.format(zz))

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
                ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 2, color='m', fill=True)
                patches.append(ci)

            p = PatchCollection(patches)
            ax.add_collection(p)

            # plot target trajectory
            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
            ax.plot(y_target[:, 0], y_target[:, 1], '-b')
            ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

            print("TARGET FINAL POS: ", y_target[-1])
            print("AGENT FINAL POS: ", y_agent[-1])

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
                ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 5, color='m', fill=True)
                patches.append(ci)

            p = PatchCollection(patches)
            ax.add_collection(p)

            # plot target trajectory
            y_target = yout[:, (zz+nagents)*4:(zz+nagents+1)*4]
            ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
            ax.plot(y_target[:, 0], y_target[:, 1], '-b')
            ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

            print("TARGET FINAL POS: ", y_target[-1])
            print("AGENT FINAL POS: ", y_agent[-1])

        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

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

def post_process_identical_3d_doubleint(df, poltrack, poltargets, nagents, ntargets, ot_costs, polagents, opt_asst):

    dx = 6
    du = 3

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()
    # print(yout)

    yout = copy.deepcopy(yout)
    assignment_switches = find_switches(tout, yout, nagents, nagents, 6, 6)

    # P = poltrack.get_P()
    # Q = poltrack.get_Q()
    # R = poltrack.get_R()

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
    assignments = yout[:, nagents*2*dx:].astype(np.int32)

    # PLOT COSTS
    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    optimal_cost = np.zeros((1, nagents))

    xss = np.zeros((yout.shape[0], nagents*2*dx))
    for zz in range(nagents):
        y_agent = yout[:, zz*dx:(zz+1)*dx]

        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # AUGMENTED TRACKER
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)

            # NEW
            # Get agent policy in correct tracking state for P, Q, p at time ii
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            controls[ii, :] = polagents[zz].evaluate(tout[ii], y_agent[ii, :], y_target, controls_targ)

        # COSTS

        # post-process for t=0
        y_target = yout[0, (assignments[0][zz]+nagents)*dx:(assignments[0][zz]+nagents+1)*dx]

        # Get agent policy in correct tracking state for P, Q, p at t=0
        asst_0 = assignments[0] # assignments at time ii
        sigma_i = asst_0[zz] # target assigned-to agent zz
        Acl_0 = poltargets[sigma_i].get_closed_loop_A()
        gcl_0 = poltargets[sigma_i].get_closed_loop_g()
        polagents[zz].track(0, sigma_i, Acl_0, gcl_0)

        R = polagents[zz].get_R()
        Q_0 = polagents[zz].get_Q()
        P_0 = polagents[zz].get_P()
        p_0 = polagents[zz].get_p()

        uss_0 = polagents[zz].get_uss()
        Xss_0 = polagents[zz].get_xss()
        X_0 = np.hstack((y_agent[0, :], y_target))
        xp[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        stage_cost[0, zz] = np.dot(X_0, np.dot(Q_0, X_0)) + np.dot(controls[0, :], np.dot(R, controls[0, :])) -\
            (np.dot(Xss_0, np.dot(Q_0, Xss_0)) + np.dot(uss_0, np.dot(R, uss_0)))

        # optimal cost (ie. DYN)
        opt_asst_y_target = yout[0, (opt_asst[zz]+nagents)*dx:(opt_asst[zz]+nagents+1)*dx]
        X_0 = np.hstack((y_agent[0, :], opt_asst_y_target))

        # Get agent policy in correct tracking state for P, Q, p at t=0
        optasst_0 = opt_asst # assignments at time ii
        optasst_sigma_i = asst_0[zz] # target assigned-to agent zz
        optasst_Acl_0 = poltargets[optasst_sigma_i].get_closed_loop_A()
        optasst_gcl_0 = poltargets[optasst_sigma_i].get_closed_loop_g()
        polagents[zz].track(0, optasst_sigma_i, optasst_Acl_0, optasst_gcl_0)

        R = polagents[zz].get_R()
        optasst_Q_0 = polagents[zz].get_Q()
        optasst_P_0 = polagents[zz].get_P()
        optasst_p_0 = polagents[zz].get_p()

        optasst_uss_0 = polagents[zz].get_uss()
        optasst_Xss_0 = polagents[zz].get_xss()
        optimal_cost[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))


        # continue post-processing for rest of time points
        for ii in range(1, yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # TEST
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)
            X = np.hstack((y_agent[ii, :], y_target))

            # Get agent policy in correct tracking state for P, Q, p at time ii
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            R = polagents[zz].get_R()
            Q = polagents[zz].get_Q()
            P = polagents[zz].get_P()
            p = polagents[zz].get_p()

            # STEADY-STATE TERMS
            uss = polagents[zz].get_uss()
            Xss = polagents[zz].get_xss()

            # STAGE COST
            stage_cost[ii, zz] = np.dot(X, np.dot(Q, X)) + np.dot(controls[ii, :], np.dot(R, controls[ii, :])) -\
                (np.dot(Xss, np.dot(Q, Xss)) + np.dot(uss, np.dot(R, uss)))

            # COST-TO-GO
            xp[ii, zz] = np.dot(X, np.dot(P, X)) + 2*np.dot(p, X) -\
                (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        for ii in range(tout.shape[0]):
            final_cost[ii, zz] = np.trapz(stage_cost[:ii, zz], x=tout[:ii])

    optcost = np.sum(optimal_cost[0, :])

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

    print("POLICY: ", poltrack.__class__.__name__)
    print("FINAL TIME: ", tout[-1])
    print("initial optimal cost: ", optcost)
    print("initial incurred cost: ", final_cost[0])
    print("final cost-to-go value: ", np.sum(xp, axis=1)[-1])
    print("final incurred cost value: ", np.sum(final_cost, axis=1)[-1]) # last stage cost
    print("initial optimal cost - final incurred cost value = ", optcost - np.sum(final_cost, axis=1)[-1])
    print("INITIAL CONDITIONS")
    print(yout[0, :])
    print("FINAL STATE")
    print(yout[-1, :])
    print("OFFSET")
    for pt in poltargets:
        print(pt.const)

    print("FINAL CONTROL")
    print(controls[-1, :])


    # plt.figure()
    # for zz in range(nagents):
    #     plt.plot(tout, xss[:, zz*2*dx:(zz+1)*2*dx][0], 'r')
    #     plt.plot(tout, xss[:, zz*2*dx:(zz+1)*2*dx][1], 'b')
    #     plt.plot(tout, xss[:, zz*2*dx:(zz+1)*2*dx][2], 'm')
    #     plt.plot(tout, yout[:, zz*2*dx:(zz+1)*2*dx][0], 'rx') 
    #     plt.plot(tout, yout[:, zz*2*dx:(zz+1)*2*dx][1], 'bx') 
    #     plt.plot(tout, yout[:, zz*2*dx:(zz+1)*2*dx][2], 'mx') 

    # if nagents == 2:
    #     plt.plot(tout, final_cost[:, 0], '-.c', label='Cum. Stage Cost (0)')    
    #     plt.plot(tout, final_cost[:, 1], '--c', label='Cum. Stage Cost (1)')        
    #     plt.plot(tout, xp[:, 0], '-.r', label='Cost-to-go (assuming no switch) (0)')
    #     plt.plot(tout, xp[:, 1], '--r', label='Cost-to-go (assuming no switch) (1)')
    # # plt.legend()


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
    if nagents:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # destination location
        # IF zeropol just set to target last position
        for zz in range(ntargets):
            pt = poltargets[zz]
            offset = pt.const
            ax.scatter3D(offset[0], offset[1], offset[2], color='k')
            ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

        # agent/target trajectories
        for zz in range(nagents):

            # agent state over time
            y_agent = yout[:, zz*dx:(zz+1)*dx]

            # plot agent trajectory with text
            ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
            ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r') # returns line2d object
            ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz))

            # plot location of assignment switches
            # patches = []
            for switch_ind in assignment_switches[zz]:
                ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO
                # ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='m', fill=True)
                # patches.append(ci)

            # p = PatchCollection(patches)
            # ax.add_collection(p)

            # plot target trajectory
            y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
            ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
            ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
            ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

def post_process_batch_simulation(batch_results):

    sim_names = []
    batch_performance_metrics = {} # performance metrics
    sim_components = {} # useful parameters and objects used within the simulation

    dim = 2 # default value. also uniform across batch simulations

    # for every simulation within a batch, post-process results
    for sim_name, sim in batch_results.items():
        sim_names.append(sim_name)
        parameters = sim[0]
        sim_results = sim[1]

        dx = parameters['dx']
        du = parameters['du']
        dim = parameters['dim']
        agent_model = parameters['agent_model']
        target_model = parameters['target_model']
        agent_control_policy = parameters['agent_control_policy']
        target_control_policy = parameters['target_control_policy']

        nagents = sim_results['nagents']
        ntargets = sim_results['ntargets']
        poltargets = sim_results['target_pol']
        apol = sim_results['asst_policy']

        components = {'dx': dx, 'du': du, 'dim': dim, 'agent_model': agent_model, 'target_model': target_model,
                'agent_control_policy': agent_control_policy, 'target_control_policy': target_control_policy, 'nagents': nagents, 'ntargets': ntargets, 'poltargets': poltargets, 'apol': apol}

        sim_components.update({sim_name: components})

        post_processed_results_df = None

        # post-process each sim within batch
        if parameters['dim'] == 2:
            if parameters['agent_model'] == 'Double_Integrator':
                post_processed_results_df = post_process_identical_doubleint_TEST(parameters, sim_results)

            # if parameters['agent_model'] == 'Linearized Quadcopter':
            #     post_process_identical_2d_doubleint()


        if parameters['dim'] == 3:
            if parameters['agent_model'] == 'Double_Integrator':
                post_processed_results_df = post_process_identical_doubleint_TEST(parameters, sim_results)

            # if parameters['agent_model'] == 'Linearized Quadcopter':
            #     post_process_identical_3d_doubleint()

        # collect post-processed performance metrics
        batch_performance_metrics.update({sim_name: post_processed_results_df})

    return batch_performance_metrics


def unpack_performance_metrics(batch_performance_metrics):

    unpacked_batch_metrics = {}

    for sim_name, metrics_df in batch_performance_metrics.items():

        ### unpack simulation metrics ###

        # simulation parameters
        parameter_cols = 7 # see stored data spec
        parameters = metrics_df.iloc[0, 0:7].to_numpy()

        dt = int(parameters[0])
        dim = int(parameters[1])
        collisions = int(parameters[2])
        dx = int(parameters[3])
        du = int(parameters[4])
        nagents = int(parameters[5])
        ntargets = int(parameters[6])

        # simulation outputs
        output_cols = 1 + nagents*dx + ntargets*dx + nagents + ntargets*dx + nagents*du
        outputs = metrics_df.iloc[:, 7: parameter_cols + output_cols].to_numpy()

        tout = outputs[:, 0]
        yout_cols = 1 + nagents*dx + ntargets*dx + nagents
        yout = outputs[:, 1: yout_cols] # good
        ss_cols = yout_cols + ntargets*dx
        stationary_states = outputs[0, yout_cols: ss_cols]
        ctrl_cols = ss_cols + nagents*du
        agent_controls = outputs[:, ss_cols: 1+ ctrl_cols]

        # simulation costs
        costs = metrics_df.iloc[:, parameter_cols + output_cols: ].to_numpy()

        fc_cols = nagents
        final_cost = costs[:, 0:fc_cols]
        sc_cols = fc_cols + nagents
        stage_cost = costs[:, fc_cols: sc_cols]
        ctg_cols = sc_cols + nagents
        cost_to_go = costs[:, sc_cols: ctg_cols]
        optimal_cost = costs[:, ctg_cols: ]

        unpacked = [dt, dim, collisions, dx, du, nagents, ntargets, tout, yout, stationary_states, agent_controls,
                final_cost, stage_cost, cost_to_go, optimal_cost]

        ### end unpack ###

        columns = [
            "dt",
            "dim",
            "collisions",
            "dx",
            "du",
            "nagents",
            "ntargets",
            "tout",
            "yout",
            "stationary_states",
            "agent_controls",
            "final_cost",
            "stage_cost",
            "cost_to_go",
            "optimal_cost"
        ]

        metrics = {}
        for (col, met) in zip(columns, unpacked):
            metrics.update({col: met})

        unpacked_batch_metrics.update({sim_name: metrics})

    return unpacked_batch_metrics

def plot_batch_performance_metrics(batch_performance_metrics):

    unpacked = unpack_performance_metrics(batch_performance_metrics)

    # dim is universal across a batch
    dim = 2

    ####### PLOT RESULTS
    # TODO refactor into individual functions - cost_plots, assignment_plots, trajectory_plots depending on agent/target models
    # plot batch performance metrics

    # NOTE independent of dimension
    fig, axs = plt.subplots(1,1)
    for sim_name, metrics in unpacked.items():

        tout = metrics['tout']
        yout = metrics['yout']
        final_cost = metrics['final_cost']
        cost_to_go = metrics['cost_to_go']
        optimal_cost = metrics['optimal_cost']

        summed_opt_cost = np.sum(optimal_cost[0, :])

        ### cost plots
        if sim_name == 'AssignmentDyn':
            axs.plot(tout, summed_opt_cost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching - DYN')
            axs.plot(tout, np.sum(final_cost, axis=1), '--c', label='Cum. Stage Cost'+' '+sim_name)
            axs.plot(tout, np.sum(cost_to_go, axis=1), '--r', label='Cost-to-go'+' '+sim_name)
        else:
            axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost'+' '+sim_name)
            axs.plot(tout, np.sum(cost_to_go, axis=1), '-r', label='Cost-to-go'+' '+sim_name)

        axs.legend()

    plt.figure()
    for sim_name, metrics in unpacked.items():

        nagents = metrics['nagents']
        tout = metrics['tout']
        final_cost = metrics['final_cost']
        cost_to_go = metrics['cost_to_go']

        for zz in range(nagents):
            plt.plot(tout, final_cost[:, zz], '-.c', label='Cum. Stage Cost ({0})'.format(zz))
            plt.plot(tout, cost_to_go[:, zz], '-.r', label='Cost-to-go (assuming no switch) ({0})'.format(zz))

        plt.legend()

    # NOTE independent of dimension
    ### assignment plots
    for sim_name, metrics in unpacked.items():

        dx = metrics['dx']
        nagents = metrics['nagents']
        tout = metrics['tout']
        yout = metrics['yout']

        assignments = yout[:, nagents*2*dx:].astype(np.int32)

        plt.figure()
        for ii in range(nagents):
            plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))
            plt.title("Assignments")
            plt.legend()

    ### trajectory plots
    for sim_name, metrics in unpacked.items():
        dim = metrics['dim']

    # want to display all trajectories on same figure
    if dim == 2:
        fig, ax = plt.subplots()
    if dim == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    # for sim_name in sim_names:
    for sim_name, metrics in unpacked.items():

        dx = metrics['dx']
        du = metrics['du']
        dim = metrics['dim']
        nagents = metrics['nagents']
        ntargets = metrics['ntargets']
        tout = metrics['tout']
        yout = metrics['yout']
        stationary_states = metrics['stationary_states']

        assignment_switches = find_switches(tout, yout, nagents, ntargets, dx, dx)

        if dim == 2: # and agent/target models both double integrator (omit requirement for now)
            ### stationary points
            for zz in range(ntargets):
                offset = stationary_states[zz*dx:(zz+1)*dx]
                ax.plot(offset[0], offset[1], 'ks')
                ax.text(offset[0], offset[1], 'C{0}'.format(zz))

            ### Agent / Target Trajectories
            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentDyn':
                for zz in range(nagents):

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
                    ax.plot(y_agent[:, 0], y_agent[:, 1], '-r') # returns line2d object
                    ax.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

                    # plot location of assignment switches
                    patches = []
                    for switch_ind in assignment_switches[zz]:
                        ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='b', fill=True)
                        patches.append(ci)

                    p = PatchCollection(patches)
                    ax.add_collection(p)

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
                    ax.plot(y_target[:, 0], y_target[:, 1], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

                    print("TARGET FINAL POS: ", y_target[-1])
                    print("AGENT FINAL POS: ", y_agent[-1])

                ax.set_title("Trajectory")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            elif sim_name == 'AssignmentEMD':
                # non-optimal trajectories (dotted lines)
                for zz in range(nagents):

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
                    ax.plot(y_agent[:, 0], y_agent[:, 1], '--r') # returns line2d object
                    ax.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

                    # plot location of assignment switches
                    # patches = []
                    for switch_ind in assignment_switches[zz]:
                        # ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 2, color='m', fill=True)
                        # patches.append(ci)
                        ax.plot(y_agent[switch_ind, 0], y_agent[switch_ind, 1], 'ms')

                    # p = PatchCollection(patches)
                    # ax.add_collection(p)

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.plot(y_target[0, 0], y_target[0, 1], 'bs')
                    ax.plot(y_target[:, 0], y_target[:, 1], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

                    print("TARGET FINAL POS: ", y_target[-1])
                    print("AGENT FINAL POS: ", y_agent[-1])

                ax.set_title("Trajectory")
                ax.set_xlabel("x")
                ax.set_ylabel("y")


        if dim == 3:

            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentDyn':
                ### stationary points
                for zz in range(ntargets):
                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k')
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                # agent/target trajectories
                for zz in range(nagents):

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r') # returns line2d object
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz))

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                ax.set_title("Trajectory")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()

            elif sim_name == 'AssignmentEMD':
                # non-optimal trajectories (dotted lines)

                # stationary locations
                for zz in range(ntargets):
                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k')
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                # agent/target trajectories
                for zz in range(nagents):

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '--r') # returns line2d object
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz))

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                ax.set_title("Trajectory")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()


# COMPUTE CONTROLS
def compute_controls(dx, du, yout, tout, assignments, nagents, poltargets, polagents):

    agent_controls = np.zeros((tout.shape[0], du*nagents))
    for zz in range(nagents):
        y_agent = yout[:, zz*dx:(zz+1)*dx]

        agent_zz_controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # AUGMENTED TRACKER
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)

            # NEW
            # Get agent policy in correct tracking state for P, Q, p at time ii
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            agent_zz_controls[ii, :] = polagents[zz].evaluate(tout[ii], y_agent[ii, :], y_target, controls_targ)

        agent_controls[:, zz*du:(zz+1)*du] = agent_zz_controls

    return agent_controls

# 2d or 3d identical agent/target double integrators
def post_process_identical_doubleint_TEST(parameters, sim_results):

    df = sim_results['data']
    poltrack = sim_results['tracking_policy']
    poltargets = sim_results['target_pol']
    nagents = sim_results['nagents']
    ntargets = sim_results['ntargets']
    ot_costs = sim_results['asst_cost']
    polagents = sim_results['agent_pol']
    opt_asst = sim_results['optimal_asst']
    asst_policy = sim_results['asst_policy']

    dt = parameters['dt']
    dim = parameters['dim']
    dx = parameters['dx']
    du = parameters['du']
    collisions = parameters['collisions']

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()

    yout = copy.deepcopy(yout)
    assignment_switches = find_switches(tout, yout, nagents, nagents, dx, dx)

    print("INITIAL CONDITION: ", yout[0])

    # assignments = yout[:, nagents*2*4:].astype(np.int32)
    assignments = yout[:, nagents*2*dx:].astype(np.int32)

    # TEST
    test = compute_controls(dx, du, yout, tout, assignments, nagents, poltargets, polagents)

    # PLOT COSTS
    final_cost = np.zeros((tout.shape[0], nagents))
    stage_cost = np.zeros((tout.shape[0], nagents))
    xp = np.zeros((yout.shape[0], nagents))
    optimal_cost = np.zeros((1, nagents))

    xss = np.zeros((yout.shape[0], nagents*2*dx))
    for zz in range(nagents):
        y_agent = yout[:, zz*dx:(zz+1)*dx]

        # COMPUTE CONTROLS
        # yout, assignments, nagents, dx, poltargets, polagents, 
        controls = np.zeros((yout.shape[0], du))
        for ii in range(yout.shape[0]): # compute controls
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # AUGMENTED TRACKER
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)

            # NEW
            # Get agent policy in correct tracking state for P, Q, p at time ii
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            controls[ii, :] = polagents[zz].evaluate(tout[ii], y_agent[ii, :], y_target, controls_targ)

        # COSTS

        # post-process for t=0
        y_target = yout[0, (assignments[0][zz]+nagents)*dx:(assignments[0][zz]+nagents+1)*dx]

        # Get agent policy in correct tracking state for P, Q, p at t=0
        asst_0 = assignments[0] # assignments at time ii
        sigma_i = asst_0[zz] # target assigned-to agent zz
        Acl_0 = poltargets[sigma_i].get_closed_loop_A()
        gcl_0 = poltargets[sigma_i].get_closed_loop_g()
        polagents[zz].track(0, sigma_i, Acl_0, gcl_0)

        R = polagents[zz].get_R()
        Q_0 = polagents[zz].get_Q()
        P_0 = polagents[zz].get_P()
        p_0 = polagents[zz].get_p()

        uss_0 = polagents[zz].get_uss()
        Xss_0 = polagents[zz].get_xss()
        X_0 = np.hstack((y_agent[0, :], y_target))
        xp[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        stage_cost[0, zz] = np.dot(X_0, np.dot(Q_0, X_0)) + np.dot(controls[0, :], np.dot(R, controls[0, :])) -\
            (np.dot(Xss_0, np.dot(Q_0, Xss_0)) + np.dot(uss_0, np.dot(R, uss_0)))

        # optimal cost (ie. DYN)
        opt_asst_y_target = yout[0, (opt_asst[zz]+nagents)*dx:(opt_asst[zz]+nagents+1)*dx]
        X_0 = np.hstack((y_agent[0, :], opt_asst_y_target))

        # Get agent policy in correct tracking state for P, Q, p at t=0
        optasst_0 = opt_asst # assignments at time ii
        optasst_sigma_i = asst_0[zz] # target assigned-to agent zz
        optasst_Acl_0 = poltargets[optasst_sigma_i].get_closed_loop_A()
        optasst_gcl_0 = poltargets[optasst_sigma_i].get_closed_loop_g()
        polagents[zz].track(0, optasst_sigma_i, optasst_Acl_0, optasst_gcl_0)

        R = polagents[zz].get_R()
        optasst_Q_0 = polagents[zz].get_Q()
        optasst_P_0 = polagents[zz].get_P()
        optasst_p_0 = polagents[zz].get_p()

        optasst_uss_0 = polagents[zz].get_uss()
        optasst_Xss_0 = polagents[zz].get_xss()
        optimal_cost[0, zz] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        # continue post-processing for rest of time points
        for ii in range(1, yout.shape[0]):
            y_target = yout[ii, (assignments[ii][zz]+nagents)*dx:(assignments[ii][zz]+nagents+1)*dx]

            # TEST
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            controls_targ = poltargets[sigma_i].evaluate(tout[ii], y_target)
            X = np.hstack((y_agent[ii, :], y_target))

            # Get agent policy in correct tracking state for P, Q, p at time ii
            asst_ii = assignments[ii] # assignments at time ii
            sigma_i = asst_ii[zz] # target assigned-to agent zz
            Acl = poltargets[sigma_i].get_closed_loop_A()
            gcl = poltargets[sigma_i].get_closed_loop_g()
            polagents[zz].track(ii, sigma_i, Acl, gcl)

            R = polagents[zz].get_R()
            Q = polagents[zz].get_Q()
            P = polagents[zz].get_P()
            p = polagents[zz].get_p()

            # STEADY-STATE TERMS
            uss = polagents[zz].get_uss()
            Xss = polagents[zz].get_xss()

            # STAGE COST
            stage_cost[ii, zz] = np.dot(X, np.dot(Q, X)) + np.dot(controls[ii, :], np.dot(R, controls[ii, :])) -\
                (np.dot(Xss, np.dot(Q, Xss)) + np.dot(uss, np.dot(R, uss)))

            # COST-TO-GO
            xp[ii, zz] = np.dot(X, np.dot(P, X)) + 2*np.dot(p, X) -\
                (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

        for ii in range(tout.shape[0]):
            final_cost[ii, zz] = np.trapz(stage_cost[:ii, zz], x=tout[:ii])

    optcost = np.sum(optimal_cost[0, :])

    print("POLICY: ", poltrack.__class__.__name__)
    print("FINAL TIME: ", tout[-1])
    print("initial optimal cost: ", optcost)
    print("initial incurred cost: ", final_cost[0])
    print("final cost-to-go value: ", np.sum(xp, axis=1)[-1])
    print("final incurred cost value: ", np.sum(final_cost, axis=1)[-1]) # last stage cost
    print("initial optimal cost - final incurred cost value = ", optcost - np.sum(final_cost, axis=1)[-1])
    print("INITIAL CONDITIONS")
    print(yout[0, :])
    print("FINAL STATE")
    print(yout[-1, :])
    print("OFFSET")
    for pt in poltargets:
        print(pt.const)

    # final dataset = dim, dx, du, nagents, ntargets, yout, tout, final_cost, stage_cost, cost_to_go, optimal_cost, city states
    columns = ['dim', 'dx', 'du', 'nagents', 'ntargets', 'tout', 'yout', 'city_states', 'final_cost', 'stage_cost',
            'cost_to_go', 'optimal_cost']
    # eng.df = [tout, yout, asst history] dataframe

    #### PACK INTO SINGLE DATAFRAME
    if collisions:
        col_df = pd.DataFrame([1])
    else:
        col_df = pd.DataFrame([0])

    dt_df = pd.DataFrame([dt])
    dim_df = pd.DataFrame([dim])
    dx_df = pd.DataFrame([dx])
    du_df = pd.DataFrame([du])
    nagents_df = pd.DataFrame([nagents])
    ntargets_df = pd.DataFrame([ntargets])
    parameters_df = pd.concat([dt_df, dim_df, col_df, dx_df, du_df, nagents_df, ntargets_df], axis=1)

    fc_df = pd.DataFrame(final_cost)
    sc_df = pd.DataFrame(stage_cost)
    ctg_df = pd.DataFrame(xp)
    oc_df = pd.DataFrame(optimal_cost)
    costs_df = pd.concat([fc_df, sc_df, ctg_df, oc_df], axis=1)

    cities = np.zeros((1, ntargets*dx))
    for jj in range(ntargets):
        cities[0, jj*dx:(jj+1)*dx] = poltargets[jj].const
    stationary_states_df = pd.DataFrame(cities)

    controls_df = pd.DataFrame(compute_controls(dx, du, yout, tout, assignments, nagents, poltargets, polagents))

    outputs_df = pd.concat([df, stationary_states_df, controls_df], axis=1)

    return_df = pd.concat([parameters_df, outputs_df, costs_df], axis=1)

    return return_df

    # return yout, tout, assignments, assignment_switches, final_cost, stage_cost, xp, optcost


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



