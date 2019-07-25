
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker

import post_process

# TeX fonts
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


# TODO move plotting functions to plot.py

def plot_costs(unpacked):

    fontsize = 14
    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('time (s)', fontsize=fontsize)
    # axs.set_ylabel('Cost', fontsize=fontsize)

    # TODO normalized costs
    axs.set_ylabel('Normalized Cost', fontsize=fontsize)
    # axs.set_title('Cost VS Time')
    for sim_name, metrics in unpacked.items():

        tout = metrics['tout']
        yout = metrics['yout']
        final_cost = metrics['final_cost']
        cost_to_go = metrics['cost_to_go']
        optimal_cost = metrics['optimal_cost']

        summed_opt_cost = np.sum(optimal_cost[0, :])

        ### cost plots
        if sim_name == 'AssignmentDyn':
            # axs.plot(tout, summed_opt_cost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')
            # axs.plot(tout, np.sum(final_cost, axis=1), '--c', label='Cum. Stage Cost'+' '+sim_name)
            # axs.plot(tout, np.sum(cost_to_go, axis=1), '--r', label='Cost-to-go'+' '+sim_name)

            # TODO normalized costs
            axs.plot(tout, np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')
            axs.plot(tout, np.sum(final_cost, axis=1)/summed_opt_cost, '--c', label='Cum. Stage Cost'+' '+sim_name)
            axs.plot(tout, np.sum(cost_to_go, axis=1)/summed_opt_cost, '--r', label='Cost-to-go'+' '+sim_name)
        else:
            # axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost'+' '+sim_name)
            ## axs.plot(tout, np.sum(cost_to_go, axis=1), '-r', label='Cost-to-go'+' '+sim_name)

            # TODO normalized costs
            axs.plot(tout, np.sum(final_cost, axis=1)/summed_opt_cost, '-c', label='Cum. Stage Cost'+' '+sim_name)

        axs.legend(fontsize=fontsize)

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

def plot_cost_histogram(unpacked_ensemble_metric):

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Final Cost - Optimal Cost', fontsize=14)
    axs.set_ylabel('Frequency', fontsize=14)

    axs.hist(unpacked_ensemble_metric)

def plot_asst_histogram(unpacked_ensemble_metric):

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Assignment Switches', fontsize=14)
    axs.set_ylabel('Frequency', fontsize=14)

    axs.hist(unpacked_ensemble_metric)

def plot_assignments(unpacked):

    for sim_name, metrics in unpacked.items():

        dx = metrics['dx']
        nagents = metrics['nagents']
        ntargets = metrics['ntargets']
        tout = metrics['tout']
        yout = metrics['yout']

        assignments = yout[:, nagents*2*dx:].astype(np.int32)
        assignment_switches = post_process.find_switches(tout, yout, nagents, nagents, dx, dx)

        # recreate assignments per switch
        asst_switch_indices = set()
        asst_switch_indices.add(0) # add the origin assignment
        for ii in range(nagents):
           switch_indices = assignment_switches[ii]
           for ind in switch_indices:
               asst_switch_indices.add(ind)

        # order the switch time
        asst_switch_indices = sorted(asst_switch_indices) # becomes ordered list

        # get assignment switches in increasing time order
        asst_to_plot = np.zeros((len(asst_switch_indices), nagents)) # (starting assignment + switches)
        asst_to_plot[0, :] = assignments[0, :]
        for tt, ind in enumerate(asst_switch_indices):
            asst_to_plot[tt, :] = assignments[ind, :]

        # PLOT TOO BUSY, deprecate
        plt.figure()
        # plt.title("Agent-Target Assignments")
        plt.xlabel('time (s)')
        plt.ylabel('Assigned-to Target')
        for ii in range(nagents):
            plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))
            plt.legend()

        # TEST
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # fig, ax = plt.subplots()

        ax.set_title(sim_name)

        asst_array = np.zeros((nagents, tout.shape[0], ntargets)) # want to show propogation of assignment over time in y-axis

        # construct assignment array
        for tt in range(tout.shape[0]):
            time = tout[tt]
            for ii in range(nagents): # iterate consecutively through agents
                # ax.plot3D(agent_i, tout, target_j, '-r', label=agent_traj_label)
                jj = assignments[tt, ii]
                asst_array[ii, tt, jj] = 1
                # change color and marker if there's a switch


        # # stack plots on top of each other
        # agents = np.arange(nagents)
        # for asst_num, (switch_ind, assignment) in enumerate(zip(asst_switch_indices, asst_to_plot)):
        #     assigned_to_targets = assignment
        #     # ax.plot(agents, assigned_to_targets, marker='s', label='Assignment{0}'.format(asst_num))
        #     ax.plot(agents, assigned_to_targets, label='Assignment{0}'.format(asst_num))
        #     # if sim_name != 'AssignmentDyn':
        #     #     ax.fill_between(agents, assigned_to_targets, asst_to_plot[1], color='blue')
        # ax.set_xlabel('agents')
        # ax.set_ylabel('targets')
        # ax.legend()


        # plot 2d assignment plots in 3d at correct time step
        cumulative_asst_label = 'Cumulative Assignment Projection'
        agents = np.arange(nagents)
        for asst_num, (switch_ind, assignment) in enumerate(zip(asst_switch_indices, asst_to_plot)):
            switch_time = tout[switch_ind]
            assigned_to_targets = assignment

            if asst_num >= 1:
                cumulative_asst_label = '__nolabel__'
            ax.plot(agents, assigned_to_targets, tout[-1], zdir='y', color='blue', label=cumulative_asst_label)

            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(agents, assigned_to_targets, switch_time, '-s', color=color, zdir='y', label='Assignment{0}'.format(asst_num))
            ax.scatter(agents, assigned_to_targets, tout[-1], color=color, zdir='y')
            ax.add_collection3d(plt.fill_between(agents, assigned_to_targets, asst_to_plot[0], color='blue'), zs=tout[-1], zdir='y')


        ax.set_xlabel('agents')
        ax.set_ylabel('time (s)')
        ax.set_zlabel('targets')
        ax.legend()
        ax.set_ylim3d(0, tout[-1])
        ax.xaxis.set_ticks(np.arange(nagents))
        ax.zaxis.set_ticks(np.arange(ntargets))



        # import ipdb; ipdb.set_trace()
        # asst_density_grid = np.sum(asst_array, axis=1) # sum over time
        # ax.imshow(asst_density_grid, cmap='hot')

        # for tt in range(tout.shape[0]):
        #     ax.imshow(asst_array[:, tt, :])


        # agents = np.arange(nagents)
        # assigned_to_targets = assignments[:, agents]
        # # X, Y = np.meshgrid(agents, tout)
        # # ax.contour(X, Y, assigned_to_targets)
        # X, Y = np.meshgrid(tout, agents)
        # ax.contour(X, Y, assigned_to_targets.T)
        # ax.set_xlabel('time (s)')
        # ax.set_ylabel('agent')


        # agents = np.arange(nagents)
        # assigned_to_targets = assignments[:, agents]
        # X, Y = np.meshgrid(agents, assigned_to_targets)
        # U, V = np.meshgrid(assigned_to_targets, tout)
        # # import ipdb; ipdb.set_trace()
        # ax.contour(X, Y, V)

        # agents = np.arange(nagents)
        # assigned_to_targets = assignments[:, agents]
        # X, Y = np.meshgrid(tout, agents)
        # ax.contourf3D(X, Y, assigned_to_targets.T)
        # ax.plot_surface(X, Y, assigned_to_targets.T)
        # ax.set_xlabel('time (s)')
        # ax.set_ylabel('agent')
        # ax.set_zlabel('assigned-to target')

        # import ipdb; ipdb.set_trace()

        # for ii in range(nagents):
        #     jj = assignments[:, ii]
        #     ax.scatter(ii, tout, jj)

        # import ipdb; ipdb.set_trace()


def plot_trajectory(unpacked):

    dim = 2 # default value

    # update dim
    for sim_name, metrics in unpacked.items():
        dim = metrics['dim']

    # want to display all trajectories on same figure
    fontsize = 26
    fontweight = 'bold'
    labelsize = 26

    if dim == 2:
        fig, ax = plt.subplots()
    if dim == 3:
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        ax = plt.axes(projection='3d')

        # TEST
        # TODO 2d slice
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

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

        assignment_switches = post_process.find_switches(tout, yout, nagents, ntargets, dx, dx)

        agent_traj_label = 'Agent Trajectory - AssignmentDyn'
        agent_start_pt_label = 'Agent Start Position'
        target_start_pt_label = 'Target Start Position'
        target_traj_label = 'Target Trajectory'
        stationary_pt_label = 'Target Terminal Positions'

        # TEST # TODO REMOVE EVENTUALLY
        if dx == 12:
            agent_model = 'Linearized_Quadcopter'
            target_model = 'Linearized_Quadcopter'
            labels = [agent_traj_label, agent_start_pt_label, target_start_pt_label, target_traj_label, stationary_pt_label]
            plot_trajectory_qc(fig, ax, sim_name, dx, du, dim, nagents, ntargets, tout, yout, stationary_states,
                    assignment_switches, labels)
            continue
        if dx == 6:
            agent_model = 'Double_Integrator'
            target_model = 'Double_Integrator'

        if dim == 2: # and agent/target models both double integrator (omit requirement for now)

            ### Agent / Target Trajectories
            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentDyn':

                for zz in range(nagents):

                    if zz >= 1:
                        agent_traj_label = '__nolabel__'
                        agent_start_pt_label = '__nolabel__'
                        target_start_pt_label = '__nolabel__'
                        target_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs', label=agent_start_pt_label)
                    ax.plot(y_agent[:, 0], y_agent[:, 1], '-r', label=agent_traj_label)
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
                    ax.plot(y_target[0, 0], y_target[0, 1], 'bs', label=target_start_pt_label)
                    ax.plot(y_target[:, 0], y_target[:, 1], '-b', label=target_traj_label)
                    ax.text(y_target[0, 0], y_target[0, 1], 'T{0}'.format(zz))

                ### stationary points
                for zz in range(ntargets):

                    if zz >= 1:
                        stationary_pt_label = '__nolabel__'

                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.plot(offset[0], offset[1], 'ks', label=stationary_pt_label)
                    ax.text(offset[0], offset[1], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)

            elif sim_name == 'AssignmentEMD':

                agent_traj_label = 'Agent Trajectory - AssignmentEMD'

                # non-optimal trajectories (dotted lines)
                for zz in range(nagents):

                    if zz >= 1:
                        agent_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
                    ax.plot(y_agent[:, 0], y_agent[:, 1], '--r', label=agent_traj_label)
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

                ### stationary points
                for zz in range(ntargets):

                    if zz >= 1:
                        stationary_pt_label = '__nolabel__'

                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.plot(offset[0], offset[1], 'ks')
                    ax.text(offset[0], offset[1], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)

            # dim == 2
            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)

        # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
        ax.legend(loc='lower right', fontsize=14)

        if dim == 3:

            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentDyn':

                # agent/target trajectories
                for zz in range(nagents):

                    # avoid repeated legend entries
                    if zz >= 1:
                        agent_traj_label = '__nolabel__'
                        agent_start_pt_label = '__nolabel__'
                        target_start_pt_label = '__nolabel__'
                        target_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r', label=agent_start_pt_label)
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r', label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(y_agent[0, 0], y_agent[0, 1], 'rs', label=agent_start_pt_label)
                    ax2.plot(y_agent[:, 0], y_agent[:, 1], '-r', label=agent_traj_label)
                    ax2.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b', label=target_start_pt_label)
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b', label=target_traj_label)
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(y_target[0, 0], y_target[0, 1], 'bs', label=target_start_pt_label)
                    ax2.plot(y_target[:, 0], y_target[:, 1], '-b', label=target_traj_label)
                    ax2.text(y_target[0, 0], y_agent[0, 1], 'T{0}'.format(zz))

                ### stationary points
                for zz in range(ntargets):

                    if zz >= 1:
                        stationary_pt_label = '__nolabel__'

                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k', label=stationary_pt_label)
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(offset[0], offset[1], 'ks', label=stationary_pt_label)
                    ax2.text(offset[0], offset[1], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

                ax2.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax2.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)

            elif sim_name == 'AssignmentEMD':
                # non-optimal trajectories (dotted lines)

                agent_traj_label = 'Agent Trajectory - AssignmentEMD'

                # agent/target trajectories
                for zz in range(nagents):

                    # avoid repeated legend entries
                    if zz >= 1:
                        agent_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '--r', label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
                    ax2.plot(y_agent[:, 0], y_agent[:, 1], '--r', label=agent_traj_label)
                    ax2.text(y_agent[0, 0], y_agent[0, 1], 'A{0}'.format(zz))

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(y_target[0, 0], y_target[0, 1], 'bs')
                    ax2.plot(y_target[:, 0], y_target[:, 1], '-b')
                    ax2.text(y_target[0, 0], y_agent[0, 1], 'T{0}'.format(zz))

                # stationary locations
                for zz in range(ntargets):
                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k')
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                    # TEST
                    # TODO 2d slice
                    ax2.plot(offset[0], offset[1], 'ks')
                    ax2.text(offset[0], offset[1], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize+4)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize+4)
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize+4)

                ax2.set_xlabel("x", fontweight=fontweight, fontsize=fontsize+4)
                ax2.set_ylabel("y", fontweight=fontweight, fontsize=fontsize+4)

            # dim = 3

            tick_spacing = 1000
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            ax.xaxis.set_tick_params(labelsize=labelsize+4)
            ax.yaxis.set_tick_params(labelsize=labelsize+4)
            ax.zaxis.set_tick_params(labelsize=labelsize+4)

            ax.tick_params(axis='x', which='major', pad=15)
            ax.tick_params(axis='y', which='major', pad=15)
            ax.tick_params(axis='z', which='major', pad=15)

            ax.xaxis.labelpad = 25
            ax.yaxis.labelpad = 25
            ax.zaxis.labelpad = 25

            # TEST
            # TODO 2d slice
            ax2.xaxis.set_tick_params(labelsize=labelsize)
            ax2.yaxis.set_tick_params(labelsize=labelsize)

        # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
        # ax.legend(loc='lower right', fontsize=fontsize)
        ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize+4)
        # ax2.legend(loc='lower right', fontsize=fontsize)
        ax2.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize+4)

# ************ TEST LINEARIZED QC ***************
def plot_trajectory_qc(fig, ax, fontsize, fontweight, labelsize, sim_name, dx, du, dim, nagents, ntargets, tout, yout, stationary_states,
        assignment_switches, labels):

    agent_traj_label = labels[0]
    agent_start_pt_label = labels[1]
    target_start_pt_label = labels[2]
    target_traj_label = labels[3]
    stationary_pt_label = labels[4]

    if dim == 3:

        # optimal trajectories (solid lines)
        if sim_name == 'AssignmentDyn':

            # agent/target trajectories
            for zz in range(nagents):

                # avoid repeated legend entries
                if zz >= 1:
                    agent_traj_label = '__nolabel__'
                    agent_start_pt_label = '__nolabel__'
                    target_start_pt_label = '__nolabel__'
                    target_traj_label = '__nolabel__'

                # agent state over time
                y_agent = yout[:, zz*dx:(zz+1)*dx]

                # plot agent trajectory with text
                ax.scatter3D(y_agent[0, 9], y_agent[0, 10], y_agent[0, 11], color='r', label=agent_start_pt_label)
                ax.plot3D(y_agent[:, 9], y_agent[:, 10], y_agent[:, 11], '-r', label=agent_traj_label)
                ax.text(y_agent[0, 9], y_agent[0, 10], y_agent[0, 11], 'A{0}'.format(zz))

                # # plot location of assignment switches
                # for switch_ind in assignment_switches[zz]:
                #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                # plot target trajectory
                y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                ax.scatter3D(y_target[0, 9], y_target[0, 10], y_target[0, 11], color='b', label=target_start_pt_label)
                ax.plot3D(y_target[:, 9], y_target[:, 10], y_target[:, 11], '-b', label=target_traj_label)
                ax.text(y_target[0, 9], y_target[0, 10], y_target[0, 11], 'T{0}'.format(zz))

            ### stationary points
            for zz in range(ntargets):

                if zz >= 1:
                    stationary_pt_label = '__nolabel__'

                offset = stationary_states[zz*dx:(zz+1)*dx]
                ax.scatter3D(offset[9], offset[10], offset[11], color='k', label=stationary_pt_label)
                ax.text(offset[9], offset[10], offset[11], 'C{0}'.format(zz))

            ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
            ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

        elif sim_name == 'AssignmentEMD':
            # non-optimal trajectories (dotted lines)

            agent_traj_label = 'Agent Trajectory - AssignmentEMD'

            # agent/target trajectories
            for zz in range(nagents):

                # avoid repeated legend entries
                if zz >= 1:
                    agent_traj_label = '__nolabel__'

                # agent state over time
                y_agent = yout[:, zz*dx:(zz+1)*dx]

                # plot agent trajectory with text
                ax.scatter3D(y_agent[0, 9], y_agent[0, 10], y_agent[0, 11], color='r')
                ax.plot3D(y_agent[:, 9], y_agent[:, 10], y_agent[:, 11], '--r', label=agent_traj_label)
                ax.text(y_agent[0, 9], y_agent[0, 10], y_agent[0, 11], 'A{0}'.format(zz))

                # # plot location of assignment switches
                # for switch_ind in assignment_switches[zz]:
                #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                # plot target trajectory
                y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                ax.scatter3D(y_target[0, 9], y_target[0, 10], y_target[0, 11], color='b')
                ax.plot3D(y_target[:, 9], y_target[:, 10], y_target[:, 11], '-b')
                ax.text(y_target[0, 9], y_target[0, 10], y_target[0, 11], 'T{0}'.format(zz))

            # stationary locations
            for zz in range(ntargets):
                offset = stationary_states[zz*dx:(zz+1)*dx]
                ax.scatter3D(offset[9], offset[10], offset[11], color='k')
                ax.text(offset[9], offset[10], offset[11], 'C{0}'.format(zz))

            ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
            ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)
            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)
            ax.zaxis.set_tick_params(labelsize=labelsize)

        # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
        ax.legend(loc='lower right', fontsize=13)

def plot_assignment_comp_time(unpacked):

    for sim_name, sim_diagnostics in unpacked.items():

        runtime_diagnostics = sim_diagnostics['runtime_diagnostics']

        tout = runtime_diagnostics.iloc[:, 0].to_numpy()
        assign_comp_cost = runtime_diagnostics.iloc[:, 1].to_numpy()
        dynamics_comp_cost = runtime_diagnostics.iloc[:, 2].to_numpy()

        fig, axs = plt.subplots(1,1)
        axs.plot(tout, np.cumsum(assign_comp_cost))
        axs.set_xlabel('time (s)')
        axs.set_ylabel('assignment cumulative computational cost (s)')
        axs.set_title(sim_name)


