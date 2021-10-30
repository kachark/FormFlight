
""" @file ploy.py
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
import scipy.stats as sts

from . import post_process

# # TeX fonts
# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# # matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# # matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
# rc('font', size=14)
# rc('legend', fontsize=13)
# rc('text.latex', preamble=r'\usepackage{cmbright}')


def plot_costs(unpacked):

    """ Plots costs

    DATE: 02/07/2020

    """

    linewidth = 2

    labelsize = 18
    fontsize = 18
    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Time (s)', fontsize=fontsize)
    # axs.set_ylabel('Cost', fontsize=fontsize)

    axs.set_ylabel('Normalized Cost', fontsize=fontsize)
    # axs.set_title('Cost VS Time')

    unpacked_worlds = unpacked[0]
    unpacked_batch_metrics = unpacked[1]

    emd = 'emd'
    emd_cost_to_go = unpacked_batch_metrics[emd]['cost_to_go']
    # summed_opt_cost = np.sum(opt_cost_to_go[0, :])
    summed_opt_cost = 10E9

    for (sim_name, world), (_, metrics) in zip(unpacked_worlds.items(),
            unpacked_batch_metrics.items()):

        tout = metrics['tout']
        yout = metrics['yout']
        final_cost = metrics['final_cost']
        cost_to_go = metrics['cost_to_go']

        label = sim_name

        # normalized costs
        axs.plot(tout, np.sum(final_cost, axis=1)/summed_opt_cost, '-c', linewidth=linewidth,
                label='Cum. Stage Cost'+' '+label)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    # t = axs.yaxis.get_offset_text()
    # t.set_size(fontsize)

    # reorder the legend terms
    # handles, labels = axs.get_legend_handles_labels()
    # labels = [labels[1], labels[0], labels[2], labels[3]]
    # handles = [handles[1], handles[0], handles[2], handles[3]]

    # axs.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.25), fontsize=fontsize)

    # Agent-by-agent cost plots on 1 figure
    # plt.figure()
    # for sim_name, metrics in unpacked.items():

    #     nagents = metrics['nagents']
    #     tout = metrics['tout']
    #     final_cost = metrics['final_cost']
    #     cost_to_go = metrics['cost_to_go']

    #     for zz in range(nagents):
    #         plt.plot(tout, final_cost[:, zz], '-.c', label='Cum. Stage Cost ({0})'.format(zz))
    #         plt.plot(tout, cost_to_go[:, zz], '-.r', label='Cost-to-go (assuming no switch) ({0})'.format(zz))

    #     plt.legend()
    plt.tight_layout()
    plt.savefig("normalized_cost.pdf", bbox_inches='tight')

def plot_cost_histogram(unpacked_ensemble_metric):

    """ Plots histogram of costs
    """

    fontsize = 32
    labelsize = 32

    labels = ['Dyn', 'EMD']

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Control Expenditure Difference (EMD - Dyn)/Dyn', fontsize=fontsize)
    axs.set_ylabel('Frequency', fontsize=fontsize)

    axs.hist(unpacked_ensemble_metric, histtype='bar', stacked=True, bins=10, align='left', label=labels)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    axs.xaxis.offsetText.set_fontsize(fontsize)

    axs.legend(fontsize=fontsize)

# TODO move to a different file
def atoi(text):
    return int(text) if text.isdigit() else text

# TODO move to a different file
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def plot_ensemble_cost_histogram(metrics_to_compare):

    """ Plots histogram of agent swarm LQ costs for multiple ensembles
    """

    fontsize = 40
    labelsize = 40

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Control Expenditure Difference (EMD - Dyn)/Dyn', fontsize=fontsize)
    axs.set_ylabel('Frequency', fontsize=fontsize)

    # Using DataFrames
    labels = []
    for ensemble_name in metrics_to_compare.keys():
        labels.append(re.search('\d+v\d+', ensemble_name).group())

    metrics_df = pd.DataFrame.from_dict(metrics_to_compare)
    metrics_df.columns = labels

    # order data by number of agents
    labels.sort(key=natural_keys)
    metrics_df = metrics_df[labels]

    for i, (label, data) in enumerate(metrics_df.iteritems()):
        nbins = int(len(data)/4)
        data.hist(ax=axs, bins=nbins, align='left', edgecolor='k', alpha=0.5, label=label)
        # data.plot.kde(ax=axs)

    axs.grid(False)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    axs.xaxis.offsetText.set_fontsize(fontsize)

    axs.legend(fontsize=fontsize)

def plot_assignments(unpacked):

    """ Plots assignments
    """

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

def plot_ensemble_switch_histogram(metrics_to_compare):

    """ Plots histogram of assignment switches for multiple ensembles
    """

    fontsize = 40
    labelsize = 40

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Assignment Switches', fontsize=fontsize)
    axs.set_ylabel('Frequency', fontsize=fontsize)

    # Using DataFrames
    labels = []
    for ensemble_name in metrics_to_compare.keys():
        labels.append(re.search('\d+v\d+', ensemble_name).group())

    metrics_df = pd.DataFrame.from_dict(metrics_to_compare)
    metrics_df.columns = labels

    # order data by number of agents
    labels.sort(key=natural_keys)
    metrics_df = metrics_df[labels]

    for i, (label, data) in enumerate(metrics_df.iteritems()):
        nbins = int(len(data)/4)
        data.hist(ax=axs, bins=nbins, align='left', edgecolor='k', alpha=0.5, label=label)
        # data.plot.kde(ax=axs)

    axs.grid(False)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    tick_spacing = 1
    axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    axs.xaxis.offsetText.set_fontsize(fontsize)

    axs.legend(fontsize=fontsize)

def plot_ensemble_avg_switch(metrics_to_compare):

    """ Plots average number of assignment switches over time for multiple ensembles
    """

    fontsize = 40
    labelsize = 40

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Agents', fontsize=fontsize)
    axs.set_ylabel('Average \# Assign. Switches', fontsize=fontsize)

    # Using DataFrames
    labels = []
    for ensemble_name in metrics_to_compare.keys():
        labels.append(re.search('\d+v\d+', ensemble_name).group())

    metrics_df = pd.DataFrame(metrics_to_compare, index=[0])
    metrics_df.columns = labels

    # order data by number of agents
    labels.sort(key=natural_keys)
    metrics_df = metrics_df[labels]

    metrics = {'Ensemble': labels, 'Average Assignment Switches': metrics_df.values.tolist()[0]}
    metrics_df = pd.DataFrame(metrics)

    # metrics_df.plot.bar(x='Ensemble', rot=0, fontsize=fontsize)

    values = metrics_df['Average Assignment Switches'].values.tolist()
    xpos = [i for i, _ in enumerate(labels)]

    axs.bar(xpos, values, alpha=0.5)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    # tick_spacing = 1
    # axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # axs.xaxis.offsetText.set_fontsize(fontsize)

    axs.set_xticks(xpos)
    axs.set_xticklabels(labels)

    # axs.legend(fontsize=fontsize)

def plot_trajectory(unpacked):

    """ Plots trajectory in 2D or 3D for homogeneous identical double integrator and linearized quadcopters

    DATE: 02/07/2020

    """

    unpacked_worlds = unpacked[0]
    unpacked_batch_metrics = unpacked[1]

    dim = 2 # default value

    # update dim
    for _, metrics in unpacked_batch_metrics.items():
        dim = metrics['dim']

    # want to display all trajectories on same figure
    linewidth_3d = 2
    linewidth = 2
    markersize = 8
    scatter_width = markersize**2
    textsize = 16

    fontsize = 16
    fontweight = 'light'
    labelsize = 20

    axispad = 18
    labelpad = 20

    ax = None
    if dim == 2:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    if dim == 3:
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        ax = plt.axes(projection='3d')

        # TEST
        # TODO 2d slice
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

    optimal_name = 'dyn'

    for (sim_name, world), (_, metrics) in zip(unpacked_worlds.items(),
            unpacked_batch_metrics.items()):

        dim = metrics['dim']
        tout = metrics['tout']
        yout = metrics['yout']

        # TODO assumes scenario='Intercept'
        agent_mas = world.get_multi_object('Agent_MAS')
        target_mas = world.get_multi_object('Target_MAS')
        nagents = agent_mas.nagents
        ntargets = target_mas.nagents
        total_mas_dx = np.sum([agent.dx for agent in agent_mas.agent_list])
        total_mas_du = np.sum([agent.du for agent in agent_mas.agent_list])
        total_target_dx = np.sum([target.dx for target in target_mas.agent_list])
        total_target_du = np.sum([target.du for target in target_mas.agent_list])

        # assignment_switches = post_process.find_switches(tout, yout, nagents, ntargets, dx, dx)

        agent_traj_label = 'Agent Path (Dyn)'
        agent_start_pt_label = 'Agent Start'
        target_start_pt_label = 'Target Start'
        target_traj_label = 'Target Path'
        stationary_pt_label = 'Terminal State'

        # # TEST # TODO REMOVE EVENTUALLY
        # if dx == 12:
        #     agent_model = 'Linearized_Quadcopter'
        #     target_model = 'Linearized_Quadcopter'
        #     labels = [agent_traj_label, agent_start_pt_label, target_start_pt_label, target_traj_label, stationary_pt_label]
        #     plot_params = [linewidth, linewidth_3d, markersize, scatter_width, textsize, fontsize, fontweight, labelsize, axispad, labelpad]
        #     figures = [(fig, ax), (fig2, ax2)]
        #     plot_trajectory_qc(figures, plot_params, sim_name, dx, du, dim, nagents, ntargets, tout, yout, stationary_states,
        #             assignment_switches, labels)
        #     continue
        # if dx == 6:
        #     agent_model = 'Double_Integrator'
        #     target_model = 'Double_Integrator'

        ### Agent / Target Trajectories
        # optimal trajectories (solid lines)
        if sim_name == 'dyn':

            for agent in agent_mas.agent_list:

                dyn_model = agent.type

                statespace = agent.get_statespace()
                agent_dim_pos = statespace['position']
                x_pos = agent_dim_pos[0]
                y_pos = agent_dim_pos[1]
                z_pos = None
                if dim == 3:
                    z_pos = agent_dim_pos[2]

                # only need one label representing 'agents'
                if agent.ID >= 1:
                    agent_traj_label = '__nolabel__'
                    agent_start_pt_label = '__nolabel__'

                # agent state over time
                ag_start_ind, ag_end_ind = world.get_object_world_state_index(agent.ID)
                agent_state_history = yout[:, ag_start_ind:ag_end_ind]
                y_agent = agent_state_history

                # plot agent trajectory with text
                if dim == 2:
                    ax.plot(y_agent[0, x_pos], y_agent[0, y_pos], 'rs', markersize=markersize,
                            label=agent_start_pt_label)
                    ax.plot(y_agent[:, x_pos], y_agent[:, y_pos], '-r', linewidth=linewidth,
                            label=agent_traj_label)
                    ax.text(y_agent[0, x_pos], y_agent[0, y_pos], 'A{0}'.format(agent.ID),
                            fontsize=textsize)
                elif dim == 3:
                    ax.scatter3D(y_agent[0, x_pos], y_agent[0, y_pos], y_agent[0, z_pos], color='r',
                            s=scatter_width, label=agent_start_pt_label)
                    ax.plot3D(y_agent[:, x_pos], y_agent[:, y_pos], y_agent[:, z_pos], '-r',
                            linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, x_pos], y_agent[0, y_pos], y_agent[0, z_pos],
                            'A{0}'.format(agent.ID), fontsize=textsize)

                    # >> 2d slice view <<
                    ax2.plot(y_agent[:, x_pos], y_agent[:, y_pos], '-r', linewidth=linewidth,
                            label=agent_traj_label)
                    # points
                    ax2.plot(y_agent[0, x_pos], y_agent[0, y_pos], 'ro', markersize=markersize,
                            label=agent_start_pt_label)
                    # text
                    ax2.text(y_agent[0, x_pos], y_agent[0, y_pos], 'A{0}'.format(agent.ID),
                            fontsize=textsize)

                # # plot location of assignment switches
                # patches = []
                # for switch_ind in assignment_switches[zz]:
                #     ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='b', fill=True)
                #     patches.append(ci)
                # p = PatchCollection(patches)
                # ax.add_collection(p)

            for target in target_mas.agent_list:

                dyn_model = target.type

                statespace = target.get_statespace()
                target_dim_pos = statespace['position']
                x_pos = target_dim_pos[0]
                y_pos = target_dim_pos[1]
                z_pos = None
                if dim == 3:
                    z_pos = target_dim_pos[2]

                # only need one label representing 'agents'
                if target.ID >= 1:
                    target_start_pt_label = '__nolabel__'
                    target_traj_label = '__nolabel__'

                # agent state over time
                target_start_ind, target_end_ind = world.get_object_world_state_index(target.ID)
                target_state_history = yout[:, target_start_ind:target_end_ind]
                y_target = target_state_history

                # plot target trajectory
                if dim == 2:
                    ax.plot(y_target[0, x_pos], y_target[0, y_pos], 'bs', markersize=markersize,
                            label=target_start_pt_label)
                    ax.plot(y_target[:, x_pos], y_target[:, y_pos], '-b', linewidth=linewidth,
                        label=target_traj_label)
                    ax.text(y_target[0, x_pos], y_target[0, y_pos], 'T{0}'.format(target.ID),
                            fontsize=textsize)
                elif dim == 3:
                    ax.scatter3D(y_target[0, x_pos], y_target[0, y_pos], y_target[0, z_pos], color='b',
                            s=scatter_width, label=target_start_pt_label)
                    ax.plot3D(y_target[:, x_pos], y_target[:, y_pos], y_target[:, z_pos], '-b',
                            linewidth=linewidth_3d, label=target_traj_label)
                    ax.text(y_target[0, x_pos], y_target[0, y_pos], y_target[0, z_pos],
                            'T{0}'.format(target.ID), fontsize=textsize)

                    # >> 2d slice view <<
                    ax2.plot(y_target[:, x_pos], y_target[:, y_pos], '-b', linewidth=linewidth,
                            label=target_traj_label)

                    # points
                    ax2.plot(y_target[0, x_pos], y_target[0, y_pos], 'bo', markersize=markersize,
                            label=target_start_pt_label)

                    # text
                    ax2.text(y_target[0, x_pos], y_target[0, y_pos], 'T{0}'.format(target.ID),
                            fontsize=textsize)

                # ### stationary points
                # if target.ID >= 1:
                #     stationary_pt_label = '__nolabel__'

                # terminal_state = target.pol.const

                # if dim == 2:
                #     ax.plot(terminal_state[x_pos], terminal_state[y_pos], 'ks',
                #             markersize=markersize, label=stationary_pt_label)
                #     ax.text(terminal_state[x_pos], terminal_state[y_pos], 'C{0}'.format(target.ID),
                #             fontsize=textsize)
                # if dim == 3:
                #     ax.scatter3D(terminal_state[x_pos], terminal_state[y_pos],
                #             terminal_state[z_pos], color='k', s=scatter_width,
                #             label=stationary_pt_label)
                #     ax.text(terminal_state[x_pos], terminal_state[y_pos], terminal_state[z_pos],
                #             'C{0}'.format(target.ID), fontsize=textsize)

                #     # >> 2d slice view <<
                #     ax2.plot(terminal_state[0], terminal_state[1], 'ko', markersize=markersize,
                #             label=stationary_pt_label)
                #     ax2.text(terminal_state[0], terminal_state[1], 'C{0}'.format(target.ID),
                #             fontsize=textsize)


            ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
            if dim == 3:
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

        elif sim_name == 'emd':

            # dotted lines for emd

            agent_traj_label = 'Agent Path (EMD)'

            for agent in agent_mas.agent_list:

                dyn_model = agent.type

                statespace = agent.get_statespace()
                agent_dim_pos = statespace['position']
                x_pos = agent_dim_pos[0]
                y_pos = agent_dim_pos[1]
                z_pos = None
                if dim == 3:
                    z_pos = agent_dim_pos[2]

                # only need one label representing 'agents'
                if agent.ID >= 1:
                    agent_traj_label = '__nolabel__'

                # agent state over time
                ag_start_ind, ag_end_ind = world.get_object_world_state_index(agent.ID)
                agent_state_history = yout[:, ag_start_ind:ag_end_ind]
                y_agent = agent_state_history

                # plot agent trajectory with text
                if dim == 2:
                    ax.plot(y_agent[0, x_pos], y_agent[0, y_pos], 'rs', markersize=markersize,
                            label=agent_start_pt_label)
                    ax.plot(y_agent[:, x_pos], y_agent[:, y_pos], '--r', linewidth=linewidth,
                            label=agent_traj_label)
                    ax.text(y_agent[0, x_pos], y_agent[0, y_pos], 'A{0}'.format(agent.ID),
                            fontsize=textsize)
                elif dim == 3:
                    ax.scatter3D(y_agent[0, x_pos], y_agent[0, y_pos], y_agent[0, z_pos], color='r',
                            s=scatter_width, label=agent_start_pt_label)
                    ax.plot3D(y_agent[:, x_pos], y_agent[:, y_pos], y_agent[:, z_pos], '--r',
                            linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, x_pos], y_agent[0, y_pos], y_agent[0, z_pos],
                            'A{0}'.format(agent.ID), fontsize=textsize)

                    # >> 2d slice view <<
                    ax2.plot(y_agent[:, x_pos], y_agent[:, y_pos], '--r', linewidth=linewidth,
                            label=agent_traj_label)
                    # points
                    ax2.plot(y_agent[0, x_pos], y_agent[0, y_pos], 'ro', markersize=markersize,
                            label=agent_start_pt_label)
                    # text
                    ax2.text(y_agent[0, x_pos], y_agent[0, y_pos], 'A{0}'.format(agent.ID),
                            fontsize=textsize)

                # # plot location of assignment switches
                # patches = []
                # for switch_ind in assignment_switches[zz]:
                #     ci = Circle( (y_agent[switch_ind, 0], y_agent[switch_ind, 1]), 0.2, color='b', fill=True)
                #     patches.append(ci)
                # p = PatchCollection(patches)
                # ax.add_collection(p)

            for target in target_mas.agent_list:

                dyn_model = target.type

                statespace = target.get_statespace()
                target_dim_pos = statespace['position']
                x_pos = target_dim_pos[0]
                y_pos = target_dim_pos[1]
                z_pos = None
                if dim == 3:
                    z_pos = target_dim_pos[2]

                # only need one label representing 'agents'
                if target.ID >= 1:
                    target_start_pt_label = '__nolabel__'
                    target_traj_label = '__nolabel__'

                # agent state over time
                target_start_ind, target_end_ind = world.get_object_world_state_index(target.ID)
                target_state_history = yout[:, target_start_ind:target_end_ind]
                y_target = target_state_history

                # plot target trajectory
                if dim == 2:
                    ax.plot(y_target[0, x_pos], y_target[0, y_pos], 'bs', markersize=markersize,
                            label=target_start_pt_label)
                    ax.plot(y_target[:, x_pos], y_target[:, y_pos], '-b', linewidth=linewidth,
                        label=target_traj_label)
                    ax.text(y_target[0, x_pos], y_target[0, y_pos], 'T{0}'.format(target.ID),
                            fontsize=textsize)
                elif dim == 3:
                    ax.scatter3D(y_target[0, x_pos], y_target[0, y_pos], y_target[0, z_pos], color='b',
                            s=scatter_width, label=target_start_pt_label)
                    ax.plot3D(y_target[:, x_pos], y_target[:, y_pos], y_target[:, z_pos], '-b',
                            linewidth=linewidth_3d, label=target_traj_label)
                    ax.text(y_target[0, x_pos], y_target[0, y_pos], y_target[0, z_pos],
                            'T{0}'.format(target.ID), fontsize=textsize)

                    # >> 2d slice view <<
                    ax2.plot(y_target[:, x_pos], y_target[:, y_pos], '-b', linewidth=linewidth,
                            label=target_traj_label)

                    # points
                    ax2.plot(y_target[0, x_pos], y_target[0, y_pos], 'bo', markersize=markersize,
                            label=target_start_pt_label)

                    # text
                    ax2.text(y_target[0, x_pos], y_target[0, y_pos], 'T{0}'.format(target.ID),
                            fontsize=textsize)

                # ### stationary points
                # if target.ID >= 1:
                #     stationary_pt_label = '__nolabel__'

                # terminal_state = target.pol.const

                # if dim == 2:
                #     ax.plot(terminal_state[x_pos], terminal_state[y_pos], 'ks',
                #             markersize=markersize, label=stationary_pt_label)
                #     ax.text(terminal_state[x_pos], terminal_state[y_pos], 'C{0}'.format(target.ID),
                #             fontsize=textsize)
                # if dim == 3:
                #     ax.scatter3D(terminal_state[x_pos], terminal_state[y_pos],
                #             terminal_state[z_pos], color='k', s=scatter_width,
                #             label=stationary_pt_label)
                #     ax.text(terminal_state[x_pos], terminal_state[y_pos], terminal_state[z_pos],
                #             'C{0}'.format(target.ID), fontsize=textsize)

                #     # >> 2d slice view <<
                #     ax2.plot(terminal_state[0], terminal_state[1], 'ko', markersize=markersize,
                #             label=stationary_pt_label)
                #     ax2.text(terminal_state[0], terminal_state[1], 'C{0}'.format(target.ID),
                #             fontsize=textsize)

            ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
            if dim == 3:
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

        if dim == 2:
            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)
        if dim == 3:
            ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
            ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

            ax2.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
            ax2.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)

        # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
        # ax.legend(loc='lower right', fontsize=14)

            tick_spacing = 1000
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)
            ax.zaxis.set_tick_params(labelsize=labelsize)

            ax.tick_params(axis='x', which='major', pad=axispad)
            ax.tick_params(axis='y', which='major', pad=axispad)
            ax.tick_params(axis='z', which='major', pad=axispad)

            ax.xaxis.labelpad = labelpad
            ax.yaxis.labelpad = labelpad
            ax.zaxis.labelpad = labelpad

            # TEST
            # TODO 2d slice
            ax2.xaxis.set_tick_params(labelsize=labelsize)
            ax2.yaxis.set_tick_params(labelsize=labelsize)

        # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
        # ax.legend(loc='lower right', fontsize=fontsize)

        # # reorder the legend terms
        # handles, labels = ax.get_legend_handles_labels()
        # labels = [labels[1], labels[0], labels[2], labels[3]]
        # handles = [handles[1], handles[0], handles[2], handles[3]]

        # legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize)
        # legend.remove()

        # if dim == 2:
        #     ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize)

        # if dim == 3:
        #     # ax2.legend(loc='lower right', fontsize=fontsize-4)
        #     ax2.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize)

    plt.tight_layout()
    plt.savefig("trajectories.pdf", bbox_inches='tight')


def plot_assignment_comp_time(unpacked):

    """ Plots assignment computational cost over time
    """

    linewidth = 4

    fontsize = 40
    labelsize = 40

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Time (s)', fontsize=fontsize)
    axs.set_ylabel('Assignment Cum. Cost (s)', fontsize=fontsize)

    for sim_name, sim_diagnostics in unpacked.items():

        label = sim_name.split('Assignment', 1)[1]
        runtime_diagnostics = sim_diagnostics['runtime_diagnostics']

        tout = runtime_diagnostics.iloc[:, 0].to_numpy()
        assign_comp_cost = runtime_diagnostics.iloc[:, 1].to_numpy()
        dynamics_comp_cost = runtime_diagnostics.iloc[:, 2].to_numpy()

        axs.plot(tout, np.cumsum(assign_comp_cost), linewidth=linewidth, label=label)

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)
    axs.legend(fontsize=fontsize)

def plot_runtime_histogram(unpacked_ensemble_diagnostic):

    """ Plots histogram of simulation runtime over ensemble of batch simulations
    """

    fontsize = 32
    labelsize = 32

    # fig, axs = plt.subplots(1,1)
    # axs.set_xlabel('Simulation runtime (s)', fontsize=fontsize)
    # axs.set_ylabel('Frequency', fontsize=fontsize)
    # labels = ['Dyn', 'EMD']
    # axs.hist(unpacked_ensemble_diagnostic, histtype='bar', stacked=True, bins=10, align='left', label=labels)
    # axs.legend(fontsize=fontsize)

    fig, axs = plt.subplots(1,1)
    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    labels = 'EMD Runtime - Dyn Runtime'
    axs.set_xlabel('Runtime Difference (EMD - Dyn)', fontsize=fontsize)
    axs.set_ylabel('Frequency', fontsize=fontsize)
    axs.hist(unpacked_ensemble_diagnostic, histtype='bar', stacked=True, bins=10, align='left')
    # axs.legend(fontsize=fontsize)


def plot_runtimes(unpacked_ensemble_diagnostic):

    """ Plots runtime
    """

    fontsize = 32
    labelsize = 32

    fig, axs = plt.subplots(1,1)
    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    labels = ['Dyn', 'EMD']
    axs.set_xlabel('Simulation', fontsize=fontsize)
    axs.set_ylabel('Runtime (s)', fontsize=fontsize)

    # NOTE make sure that label is matching up with diagnostics
    axs.plot(unpacked_ensemble_diagnostic[0], marker='.', label=labels[0])
    axs.plot(unpacked_ensemble_diagnostic[1], marker='.', label=labels[1])

    axs.legend(fontsize=fontsize)

def plot_ensemble_avg_runtime(ensemble_diagnostic):

    """ Plots average runtimes for multiple ensembles
    """

    fontsize = 40
    labelsize = 40

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('Agents', fontsize=fontsize)
    axs.set_ylabel('Average Runtime (s)', fontsize=fontsize)

    # Using DataFrames
    labels = []
    for ensemble_name in ensemble_diagnostic.keys():
        labels.append(re.search('\d+v\d+', ensemble_name).group())

    metrics_df = pd.DataFrame(ensemble_diagnostic)
    metrics_df.columns = labels

    # order data by number of agents
    labels.sort(key=natural_keys)
    metrics_df = metrics_df[labels]

    metrics = {'Ensemble': labels, 'Average Runtime (s) - EMD': metrics_df.values[0, :], 'Average Runtime (s) - Dyn': metrics_df.values[1, :]}
    # metrics = {'Ensemble': labels, 'Average Runtime (s)': metrics_df.values.tolist()[0]}

    metrics_df = pd.DataFrame(metrics)

    # metrics_df.plot.bar(x='Ensemble', rot=0, fontsize=fontsize)

    nensembles = len(ensemble_diagnostic)
    xpos = np.arange(nensembles)
    width = 0.35

    axs.bar(xpos, metrics_df['Average Runtime (s) - Dyn'].values, width, alpha=0.5, label='Dyn')
    axs.bar(xpos+width, metrics_df['Average Runtime (s) - EMD'].values, width, alpha=0.5, label='EMD')

    axs.xaxis.set_tick_params(labelsize=labelsize)
    axs.yaxis.set_tick_params(labelsize=labelsize)

    # tick_spacing = 1
    # axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # axs.xaxis.offsetText.set_fontsize(fontsize)

    axs.set_xticks(xpos + width / 2)
    axs.set_xticklabels(labels)

    axs.legend(fontsize=fontsize)


    # linewidth = 4
    # fontsize = 32
    # labelsize = 32

    # fig, axs = plt.subplots(1,1)
    # axs.set_xlabel('Agents', fontsize=fontsize)
    # axs.set_ylabel('Average Runtime (s)', fontsize=fontsize)

    # # Using DataFrames
    # labels = []
    # for ensemble_name in ensemble_diagnostic.keys():
    #     labels.append(re.search('\d+v\d+', ensemble_name).group())

    # diag_df = pd.DataFrame.from_dict(ensemble_diagnostic)
    # diag_df.columns = labels

    # # order data by number of agents
    # labels.sort(key=natural_keys)
    # diag_df = diag_df[labels]

    # for i, (label, data) in enumerate(diag_df.iteritems()):
    #     nbins = int(len(data)/4)
    #     for i, d in enumerate(data):
    #         if i == 0:
    #             asst_type = 'EMD'
    #         elif i == 1:
    #             asst_type = 'Dyn'
    #         if label == '5v5':
    #             nagents = 5
    #         elif label == '10v10':
    #             nagents = 10
    #         elif label == '20v20':
    #             nagents = 20
    #         axs.plot(nagents, d, 'o', linewidth=linewidth, label=label+' '+asst_type)

    # axs.grid(False)

    # axs.xaxis.set_tick_params(labelsize=labelsize)
    # axs.yaxis.set_tick_params(labelsize=labelsize)

    # # tick_spacing = 1
    # # axs.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # axs.xaxis.offsetText.set_fontsize(fontsize)
    # axs.legend(loc='lower right', fontsize=fontsize)



