
## @include .
## @example ./examples/linearized_quadcopter_3D/animate_3D_trajectory.py
# An Example of how to plot an animation in 3D.
# See the SETUP section for an example of loading the simulation
# Adjust 'batch_num' to select the desired batch to load within the ensemble folder


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import matplotlib.ticker as ticker

from DOT_assignment import log
from DOT_assignment.post_process import post_process

### TEST ANIMATION
# TODO CLEANUP

def get_trajectory(unpacked):

    linewidth_3d = 2
    linewidth = 4
    markersize = 8
    scatter_width = markersize**2
    textsize = 32

    fontsize = 40
    fontweight = 'bold'
    labelsize = 40

    axispad = 18
    labelpad = 40

    fontsize = 32
    fontweight = 'bold'
    labelsize = 32

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # lines: emd lines, dyn lines, target lines (only once)
    # pts: emd/dyn pts (only once), target pts, city pts
    # textpts: emd text, dyn text, target text, city text
    # lines = [ax.plot([], [], []) for a in range(2*nagents + 2*ntargets)]
    # pts = [ax.plot([], [], [], 'o') for a in range(nagents + 2*ntargets)]
    # textpts = [ax.plot([], [], [], 'o') for a in range(2*nagents + 2*nagents)]

    dyn_agents = []
    emd_agents = []
    targets = []

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

        agent_traj_label = 'Agent Path (Dyn)'
        agent_start_pt_label = 'Agent Start'
        target_start_pt_label = 'Target Start'
        target_traj_label = 'Target Path'
        stationary_pt_label = 'Terminal State'

        # TEST # TODO REMOVE EVENTUALLY
        if dx == 12:
            agent_model = 'Linearized_Quadcopter'
            labels = [agent_traj_label, agent_start_pt_label, target_start_pt_label, target_traj_label, stationary_pt_label]
            get_trajectory_qc(unpacked)
            continue
        if dx == 6:
            agent_model = 'Double_Integrator'

        if dim == 3:

            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentCustom':

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

                    dyn_agents.append(y_agent[:, 0:3])

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r', s=scatter_width, label=agent_start_pt_label)
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r', linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz), fontsize=textsize)

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b', s=scatter_width, label=target_start_pt_label)
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b', linewidth=linewidth_3d, label=target_traj_label)
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz), fontsize=textsize)

                    targets.append(y_target[:, 0:3])

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

            elif sim_name == 'AssignmentEMD':
                # non-optimal trajectories (dotted lines)

                agent_traj_label = 'Agent Path (EMD)'

                # agent/target trajectories
                for zz in range(nagents):

                    # avoid repeated legend entries
                    if zz >= 1:
                        agent_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    emd_agents.append(y_agent[:, 0:3])

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '--r', linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz), fontsize=textsize)

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz), fontsize=textsize)

        # dim = 3

        # tick_spacing = 1000
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.xaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.zaxis.set_tick_params(labelsize=labelsize)

        ax.tick_params(axis='x', which='major', pad=axispad)
        ax.tick_params(axis='y', which='major', pad=axispad)
        ax.tick_params(axis='z', which='major', pad=axispad)

        ax.xaxis.labelpad = labelpad
        ax.yaxis.labelpad = labelpad
        ax.zaxis.labelpad = labelpad

        ax.set_zlim3d(-100, 100)

    # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
    # ax.legend(loc='lower right', fontsize=fontsize)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize)
    legend.remove()

    return fig, ax, dyn_agents, emd_agents, targets


def get_trajectory_qc(unpacked):

    linewidth_3d = 2
    linewidth = 4
    markersize = 8
    scatter_width = markersize**2
    textsize = 32

    fontsize = 40
    fontweight = 'bold'
    labelsize = 40

    axispad = 18
    labelpad = 40

    fontsize = 32
    fontweight = 'bold'
    labelsize = 32

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    dyn_agents = []
    emd_agents = []
    targets = []

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

        agent_traj_label = 'Agent Path (Dyn)'
        agent_start_pt_label = 'Agent Start'
        target_start_pt_label = 'Target Start'
        target_traj_label = 'Target Path'
        stationary_pt_label = 'Terminal State'

        # TEST # TODO REMOVE EVENTUALLY
        if dx == 12:
            agent_model = 'Linearized_Quadcopter'
            labels = [agent_traj_label, agent_start_pt_label, target_start_pt_label, target_traj_label, stationary_pt_label]

        if dim == 3:

            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentCustom':

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

                    dyn_agents.append(y_agent[:, 0:3])

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r', s=scatter_width, label=agent_start_pt_label)
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '-r', linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz), fontsize=textsize)

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b', s=scatter_width, label=target_start_pt_label)
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b', linewidth=linewidth_3d, label=target_traj_label)
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz), fontsize=textsize)

                    targets.append(y_target[:, 0:3])

                ax.set_xlabel("x", fontweight=fontweight, fontsize=fontsize)
                ax.set_ylabel("y", fontweight=fontweight, fontsize=fontsize)
                ax.set_zlabel("z", fontweight=fontweight, fontsize=fontsize)

            elif sim_name == 'AssignmentEMD':
                # non-optimal trajectories (dotted lines)

                agent_traj_label = 'Agent Path (EMD)'

                # agent/target trajectories
                for zz in range(nagents):

                    # avoid repeated legend entries
                    if zz >= 1:
                        agent_traj_label = '__nolabel__'

                    # agent state over time
                    y_agent = yout[:, zz*dx:(zz+1)*dx]

                    emd_agents.append(y_agent[:, 0:3])

                    # plot agent trajectory with text
                    ax.scatter3D(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], color='r')
                    ax.plot3D(y_agent[:, 0], y_agent[:, 1], y_agent[:, 2], '--r', linewidth=linewidth_3d, label=agent_traj_label)
                    ax.text(y_agent[0, 0], y_agent[0, 1], y_agent[0, 2], 'A{0}'.format(zz), fontsize=textsize)

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz), fontsize=textsize)

        # dim = 3

        # tick_spacing = 1000
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.xaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.zaxis.set_tick_params(labelsize=labelsize)

        ax.tick_params(axis='x', which='major', pad=axispad)
        ax.tick_params(axis='y', which='major', pad=axispad)
        ax.tick_params(axis='z', which='major', pad=axispad)

        ax.xaxis.labelpad = labelpad
        ax.yaxis.labelpad = labelpad
        ax.zaxis.labelpad = labelpad

        ax.set_zlim3d(-100, 100)

    # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
    # ax.legend(loc='lower right', fontsize=fontsize)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=fontsize)
    legend.remove()

    return fig, ax, dyn_agents, emd_agents, targets


# SETUP
# .###################################################################

# loads desired batch within a single ensemble which the user specifies

# EDIT the following set of parameters used in the desired ensemble test folder
dim = 3

nagents = 5
ntargets = 5

agent_model = 'Linearized_Quadcopter'
agent_formation = 'uniform_distribution'

# EDIT the date here to match the ensemble test folder, you would like to load
ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+agent_formation+'_LQR_LINEARIZED_QUADCOPTER_3D_EXAMPLE'

root_directory = os.getcwd() + '/'
ensemble_directory = root_directory + ensemble_name

# get number of batches
batch_dirs = [x[0] for x in os.walk(ensemble_directory)]
nbatches = len(batch_dirs[1:])

# load batches and plot
sim_name_list = ['AssignmentCustom', 'AssignmentEMD']

# EDIT select specific batch you want to animate
# load and plot a specific batch
batch_num = 0
batch_name = 'batch_{0}'.format(batch_num)
loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)

unpacked = post_process.unpack_performance_metrics(loaded_batch)

# .###################################################################

if agent_model == "Double_Integrator":
    fig, ax, dyn_agents, emd_agents, targets = get_trajectory(unpacked)

if agent_model == "Linearized_Quadcopter":
    fig, ax, dyn_agents, emd_agents, targets = get_trajectory_qc(unpacked)

dyn_agent_lines = sum([ax.plot([], [], [], 'ro') for dat in dyn_agents], [])
emd_agent_lines = sum([ax.plot([], [], [], 'r*') for dat in emd_agents], [])
target_lines = sum([ax.plot([], [], [], 'bo') for dat in targets], [])

def init():
    for dyn_line, emd_line, target_line in zip(dyn_agent_lines, emd_agent_lines, target_lines):
        dyn_line.set_data([], [])
        dyn_line.set_3d_properties([])

        emd_line.set_data([], [])
        emd_line.set_3d_properties([])

        target_line.set_data([], [])
        target_line.set_3d_properties([])
    return dyn_agent_lines + emd_agent_lines + target_lines

def animate(i):
    i = (50*i) % dyn_agents[0].shape[0]

    try:
        for line, data in zip(dyn_agent_lines, dyn_agents) :
            # NOTE: there is no .set_data() for 3 dim data...
            x, y, z = data[i]
            line.set_data(x,y)
            line.set_3d_properties(z)

        for line, data in zip(emd_agent_lines, emd_agents) :
            x, y, z = data[i]
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(x,y)
            line.set_3d_properties(z)

        for line, data in zip(target_lines, targets) :
            x, y, z = data[i]
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(x,y)
            line.set_3d_properties(z)
    except IndexError:
        return dyn_agent_lines + emd_agent_lines + target_lines

    fig.canvas.draw()
    return dyn_agent_lines + emd_agent_lines + target_lines

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=10, blit=True, repeat=True)

plt.show()
