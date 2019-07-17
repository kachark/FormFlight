import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from matplotlib.collections import PatchCollection
import pandas as pd
import copy
from controls import *

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
                post_processed_results_df = post_process_identical_doubleint(parameters, sim_results)

            # if parameters['agent_model'] == 'Linearized Quadcopter':
            #     post_process_identical_2d_doubleint()


        if parameters['dim'] == 3:
            if parameters['agent_model'] == 'Double_Integrator':
                post_processed_results_df = post_process_identical_doubleint(parameters, sim_results)

            # if parameters['agent_model'] == 'Linearized Quadcopter':
            #     post_process_identical_3d_doubleint()

        # collect post-processed performance metrics
        batch_performance_metrics.update({sim_name: post_processed_results_df})

    return batch_performance_metrics

# 2d or 3d identical agent/target double integrators
def post_process_identical_doubleint(parameters, sim_results):

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

    # # TEST
    # test = compute_controls(dx, du, yout, tout, assignments, nagents, poltargets, polagents)

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

    ##DEBUG
    #print("POLICY: ", poltrack.__class__.__name__)
    #print("FINAL TIME: ", tout[-1])
    #print("initial optimal cost: ", optcost)
    #print("initial incurred cost: ", final_cost[0])
    #print("final cost-to-go value: ", np.sum(xp, axis=1)[-1])
    #print("final incurred cost value: ", np.sum(final_cost, axis=1)[-1]) # last stage cost
    #print("initial optimal cost - final incurred cost value = ", optcost - np.sum(final_cost, axis=1)[-1])
    #print("INITIAL CONDITIONS")
    #print(yout[0, :])
    #print("FINAL STATE")
    #print(yout[-1, :])
    #print("OFFSET")
    #for pt in poltargets:
    #    print(pt.const)

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

def unpack_performance_metrics(batch_performance_metrics):
    """
    unpacks pandas DataFrame into a python standard dictionary
    """

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

def plot_costs(unpacked):

    fig, axs = plt.subplots(1,1)
    axs.set_xlabel('time (s)', fontsize=14)
    axs.set_ylabel('Cost', fontsize=14)
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
            axs.plot(tout, summed_opt_cost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')
            axs.plot(tout, np.sum(final_cost, axis=1), '--c', label='Cum. Stage Cost'+' '+sim_name)
            axs.plot(tout, np.sum(cost_to_go, axis=1), '--r', label='Cost-to-go'+' '+sim_name)
        else:
            axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost'+' '+sim_name)
            # axs.plot(tout, np.sum(cost_to_go, axis=1), '-r', label='Cost-to-go'+' '+sim_name)

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


def plot_assignments(unpacked):

    for sim_name, metrics in unpacked.items():

        dx = metrics['dx']
        nagents = metrics['nagents']
        tout = metrics['tout']
        yout = metrics['yout']

        assignments = yout[:, nagents*2*dx:].astype(np.int32)

        # TOO BUSY
        # plt.figure()
        # # plt.title("Agent-Target Assignments")
        # plt.xlabel('time (s)')
        # plt.ylabel('Assigned-to Target')
        # for ii in range(nagents):
        #     plt.plot(tout, assignments[:, ii], '-', label='A{0}'.format(ii))
        #     plt.legend()

        # # TEST
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # plt.xlabel('time (s)')
        # plt.ylabel('Assigned-to Target')

        # for tt in range(tout.shape[0]):
        #     time = tout[tt]
        #     for ii in range(nagents): # iterate consecutively through agents
        #         # ax.plot3D(agent_i, tout, target_j, '-r', label=agent_traj_label)
        #         ax.scatter(ii, time, assignments[tt, ii], '-r')
        #         # change color and marker if there's a switch


def plot_trajectory(unpacked):

    dim = 2 # default value

    # update dim
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

        agent_traj_label = 'Agent Trajectory - AssignmentDyn'
        agent_start_pt_label = 'Agent Start Position'
        target_start_pt_label = 'Target Start Position'
        target_traj_label = 'Target Trajectory'
        stationary_pt_label = 'Target Terminal Positions'

        if dim == 2: # and agent/target models both double integrator (omit requirement for now)

            ### Agent / Target Trajectories
            # optimal trajectories (solid lines)
            if sim_name == 'AssignmentDyn':

                for zz in range(nagents):

                    if zz >= 1:
                        agent_traj_label = '__nolabel__'
                        agent_start_pt_label = '__nolabel__'
                        target_start_pt_label = '__nolabel__'

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

                ax.set_xlabel("x")
                ax.set_ylabel("y")

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
                    ax.plot(offset[0], offset[1], 'ks', label=stationary_pt_label)
                    ax.text(offset[0], offset[1], 'C{0}'.format(zz))

                ax.set_xlabel("x")
                ax.set_ylabel("y")


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

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b', label=target_start_pt_label)
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b', label=target_traj_label)
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                ### stationary points
                for zz in range(ntargets):

                    if zz >= 1:
                        stationary_pt_label = '__nolabel__'

                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k', label=stationary_pt_label)
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight='bold', fontsize=14)
                ax.set_ylabel("y", fontweight='bold', fontsize=14)
                ax.set_zlabel("z", fontweight='bold', fontsize=14)

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

                    # # plot location of assignment switches
                    # for switch_ind in assignment_switches[zz]:
                    #     ax.scatter3D(y_agent[switch_ind, 0], y_agent[switch_ind, 1], y_agent[switch_ind, 2], color='m') # TODO

                    # plot target trajectory
                    y_target = yout[:, (zz+nagents)*dx:(zz+nagents+1)*dx]
                    ax.scatter3D(y_target[0, 0], y_target[0, 1], y_target[0, 2], color='b')
                    ax.plot3D(y_target[:, 0], y_target[:, 1], y_target[:, 2], '-b')
                    ax.text(y_target[0, 0], y_target[0, 1], y_target[0, 2], 'T{0}'.format(zz))

                # stationary locations
                for zz in range(ntargets):
                    offset = stationary_states[zz*dx:(zz+1)*dx]
                    ax.scatter3D(offset[0], offset[1], offset[2], color='k')
                    ax.text(offset[0], offset[1], offset[2], 'C{0}'.format(zz))

                ax.set_xlabel("x", fontweight='bold', fontsize=16)
                ax.set_ylabel("y", fontweight='bold', fontsize=16)
                ax.set_zlabel("z", fontweight='bold', fontsize=16)

            # ax.text2D(0.40, 0.95, 'Agent-Target Trajectories', fontweight='bold', fontsize=14, transform=ax.transAxes)
            ax.legend(loc='lower right', fontsize=12)

def plot_batch_performance_metrics(batch_performance_metrics):

    unpacked = unpack_performance_metrics(batch_performance_metrics)

    plot_costs(unpacked)
    plot_assignments(unpacked)
    plot_trajectory(unpacked)

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



