
""" @file post_process.py
"""

import numpy as np
import pandas as pd
import copy
import os

# DOT_assignment.post_process
from . import plot

def post_process_batch_simulation(batch_results):

    """ Calls utilities to parse results and package into pandas DataFrame

    Input:
    - batch_results:            dict of simulation parameters and results (np.array)

    Output:
    - batch_performance_metrics:dict containing DataFrames simulation parameters, results, costs per simulation within
    batch

    """

    batch_performance_metrics = {}

    # for every simulation within a batch, post-process results
    for sim_name, sim in batch_results.items():

        sim_params = sim[0]
        sim_results = sim[1]
        world = sim_results['world']
        post_processed_results_df = None

        # post-process each sim within batch
        post_processed_results_df = post_process(sim_params, sim_results)

        # collect post-processed performance metrics
        batch_performance_metrics.update({sim_name: [world, post_processed_results_df]})

    return batch_performance_metrics

def post_process_batch_diagnostics(batch_diagnostics):

    """ Calls utilities to parse diagnostics and package into pandas DataFrame

    Input:
    - batch_diagnostics:            dict of simulation parameters and results (np.array)

    Output:
    - packed_batch_diagnostics:     dict containing DataFrames simulation runtime diagnostics per simulation in batch

    """

    sim_names = []
    packed_batch_diagnostics = {} # performance metrics

    # dim = 2 # default value. also uniform across batch simulations

    # for every simulation within a batch, post-process results
    for sim_name, sim_diagnostics in batch_diagnostics.items():
        sim_names.append(sim_name)
        parameters = sim_diagnostics[0]
        diagnostics = sim_diagnostics[1]

        runtime_diagnostics = diagnostics['runtime_diagnostics'] # runtime_diagnostics are a df from engine.py

        # TODO diagnostics packaging : finish this
        # pack
        diagnostics_df = runtime_diagnostics
        packed_batch_diagnostics.update({sim_name: diagnostics_df})

    return packed_batch_diagnostics

def post_process(parameters, sim_results):

    # TODO need to recompute the control costs
    # eventually just record separately and save

    scenario = sim_results['scenario']
    df = sim_results['data']
    world = sim_results['world']
    world = copy.deepcopy(world) # because we need to adjust world objects

    yout = df.iloc[:, 1:].to_numpy()
    tout = df.iloc[:, 0].to_numpy()

    scenario_results = []

    if scenario == 'Intercept':
        engagement = [('Agent_MAS', 'Target_MAS')]
        engagement_results = {}
        for pair in engagement:
            system_of_interest = pair[0]
            targettable_set = pair[1]
            mas = world.get_multi_object(system_of_interest)
            target_mas = world.get_multi_object(targettable_set)
            nagents = mas.nagents
            ntargets = target_mas.nagents

            decision_epoch = mas.decision_epoch

            mas_models = [agent.type for agent in mas.agent_list]
            target_models = [target.type for target in target_mas.agent_list]

            # TODO need a better way to store this info for multiple systems of interest
            mas_dx = [agent.dx for agent in mas.agent_list]
            mas_du = [agent.du for agent in mas.agent_list]
            total_mas_dx = np.sum(mas_dx)
            total_mas_du = np.sum(mas_du)
            target_dx = [agent.dx for agent in target_mas.agent_list]
            total_target_dx = np.sum(target_dx)
            # remember assignments match global IDs of Agent_MAS agents and Target_MAS
            assignments = yout[:, total_mas_dx+(2*total_target_dx):].astype(np.int32)

            final_cost = np.zeros((tout.shape[0], nagents))
            stage_cost = np.zeros((tout.shape[0], nagents))
            xp = np.zeros((yout.shape[0], nagents))
            xss = np.zeros((yout.shape[0], total_mas_dx+total_target_dx))

            agent_controls = []

            # RECOMPUTE CONTROLS HISTORY for each agent
            for agent in mas.agent_list:
                dx = agent.dx
                du = agent.du
                # NOTE assumes that 'Agent_MAS' states start at beginning of yout
                ag_start_ind, ag_end_ind = world.get_object_world_state_index(agent.ID)
                agent_state_history = yout[:, ag_start_ind:ag_end_ind]

                controls = np.zeros((tout.shape[0], du))
                for k, time in enumerate(tout):
                    target_ID_k = assignments[k, agent.ID]

                    target = world.objects[target_ID_k]
                    target_start_ind, target_end_ind = world.get_object_world_state_index(target_ID_k)
                    target_state_k = yout[k, target_start_ind:target_end_ind]

                    u_target = target.pol.evaluate(time, target_state_k)

                    # update the linear augmented tracker
                    if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                        # Get agent policy in correct tracking state for P, Q, p at time ii
                        Acl = target.pol.get_closed_loop_A()
                        gcl = target.pol.get_closed_loop_g()
                        agent.pol.track(time, target_ID_k, Acl, gcl)

                        # TEST inten learning
                        # agent.pol.track(time, target)

                        controls[k, :] = agent.pol.evaluate(time, agent_state_history[k, :], target_state_k,
                                feedforward=u_target)
                    else:
                        controls[k, :] = agent.pol.evaluate(time, agent_state_history[k, :], target_state_k)


                agent_controls.append(controls)

                # CONTROL COSTS

                # post-process for t=0
                target_ID_0 = assignments[0, agent.ID]
                target_0 = world.objects[target_ID_0]
                target_start_ind, target_end_ind = world.get_object_world_state_index(target_ID_0)
                target_state_0 = yout[0, target_start_ind:target_end_ind]

                u_target = target.pol.evaluate(time, target_state_0)

                # update the linear augmented tracker
                if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                    # Get agent policy in correct tracking state for P, Q, p at time ii
                    Acl = target_0.pol.get_closed_loop_A()
                    gcl = target_0.pol.get_closed_loop_g()
                    agent.pol.track(0, target_ID_0, Acl, gcl)

                    # TEST inten learning
                    # agent.pol.track(0, target)

                    # NOTE assumes LQT
                    # Get agent controller properties
                    R = agent.pol.get_R()
                    Q_0 = agent.pol.get_Q()
                    P_0 = agent.pol.get_P()
                    p_0 = agent.pol.get_p()
                    uss_0 = agent.pol.get_uss()
                    Xss_0 = agent.pol.get_xss()

                    # agent LQT controller internal state representation - "augmented state"
                    X_0 = np.hstack((agent_state_history[0, :], target_state_0))

                    xp[0, agent.ID] = np.dot(X_0, np.dot(P_0, X_0)) + 2*np.dot(p_0, X_0) -\
                            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

                    stage_cost[0, agent.ID] = np.dot(X_0, np.dot(Q_0, X_0)) + \
                            np.dot(controls[0, :], np.dot(R, controls[0, :])) -\
                            (np.dot(Xss_0, np.dot(Q_0, Xss_0)) + np.dot(uss_0, np.dot(R, uss_0)))
                else:
                    x_0 = agent_state_history[0, :]
                    u_0 = controls[0, :]

                    r = 0.0

                    # STAGE COST
                    stage_cost[0, agent.ID] = 0.5*np.linalg.norm(x_0 - target_state_0)**2 + r*0.5*u_0[0]**2

                    # COST-TO-GO
                    xp[0, agent.ID] = 1

                # continue post-processing for rest of time points
                for k, time in enumerate(tout[1:]):
                    target_ID_k = assignments[k, agent.ID]

                    target = world.objects[target_ID_k]
                    target_start_ind, target_end_ind = world.get_object_world_state_index(target_ID_k)
                    target_state_k = yout[k, target_start_ind:target_end_ind]

                    u_target = target.pol.evaluate(time, target_state_k)

                    if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                        # Get agent policy in correct tracking state for P, Q, p at iteration k
                        Acl = target.pol.get_closed_loop_A()
                        gcl = target.pol.get_closed_loop_g()
                        agent.pol.track(time, target_ID_k, Acl, gcl)

                        # TEST inten learning
                        # agent.pol.track(time, target)

                        # NOTE assumes LQT
                        # Get agent controller properties
                        R = agent.pol.get_R()
                        Q = agent.pol.get_Q()
                        P = agent.pol.get_P()
                        p = agent.pol.get_p()
                        uss = agent.pol.get_uss()
                        Xss = agent.pol.get_xss()

                        X = np.hstack((agent_state_history[k, :], target_state_k))

                        # STAGE COST
                        stage_cost[k, agent.ID] = np.dot(X, np.dot(Q, X)) +\
                                np.dot(controls[k, :], np.dot(R, controls[k, :])) -\
                                (np.dot(Xss, np.dot(Q, Xss)) + np.dot(uss, np.dot(R, uss)))

                        # COST-TO-GO
                        xp[k, agent.ID] = np.dot(X, np.dot(P, X)) + 2*np.dot(p, X) -\
                            (np.dot(Xss_0, np.dot(P_0, Xss_0)) + 2*np.dot(p_0.T, Xss_0))

                    else:
                        x_k = agent_state_history[k, :]
                        u_k = controls[k, :]

                        r = 0.0

                        # STAGE COST
                        stage_cost[k, agent.ID] = 0.5*np.linalg.norm(x_k - target_state_k)**2 + r*0.5*u_k[0]**2

                        # COST-TO-GO
                        xp[k, agent.ID] = 1


                for k in range(tout.shape[0]):
                    # final_cost[k, agent.ID] = np.trapz(stage_cost[:k, agent.ID], x=tout[:k])
                    final_cost[k, agent.ID] = np.sum(stage_cost[:k, agent.ID])
 
            agent_controls = np.concatenate(agent_controls, axis=1)
            engagement_results.update({
                'engagement': pair,
                'system_of_interest': system_of_interest,
                'decision_epoch': decision_epoch,
                'targettable_set': targettable_set,
                'nagents': nagents,
                'ntargets': ntargets,
                'agent_models': mas_models,
                'target_models': target_models,
                'agent_dx': mas_dx,
                'target_dx': target_dx,
                'agent_controls': agent_controls,
                'final_cost': final_cost,
                'stage_cost': stage_cost,
                'cost_to_go': xp})

            scenario_results.append(engagement_results)

    # collect results of this engagement

    # TODO separate function
    # PACK RESULTS
    # sim params : system_of_interest : data : costs
    columns = ['dim', 'dx', 'du', 'nagents', 'ntargets', 'tout', 'yout', 'city_states',
            'final_cost', 'stage_cost', 'cost_to_go']
    # eng.df = [tout, yout, asst history] dataframe

    #### PACK INTO SINGLE DATAFRAME
    collisions = parameters['collisions']
    collision_tol = parameters['collision_tol']
    dt = parameters['dt']
    dim = parameters['dim']
    maxtime = parameters['maxtime']

    if collisions:
        col_df = pd.DataFrame([1])
    else:
        col_df = pd.DataFrame([0])

    col_tol_df = pd.DataFrame([collision_tol])

    dt_df = pd.DataFrame([dt])
    dim_df = pd.DataFrame([dim])
    maxtime_df = pd.DataFrame([maxtime])
    parameters_df = pd.concat([dt_df, dim_df, col_df, col_tol_df, maxtime_df], axis=1)

    # save the World state instead
#     engagement = scenario_results[0]['engagement']
#     system_of_interest = scenario_results[0]['system_of_interest']
#     decision_epoch = scenario_results[0]['decision_epoch']
#     targettable_set = scenario_results[0]['targettable_set']
#     nagents = scenario_results[0]['nagents']
#     ntargets = scenario_results[0]['ntargets']
#     agent_models = scenario_results[0]['agent_models']
#     target_models = scenario_results[0]['target_models']
#     agent_dx = scenario_results[0]['agent_dx']
#     target_dx = scenario_results[0]['target_dx']

#     agent_dx_df = pd.DataFrame([agent_dx])
#     target_dx_df = pd.DataFrame([agent_dx])
#     du_df = pd.DataFrame([du])
#     decision_epoch_df = pd.DataFrame([decision_epoch])
#     nagents_df = pd.DataFrame([nagents])
#     ntargets_df = pd.DataFrame([ntargets])

    # system_of_interest_df = pd.concat([decision_epoch, dx_df, du_df, nagents_df, ntargets_df], axis=1)

    final_cost = scenario_results[0]['final_cost']
    stage_cost = scenario_results[0]['stage_cost']
    cost_to_go = scenario_results[0]['cost_to_go']

    fc_df = pd.DataFrame(final_cost)
    sc_df = pd.DataFrame(stage_cost)
    ctg_df = pd.DataFrame(cost_to_go)
    costs_df = pd.concat([fc_df, sc_df, ctg_df], axis=1)

    # cities = np.zeros((1, ntargets*dx))
    # for jj in range(ntargets):
    #     cities[0, jj*dx:(jj+1)*dx] = poltargets[jj].const
    # stationary_states_df = pd.DataFrame(cities)

    controls_df = pd.DataFrame(scenario_results[0]['agent_controls'])

    outputs_df = pd.concat([df, controls_df], axis=1)

    return_df = pd.concat([parameters_df, outputs_df, costs_df], axis=1)

    return return_df

def unpack_performance_metrics(batch_performance_metrics):

    """ Unpacks batch performance metrics DataFrame into a python standard dictionary

    DATE: 02/07/2020

    Input:
    - batch_performance_metrics:           pandas DataFrame

    Output:
    - unpacked_batch_metrics:              dict containing simulation parameters, results, post-processed results (costs)

    """

    # parameters_df = pd.concat([dt_df, dim_df, col_df, col_tol_df, maxtime_df], axis=1)
    # outputs_df = pd.concat([df, controls_df], axis=1)
    # costs_df = pd.concat([fc_df, sc_df, ctg_df], axis=1)

    unpacked_batch_metrics = {}
    unpacked_worlds = {}

    for sim_name, sim_metrics in batch_performance_metrics.items():

        ### unpack simulation metrics ###
        world = sim_metrics[0]
        metrics_df = sim_metrics[1]

        # simulation parameters
        parameter_cols = 5 # see stored data spec
        parameters = metrics_df.iloc[0, 0:parameter_cols].to_numpy()

        dt = float(parameters[0])
        dim = int(parameters[1])
        collisions = int(parameters[2])
        collision_tol = float(parameters[3])
        maxtime = float(parameters[4])

        # TODO assumes scenario='Intercept'
        agent_mas = world.get_multi_object('Agent_MAS')
        target_mas = world.get_multi_object('Target_MAS')
        terminal_mas = world.get_multi_object('Region')
        nagents = agent_mas.nagents
        ntargets = target_mas.nagents
        nterminal_states = terminal_mas.nagents
        total_mas_dx = np.sum([agent.dx for agent in agent_mas.agent_list])
        total_mas_du = np.sum([agent.du for agent in agent_mas.agent_list])
        total_target_dx = np.sum([target.dx for target in target_mas.agent_list])
        total_target_du = np.sum([target.du for target in target_mas.agent_list])
        total_target_terminal_dx = np.sum([point.dx for point in terminal_mas.agent_list])

        # simulation outputs
        output_cols = 1 + total_mas_dx + total_target_dx + total_target_terminal_dx + nagents + \
                total_mas_du
        outputs = metrics_df.iloc[:, parameter_cols: parameter_cols + output_cols].to_numpy()

        tout = outputs[:, 0]
        # agent states, target states, asst
        yout_cols = 1 + total_mas_dx + total_target_dx + total_target_terminal_dx + nagents
        yout = outputs[:, 1: yout_cols]
        ctrl_cols = yout_cols + total_mas_du
        agent_controls = outputs[:, yout_cols: ctrl_cols]

        # simulation costs
        costs = metrics_df.iloc[:, parameter_cols + output_cols: ].to_numpy()

        fc_cols = nagents
        final_cost = costs[:, 0:fc_cols]
        sc_cols = fc_cols + nagents
        stage_cost = costs[:, fc_cols: sc_cols]
        ctg_cols = sc_cols + nagents
        cost_to_go = costs[:, sc_cols: ]

        unpacked = [dt, dim, collisions, collision_tol, maxtime, tout, yout, agent_controls,
                final_cost, stage_cost, cost_to_go]

        columns = [
            "dt",
            "dim",
            "collisions",
            "collision_tol",
            "maxtime",
            "tout",
            "yout",
            "agent_controls",
            "final_cost",
            "stage_cost",
            "cost_to_go"
        ]

        metrics = {}
        for (col, met) in zip(columns, unpacked):
            metrics.update({col: met})

        unpacked_batch_metrics.update({sim_name: metrics})
        unpacked_worlds.update({sim_name: world})

    return [unpacked_worlds, unpacked_batch_metrics]

def unpack_batch_diagnostics(batch_diagnostics):

    """ Unpacks batch diagnostics DataFrame into a python standard dictionary

    Targets old data storage schematic

    Input:
    - batch_diagnostics:                    pandas DataFrame

    Output:
    - unpack_batch_diagnostics:             dict containing simulation runtime diagnostics

    """

    unpack_batch_diagnostics = {}

    for sim_name, sim_diagnostics in batch_diagnostics.items():

        ### unpack simulation diagnostics ###

        # runtime diagnostics
        runtime_diagnostics = sim_diagnostics # the entire pandas df contains runtime diagnostics

        unpacked = [runtime_diagnostics]

        ### end unpack ###

        columns = [
            "runtime_diagnostics"
        ]

        diagnostics = {}
        for (col, diag) in zip(columns, unpacked):
            diagnostics.update({col: diag})

        unpack_batch_diagnostics.update({sim_name: diagnostics})

    return unpack_batch_diagnostics

# COMPUTE CONTROLS
def compute_controls(dx, du, yout, tout, assignments, nagents, poltargets, polagents):

    """ Recreate the agent swarm member control inputs

    """

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

    """ Returns time indices at which there is an assignment switch

    """

    # Output:
    # assignment switch indices per agent

    switches = []
    yout = copy.deepcopy(yout)
    tout = copy.deepcopy(tout)

    for zz in range(nagents):
        y_agent_assignments = yout[:, (nagents+ntargets)*agent_config_size + zz]

        # import ipdb; ipdb.set_trace()
        # identify indices where assignment switches
        assignment_switch_ind = np.where(y_agent_assignments[:-1] != y_agent_assignments[1:])[0]

        switches.append(assignment_switch_ind)

    return switches

def index_at_time(tout, time):

    """ Returns the index in data at a time point

    Input: time history, time point
    Output: index in data of time point

    """

    return (np.abs(tout-time)).argmin()

def collision_time(tout, yout):

    """ Returns the time at which all agents have collided

    Input: time history, trajectories, assignments
    Output: time for all agents to be collided

    """
    pass

def agent_agent_collisions(unpacked):

    """ Returns a list of agent indices corresponding to agent-agent collisions

    Input: time history, trajectories, assignments, agent statesize
    Output: agents that collided with eachother

    """
    # assignment_switch_ind = np.where(y_agent_assignments[:-1] != y_agent_assignments[1:])[0]

    # NOTE assumes homogeneous agents
    agent_collisions = {}
    for sim_name, metrics in unpacked.items():

        nagents = metrics['nagents']
        dx = metrics['dx']
        tout = metrics['tout']
        yout = metrics['yout']

        # TODO add collision_tol
        # collision_tol = metrics['collision_tol']
        collision_tol = 1e-1

        agent_states = []
        for zz in range(nagents):
            agent_states.append(np.zeros((tout.shape[0], 3)))

        for zz in range(nagents):
            y_agent = yout[:, zz*dx:(zz+1)*dx]

            # TODO make this not hardcoded
            # get positions
            if dx == 12: # 3D linearized quadcopter
                agent_pos = np.array(y_agent[:, 9:12])
            if dx == 6: # 3D double integrator
                agent_pos = np.array(y_agent[:, 0:3])
            if dx == 4: # 2D double integrator
                agent_pos = np.array(y_agent[:, 0:2])
            if dx == 8: # 2D linearized quadcopter
                agent_pos = np.array(y_agent[:, 0:2])

            agent_states[zz] = agent_pos

        # compare agent positions
        collisions_list = []
        for zz in range(nagents):
            for other_agent in range(nagents):
                if zz == other_agent:
                    continue

                agent_zz = agent_states[zz]
                agent_other = agent_states[other_agent]
                check_collisions = np.abs(agent_zz - agent_other) <= collision_tol
                for collision_at_time in check_collisions:
                    if collision_at_time.all():
                        collisions_list.append( (zz, other_agent) )

                # check_collisions = np.isclose(agent_zz, agent_other, rtol=1e-2, atol=1e-5)
                # for c in check_collisions:
                #     if c.all():
                #         collisions.append( (zz, other_agent) )

        agent_collisions.update({sim_name: collisions_list})

    return agent_collisions

def plot_batch_performance_metrics(batch_performance_metrics):

    """ Function which unpacks batch metrics and calls relevant plotting utilities
    """

    unpacked = unpack_performance_metrics(batch_performance_metrics)
    # unpacked = unpack_performance_metrics_OLD3(batch_performance_metrics)
    # unpacked = unpack_performance_metrics_OLD2(batch_performance_metrics)
    # unpacked = unpack_performance_metrics_OLD(batch_performance_metrics)

    plot.plot_costs(unpacked)
    # plot.plot_assignments(unpacked)
    plot.plot_trajectory(unpacked)

    # collisions = agent_agent_collisions(unpacked)

    # TODO trajectory movie
    # plot_animated_trajectory(unpacked)

def plot_ensemble_histograms(ensemble_performance_metrics):

    """ Function which unpacks ensemble of batch simulations metrics and calls relevant plotting utilities
    """

    nbatches = len(ensemble_performance_metrics)
    unpacked_ensemble_metrics_emd = np.zeros((nbatches, 4))
    unpacked_ensemble_metrics_dyn = np.zeros((nbatches, 4))
    for i, batch in enumerate(ensemble_performance_metrics):
        unpacked = unpack_performance_metrics(batch)

        # extract cost metrics

        # TEST
        for sim_name, metrics in unpacked.items():

            tout = metrics['tout']
            yout = metrics['yout']
            nagents = metrics['nagents']
            dx = metrics['dx']
            final_cost = metrics['final_cost']
            cost_to_go = metrics['cost_to_go']
            optimal_cost = metrics['optimal_cost']

            assignments = yout[:, nagents*2*dx:].astype(np.int32)
            assignment_switches = find_switches(tout, yout, nagents, nagents, dx, dx)
            # recreate assignments per switch
            asst_switch_indices = set()
            # asst_switch_indices.add(0) # add the origin assignment
            for ii in range(nagents):
               switch_indices = assignment_switches[ii]
               for ind in switch_indices:
                   asst_switch_indices.add(ind)
            nswitches = len(asst_switch_indices)

            summed_opt_cost = np.sum(optimal_cost[0, :])

            if sim_name == 'AssignmentDyn':
                unpacked_ensemble_metrics_dyn[i, 0] = np.sum(final_cost, axis=1)[-1]
                unpacked_ensemble_metrics_dyn[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                unpacked_ensemble_metrics_dyn[i, 2] = summed_opt_cost
                unpacked_ensemble_metrics_dyn[i, 3] = nswitches
            if sim_name == 'AssignmentEMD':
                unpacked_ensemble_metrics_emd[i, 0] = np.sum(final_cost, axis=1)[-1]
                unpacked_ensemble_metrics_emd[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                unpacked_ensemble_metrics_emd[i, 2] = summed_opt_cost
                unpacked_ensemble_metrics_emd[i, 3] = nswitches

            # ### cost plots
            # if sim_name == 'AssignmentDyn':
            #     axs.plot(tout, summed_opt_cost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')
            #     axs.plot(tout, np.sum(final_cost, axis=1), '--c', label='Cum. Stage Cost'+' '+sim_name)
            #     axs.plot(tout, np.sum(cost_to_go, axis=1), '--r', label='Cost-to-go'+' '+sim_name)
            # else:
            #     axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost'+' '+sim_name)
            #     # axs.plot(tout, np.sum(cost_to_go, axis=1), '-r', label='Cost-to-go'+' '+sim_name)

    # now, we have the data split by assignment policy
    emd_finalcost_optcost = (unpacked_ensemble_metrics_emd[:, 0] - unpacked_ensemble_metrics_emd[:, 2]) # final_cost - optimal_cost
    emd_asst_switches = unpacked_ensemble_metrics_emd[:, 3]

    # plot_cost_histogram(emd_finalcost_optcost)
    plot.plot_cost_histogram([unpacked_ensemble_metrics_dyn[:, 0], unpacked_ensemble_metrics_emd[:, 0]])
    plot.plot_asst_histogram(emd_asst_switches)

def plot_batch_diagnostics(batch_diagnostics):

    """ Function which unpacks batch diagnostics and calls relevant plotting utilities
    """

    unpacked = unpack_batch_diagnostics(batch_diagnostics)

    plot.plot_assignment_comp_time(unpacked)

def plot_ensemble_diagnostics(ensemble_diagnostics):

    """ Function which unpacks ensemble of batch simulations diagnostics and calls relevant plotting utilities
    """

    nbatches = len(ensemble_diagnostics)
    unpacked_ensemble_diagnostics_emd = np.zeros((nbatches, 3))
    unpacked_ensemble_diagnostics_dyn = np.zeros((nbatches, 3))

    for i, batch_diagnostics in enumerate(ensemble_diagnostics):
        unpacked = unpack_batch_diagnostics(batch_diagnostics)

        # extract cost metrics

        for sim_name, sim_diagnostics in unpacked.items():

            runtime_diagnostics = sim_diagnostics['runtime_diagnostics']

            tout = runtime_diagnostics.iloc[:, 0].to_numpy()
            assign_comp_cost = runtime_diagnostics.iloc[:, 1].to_numpy()
            dynamics_comp_cost = runtime_diagnostics.iloc[:, 2].to_numpy()
            runtime = runtime_diagnostics.iloc[0, 3]

            if sim_name == 'AssignmentDyn':
                unpacked_ensemble_diagnostics_dyn[i, 0] = assign_comp_cost[0] # time to perform initial assignment
                unpacked_ensemble_diagnostics_dyn[i, 1] = np.sum(dynamics_comp_cost)/dynamics_comp_cost.shape[0]
                unpacked_ensemble_diagnostics_dyn[i, 2] = runtime
            if sim_name == 'AssignmentEMD':
                unpacked_ensemble_diagnostics_emd[i, 0] = assign_comp_cost[0]
                unpacked_ensemble_diagnostics_emd[i, 1] = np.sum(dynamics_comp_cost)/dynamics_comp_cost.shape[0]
                unpacked_ensemble_diagnostics_emd[i, 2] = runtime

    # now, we have the data listed per batch
    runtime_diff = (unpacked_ensemble_diagnostics_emd[:, 2] - unpacked_ensemble_diagnostics_dyn[:, 2]) # dyn runtime - emd runtime
    # emd_asst_switches = unpacked_ensemble_metrics_emd[:, 3]

    # plot_runtime_histogram(runtime_diff)
    runtimes = [unpacked_ensemble_diagnostics_dyn[:, 2], unpacked_ensemble_diagnostics_emd[:,2]]
    # plot_runtime_histogram(runtimes)
    plot.plot_runtime_histogram(runtime_diff)
    plot.plot_runtimes(runtimes)


# TODO MOVE TO log.py
def save_ensemble_metrics(ensemble_performance_metrics, ensemble_name):

    """ Function to parse and repack ensemble of batch simulation metrics and save
    """

    nbatches = len(ensemble_performance_metrics)
    unpacked_ensemble_metrics_emd = np.zeros((nbatches, 4))
    unpacked_ensemble_metrics_dyn = np.zeros((nbatches, 4))

    # final_cost dyn, final_cost emd, assignments (emd)
    unpacked_ensemble_fc_asst = np.zeros((nbatches, 3))

    for i, batch in enumerate(ensemble_performance_metrics):
        unpacked = unpack_performance_metrics(batch)

        # extract cost metrics

        # TEST
        for sim_name, metrics in unpacked.items():

            tout = metrics['tout']
            yout = metrics['yout']
            nagents = metrics['nagents']
            dx = metrics['dx']
            final_cost = metrics['final_cost']
            cost_to_go = metrics['cost_to_go']
            optimal_cost = metrics['optimal_cost']

            assignments = yout[:, nagents*2*dx:].astype(np.int32)
            assignment_switches = find_switches(tout, yout, nagents, nagents, dx, dx)
            # recreate assignments per switch
            asst_switch_indices = set()
            # asst_switch_indices.add(0) # add the origin assignment
            for ii in range(nagents):
               switch_indices = assignment_switches[ii]
               for ind in switch_indices:
                   asst_switch_indices.add(ind)
            nswitches = len(asst_switch_indices)

            summed_opt_cost = np.sum(optimal_cost[0, :])

            if sim_name == 'AssignmentDyn':
                unpacked_ensemble_metrics_dyn[i, 0] = np.sum(final_cost, axis=1)[-1]
                unpacked_ensemble_metrics_dyn[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                unpacked_ensemble_metrics_dyn[i, 2] = summed_opt_cost
                unpacked_ensemble_metrics_dyn[i, 3] = nswitches

                unpacked_ensemble_fc_asst[i, 0] = np.sum(final_cost, axis=1)[-1]
            if sim_name == 'AssignmentEMD':
                unpacked_ensemble_metrics_emd[i, 0] = np.sum(final_cost, axis=1)[-1]
                unpacked_ensemble_metrics_emd[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                unpacked_ensemble_metrics_emd[i, 2] = summed_opt_cost
                unpacked_ensemble_metrics_emd[i, 3] = nswitches

                unpacked_ensemble_fc_asst[i, 1] = np.sum(final_cost, axis=1)[-1]
                unpacked_ensemble_fc_asst[i, 2] = nswitches

            # ### cost plots
            # if sim_name == 'AssignmentDyn':
            #     axs.plot(tout, summed_opt_cost*np.ones((yout.shape[0])), '--k', label='Optimal cost with no switching')
            #     axs.plot(tout, np.sum(final_cost, axis=1), '--c', label='Cum. Stage Cost'+' '+sim_name)
            #     axs.plot(tout, np.sum(cost_to_go, axis=1), '--r', label='Cost-to-go'+' '+sim_name)
            # else:
            #     axs.plot(tout, np.sum(final_cost, axis=1), '-c', label='Cum. Stage Cost'+' '+sim_name)
            #     # axs.plot(tout, np.sum(cost_to_go, axis=1), '-r', label='Cost-to-go'+' '+sim_name)

    df = pd.DataFrame(unpacked_ensemble_fc_asst)

    root_directory = '/Users/koray/Box Sync/TargetAssignment/draper_paper/raw_data/final_costs_assignments/'
    directory = root_directory + ensemble_name

    try:
        os.makedirs(directory)
    except FileExistsError:
        # directory already exists
        pass

    path = directory + '/' + 'final_costs_assignments.csv'

    df.to_csv(path, index=False, header=False)

def plot_ensemble_metric_comparisons(ensemble_performance_metrics):

    """ Function to parse ensemble of batch simulation metrics and call relevant plotting utilities
    """

    control_exp_metrics = {}
    switch_metrics = {}
    avg_switch_metrics = {}
    for ensemble_name, ensemble_metrics in ensemble_performance_metrics.items():
        nbatches = len(ensemble_metrics)
        unpacked_ensemble_metrics_emd = np.zeros((nbatches, 4))
        unpacked_ensemble_metrics_dyn = np.zeros((nbatches, 4))
        for i, batch in enumerate(ensemble_metrics):

            if '20v20' in ensemble_name:
                unpacked = unpack_performance_metrics_OLD(batch)
            else:
                unpacked = unpack_performance_metrics(batch)

            # extract cost metrics

            # TEST
            for sim_name, metrics in unpacked.items():

                tout = metrics['tout']
                yout = metrics['yout']
                nagents = metrics['nagents']
                dx = metrics['dx']
                final_cost = metrics['final_cost']
                cost_to_go = metrics['cost_to_go']
                optimal_cost = metrics['optimal_cost']

                assignments = yout[:, nagents*2*dx:].astype(np.int32)
                assignment_switches = find_switches(tout, yout, nagents, nagents, dx, dx)
                # recreate assignments per switch
                asst_switch_indices = set()
                # asst_switch_indices.add(0) # add the origin assignment
                for ii in range(nagents):
                   switch_indices = assignment_switches[ii]
                   for ind in switch_indices:
                       asst_switch_indices.add(ind)
                nswitches = len(asst_switch_indices)

                summed_opt_cost = np.sum(optimal_cost[0, :])

                if sim_name == 'AssignmentDyn':
                    unpacked_ensemble_metrics_dyn[i, 0] = np.sum(final_cost, axis=1)[-1]
                    unpacked_ensemble_metrics_dyn[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                    unpacked_ensemble_metrics_dyn[i, 2] = summed_opt_cost
                    unpacked_ensemble_metrics_dyn[i, 3] = nswitches
                if sim_name == 'AssignmentEMD':
                    unpacked_ensemble_metrics_emd[i, 0] = np.sum(final_cost, axis=1)[-1]
                    unpacked_ensemble_metrics_emd[i, 1] = np.sum(cost_to_go, axis=1)[-1]
                    unpacked_ensemble_metrics_emd[i, 2] = summed_opt_cost
                    unpacked_ensemble_metrics_emd[i, 3] = nswitches

        control_expenditure_diff = (unpacked_ensemble_metrics_emd[:, 0] - unpacked_ensemble_metrics_dyn[:, 0])\
                / unpacked_ensemble_metrics_dyn[:, 0] # final_cost - optimal_cost
        ensemble_switches = unpacked_ensemble_metrics_emd[:, 3]

        control_exp_metrics.update({ensemble_name: control_expenditure_diff})
        switch_metrics.update({ensemble_name: ensemble_switches})
        avg_switch_metrics.update({ensemble_name: np.sum(ensemble_switches, axis=0)/ensemble_switches.shape[0]})

    plot.plot_ensemble_cost_histogram(control_exp_metrics)
    plot.plot_ensemble_switch_histogram(switch_metrics)
    plot.plot_ensemble_avg_switch(avg_switch_metrics)

def plot_ensemble_diagnostic_comparison(ensemble_diagnostics):

    """ Function to parse ensemble of batch simulation diagnostics and call relevant plotting utilities
    """

    avg_runtime_diagnostic = {}
    runtime_diagnostic = {}
    for ensemble_name, ensemble_diag in ensemble_diagnostics.items():
        nbatches = len(ensemble_diag)
        unpacked_ensemble_diagnostics_emd = np.zeros((nbatches, 4))
        unpacked_ensemble_diagnostics_dyn = np.zeros((nbatches, 4))

        for i, batch_diagnostics in enumerate(ensemble_diag):
            unpacked = unpack_batch_diagnostics(batch_diagnostics)

            # extract cost metrics

            for sim_name, sim_diagnostics in unpacked.items():

                runtime_diagnostics = sim_diagnostics['runtime_diagnostics']

                tout = runtime_diagnostics.iloc[:, 0].to_numpy()
                assign_comp_cost = runtime_diagnostics.iloc[:, 1].to_numpy()
                dynamics_comp_cost = runtime_diagnostics.iloc[:, 2].to_numpy()
                runtime = runtime_diagnostics.iloc[0, 3]

                if sim_name == 'AssignmentDyn':
                    unpacked_ensemble_diagnostics_dyn[i, 0] = assign_comp_cost[0] # time to perform initial assignment
                    unpacked_ensemble_diagnostics_dyn[i, 1] = np.sum(dynamics_comp_cost)/dynamics_comp_cost.shape[0]
                    unpacked_ensemble_diagnostics_dyn[i, 2] = runtime
                if sim_name == 'AssignmentEMD':
                    unpacked_ensemble_diagnostics_emd[i, 0] = assign_comp_cost[0]
                    unpacked_ensemble_diagnostics_emd[i, 1] = np.sum(dynamics_comp_cost)/dynamics_comp_cost.shape[0]
                    unpacked_ensemble_diagnostics_emd[i, 2] = runtime

        # now, we have the data listed per batch
        runtime_diff = (unpacked_ensemble_diagnostics_emd[:, 2] - unpacked_ensemble_diagnostics_dyn[:, 2]) # dyn runtime - emd runtime
        # emd_asst_switches = unpacked_ensemble_metrics_emd[:, 3]

        # plot_runtime_histogram(runtime_diff)
        runtimes = [unpacked_ensemble_diagnostics_dyn[:, 2], unpacked_ensemble_diagnostics_emd[:,2]]

        avg_runtime_emd = np.sum(unpacked_ensemble_diagnostics_emd[:, 2])/nbatches
        avg_runtime_dyn = np.sum(unpacked_ensemble_diagnostics_dyn[:, 2])/nbatches

        avg_runtime_diagnostic.update({ensemble_name: [avg_runtime_emd, avg_runtime_dyn]})
        runtime_diagnostic.update({ensemble_name: [runtimes[1], runtimes[0]]})

    plot.plot_ensemble_avg_runtime(avg_runtime_diagnostic)
    # plot.plot_ensemble_avg_runtime(runtime_diagnostic)
