
""" @file post_process.py
"""

import numpy as np
import pandas as pd
import copy
import os

# DOT_assignment
from DOT_assignment import plot

def post_process_batch_simulation(batch_results):

    """ Calls utilities to parse results and package into pandas DataFrame

    Input:
    - batch_results:            dict of simulation parameters and results (np.array)

    Output:
    - batch_performance_metrics:dict containing DataFrames simulation parameters, results, costs per simulation within
    batch

    """

    sim_names = []
    batch_performance_metrics = {} # performance metrics
    sim_components = {} # useful parameters and objects used within the simulation

    dim = 2 # default value. also uniform across batch simulations

    # for every simulation within a batch, post-process results
    for sim_name, sim in batch_results.items():
        sim_names.append(sim_name)
        parameters = sim[0]
        sim_results = sim[1]
        post_processed_results_df = None

        # post-process each sim within batch
        # NOTE assumes agent/target homogeneity
        if parameters['dim'] == 2:
            if parameters['agent_model'] == 'Double_Integrator' and parameters['target_model'] == 'Double_Integrator':
                post_processed_results_df = post_process_homogeneous_identical(parameters, sim_results)

            if parameters['agent_model'] == 'Linearized_Quadcopter' and parameters['target_model'] == 'Linearized_Quadcopter':
                post_processed_results_df = post_process_homogeneous_identical(parameters, sim_results)

        # NOTE assumes agent/target homogeneity
        if parameters['dim'] == 3:
            if parameters['agent_model'] == 'Double_Integrator' and parameters['target_model'] == 'Double_Integrator':
                post_processed_results_df = post_process_homogeneous_identical(parameters, sim_results)

            if parameters['agent_model'] == 'Linearized_Quadcopter' and parameters['target_model'] == 'Linearized_Quadcopter':
                post_processed_results_df = post_process_homogeneous_identical(parameters, sim_results)

        # collect post-processed performance metrics
        batch_performance_metrics.update({sim_name: post_processed_results_df})

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

# TODO rename: post_process_homogeneous_identical(parameters, sim_results):
def post_process_homogeneous_identical(parameters, sim_results):

    """ Post-process and package simulation parameters, results, and post-processed data (costs)

    Post-process and packing function for homogeneous identical agent/target swarms

    Input:
    - parameters:           dict containing simulation parameters
    - sim_results:          dict containing simulaton results

    Output:
    - return_df:            pandas Dataframe with simulation parameters, results, costs

    """

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
    collision_tol = parameters['collision_tol']
    assignment_epoch = parameters['assignment_epoch']

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

    col_tol_df = pd.DataFrame([collision_tol])

    dt_df = pd.DataFrame([dt])
    dim_df = pd.DataFrame([dim])
    dx_df = pd.DataFrame([dx])
    du_df = pd.DataFrame([du])
    assignment_epoch_df = pd.DataFrame([assignment_epoch])
    nagents_df = pd.DataFrame([nagents])
    ntargets_df = pd.DataFrame([ntargets])
    parameters_df = pd.concat([dt_df, dim_df, col_df, col_tol_df, assignment_epoch_df, dx_df, du_df, nagents_df, ntargets_df], axis=1)

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

def post_process_homogeneous_nonidentical(parameters, sim_results):

    """ Post-process and package simulation parameters, results, and post-processed data (costs)

    Post-process and packing function for homogeneous non-identical agent and target swarms
    Homogeneous Non-identical implies that the agent and targets have different dynamic models, but are uniform
    within their respective swarms

    Input:
    - parameters:           dict containing simulation parameters
    - sim_results:          dict containing simulaton results

    Output:
    - return_df:            pandas Dataframe with simulation parameters, results, costs

    """

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
    dx_a = parameters['dx']
    du = parameters['du']
    collisions = parameters['collisions']
    assignment_epoch = parameters['assignment_epoch']

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

    col_tol_df = pd.DataFrame([collision_tol])

    dt_df = pd.DataFrame([dt])
    dim_df = pd.DataFrame([dim])
    dx_df = pd.DataFrame([dx])
    du_df = pd.DataFrame([du])
    assignment_epoch_df = pd.DataFrame([assignment_epoch])
    nagents_df = pd.DataFrame([nagents])
    ntargets_df = pd.DataFrame([ntargets])
    parameters_df = pd.concat([dt_df, dim_df, col_df, col_tol_df, assignment_epoch_df, dx_df, du_df, nagents_df, ntargets_df], axis=1)

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

def post_process_heterogeneous(parameters, sim_results):

    """ Post-process and package simulation parameters, results, and post-processed data (costs)

    Post-process and packing function for homogeneous non-identical agent and target swarms
    Heterogeneous implies that the agent and targets have different dynamic models and each member may have different dynamic models

    Input:
    - parameters:           dict containing simulation parameters
    - sim_results:          dict containing simulaton results

    Output:
    - return_df:            pandas Dataframe with simulation parameters, results, costs

    """

    pass

# TODO
# def unpack_performance_metrics(batch_performance_metrics):

#     if homogeneous and identical:
#         unpack_homogeneous_identical(batch_performance_metrics)
#     elif homogeneous and not identical:
#         unpack_homogeneous_nonidentical(batch_performance_metrics)
#     else:
#         unpack_heterogeneous(batch_performance_metrics)

# TODO rename to unpack_homogeneous_identical(batch_performance_metrics):

def unpack_performance_metrics(batch_performance_metrics):

    """ Unpacks batch performance metrics DataFrame into a python standard dictionary

    Input:
    - batch_performance_metrics:           pandas DataFrame

    Output:
    - unpacked_batch_metrics:              dict containing simulation parameters, results, post-processed results (costs)

    """

    unpacked_batch_metrics = {}

    for sim_name, metrics_df in batch_performance_metrics.items():

        ### unpack simulation metrics ###

        # simulation parameters
        parameter_cols = 9 # see stored data spec
        parameters = metrics_df.iloc[0, 0:parameter_cols].to_numpy()

        dt = float(parameters[0])
        dim = int(parameters[1])
        collisions = int(parameters[2])
        collision_tol = float(parameters[3])
        assignment_epoch = int(parameters[4])
        dx = int(parameters[5])
        du = int(parameters[6])
        nagents = int(parameters[7])
        ntargets = int(parameters[8])

        # simulation outputs
        output_cols = 1 + nagents*dx + ntargets*dx + nagents + ntargets*dx + nagents*du
        outputs = metrics_df.iloc[:, parameter_cols: parameter_cols + output_cols].to_numpy()

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

        unpacked = [dt, dim, assignment_epoch, collisions, collision_tol, dx, du, nagents, ntargets, tout, yout, stationary_states, agent_controls, final_cost, stage_cost, cost_to_go, optimal_cost]

        columns = [
            "dt",
            "dim",
            "assignment_epoch",
            "collisions",
            "collision_tol",
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


def unpack_performance_metrics_OLD_2(batch_performance_metrics):
    """ Unpacks batch performance metrics DataFrame into a python standard dictionary

    Targets old data storage schematic PRIOR TO AUGUST 13, 2019, AFTER JULY 25, 2019

    Input:
    - batch_performance_metrics:           pandas DataFrame

    Output:
    - unpacked_batch_metrics:              dict containing simulation parameters, results, post-processed results (costs)

    """

    unpacked_batch_metrics = {}

    for sim_name, metrics_df in batch_performance_metrics.items():

        ### unpack simulation metrics ###

        # simulation parameters
        parameter_cols = 8 # see stored data spec
        parameters = metrics_df.iloc[0, 0:parameter_cols].to_numpy()

        dt = float(parameters[0])
        dim = int(parameters[1])
        collisions = int(parameters[2])
        assignment_epoch = int(parameters[3])
        dx = int(parameters[4])
        du = int(parameters[5])
        nagents = int(parameters[6])
        ntargets = int(parameters[7])

        # simulation outputs
        output_cols = 1 + nagents*dx + ntargets*dx + nagents + ntargets*dx + nagents*du
        outputs = metrics_df.iloc[:, parameter_cols: parameter_cols + output_cols].to_numpy()

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

        unpacked = [dt, dim, assignment_epoch, collisions, dx, du, nagents, ntargets, tout, yout, stationary_states, agent_controls,
                final_cost, stage_cost, cost_to_go, optimal_cost]

        columns = [
            "dt",
            "dim",
            "assignment_epoch",
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

def unpack_performance_metrics_OLD(batch_performance_metrics):

    """ Unpacks batch performance metrics DataFrame into a python standard dictionary

    Targets old data storage schematic PRIOR TO JULY 25, 2019

    Input:
    - batch_performance_metrics:           pandas DataFrame

    Output:
    - unpacked_batch_metrics:              dict containing simulation parameters, results, post-processed results (costs)

    """


    unpacked_batch_metrics = {}

    for sim_name, metrics_df in batch_performance_metrics.items():

        ### unpack simulation metrics ###

        #### OLD DATA (earlier than 7/25) ####
        # simulation parameters
        parameter_cols = 7 # see stored data spec
        parameters = metrics_df.iloc[0, 0:parameter_cols].to_numpy()

        dt = float(parameters[0])
        dim = int(parameters[1])
        collisions = int(parameters[2])
        dx = int(parameters[3])
        du = int(parameters[4])
        nagents = int(parameters[5])
        ntargets = int(parameters[6])


        # simulation outputs
        output_cols = 1 + nagents*dx + ntargets*dx + nagents + ntargets*dx + nagents*du
        outputs = metrics_df.iloc[:, parameter_cols: parameter_cols + output_cols].to_numpy()

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

        #### OLD DATA (earlier than 7/25) ####
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
    # unpacked = unpack_performance_metrics_OLD2(batch_performance_metrics)
    # unpacked = unpack_performance_metrics_OLD(batch_performance_metrics)

    plot.plot_costs(unpacked)
    plot.plot_assignments(unpacked)
    plot.plot_trajectory(unpacked)

    collisions = agent_agent_collisions(unpacked)

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
