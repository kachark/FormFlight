import numpy as np
import pandas as pd
import copy
from controls import *
from plot import *


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

            if parameters['agent_model'] == 'Linearized_Quadcopter':
                post_processed_results_df = post_process_identical_doubleint(parameters, sim_results)

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

        # import ipdb; ipdb.set_trace()
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


def plot_batch_performance_metrics(batch_performance_metrics):

    unpacked = unpack_performance_metrics(batch_performance_metrics)

    plot_costs(unpacked)
    plot_assignments(unpacked)
    plot_trajectory(unpacked)
