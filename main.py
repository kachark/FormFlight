
import copy
import os
import atexit
from time import time, strftime, localtime
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
import pandas as pd

# FormFlight
from FormFlight.setup import (setup_simulation, assign_decision_pol,
        generate_initial_conditions)
from FormFlight.post_process.post_process import (
        post_process_batch_simulation,
        post_process_batch_diagnostics,
        plot_batch_performance_metrics,
        plot_batch_diagnostics
)
from FormFlight.log import (
        save_batch_metrics,
        save_batch_diagnostics_to_csv,
        save_test_info_to_txt
)
from FormFlight.scenarios.intercept_init import (create_world)
from FormFlight.run import (OneVOne_runner)

def get_ensemble_name(nensemble, dim, nagents, ntargets, agent_model, target_model, agent_control_policy, target_control_policy):

    """ Returns ensemble name

    """

    identical = (agent_model==target_model)
    if identical:
        ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + '_' +\
                str(nagents) + 'v' + str(ntargets) + '_identical_' + agent_model + '_' + agent_control_policy + '_' +\
                target_control_policy + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + \
                '_' + str(nagents) + 'v' + str(ntargets) + agent_model + '_' + target_model + '_' + agent_control_policy + '_' +\
                target_control_policy + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    return ensemble_name


def construct_ensemble():

    """

    Setup ensemble, batch, and individual simulation parameters
    Create new directory to store ensemble, batch, and individual simulation results
    Call functions to perform simulations, pack results and diagnostics, and save to .csv

    ####### INFO ######
    # simulation: set of initial conditions with 1 asst pol
    # batch_simulation: set of simulations with SAME initial conditions
    # ensemble_simulation: set of different batch simulations

    # ex. ic_i = ith initial_conditions
    # sim1 = ic_1 and EMD
    # sim2 = ic_1 and DYN
    # batch_1 = [sim1, sim2] = [ic_1 and EMD, ic_1 and DYN]
    # sim3 = ic_2 and EMD
    # sim4 = ic_2 and DYN
    # batch_2 = [sim3, sim4] = [ic_2 and EMD, ic_2 and DYN]
    # ensemble_1 = [batch_1, batch_2] = [ [sim1, sim2], [sim3, sim4] ]
    #            = [ [ic_1 and EMD, ic_1 and DYN], [ic_2 and EMD, ic_2 and DYN] ]

    """


    # SIM PARAMETERS CONSTANT ACROSS ENSEMBLE
    dt = 0.5
    maxtime = 40
    dim = 2
    collisions = False
    collision_tol = 1e-1
    sim_params = {'dt': dt, 'maxtime': maxtime, 'dim': dim, 'collisions': collisions,
            'collision_tol': collision_tol}

    # SCENARIO PARAMETERS
    scenario = "Intercept"
    ndynamic_objects = 20
    nstatic_objects = 10

    # MONTE CARLO SIMULATION PARAMETERS
    ensemble_simulation = []
    batch_simulation = []
    nbatches = 1

    # Create directory for storage
    # TODO ensemble should not default to 'identical'
    # TODO deprecated
    nagents = 3
    ntargets = 3
    nterminal_states = ntargets
    agent_model = "Linearized_Quadcopter"
    target_model = "Linearized_Quadcopter"
    agent_formation = "circle"
    target_formation = "uniform_distribution"
    terminal_states_formation = "fibonacci_sphere"
    agent_control_policy = "LQT"
    target_control_policy = "LQR"
    assignment_epoch = 10

    nensemble = 0
    ensemble_name = get_ensemble_name(nensemble, dim, nagents, ntargets, agent_model, target_model,
            agent_control_policy, target_control_policy)

    # root_directory = './'
    root_directory = os.getcwd() + '/'
    ensemble_directory = root_directory + ensemble_name

    monte_carlo_params = {'nensemble': nensemble, 'nbatches': nbatches, 'ensemble_name':
            ensemble_name, 'ensemble_directory': ensemble_directory}

    # run 'python main.py' to disable debugging
    if __debug__:
        print('DEBUG ACTIVE')
        pass
    else:
        # create directory to store ensemble
        try:
            os.makedirs(ensemble_directory)
        except FileExistsError:
            # directory already exists
            pass

    # ### SCENARIO SPECIFIC
    # # generate schemas which outline all of the objects (ie. agents, points) in a given scenario
    # provides information related to dynamics,controllers,starting formations/locations of objects
    world = create_world(ndynamic_objects, nstatic_objects)
    ###

    # CONSTRUCT ENSEMBLE OF SIMULATIONS
    for batch_i in range(nbatches):

        # Create a batch of simulations (ie. group of sim with same initial state conditions)
        batch = {}

        # SIM SETUP

        # numerical representation of 'geometric information' provided by scenario schemas
        initial_conditions = generate_initial_conditions(dim, world)

        ###### DEFINE SIMULATION PROFILES ######
        sim_profiles = {}

        emd_world = copy.deepcopy(world)
        dyn_world = copy.deepcopy(world)

        # updates the geometric scenario information with extraneous information
        emd_world.set_name('emd')
        assign_decision_pol(emd_world, 'Agent_MAS', 'EMD', assignment_epoch)
        sim_profile_name = 'emd'
        emd_sim_profile = {'scenario': scenario, 'initial_conditions': initial_conditions,
                'world': emd_world, 'sim_params': sim_params}
        sim_profiles.update({sim_profile_name: emd_sim_profile})

        ########################################

        for profile_name, profile in sim_profiles.items():
            # sim = setup_simulation(profile)
            # sim_name = sim['asst_pol'].__class__.__name__
            setup_simulation(profile)
            sim_name = profile['world'].name
            batch.update({sim_name: profile})

        # add batch to ensemble
        ensemble_simulation.append(batch)

    # TODO need to update
    # parameters constant across the test ensemble
    # test_conditions = {'nbatches': nbatches, 'default_dt': dt, 'maxtime': maxtime, 'dim': dim,
    #         'nagents': nagents, 'ntargets': ntargets, 'agent_model': agent_model, 'target_model':
    #         target_model, 'nterminal_states': nterminal_states, 'stationary_states_formation':
    #         terminal_states_formation, 'collisions': collisions, 'collision_tol': collision_tol,
    #         'agent_control_policy': agent_control_policy, 'target_control_policy':
    #         target_control_policy, 'assignment_epoch': assignment_epoch, 'ensemble_name':
    #         ensemble_name, 'ensemble_directory': ensemble_directory}

    test_conditions = {'scenario': scenario, 'sim_params': sim_params, 'monte_carlo_params':
            monte_carlo_params}

    return ensemble_simulation, ensemble_directory, test_conditions

def run_ensemble_simulation(test_conditions, ensemble_simulation, ensemble_directory):

    """

    Run the ensemble of tests

    """

    # RUN SIMULATION
    ensemble_results = {}
    for ii, batch in enumerate(ensemble_simulation):

        batch_name = 'batch_{0}'.format(ii)
        batch_results = {}
        batch_diagnostics = {}

        for sim_name, profile in batch.items():

            scenario = profile['scenario']
            sim_params = profile['sim_params']
            world_i = profile['world']
            initial_world_state = profile['initial_conditions']

            # NOTE temporary
            runner = None
            if scenario == 'Intercept':
                runner = OneVOne_runner

            # run simulation
            results, diagnostics = runner(scenario, initial_world_state, world_i, sim_params)

            # results components
            components = [
                "scenario",
                "data",
                "world"
            ]

            # diagnostics components
            diag_components = [
                "runtime_diagnostics"
            ]

            # organize results according to components
            sim_results = {}
            for (comp, res) in zip(components, results):
                sim_results.update({comp: res})

            # organize diagnostics
            sim_diagnostics = {}
            for (diag_comp, diag) in zip(diag_components, diagnostics):
                sim_diagnostics.update({diag_comp: diag})

            # store sim results into a batch
            batch_results.update({sim_name: [sim_params, sim_results]}) # dict

            # store sim diagnostics into a batch
            batch_diagnostics.update({sim_name: [sim_params, sim_diagnostics]}) # dict

        # post-process and save
        batch_performance_metrics = post_process_batch_simulation(batch_results) # returns dict
        # collect diagnostics
        packed_batch_diagnostics = post_process_batch_diagnostics(batch_diagnostics) # returns dict

        if __debug__:
            # DEBUG
            plot_batch_performance_metrics(batch_performance_metrics)
            # plot_batch_diagnostics(packed_batch_diagnostics) # TODO need to fix
            plt.show()

        save_batch_metrics(batch_performance_metrics, ensemble_directory, batch_name)
        save_batch_diagnostics_to_csv(packed_batch_diagnostics, ensemble_directory, batch_name)

        # store batch results (useful for saving multiple ensembles)
        # ensemble_results.update({batch_name: batch_results})

def main():
    """ Main function

    Call functions to gather user-defined test conditions, perform simulations, pack results and
    diagnostics, and save to .csv

    """

    # define ensemble simulation parameters
    ensemble_simulation, ensemble_directory, test_conditions = construct_ensemble()

    # perform ensemble of simulations and save results
    run_ensemble_simulation(test_conditions, ensemble_simulation, ensemble_directory)

    print("done!")

    return test_conditions


# TODO move to util file
# Utilities
def secondsToStr(elapsed=None):

    """ Converts seconds to strings

    """

    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):

    """ start logging of elapsed time

    """

    line = "="*40
    print(line)
    print(secondsToStr(), '-', s)
    ss = secondsToStr() + ' - ' + s
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()
    return ss

def endlog():

    """ end log of elapsed time

    """

    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))


if __name__ == "__main__":

    start = time()
    # atexit.register(endlog) # print end time at program termination
    starttime = log("Start Program")

    # PERFORM TEST
    test_conditions = main()

    ensemble_name = test_conditions['monte_carlo_params']['ensemble_name']
    ensemble_directory = test_conditions['monte_carlo_params']['ensemble_directory']

    # PRINT TEST INFO TO TERMINAL
    print()
    line = "*"*40
    print(line)
    for param_type, parameter in test_conditions.items():
        if isinstance(parameter, str):
            print(param_type, ': ', parameter)
        else:
            for condition, value in parameter.items():
                print(condition, ': ', value)
    print(line)
    print()

    # display starttime at the end as well as beginning
    line = "="*40
    print(line)
    print(starttime)
    print(line)
    print()

    end = time() # print end time at end of simulation
    elapsed = end-start
    elapsedstr = secondsToStr(elapsed)
    endtime = log("End Program", elapsedstr)

    save_test_info_to_txt(ensemble_name, test_conditions, ensemble_directory, starttime, endtime, elapsedstr)


