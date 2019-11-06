
import os
import atexit
from time import time, strftime, localtime
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
import pandas as pd

# DOT_assignment
from DOT_assignment.setup import (setup_simulation, generate_initial_conditions)
from DOT_assignment.post_process.post_process import (
        post_process_batch_simulation,
        post_process_batch_diagnostics,
        plot_batch_performance_metrics,
        plot_batch_diagnostics
)
from DOT_assignment.log import (
        save_batch_metrics_to_csv,
        save_batch_diagnostics_to_csv,
        save_test_info_to_txt
)


def get_ensemble_name(nensemble, dim, nagents, ntargets, agent_model, agent_control_policy,
        target_formation):

    """ Returns ensemble name

    User defined ensemble naming convention

    """

    ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + '_' +\
            str(nagents) + '_' + str(ntargets) + '_' + target_formation + '_' + agent_control_policy + '_' +\
            '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    return ensemble_name

def construct_ensemble():

    """

    Setup ensemble, batch, and individual simulation parameters
    Create new directory to store ensemble, batch, and individual simulation results

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

    ensemble_simulation = []
    batch_simulation = []
    nbatches = 1

    # SIM PARAMETERS CONSTANT ACROSS ENSEMBLE
    dt = 0.01
    maxtime = 5
    dim = 3
    nagents = 5
    ntargets = 5
    # agent_model = "Double_Integrator"
    agent_model = "Linearized_Quadcopter"
    agent_formation = 'uniform_distribution'
    target_formation = 'circle'
    collisions = False
    collision_tol = 1e-1
    agent_control_policy = "LQR"
    assignment_epoch = 10

    # Create directory for storage
    nensemble = 0

    # TODO ensemble should not default to 'identical'
    ensemble_name = get_ensemble_name(nensemble, dim, nagents, ntargets, agent_model, agent_control_policy, target_formation)

    # root_directory = './'
    root_directory = os.getcwd() + '/'
    ensemble_directory = root_directory + ensemble_name

# # TODO saving results disabled! - DEBUGGING
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

    # TODO assumes homogeneous swarms
    # formations: uniform_distribution, circle, fibonacci_sphere
    initial_formation_params = {
            'nagents': nagents, 'agent_model': agent_model, 'agent_swarm_formation': agent_formation,
            'ntargets': ntargets, 'target_swarm_formation': target_formation
            }

    # CONSTRUCT ENSEMBLE OF SIMULATIONS
    for batch_i in range(nbatches):

        # Create a batch of simulations (ie. group of sim with same initial state conditions)
        batch = {}

        # SIM SETUP

        initial_conditions = generate_initial_conditions(dim, initial_formation_params)

        ###### DEFINE SIMULATION PROFILES ######
        sim_profiles = {}

        # EMD parameters
        dt = dt
        asst = 'AssignmentEMD'
        sim_profile_name = 'emd'
        sim_profiles.update({sim_profile_name: {'agent_model': agent_model, 'agent_control_policy':
            agent_control_policy, 'agent_formation': agent_formation, 'target_formation':
            target_formation, 'assignment_policy': asst, 'assignment_epoch': assignment_epoch,
            'nagents': nagents, 'ntargets': ntargets, 'collisions': collisions, 'collision_tol':
            collision_tol, 'dim': dim, 'dt': dt, 'maxtime': maxtime, 'initial_conditions':
            initial_conditions}})

        # Custom Assignment parameters
        dt = dt
        asst = 'AssignmentCustom'
        sim_profile_name = 'dyn'
        sim_profiles.update({sim_profile_name: {'agent_model': agent_model, 'agent_control_policy':
            agent_control_policy, 'agent_formation': agent_formation, 'target_formation':
            target_formation, 'assignment_policy': asst, 'assignment_epoch': assignment_epoch,
            'nagents': nagents, 'ntargets': ntargets, 'collisions': collisions, 'collision_tol':
            collision_tol, 'dim': dim, 'dt': dt, 'maxtime': maxtime, 'initial_conditions':
            initial_conditions}})

        ########################################

        for profile_name, profile in sim_profiles.items():
            sim = setup_simulation(profile)
            sim_name = sim['asst_pol'].__class__.__name__
            batch.update({sim_name: sim})

        # add batch to ensemble
        ensemble_simulation.append(batch)

    # parameters constant across the test ensemble
    test_conditions = {'nbatches': nbatches, 'default_dt': dt, 'maxtime': maxtime, 'dim': dim,
            'nagents': nagents, 'ntargets': ntargets, 'agent_model': agent_model, 'agent_formation':
            agent_formation, 'target_formation': target_formation, 'collisions': collisions,
            'collision_tol': collision_tol, 'agent_control_policy': agent_control_policy,
            'assignment_epoch': assignment_epoch, 'ensemble_name': ensemble_name,
            'ensemble_directory': ensemble_directory}

    return ensemble_simulation, ensemble_directory, test_conditions

def run_ensemble_simulation(test_conditions, ensemble_simulation, ensemble_directory):

    """

    Run the ensemble of tests 

    """

    nbatches = test_conditions['nbatches']
    dim = test_conditions['dim']

    # RUN ENSEMBLE SIMULATION
    ensemble_results = {}
    for ii, batch in enumerate(ensemble_simulation):

        batch_name = 'batch_{0}'.format(ii)
        batch_results = {}
        batch_diagnostics = {}

        for sim_name, sim in batch.items():

            # TODO not the same order for heterogeneous and non-identical
            # Simulation data structures
            collisions = sim["collisions"]
            collision_tol = sim["collision_tol"]
            dt = sim["dt"]
            maxtime = sim["maxtime"]
            dx = sim["dx"]
            du = sim["du"]
            statespace = sim["statespace"]
            x0 = sim["x0"]
            ltidyn = sim["agent_dyn"]
            poltrack = sim["agent_pol"]
            assignment_pol = sim["asst_pol"]
            assignment_epoch = sim["asst_epoch"]
            nagents = sim["nagents"]
            ntargets = sim["ntargets"]
            runner = sim["runner"]

            # Other simulation information
            agent_model = sim["agent_model"]
            agent_control_policy = sim["agent_control_policy"]
            agent_formation = sim["agent_formation"]
            target_formation = sim["target_formation"]

            # run simulation
            results, diagnostics = runner(
                dx,
                du,
                statespace,
                x0,
                ltidyn,
                poltrack,
                assignment_pol,
                assignment_epoch,
                nagents,
                ntargets,
                collisions,
                collision_tol,
                dt,
                maxtime,
            )

            # results components
            components = [
                "agents",
                "targets",
                "data",
                "tracking_policy",
                "nagents",
                "ntargets",
                "asst_cost",
                "agent_pol",
                "optimal_asst",
                "asst_policy",
            ]

            # diagnostics components
            diag_components = [
                    "runtime_diagnostics"
                    ]

            # organize simulation parameters
            sim_parameters = {
                "dt": dt,
                "dim": dim,
                "dx": dx,
                "du": du,
                "statespace": statespace,
                "agent_model": agent_model,
                "agent_control_policy": agent_control_policy,
                "collisions": collisions,
                "collision_tol": collision_tol,
                "assignment_epoch": assignment_epoch
            }

            # organize results according to components
            sim_results = {}
            for (comp, res) in zip(components, results):
                sim_results.update({comp: res})

            # organize diagnostics
            sim_diagnostics = {}
            for (diag_comp, diag) in zip(diag_components, diagnostics):
                sim_diagnostics.update({diag_comp: diag})

            # store sim results into a batch
            batch_results.update({sim_name: [sim_parameters, sim_results]}) # dict

            # store sim diagnostics into a batch
            batch_diagnostics.update({sim_name: [sim_parameters, sim_diagnostics]}) # dict

        # post-process and save
        batch_performance_metrics = post_process_batch_simulation(batch_results) # returns dict
        # collect diagnostics
        packed_batch_diagnostics = post_process_batch_diagnostics(batch_diagnostics) # returns dict

        if __debug__:
            # DEBUG
            plot_batch_performance_metrics(batch_performance_metrics)
            plot_batch_diagnostics(packed_batch_diagnostics)
            plt.show()

        save_batch_metrics_to_csv(batch_performance_metrics, ensemble_directory, batch_name)
        save_batch_diagnostics_to_csv(packed_batch_diagnostics, ensemble_directory, batch_name)

        # store batch results (useful for saving multiple ensembles)
        # ensemble_results.update({batch_name: batch_results})

def main():

    """ Main function

    Call functions to gather user-defined test conditions, perform simulations, pack results and diagnostics, and save to .csv

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

    # PERFORM ENSEMBLE OF TESTS
    test_conditions = main()

    ensemble_name = test_conditions['ensemble_name']
    ensemble_directory = test_conditions['ensemble_directory']

    # PRINT TEST INFO TO TERMINAL
    print()
    line = "*"*40
    print(line)
    for condition, value in test_conditions.items():
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


