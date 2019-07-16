# import pickle
import atexit
from time import time, strftime, localtime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from setup import *
from post_process import *
from log import *



def get_ensemble_name(nensemble, dim, agent_model, target_model, agent_control_policy, target_control_policy):
    identical = (agent_model==target_model)
    if identical:
        ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + \
                '_identical_' + agent_model + '_' + agent_control_policy + '_' +\
                target_control_policy + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + \
                '_' + agent_model + '_' + target_model + '_' + agent_control_policy + '_' +\
                target_control_policy + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    return ensemble_name

def main():

    ####### INFO ######
    # simulation: set of initial conditions with 1 asst pol
    # batch_simulation: set of simulations with SAME initial conditions each with same/different single asst pol
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

    # setup_simulation() creates i sims corresponding to i assignment policies --> creates a batch

    ensemble_simulation = []
    batch_simulation = []
    nbatches = 10

    # SIM SETUP
    dt = 0.01
    maxtime = 5
    dim = 3
    nagents = 50
    ntargets = 50
    agent_model = "Double_Integrator"
    target_model = "Double_Integrator"
    collisions = True
    # initial_conditions = np.loadtxt("initial_conditions_2d.txt") # agents and targets
    # initial_conditions = np.loadtxt("initial_conditions_3d.txt")
    # cities = --> some distribution
    agent_control_policy = "LQR"
    target_control_policy = "LQR"

    ensemble_name = get_ensemble_name(0, dim, agent_model, target_model, agent_control_policy, target_control_policy)

    root_directory = '/Users/koray/Box Sync/test_results/'
    ensemble_directory = root_directory + ensemble_name

    # create directory to store ensemble
    try:
        os.makedirs(ensemble_directory)
    except FileExistsError:
        # directory already exists
        pass

    # ENSEMBLE OF SIMULATIONS
    for batch_i in range(nbatches):

        # batch simulation: collection of simulations running on the same initial conditions, but with different assignment
        # policies
        batch = setup_simulation(
            agent_model,
            target_model,
            agent_control_policy,
            target_control_policy,
            nagents,
            ntargets,
            collisions,
            dim,
            dt,
            maxtime,
        )

        ensemble_simulation.append(batch)

    # RUN SIMULATION
    ensemble_results = {}
    for ii, batch in enumerate(ensemble_simulation):

        batch_name = 'batch_{0}'.format(ii)
        batch_results = {}

        # Simulation parameters shared within batch
        collisions = batch["collisions"]
        dt = batch["dt"]
        maxtime = batch["maxtime"]
        dx = batch["dx"]
        du = batch["du"]
        x0 = batch["x0"]
        ltidyn = batch["agent_dyn"]
        target_dyn = batch["target_dyns"]
        poltrack = batch["agent_pol"]
        poltargets = batch["target_pol"]
        apol = batch["asst_pol"]
        nagents = batch["nagents"]
        ntargets = batch["ntargets"]
        runner = batch["runner"]

        # Other simulation information
        agent_model = batch["agent_model"]
        target_model = batch["target_model"]
        agent_control_policy = batch["agent_control_policy"]
        target_control_policy = batch["target_control_policy"]

        # run different assignment policies with same conditions
        for assignment_pol in apol:

            # run simulation
            results = runner(
                dx,
                du,
                x0,
                ltidyn,
                target_dyn,
                poltrack,
                poltargets,
                assignment_pol,
                nagents,
                ntargets,
                collisions,
                dt,
                maxtime,
            )

            # results components
            components = [
                "agents",
                "targets",
                "data",
                "tracking_policy",
                "target_pol",
                "nagents",
                "ntargets",
                "asst_cost",
                "agent_pol",
                "optimal_asst",
                "asst_policy",
            ]

            sim_name = assignment_pol.__class__.__name__

            # organize simulation parameters
            sim_parameters = {
                "dt": dt,
                "dim": dim,
                "dx": dx,
                "du": du,
                "agent_model": agent_model,
                "target_model": target_model,
                "agent_control_policy": agent_control_policy,
                "target_control_policy": target_control_policy,
                "collisions": collisions
            }

            # organize results according to components
            sim_results = {}
            for (comp, res) in zip(components, results):
                sim_results.update({comp: res})

            # store sim results into a batch
            batch_results.update({sim_name: [sim_parameters, sim_results]})

        # post-process and save
        batch_performance_metrics = post_process_batch_simulation(batch_results) # returns dataframe

        save_batch_metrics_to_csv(batch_performance_metrics, ensemble_directory, batch_name)

        # store batch results (useful for saving multiple ensembles)
        # ensemble_results.update({batch_name: batch_results})

    plt.show()

    print("done!")


# def save_object(obj, filename):
#     with open(filename, 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    # TEST saving objects pre-'postprocessing'
    # save_object(ensemble_results, 'test_save_pickle.pkl')
    # with open('test_save_pickle.pkl', 'rb') as input:
    #     tech_companies = pickle.load(input)


def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
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
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))


if __name__ == "__main__":

    start = time()
    atexit.register(endlog)
    starttime = log("Start Program")

    main()

    # display starttime at the end as well as beginning
    line = "="*40
    print(line)
    print(starttime)
    print(line)
    print()


