import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setup import *
from post_process import *

if __name__ == "__main__":

    ensemble_simulation = []
    batch_simulation = []

    # SIM SETUP
    dt = 0.01
    maxtime = 5
    dim = 3
    nagents = 3
    ntargets = 3
    agent_model = "Double Integrator"
    target_model = "Double Integrator"
    collisions = False
    # initial_conditions = np.loadtxt("initial_conditions_2d.txt") # agents and targets
    # initial_conditions = np.loadtxt("initial_conditions_3d.txt")
    # cities = --> some distribution
    control_policy = "LQR"

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

    # ENSEMBLE OF SIMULATIONS
    nbatches = 2
    for batch_i in range(nbatches):

        # get batch (same initial_conditions with multiple asst policies)
        batch = setup_simulation(
            agent_model,
            target_model,
            control_policy,
            nagents,
            ntargets,
            collisions,
            dim,
            dt,
            maxtime,
        )

        # batch simulation: collection of simulations running on the same initial conditions, but with different assignment
        # policies
        ensemble_simulation.append(batch)

    # RUN SIMULATION
    ensemble_results = {}
    for ii, batch in enumerate(ensemble_simulation):

        batch_results = {}

        # Simulation parameters
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
            columns = [
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
            sim_parameters = {
                "agent_model": agent_model,
                "target_model": target_model,
                "dim": dim,
                "dx": dx,
                "du": du
            }

            sim_results = {}
            for (col, res) in zip(columns, results):
                sim_results.update({col: res})

            # store sim results
            batch_results.update({sim_name: [sim_parameters, sim_results]})

        # store batch results
        ensemble_results.update({'Batch_{0}'.format(ii): batch_results})

    # post-process
    for batch_i, batch_results in ensemble_results.items():
        post_process_batch_simulation(batch_results)

    plt.show()

    print("done!")
