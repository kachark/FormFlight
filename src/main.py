import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setup import *
from post_process import *

if __name__ == "__main__":

    batch_simulation = []
    batch_results = {}

    # SIM SETUP
    dt = 0.01
    maxtime = 5
    dim = 2
    nagents = 8
    ntargets = 8
    agent_model = "Double Integrator"
    target_model = "Double Integrator"
    collisions = False
    # initial_conditions = np.loadtxt("initial_conditions_2d.txt") # agents and targets
    # initial_conditions = np.loadtxt("initial_conditions_3d.txt")
    # cities = --> some distribution
    control_policy = "LQR"
    sim = setup_simulation(
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
    batch_simulation.append(sim)

    # RUN SIMULATION
    for sim in batch_simulation:

        # Simulation parameters
        collisions = sim["collisions"]
        dt = sim["dt"]
        maxtime = sim["maxtime"]
        dx = sim["dx"]
        du = sim["du"]
        x0 = sim["x0"]
        ltidyn = sim["agent_dyn"]
        target_dyn = sim["target_dyns"]
        poltrack = sim["agent_pol"]
        poltargets = sim["target_pol"]
        apol = sim["asst_pol"]
        nagents = sim["nagents"]
        ntargets = sim["ntargets"]
        runner = sim["runner"]

        # Other simulation information
        agent_model = sim["agent_model"]
        target_model = sim["target_model"]

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
            }

            sim_results = {}
            for (col, res) in zip(columns, results):
                sim_results.update({col: res})

            # store results
            batch_results.update({sim_name: [sim_parameters, sim_results]})

    # post-process
    post_process_batch_simulation(batch_results)

    # for sim_name, sim in batch_results.items():
    # parameters = sim[0]
    # results = sim[1]

    # if parameters['dim'] == 2:
    #     if parameters['agent_model'] == 'Double Integrator':
    #         post_process_identical_2d_doubleint_TEST(sim_name, results)

    #     # if parameters['agent_model'] == 'Linearized Quadcopter':
    #     #     post_process_identical_2d_doubleint()

    # if parameters['dim'] == 3:
    #     if parameters['agent_model'] == 'Double Integrator':
    #         post_process_identical_3d_doubleint_TEST(sim_name, results)

    #     # if parameters['agent_model'] == 'Linearized Quadcopter':
    #     #     post_process_identical_3d_doubleint()

    # TODO need post-process unique to dimension and dynamic model being used

    # store results
    # results.append(yout)

    # simulation comparison post-process
    # for r in results:

    #     # superimposed trajectories from emd and dyn simulations
    #     plt.figure()
    #     for zz in range(2):
    #         y_agent = yout[:, zz*4:(zz+1)*4]
    #         plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
    #         plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

    #     for zz in range(2):
    #         y_agent = yout_dyn[:, zz*4:(zz+1)*4]
    #         plt.plot(y_agent[0, 0], y_agent[0, 1], 'rs')
    #         plt.plot(y_agent[:, 0], y_agent[:, 1], '-r')

    plt.show()

    print("done!")
