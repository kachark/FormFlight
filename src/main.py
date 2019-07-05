
import matplotlib.pyplot as plt
import numpy as np
from setup import *
from post_process import *

if __name__ == "__main__": 

    batch_simulation = []
    results = []

    # SIM SETUP
    dt = 0.01
    maxtime = 5
    dim = 2
    nagents = 5
    ntargets = 5
    agent_model = "Double Integrator"
    target_model = "Double Integrator"
    # initial_conditions = np.loadtxt("initial_conditions_2d.txt") # agents and targets
    # initial_conditions = np.loadtxt("initial_conditions_3d.txt")
    # cities = --> some distribution
    control_policy = "LQR"
    sim = setup_simulation(agent_model, target_model, control_policy, nagents, ntargets, dim, dt, maxtime)

    batch_simulation.append(sim)

    # RUN SIMULATION
    for sim in batch_simulation:
        dt = sim['dt']
        maxtime = sim['maxtime']
        dx = sim['dx']
        du = sim['du']
        x0 = sim['x0']
        ltidyn = sim['agent_dyn']
        target_dyn = sim['target_dyns']
        poltrack = sim['agent_pol']
        poltargets = sim['target_pol']
        apol = sim['asst_pol']
        nagents = sim['nagents']
        ntargets = sim['ntargets']
        runner = sim['runner']

        # run different assignment policies with same conditions
        for assignment_pol in apol:
            yout = runner(dx, du, x0, ltidyn, target_dyn, poltrack, poltargets, assignment_pol, nagents, ntargets, dt, maxtime)

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
