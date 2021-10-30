
""" @file run.py
"""

from time import time, process_time
import pandas as pd
import numpy as np
import copy

from FormFlight import agents as ag
from FormFlight import controls
from FormFlight import dynamics
from FormFlight import engine
from FormFlight import systems
from FormFlight import post_process

def OneVOne_runner(scenario, initial_world_state, world_i, sim_params):

    if __debug__:
        initial_world_state = np.array(
            [-8.32327564e+02,  2.51392215e+03, .73456152e+00,  2.73114328e+02,
            -2.37306111e+03,  2.88803109e+00, -1.62070983e+03, -2.06760102e+03,
            2.52632250e+00, 4.87454560e+02,  1.23201288e+03,  3.92638831e+00,
            1.81471761e+03,-4.23033139e+02,  4.83810828e+00,  1.42785511e+03,
            1.39889334e+03, 3.43306485e+00,  2.38943072e+03,  1.08426222e+03,
            4.69209264e+00,-3.17932838e+03,  1.72783599e+03,  1.21169148e-02,
            -2.07303607e+03, -3.14962489e+03,  1.62804333e+00, -9.47913323e+02,
            7.92323724e+02, 2.55105370e-01, -3.53428107e+03, -1.98636531e+03,
            3.67451532e+00, 3.27925272e+03, -2.04871648e+03,  1.83165140e+00,
            4.10818454e+03, -1.04673227e+03,  8.83764978e+00, -1.60666025e+03,
            -3.79479362e+03,  3.91180705e+00,  2.13306797e+03,  3.35381193e+03,
            1.08445293e-01, -1.10379235e+03, -3.87809773e+03,  1.16079642e+00,
            -2.92600538e+03, 4.26388829e+03,  7.56734885e+00, -3.01729386e+03,
            -2.97921793e+03, 2.07106134e+00,  2.79127052e+03,  4.40727985e+03,
            8.45199055e+00,  5.01305241e+03,  4.06565980e+03,  6.11656441e+00,
            3.50000000e+03,  0.00000000e+00,  0.00000000e+00,  2.83155948e+03,
            2.05724838e+03,  0.00000000e+00,  1.08155948e+03,  3.32869781e+03,
            0.00000000e+00, -1.08155948e+03,  3.32869781e+03,  0.00000000e+00,
            -2.83155948e+03,  2.05724838e+03,  0.00000000e+00, -3.50000000e+03,
            4.28626380e-13,  0.00000000e+00, -2.83155948e+03, -2.05724838e+03,
            0.00000000e+00, -1.08155948e+03, -3.32869781e+03,  0.00000000e+00,
            1.08155948e+03, -3.32869781e+03,  0.00000000e+00,  2.83155948e+03,
            -2.05724838e+03, 0.00000000e+00]
        ) # plug in custom initial condition
        print("X0: ", initial_world_state)

    # the propogator/integrator + scenario specific events (ie. decision)
    sys = systems.OneVOne(scenario, world_i)

    # physics engine
    dim = sim_params['dim']
    dt = sim_params['dt']
    maxtime = sim_params['maxtime']
    collisions = sim_params['collisions']
    collision_tol = sim_params['collision_tol']
    eng = engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions,
            collision_tol=collision_tol)

    # run simulation
    start_run_time = process_time()
    eng.run(initial_world_state, sys)
    elapsed_run_time = process_time() - start_run_time

    output = [scenario, eng.df, world_i]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics

