
""" @file run.py
"""

from time import time, process_time
import pandas as pd

from agents import *
from controls import *
from dynamics import *
from engine import *
from systems import *
from post_process import *


def run_identical_doubleint_2D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol, assignment_epoch, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    """ Setup the engine and simulation scenario

    Input:
        - dx:           agent statesize
        - du:           agent control input size
        - statespace:   dict describing agent position, velocity etc. components
        - x0:           initial agent, target, target terminal states
        - ltidyn:       agent dynamics model (homogeneous across agent swarm)
        - dyn_target:   list of target dynamic models
        - poltrack:     agent control policy (homogeneous across agent swarm)
        - poltargets:   list of target control policies
        - apol:         assignment policy
        - assignment_epoch: number of ticks at which to perform assignment
        - nagents:      number of agents
        - ntarges:      number of targets
        - collisions:   collisions on/off
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 2

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)] # give each trackingagent a tracking policy pre-assigned to Target 0
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    # agents, targets, interactions
    sys = OneVOne(agents, targets, apol, assignment_epoch)

    # tells system to update, collisions
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-3)

    # TODO time the simulation
    start_run_time = process_time()
    eng.run(x0, sys)
    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    output = [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics

def run_identical_doubleint_3D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol, assignment_epoch, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    """ Setup the engine and simulation scenario

    Input:
        - dx:           agent statesize
        - du:           agent control input size
        - statespace:   dict describing agent position, velocity etc. components
        - x0:           initial agent, target, target terminal states
        - ltidyn:       agent dynamics model (homogeneous across agent swarm)
        - dyn_target:   list of target dynamic models
        - poltrack:     agent control policy (homogeneous across agent swarm)
        - poltargets:   list of target control policies
        - apol:         assignment policy
        - assignment_epoch: number of ticks at which to perform assignment
        - nagents:      number of agents
        - ntarges:      number of targets
        - collisions:   collisions on/off
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 3

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol, assignment_epoch)
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-1)

    # TODO time the simulation
    start_run_time = process_time()

    eng.run(x0, sys)

    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    output = [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics

def run_identical_linearized_quadcopter_3D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol,
        assignment_epoch, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    """ Setup the engine and simulation scenario

    Input:
        - dx:           agent statesize
        - du:           agent control input size
        - statespace:   dict describing agent position, velocity etc. components
        - x0:           initial agent, target, target terminal states
        - ltidyn:       agent dynamics model (homogeneous across agent swarm)
        - dyn_target:   list of target dynamic models
        - poltrack:     agent control policy (homogeneous across agent swarm)
        - poltargets:   list of target control policies
        - apol:         assignment policy
        - assignment_epoch: number of ticks at which to perform assignment
        - nagents:      number of agents
        - ntarges:      number of targets
        - collisions:   collisions on/off
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 3

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    # sys = OneVOne(agents, targets, apol)
    sys = OneVOne(agents, targets, apol, assignment_epoch)
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-2)

    # TODO time the simulation
    start_run_time = process_time()
    eng.run(x0, sys)
    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    output = [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics


