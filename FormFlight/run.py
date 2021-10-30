
""" @file run.py
"""

from time import time, process_time
import pandas as pd

from DOT_assignment import agents as ag
from DOT_assignment import controls
from DOT_assignment import dynamics
from DOT_assignment import engine
from DOT_assignment import systems
from DOT_assignment import post_process

#######################
# These functions perform the individual simulations. They organize the intial conditions into
# appropriate data structures (ie. Agents, Points), setup the simulation engine, simulation scenario
# (ie. formations) and return the results and diagnostics

# TODO update function naming convention
######################

def run_identical_doubleint_2D(dx, du, statespace, x0, ltidyn, poltrack, apol,
        assignment_epoch, nagents, ntargets, collisions, collision_tol, dt=0.01, maxtime=10):

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
        - collision_tol:abosolute distance between an agent and tolerance to count as collision
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 2

    agents = [ag.TrackingAgent(dx, du, statespace, dim, ltidyn, poltrack[ii]) for ii in range(nagents)]
    targets = [ag.Point(dx, du, statespace, dim) for ii in range(ntargets)]

    # setup the scenario and engine
    sys = systems.OneVOneFormation(agents, targets, apol, assignment_epoch)
    eng = engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=collision_tol)

    # TODO time the simulation
    start_run_time = process_time()
    eng.run(x0, sys)
    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    # TODO
    output = [agents, targets, eng.df, poltrack, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics

def run_identical_doubleint_3D(dx, du, statespace, x0, ltidyn, poltrack, apol,
        assignment_epoch, nagents, ntargets, collisions, collision_tol, dt=0.01, maxtime=10):

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
        - collision_tol:abosolute distance between an agent and tolerance to count as collision
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 3

    agents = [ag.TrackingAgent(dx, du, statespace, dim, ltidyn, poltrack[ii]) for ii in range(nagents)]
    targets = [ag.Point(dx, du, statespace, dim) for ii in range(ntargets)]

    # setup the scenario and engine
    sys = systems.OneVOneFormation(agents, targets, apol, assignment_epoch)
    eng = engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=collision_tol)

    # TODO time the simulation
    start_run_time = process_time()

    eng.run(x0, sys)

    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    output = [agents, targets, eng.df, poltrack, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics

def run_identical_linearized_quadcopter_2D(dx, du, statespace, x0, ltidyn, poltrack, apol,
        assignment_epoch, nagents, ntargets, collisions, collision_tol, dt=0.01, maxtime=10):

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
        - collision_tol:abosolute distance between an agent and tolerance to count as collision
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 2

    agents = [ag.TrackingAgent(dx, du, statespace, dim, ltidyn, poltrack[ii]) for ii in range(nagents)]
    targets = [ag.Point(dx, du, statespace, dim) for ii in range(ntargets)]

    # setup the scenario and engine
    sys = systems.OneVOneFormation(agents, targets, apol, assignment_epoch)
    eng = engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=collision_tol)

    # TODO time the simulation
    start_run_time = process_time()
    eng.run(x0, sys)
    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    output = [agents, targets, eng.df, poltrack, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics


def run_identical_linearized_quadcopter_3D(dx, du, statespace, x0, ltidyn, poltrack, apol,
        assignment_epoch, nagents, ntargets, collisions, collision_tol, dt=0.01, maxtime=10):

    """ Setup the engine and simulation scenario

    Input:
        - dx:           agent statesize
        - du:           agent control input size
        - statespace:   dict describing agent position, velocity etc. components
        - x0:           initial agent, target, target terminal states
        - ltidyn:       agent dynamics model (homogeneous across agent swarm)
        - poltrack:     agent control policy (homogeneous across agent swarm)
        - apol:         assignment policy
        - assignment_epoch: number of ticks at which to perform assignment
        - nagents:      number of agents
        - ntargets:      number of targets
        - collisions:   collisions on/off
        - collision_tol:abosolute distance between an agent and tolerance to count as collision
        - dt:           engine tick size
        - maxtime:      simulation time

    Output: Returns simulation results and diagnostics

    """

    dim = 3

    agents = [ag.TrackingAgent(dx, du, statespace, dim, ltidyn, poltrack[ii]) for ii in range(nagents)]
    targets = [ag.Point(dx, du, statespace, dim) for ii in range(ntargets)]

    # setup the scenario and engine
    sys = systems.OneVOneFormation(agents, targets, apol, assignment_epoch)
    eng = engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=collision_tol)

    # time the simulation
    start_run_time = process_time()
    eng.run(x0, sys)
    elapsed_run_time = process_time() - start_run_time

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    # TODO - need to clarify what each of these fields are
    output = [agents, targets, eng.df, poltrack, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

    ### diagnostics
    runtime_diagnostics = eng.diagnostics
    runtime = pd.DataFrame([elapsed_run_time])
    runtime_diagnostics = pd.concat([runtime_diagnostics, runtime], axis=1, ignore_index=True)

    diagnostics = [runtime_diagnostics]

    return output, diagnostics


