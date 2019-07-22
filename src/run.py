
import pandas as pd

from agents import *
from controls import *
from dynamics import *
from engine import *
from systems import *
from post_process import *


def run_identical_doubleint_2D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    dim = 2

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)] # give each trackingagent a tracking policy pre-assigned to Target 0
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    # agents, targets, interactions
    sys = OneVOne(agents, targets, apol)

    # tells system to update, collisions
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-3)
    eng.run(x0, sys)

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    return [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

def run_identical_doubleint_3D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    dim = 3

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-1)
    eng.run(x0, sys)

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    return [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]

def run_identical_linearized_quadcopter_3D(dx, du, statespace, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets, collisions, dt=0.01, maxtime=10):

    dim = 3

    agents = [TrackingAgent(dx, statespace, dim, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, statespace, dim, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=1e-1)
    eng.run(x0, sys)

    opt_asst = sys.optimal_assignment

    # post processing
    polagents = [agent.pol for agent in agents]

    return [agents, targets, eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents, opt_asst, apol]


