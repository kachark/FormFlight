
import pandas as pd

from agents import *
from controls import *
from dynamics import *
from engine import *
from systems import *
from post_process import *


def run_identical_doubleint_2D(dx, du, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets, dt=0.01, maxtime=10):

    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)] # give each trackingagent a tracking policy pre-assigned to Target 0
    targets = [Agent(dx, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    # agents, targets, interactions
    sys = OneVOne(agents, targets, apol)

    # tells system to update, collisions
    eng = Engine(dt=dt, maxtime=maxtime, collision_tol=1e-3)
    eng.run(x0, sys)
    # print(eng.df.tail(10))

    # post processing
    polagents = [agent.pol for agent in agents]
    post_process_identical_2d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents)
    # post_process_identical_2d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)

    return eng.df.iloc[:, 1:].to_numpy() # yout

def run_identical_doubleint_3D(dx, du, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets, dt=0.01, maxtime=10):

    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=dt, maxtime=maxtime, collision_tol=1e-3)
    eng.run(x0, sys)
    # print(eng.df.tail(10))

    # post processing
    polagents = [agent.pol for agent in agents]
    post_process_identical_3d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs, polagents)
    # post_process_identical_3d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)

    return eng.df.iloc[:, 1:].to_numpy() # yout
