
import pandas as pd

from agents import *
from controls import *
from dynamics import *
from engine import *
from systems import *
from post_process import *


# def run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltargets, apol, nagents, ntargets):
def run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets):

    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=0.01, maxtime=6, collision_tol=1e-3)
    eng.run(x0, sys)
    # print(eng.df.tail(10))

    post_process_identical_2d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)
    # post_process_identical_3d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)

    return eng.df.iloc[:, 1:].to_numpy() # yout

def run_quadcopter_vs_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltargets, apol, nagents, ntargets):

    agents = [TrackingAgent(dx, ltidyn, poltrack) for ii in range(nagents)]
    targets = [Agent(dx, dyn_target, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = OneVOne(agents, targets, apol)
    eng = Engine(dt=0.01, maxtime=6, collision_tol=1e-3)
    eng.run(x0, sys)
    # print(eng.df.tail(10))

    post_process_identical_2d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)
    # post_process_identical_3d_doubleint(eng.df, poltrack, poltargets, nagents, ntargets, sys.costs)

    return eng.df.iloc[:, 1:].to_numpy() # yout

