
import numpy as np
from assignments import *
from controls import *
from dynamics import *
from engine import *
from linear_models_2D import *
from linear_models_3D import *
from run import *

def setup_simulation(agent_model, target_model, nagents, ntargets, dim=2):

    sim = {}
    parameters = ['dx', 'du', 'A', 'B', 'agent_dyn', 'target_dyns', 'agent_pol', 'target_pol', 'asst_pol', 'x0']
    sim.fromkeys(parameters)

    nagents=  nagents
    ntargets = ntargets

    x0 = None
    cities = None

    ##### Dynamic Model #####
    if dim == 2:

        if agent_model == "Double Integrator":

            A, B, dx, du = double_integrator_2D()

            ### Initial conditions
            x0 = np.array([-10, -9, -6, -8, 6, -2, -10, 7,
                           7, 1, 1, -8, -2, -1, 0, 8])

        # Target Terminal Location
        cities = [np.array([10, 0, 0, 0]), np.array([5, -5, 0, 0])]

    if dim == 3:

        if agent_model == "Double Integrator":

            A, B, dx, du = double_integrator_3D()

            ### Initial conditions
            x0 = np.array([-10, -9, -6, -8, -6, -8,
                           6, -2, -10, 7, -10, 7,
                           7, 1, 1, -8, 1, -8,
                           -2, -1, 0, 8, 0, 8])

            # Target Terminal Locations
            cities = [np.array([10, 0, 0, 0, 0, 0]), np.array([5, -5, 0, 0, 0, 0])]

        if agent_model == "Linearized Quadcopter":

            A, B, dx, du = quadcopter_3D()
            ### Initial conditions
            x0 = np.array([-10, -9, -6, -8, -6, -8,
                           6, -2, -10, 7, -10, 7,
                           7, 1, 1, -8, 1, -8,
                           -2, -1, 0, 8, 0, 8])

            # Target Terminal Locations
            cities = [np.array([10, 0, 0]), np.array([5, -5, 0])]

    ### Dynamics
    ltidyn = LTIDyn(A, B)
    dyn_target = LTIDyn(A, B)

    ### agent control law
    Q = np.eye(dx)
    R = np.eye(du)
    poltrack = LinearFeedbackTracking(A, B, Q, R)
    ### target control law
    Q = np.eye(dx)
    R = np.eye(du)
    poltarget = [LinearFeedbackOffset(A, B, Q, R, c) for c in cities]

    ### assignment policy
    apol = []
    # EMD
    apol.append(AssignmentEMD(nagents, ntargets)) 
    # Dyn
    apol.append(AssignmentDyn(nagents, ntargets))

    ### runner
    sim_runner = run_identical_doubleint


    sim['dx'] = dx
    sim['du'] = du
    sim['x0'] = x0
    sim['agent_dyn'] = ltidyn
    sim['target_dyns'] = dyn_target
    sim['agent_pol'] = poltrack
    sim['target_pol'] = poltarget
    sim['asst_pol'] = apol
    sim['nagents'] = nagents
    sim['ntargets'] = ntargets
    sim['runner'] = sim_runner

    return sim
















        # nagents=  2
        # ntargets = 2
        # nagents=  3
        # ntargets = 3

        # Agent dynamics
        # ltidyn = LTIDyn(A, B)
        # dyn_target = LTIDyn(A, B)
        # Agent control law
        # poltrack = LinearFeedbackTracking(A, B, Q, R)
        # poltarget = LinearFeedback(A, B, Q, R)

        # NEW
        # Agent Closed-Loop Dynamics
        # ltidyn_cl = LTIDyn_closedloop(A, B, poltrack.K)
        # ltidyn_cl = LTIDyn(A, B)


        # Target control law
        # cities = [np.array([10, 0]), np.array([5, -5])]
        # cities = [np.array([10, 0, 0, 0]), np.array([5, -5, 0, 0])]
        # cities = [np.array([10, 4, -2, 3]), np.array([2, -12, -12, 12]),
        #           np.array([-12, 10, 0, -5]), np.array([7, -2, 0, 6]), 
        #           np.array([13, -23, 1, -7]), np.array([5, -5, 0, -9]),
        #           np.array([44, 3, 0, 9]), np.array([15, -15, 0, 10])]
        # cities = [np.array([10, -3, -8, -9]), np.array([5, -5, 22, -6])]


        # ntargets = 3
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
        # cities = [np.array([-10,0], np.float_), np.array([10,0], dtype=np.float_), np.array([5,-5], dtype=np.float_)] # ntargets = 4
        # cities = [np.array([-10,0,0,0], np.float_), np.array([10,0,0,0], dtype=np.float_), np.array([5,-5,0,0], dtype=np.float_)] # ntargets = 4
        # ntargets = 4
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]
        # ntargets = 8
        # cities = [100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2), 100*np.random.rand(2)]

        # poltarget = [LinearFeedbackOffset(A, B, Q, R, c) for c in cities]
        # poltarget = [LinearFeedbackTracking(A, B, Q, R) for c in cities]

        # poltarget = [LinearFeedback(A, B, Q, R) for ii in range(ntargets)]

        # poltarget = [ZeroPol(du) for ii in range(ntargets)]
        # poltarget = ZeroPol(du)

        # 1v1
        # x0 = np.array([5, 5, 0, 0,
        #                -10, -5, 5, 5])
        # apol = AssignmentLexical(1, 1)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, [poltarget[0]], apol, 1, 1)


        # 2 v 2 Lexical
        # apol = AssignmentLexical(2, 2)
        # x0 = np.array([5, 5, 0, 0, 10, 10, -3, 8,
        #                -10, -5, 5, 5, 20, 10, -3, -8])

        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 2, 2)

        # 2 v 2 EMD
        # x0 = np.random.rand(16) * 20 - 10
        # x0 = np.array([-10, -9, -6, -8, 6, -2, -10, 7,
        #                7, 1, 1, -8, -2, -1, 0, 8])

        # 3 v 3 EMD
        # apol = AssignmentEMD(3, 3)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5])
        # x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
        #                -110, -15, 15, 15])
        # x0 = np.hstack((x0, x02))

        # x0 = np.array([-9.49472109, -9.01609684, -6.30746324, -8.61933317, -4.85049153,  8.27163463,
        #     -0.84300976, -7.39576421,  6.19783331, -1.93060319, -9.5113471,   7.13662085,
        #     -4.51410362,  4.18211928, -2.88455314,  5.88618124,  6.89237722,  0.76295034,
        #      1.18173033, -7.54980037, -2.44716163, -1.42505342,  0.22417293,  7.8352514 ], dtype=np.float_)

        # 4 v 4 EMD
        # apol = AssignmentEMD(4, 4)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5, 20, 10, -3, -8])
        # x02 = np.array([15, 15, 110, -110, 110, 110, -13, 13,
        #                -110, -15, 15, 15, 120, 110, -13, -18])
        # x0 = np.hstack((x0, x02))

        # 8 v 8 EMD
        # apol = AssignmentEMD(8, 8)
        # x0 = np.array([5, 5, 10, -10, 10, 10, -3, 3,
        #                -10, -5, 5, 5, 20, 10, -3, -8,
        #                25, 5, 10, -15, 7, 23, -1, 13,
        #                -92, -12, 33, 66, 123, 11, -1, -18])
        # x02 = np.array([15, 15, 11, -11, 11, 11, -1, 13,
        #                 -7, -15, 15, 15, 17, 13, -13, -18,
        #                 25, 35, 18, -17, -9, 10, -13, 13,
        #                -14, -45, 15, 15, 20, 14, -16, -18])
        # x0 = np.hstack((x0, x02))



        # yout = run_identical_doubleint(dx, du, x0, ltidyn, dyn_target, poltrack, poltarget, apol, 2, 2)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 4, 4)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 8, 8)

        # 2 v 2 EMD
        # apol.append(AssignmentEMD(nagents, ntargets))

        # 2 v 2 Dyn
        # apol.append(AssignmentDyn(nagents, ntargets))

        # sim_runner = run_identical_doubleint

        # 3 v 3 Dyn
        # apol = AssignmentDyn(3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 3, 3)
        # run_identical_doubleint(dx, du, x0, ltidyn, poltrack, poltarget, apol, 3, 3)

        # 4 v 4 Dyn
        # apol = AssignmentDyn(4, 4)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 4, 4)

        # 8 v 8 Dyn
        # apol = AssignmentDyn(8, 8)
        # run_identical_doubleint(dx, du, x0, ltidyn_cl, ltidyn, poltrack, poltarget, apol, 8, 8)

