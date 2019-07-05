
import numpy as np
from assignments import *
from controls import *
from dynamics import *
from engine import *
from linear_models_2D import *
from linear_models_3D import *
from run import *

def setup_simulation(agent_model, target_model, control_policy, nagents, ntargets, dim=2, dt=0.01, maxtime=10):

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

            A, B, C, D, dx, du = double_integrator_2D()

            ### Initial conditions

            # Agents
            x0 = np.zeros((nagents, dx))
            x0p = np.random.uniform(-50, 50, (nagents,dim)) # position spread
            x0v = np.random.uniform(-50, 50, (nagents,(dx-dim))) # velocity spread
            for ii, (pos, vel) in enumerate(zip(x0p, x0v)):
                x0[ii] = np.concatenate((pos, vel), axis=0)
            x0 = x0.flatten()

            # Targets
            # r = 150 # circle radius
            # x02p = [circle(r, ntargets, t) for t in range(ntargets)]
            x02p = np.random.uniform(-50, 50, (nagents,dim)) # position spread
            x02 = np.zeros((ntargets, dx))
            vel_range = 10
            for ii, tt in enumerate(x02):
                # x02[ii] = np.array([x02p[ii][0], x02p[ii][1], 0, 0])
                # x02[ii] = np.array([x02p[ii][0], x02p[ii][1], 1, 0])
                x02[ii] = np.array([x02p[ii][0], x02p[ii][1], np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))


            # Target Terminal Location
            cities = np.zeros((ntargets, dx))
            r = 50
            # cities_p = [circle(r, ntargets, t) for t in range(ntargets)]
            cities_p = np.random.uniform(-30, 30, (ntargets,dim)) # city position spread
            for ii, tt in enumerate(cities):
                cities[ii] = np.array([cities_p[ii][0], cities_p[ii][1], 0, 0])

            cities = cities.flatten()
            # cities = r*cities
            cities = np.split(cities, ntargets)



            # # TEST CASES - 1v1
            # # x0 = np.array([-10, -9, -6, -8,
            # #                7, 1, 1, -8])

            # # x0 = np.array([5, 5, 0, 0,
            # #                -10, -5, 5, 5])
            # # TEST CASES - 2v2
            # x0 = np.array([-10, -9, -6, -8, 6, -2, -10, 7,
            #                7, 1, 1, -8, -2, -1, 0, 8])

            # cities = [np.array([10, 0, 0, 0]), np.array([5, -5, 0, 0])] # 2v2 test case
            # # cities = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])] # 2v2 test case
            # # cities = [np.array([-10, 0, 0, 0])] # 1v1 test case
            # # cities = [np.array([0, 0, 0, 0])] # 1v1 test case

            ### runner
            sim_runner = run_identical_doubleint_2D

    if dim == 3:

        if agent_model == "Double Integrator":

            A, B, C, D, dx, du = double_integrator_3D()

            # Agents
            x0 = np.zeros((nagents, dx))
            x0p = np.random.uniform(-50, 50, (nagents,dim)) # position spread
            x0v = np.random.uniform(-50, 50, (nagents,(dx-dim))) # velocity spread
            for ii, (pos, vel) in enumerate(zip(x0p, x0v)):
                x0[ii] = np.concatenate((pos, vel), axis=0)
            x0 = x0.flatten()

            # Targets
            # r = 150 # circle radius
            # x02p = [circle(r, ntargets, t) for t in range(ntargets)]
            x02p = np.random.uniform(-50, 50, (nagents,dim)) # position spread
            x02 = np.zeros((ntargets, dx))
            vel_range = 10
            for ii, tt in enumerate(x02):
                # x02[ii] = np.array([x02p[ii][0], x02p[ii][1], 0, 0])
                # x02[ii] = np.array([x02p[ii][0], x02p[ii][1], 1, 0])
                x02[ii] = np.array([x02p[ii][0], 
                                    x02p[ii][1],
                                    x02p[ii][2],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))

            # Target Terminal Location
            # cities = [np.array([10, 0, 0, 0]), np.array([5, -5, 0, 0])] # 2v2 test case
            # cities = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])] # 2v2 test case
            # cities = [np.array([-10, 0, 0, 0])] # 1v1 test case
            # cities = [np.array([0, 0, 0, 0])] # 1v1 test case
            cities = np.zeros((ntargets, dx))
            r = 300
            cities_p = [circle(r, ntargets, t) for t in range(ntargets)]
            # cities_p = np.random.uniform(-300, 300, (ntargets,dim)) # city position spread
            for ii, tt in enumerate(cities):
                cities[ii] = np.array([cities_p[ii][0], 0, cities_p[ii][1], 0, 0, 0])

            cities = cities.flatten()
            cities = r*cities
            cities = np.split(cities, ntargets)

            ### runner
            sim_runner = run_identical_doubleint_3D

        if agent_model == "Linearized Quadcopter":

            A, B, C, D, dx, du = quadcopter_3D()
            ### Initial conditions
            x0 = np.array([-10, -9, -6, -8, -6, -8,
                           6, -2, -10, 7, -10, 7,
                           7, 1, 1, -8, 1, -8,
                           -2, -1, 0, 8, 0, 8])

            # Target Terminal Locations
            cities = [np.array([10, 0, 0]), np.array([5, -5, 0])]

            ### runner
            sim_runner = run_identical_doubleint_3D

    Q = np.eye(dx)
    R = np.eye(du)

    #TEST
    ######################
    if dim == 2:
        Q2 = copy.deepcopy(Q)
        Q2[2,2] = 0.0
        Q2[3,3] = 0.0

        Q3 = copy.deepcopy(Q)
        Q3[0, 0] = 1000
        Q3[1, 1] = 1000
        Q3[2,2] = 0.0
        Q3[3,3] = 0.0
        
    if dim == 3:
        Q2 = copy.deepcopy(Q)
        Q2[3,3] = 0.0
        Q2[4,4] = 0.0
        Q2[5,5] = 0.0

        Q3 = copy.deepcopy(Q)
        Q3[0, 0] = 10
        Q3[1, 1] = 10
        Q3[2, 2] = 10
        Q3[3,3] = 0.0
        Q3[4,4] = 0.0
        Q3[5,5] = 0.0
 
    ######################

    ### target control law
    # poltarget = [LinearFeedbackOffset(A, B, C, Q, R, c) for c in cities]
    # poltarget = [ZeroPol(du) for c in cities]
    poltargets = [LinearFeedbackConstTracker(A, B, Q2, 2*R, c) for c in cities]
    ### target Dynamics
    dyn_target = LTIDyn(A, B)

    ### agent control law
    if control_policy == "LQR":
        # poltrack = LinearFeedbackTracking(A, B, C, Q, R)

        #TEST
        # initialize LinearFeedbackAugmented by pre-assigning/augmenting this policy with Target 0
        Acl = poltargets[0].get_closed_loop_A()
        gcl = poltargets[0].get_closed_loop_g()
        print("gcl = ", gcl)
        poltrack = LinearFeedbackAugmented(A, B, Q3, R, Acl, gcl) # initial augmentation: agent_i tracks target_i

    if control_policy == "LQI":
        A, B, Qaug, Raug = LQI_augmented_system(A, B, C) # augmented A, B
        dx = A.shape[0]
        poltrack = LinearFeedbackIntegralTracking(A, B, Qaug, Raug)

    ### Agent Dynamics
    ltidyn = LTIDyn(A, B)

    ### assignment policy
    apol = []
    # EMD
    # apol.append(AssignmentEMD(nagents, ntargets)) 
    # Dyn
    apol.append(AssignmentDyn(nagents, ntargets))

    sim['dt'] = dt
    sim['maxtime'] = maxtime
    sim['dx'] = dx
    sim['du'] = du
    sim['x0'] = x0
    sim['agent_dyn'] = ltidyn
    sim['target_dyns'] = dyn_target
    sim['agent_pol'] = poltrack
    sim['target_pol'] = poltargets
    sim['asst_pol'] = apol
    sim['nagents'] = nagents
    sim['ntargets'] = ntargets
    sim['runner'] = sim_runner

    return sim

# Formations
def circle(r, ntargets, target):
    angle = target*(2*np.pi)/ntargets
    y = np.sin(angle)
    x = np.cos(angle)
    return x,y

