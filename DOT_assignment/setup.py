
""" @file setup.py
"""

import numpy as np
import copy

# DOT_assignment
from DOT_assignment import assignments
from DOT_assignment import controls
from DOT_assignment import dynamics
from DOT_assignment import engine
from DOT_assignment import linear_models_2D
from DOT_assignment import linear_models_3D
from DOT_assignment import run

def setup_simulation(sim_profile):

    """ Returns dictionary of controls, dynamics, decision-making policy, and initial state parameters

    Input: Standard python dict containing descriptors outlining simulation requirements
    Output: Standard python dict containing controls, dynamics, assignment, etc. data structures

    """

    x0 = None
    stationary_states = None

    agent_model = sim_profile["agent_model"]
    target_model = sim_profile["target_model"]
    agent_control_policy = sim_profile["agent_control_policy"]
    target_control_policy = sim_profile["target_control_policy"]
    assignment_policy = sim_profile["assignment_policy"]
    assignment_epoch = sim_profile["assignment_epoch"]
    nagents = sim_profile["nagents"]
    ntargets = sim_profile["ntargets"]
    collisions = sim_profile["collisions"]
    collision_tol = sim_profile["collision_tol"]
    dim = sim_profile["dim"]
    dt = sim_profile["dt"]
    maxtime = sim_profile["maxtime"]
    initial_conditions = sim_profile['initial_conditions']

    if initial_conditions == None:
        ic = generate_initial_conditions(dim, agent_model, target_model, nagents, ntargets)
        x0 = ic[0]
        stationary_states = ic[1]
    else:
        x0 = initial_conditions[0]
        stationary_states = initial_conditions[1]

    sim = {}
    parameters = ['agent_model', 'target_model', 'dx', 'du', 'A', 'B', 'agent_dyn', 'target_dyns', 'agent_pol', 'target_pol', 'asst_pol', 'x0']
    sim.fromkeys(parameters)

    ##### Dynamic Model #####
    if dim == 2:

        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_2D.double_integrator_2D()

            ### runner
            sim_runner = run.run_identical_doubleint_2D

        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_2D.quadcopter_2D()

            ### runner
            sim_runner = run.run_identical_linearized_quadcopter_2D

    if dim == 3:

        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_3D.double_integrator_3D()

            ### runner
            sim_runner = run.run_identical_doubleint_3D

        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_3D.quadcopter_3D()

            ### runner
            sim_runner = run.run_identical_linearized_quadcopter_3D

    Q = np.eye(dx)
    R = np.eye(du)

    #TEST
    Q2 = None
    Q3 = None
    ######################
    if dim == 2:
        if agent_model == 'Double_Integrator':
            Q2 = copy.deepcopy(Q)
            Q2[2,2] = 0.0
            Q2[3,3] = 0.0

            Q3 = copy.deepcopy(Q)
            Q3[0, 0] = 100
            Q3[1, 1] = 100
            Q3[2,2] = 0.0
            Q3[3,3] = 0.0
        if agent_model == 'Linearized_Quadcopter':

            Q3 = copy.deepcopy(Q)
            Q3[0, 0] = 100
            Q3[1, 1] = 100
            Q3[2,2] = 100
            Q3[3,3] = 100
            Q3[4,4] = 0.0
            Q3[5,5] = 0.0
            Q3[6, 6] = 0.0
            Q3[7, 7] = 0.0


    if dim == 3:
        if agent_model == 'Double_Integrator':
            Q2 = copy.deepcopy(Q)
            Q2[3,3] = 0.0
            Q2[4,4] = 0.0
            Q2[5,5] = 0.0

            Q3 = copy.deepcopy(Q)
            Q3[0, 0] = 1000
            Q3[1, 1] = 1000
            Q3[2, 2] = 1000
            Q3[3,3] = 0.0
            Q3[4,4] = 0.0
            Q3[5,5] = 0.0
        if agent_model == 'Linearized_Quadcopter':
            Q3 = copy.deepcopy(Q)
            Q3[0, 0] = 1000
            Q3[1, 1] = 1000
            Q3[2, 2] = 1000
            Q3[3,3] = 1000
            Q3[4,4] = 1000
            Q3[5,5] = 1000
            Q3[6,6] = 0.0
            Q3[7,7] = 0.0
            Q3[8,8] = 0.0
            Q3[9, 9] = 0.0
            Q3[10, 10] = 0.0
            Q3[11, 11] = 0.0

    ######################

    ### target control law

    if target_control_policy == "LQR":
        # poltargets = [controls.LinearFeedbackOffset(A, B, C, Q, R, c) for c in stationary_states]
        poltargets = [controls.LinearFeedbackConstTracker(A, B, Q, R, c) for c in stationary_states]

    # poltargets = [controls.ZeroPol(du) for c in stationary_states]

    ### target Dynamics
    dyn_target = dynamics.LTIDyn(A, B)

    ### agent control law
    if agent_control_policy == "LQR":
        # const = np.array([0, 0, 0, 0])
        # poltrack = LinearFeedbackConstTracker(A, B, Q2, R, const) # initial augmentation: agent_i tracks target_i

        # initialize LinearFeedbackAugmented by pre-assigning/augmenting this policy with Target 0
        Acl = poltargets[0].get_closed_loop_A()
        gcl = poltargets[0].get_closed_loop_g()
        poltrack = controls.LinearFeedbackAugmented(A, B, Q3, R, Acl, gcl) # initial augmentation: agent_i tracks target_i

    ### Agent Dynamics
    ltidyn = dynamics.LTIDyn(A, B)

    ### Assignment Policy
    if assignment_policy == 'AssignmentDyn':
        apol = assignments.AssignmentDyn(nagents, ntargets)

    if assignment_policy == 'AssignmentEMD':
        apol = assignments.AssignmentEMD(nagents, ntargets)

    ### CONSTRUCT SIMULATION DICTIONARY
    sim['agent_control_policy'] = agent_control_policy
    sim['target_control_policy'] = target_control_policy
    sim['agent_model'] = agent_model
    sim['target_model'] = target_model
    sim['collisions'] = collisions
    sim['collision_tol'] = collision_tol
    sim['dt'] = dt
    sim['maxtime'] = maxtime
    sim['dx'] = dx
    sim['du'] = du
    sim['statespace'] = statespace
    sim['x0'] = x0
    sim['agent_dyn'] = ltidyn
    sim['target_dyns'] = dyn_target
    sim['agent_pol'] = poltrack
    sim['target_pol'] = poltargets
    sim['asst_pol'] = apol
    sim['asst_epoch'] = assignment_epoch
    sim['nagents'] = nagents
    sim['ntargets'] = ntargets
    sim['runner'] = sim_runner

    return sim

# Formations
def circle(dim, radius, nsamples, sample):
    """ Computes the x,y,z position on a circle for a given number of points
    r: radius of circle
    ntargets: total number of points on circle
    target: nth point along the circle
    """

    if dim == 2:
        angle = sample*(2*np.pi)/nsamples
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return x, y
    elif dim == 3:
        angle = sample*(2*np.pi)/nsamples
        x = radius * np.cos(angle)
        y = 0
        z = radius * np.sin(angle)
        return x, y, z


def fibonacci_sphere(r, nsamples, sample):
    """ Computes the x,y,z positions on a sphere for a given number of points
    http://blog.marmakoide.org/?p=1

    r: radius of sphere / scaling factor
    nsamples: total number of points on sphere
    sample: nth point along the sphere
    """

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * sample
    z_i = (1 - 1/nsamples) * (1 - (2*sample)/(nsamples-1))
    radius = np.sqrt(1 - z_i * z_i)

    x = r * radius * np.cos(theta)
    y = r * radius * np.sin(theta)
    z = r * z_i
    return x,y,z

# def generate_initial_conditions(dim, agent_model, target_model, nagents, ntargets):
def generate_initial_conditions(dim, initial_formation_params):

    """ Returns initial states for agents, targets, and target terminal locations

    """

    x0 = None
    cities = None

    nagents = initial_formation_params['nagents']
    agent_model = initial_formation_params['agent_model']
    agent_swarm_formation = initial_formation_params['agent_swarm_formation']

    ntargets = initial_formation_params['ntargets']
    target_model = initial_formation_params['target_model']
    target_swarm_formation = initial_formation_params['target_swarm_formation']

    nstationary_states = initial_formation_params['nstationary_states']
    stationary_state_formation = initial_formation_params['stationary_states_formation']

    if dim == 2:

        ###### DOUBLE_INTEGRATOR ######
        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_2D.double_integrator_2D()

            ### Initial conditions

            # Agents
            r = 100
            if agent_swarm_formation == 'uniform_distribution':
                x0p = np.random.uniform(-100, 100, (nagents,dim)) # random position spread
            elif agent_swarm_formation == 'circle':
                x0p = [circle(dim, r, nagents, t) for t in range(nagents)] # circle
            elif agent_swarm_formation == 'fibonacci_sphere':
                x0p = [fibonacci_sphere(r, nagents, t) for t in range(nagents)] # sphere

            x0 = np.zeros((nagents, dx))
            vel_range = 500
            for ii, tt in enumerate(x0):
                x0[ii] = np.array([x0p[ii][0],
                                    x0p[ii][1],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x0 = x0.flatten()

            # Targets
            r = 100 # circle radius
            if target_swarm_formation == 'uniform_distribution':
                x02p = np.random.uniform(-100, 100, (ntargets,dim)) # random position spread
            elif target_swarm_formation == 'circle':
                x02p = [circle(dim, r, ntargets, t) for t in range(ntargets)] # circle
            elif target_swarm_formation == 'fibonacci_sphere':
                x02p = [fibonacci_sphere(r, ntargets, t) for t in range(ntargets)] # sphere

            x02 = np.zeros((ntargets, dx))
            vel_range = 50
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([x02p[ii][0],
                                    x02p[ii][1],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))

            # Target Terminal Location
            stationary_states = np.zeros((ntargets, dx))
            r = 100

            if stationary_state_formation == 'uniform_distribution':
                stationary_states_p = np.random.uniform(-100, 100, (nstationary_states,dim)) # random position spread
            elif stationary_state_formation == 'circle':
                stationary_states_p = [circle(dim, r, nstationary_states, t) for t in range(nstationary_states)] # circle
            elif stationary_state_formation == 'fibonacci_sphere':
                stationary_states_p = [fibonacci_sphere(r, nstationary_states, t) for t in range(nstationary_states)] # sphere

            for ii, tt in enumerate(stationary_states):
                stationary_states[ii] = np.array([
                    stationary_states_p[ii][0],
                    stationary_states_p[ii][1],
                    0, 0])

            stationary_states = stationary_states.flatten()
            stationary_states = np.split(stationary_states, ntargets)


        ###### LINEARIZED_QUADCOPTER ######
        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_2D.quadcopter_2D()

            # Agents
            r = 100
            if agent_swarm_formation == 'uniform_distribution':
                x0p = np.random.uniform(-100, 100, (nagents,dim)) # random position spread
            elif agent_swarm_formation == 'circle':
                x0p = [circle(dim, r, nagents, t) for t in range(nagents)] # circle
            elif agent_swarm_formation == 'fibonacci_sphere':
                x0p = [fibonacci_sphere(r, nagents, t) for t in range(nagents)] # sphere

            rot_x0p = np.random.uniform(-2*np.pi, 2*np.pi, (nagents,dim)) # position spread
            vel_range = 500
            rot_vel_range = 25
            x0 = np.zeros((nagents, dx))
            for ii, tt in enumerate(x0):
                x0[ii] = np.array([
                    x0p[ii][0],
                    x0p[ii][1],
                    rot_x0p[ii][0],
                    rot_x0p[ii][1],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])

            x0 = x0.flatten()

            # Targets
            r = 100 # circle radius
            if target_swarm_formation == 'uniform_distribution':
                x02p = np.random.uniform(-100, 100, (ntargets,dim)) # random position spread
            elif target_swarm_formation == 'circle':
                x02p = [circle(dim, r, ntargets, t) for t in range(ntargets)] # circle
            elif target_swarm_formation == 'fibonacci_sphere':
                x02p = [fibonacci_sphere(r, ntargets, t) for t in range(ntargets)] # sphere

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1],
                    rot_x02p[ii][0],
                    rot_x02p[ii][1],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))

            # Target Terminal Location
            stationary_states = np.zeros((ntargets, dx))
            r = 100

            if stationary_state_formation == 'uniform_distribution':
                stationary_states_p = np.random.uniform(-100, 100, (nstationary_states,dim)) # random position spread
            elif stationary_state_formation == 'circle':
                stationary_states_p = [circle(dim, r, nstationary_states, t) for t in range(nstationary_states)] # circle
            elif stationary_state_formation == 'fibonacci_sphere':
                stationary_states_p = [fibonacci_sphere(r, nstationary_states, t) for t in range(nstationary_states)] # sphere

            stationary_states = np.zeros((ntargets, dx))
            for ii, tt in enumerate(stationary_states):

                stationary_states[ii] = np.array([
                    stationary_states_p[ii][0],
                    stationary_states_p[ii][1],
                    0, 0, 0, 0, 0, 0])

            stationary_states = stationary_states.flatten()
            stationary_states = np.split(stationary_states, ntargets)

    if dim == 3:

        ###### DOUBLE_INTEGRATOR ######
        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_3D.double_integrator_3D()

            # Agents
            r = 100
            if agent_swarm_formation == 'uniform_distribution':
                x0p = np.random.uniform(-100, 100, (nagents,dim)) # random position spread
            elif agent_swarm_formation == 'circle':
                x0p = [circle(dim, r, nagents, t) for t in range(nagents)] # circle
            elif agent_swarm_formation == 'fibonacci_sphere':
                x0p = [fibonacci_sphere(r, nagents, t) for t in range(nagents)] # sphere

            x0 = np.zeros((nagents, dx))
            vel_range = 500
            for ii, tt in enumerate(x0):
                x0[ii] = np.array([x0p[ii][0],
                                    x0p[ii][1],
                                    x0p[ii][2],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x0 = x0.flatten()

            # Targets
            r = 100 # circle radius
            if target_swarm_formation == 'uniform_distribution':
                x02p = np.random.uniform(-100, 100, (ntargets,dim)) # random position spread
            elif target_swarm_formation == 'circle':
                x02p = [circle(dim, r, ntargets, t) for t in range(ntargets)] # circle
            elif target_swarm_formation == 'fibonacci_sphere':
                x02p = [fibonacci_sphere(r, ntargets, t) for t in range(ntargets)] # sphere

            x02 = np.zeros((ntargets, dx))
            vel_range = 50
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([x02p[ii][0],
                                    x02p[ii][1],
                                    x02p[ii][2],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))

            # Target Terminal Location
            stationary_states = np.zeros((ntargets, dx))
            r = 100

            if stationary_state_formation == 'uniform_distribution':
                stationary_states_p = np.random.uniform(-100, 100, (nstationary_states,dim)) # random position spread
            elif stationary_state_formation == 'circle':
                stationary_states_p = [circle(dim, r, nstationary_states, t) for t in range(nstationary_states)] # circle
            elif stationary_state_formation == 'fibonacci_sphere':
                stationary_states_p = [fibonacci_sphere(r, nstationary_states, t) for t in range(nstationary_states)] # sphere

            for ii, tt in enumerate(stationary_states):
                stationary_states[ii] = np.array([
                    stationary_states_p[ii][0],
                    stationary_states_p[ii][1],
                    stationary_states_p[ii][2],
                    0, 0, 0])

            stationary_states = stationary_states.flatten()
            stationary_states = np.split(stationary_states, ntargets)

        ###### LINEARIZED_QUADCOPTER ######
        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_3D.quadcopter_3D()

            # Agents
            r = 100
            if agent_swarm_formation == 'uniform_distribution':
                x0p = np.random.uniform(-100, 100, (nagents,dim)) # random position spread
            elif agent_swarm_formation == 'circle':
                x0p = [circle(dim, r, nagents, t) for t in range(nagents)] # circle
            elif agent_swarm_formation == 'fibonacci_sphere':
                x0p = [fibonacci_sphere(r, nagents, t) for t in range(nagents)] # sphere

            rot_x0p = np.random.uniform(-2*np.pi, 2*np.pi, (nagents,dim)) # position spread
            vel_range = 500
            rot_vel_range = 25
            x0 = np.zeros((nagents, dx))
            for ii, tt in enumerate(x0):
                x0[ii] = np.array([
                    x0p[ii][0],
                    x0p[ii][1],
                    x0p[ii][2],
                    rot_x0p[ii][0],
                    rot_x0p[ii][1],
                    rot_x0p[ii][2],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])

            x0 = x0.flatten()

            # Targets
            r = 100 # circle radius
            if target_swarm_formation == 'uniform_distribution':
                x02p = np.random.uniform(-100, 100, (ntargets,dim)) # random position spread
            elif target_swarm_formation == 'circle':
                x02p = [circle(dim, r, ntargets, t) for t in range(ntargets)] # circle
            elif target_swarm_formation == 'fibonacci_sphere':
                x02p = [fibonacci_sphere(r, ntargets, t) for t in range(ntargets)] # sphere

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1],
                    x02p[ii][2],
                    rot_x02p[ii][0],
                    rot_x02p[ii][1],
                    rot_x02p[ii][2],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-vel_range, vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                    np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])

            x02 = x02.flatten()
            x0 = np.hstack((x0, x02))

            # Target Terminal Location
            stationary_states = np.zeros((ntargets, dx))
            r = 100

            if stationary_state_formation == 'uniform_distribution':
                stationary_states_p = np.random.uniform(-100, 100, (nstationary_states,dim)) # random position spread
            elif stationary_state_formation == 'circle':
                stationary_states_p = [circle(dim, r, nstationary_states, t) for t in range(nstationary_states)] # circle
            elif stationary_state_formation == 'fibonacci_sphere':
                stationary_states_p = [fibonacci_sphere(r, nstationary_states, t) for t in range(nstationary_states)] # sphere

            for ii, tt in enumerate(stationary_states):

                stationary_states[ii] = np.array([
                    stationary_states_p[ii][0],
                    stationary_states_p[ii][1],
                    stationary_states_p[ii][2],
                    0, 0, 0, 0, 0, 0, 0, 0, 0])

            stationary_states = stationary_states.flatten()
            stationary_states = np.split(stationary_states, ntargets)

    return [x0, stationary_states]
