
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
from DOT_assignment import distributions

def setup_simulation(sim_profile):

    """ Returns dictionary of controls, dynamics, decision-making policy, and initial state parameters

    Input: Standard python dict containing descriptors outlining simulation requirements
    Output: Standard python dict containing controls, dynamics, assignment, etc. data structures

    """

    x0 = None
    stationary_states = None

    agent_model = sim_profile["agent_model"]
    agent_control_policy = sim_profile["agent_control_policy"]
    agent_formation = sim_profile["agent_formation"]
    target_formation = sim_profile["target_formation"]
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
        initial_formation_params = {
                'nagents': nagents, 'agent_model': agent_model, 'agent_swarm_formation': agent_formation,
                'ntargets': ntargets, 'target_swarm_formation': target_formation
                }
        initial_conditions = generate_initial_conditions(dim, initial_formation_params)
        x0 = ic[0]
        targets = ic[1]
    else:
        x0 = initial_conditions[0]
        targets = initial_conditions[1]

    sim = {}
    parameters = ['agent_model', 'dx', 'du', 'A', 'B', 'agent_dyn', 'agent_pol', 'asst_pol', 'x0']
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

    # TODO - remove
    # DEBUG control terms
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

    ### Agent control law
    if agent_control_policy == "LQR":
        poltrack = [controls.LinearFeedbackConstTracker(A, B, Q, R, t) for t in targets]

    ### Agent Dynamics
    ltidyn = dynamics.LTIDyn(A, B)

    ### Assignment Policy
    if assignment_policy == 'AssignmentCustom':
        apol = assignments.AssignmentCustom(nagents, ntargets)

    if assignment_policy == 'AssignmentEMD':
        apol = assignments.AssignmentEMD(nagents, ntargets)

    ### CONSTRUCT SIMULATION DICTIONARY
    sim['agent_control_policy'] = agent_control_policy
    sim['agent_model'] = agent_model
    sim['agent_formation'] = agent_formation
    sim['target_formation'] = target_formation
    sim['collisions'] = collisions
    sim['collision_tol'] = collision_tol
    sim['dt'] = dt
    sim['maxtime'] = maxtime
    sim['dx'] = dx
    sim['du'] = du
    sim['statespace'] = statespace
    sim['x0'] = x0
    sim['agent_dyn'] = ltidyn
    sim['agent_pol'] = poltrack
    sim['asst_pol'] = apol
    sim['asst_epoch'] = assignment_epoch
    sim['nagents'] = nagents
    sim['ntargets'] = ntargets
    sim['runner'] = sim_runner

    return sim


def generate_distribution(dim, space, num_particles, distribution):

    """

    Returns discrete distribution of states (ie. X,Y,Z positions)

    Input:
    - dim:      dimension
    - space:    range of values that distribution can take
    - num_particles: number of particles within the distribution
    - distribution: name of distribution

    Output:
    - states:     vector consisting of n-dimensional states corresponding to a desired distribution

    """

    if distribution == 'uniform_distribution':
        states = np.random.uniform(-space, space, (num_particles,dim))
    elif distribution == 'circle':
        radius = space
        states = [distributions.circle(dim, radius, num_particles, t) for t in range(num_particles)] # circle
    elif distribution == 'fibonacci_sphere':
        radius = space
        states = [distributions.fibonacci_sphere(radius, num_particles, t) for t in range(num_particles)] # sphere

    return states

# TODO breakdown into more functions
def generate_initial_conditions(dim, initial_formation_params):

    """ Returns initial states for agents, targets, and target terminal locations

    """

    x0 = None
    cities = None

    nagents = initial_formation_params['nagents']
    agent_model = initial_formation_params['agent_model']
    agent_swarm_formation = initial_formation_params['agent_swarm_formation']

    ntargets = initial_formation_params['ntargets']
    target_swarm_formation = initial_formation_params['target_swarm_formation']

    r = 100

    # TODO
    # Place these into separate function
    if dim == 2:

        ###### DOUBLE_INTEGRATOR ######
        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_2D.double_integrator_2D()

            ### Initial conditions
            # Agents
            x0p = generate_distribution(dim, r, nagents, agent_swarm_formation)

            x0 = np.zeros((nagents, dx))

            # NOTE user-defined how the intial state is constructed 
            vel_range = 500
            for ii, tt in enumerate(x0):
                x0[ii] = np.array([x0p[ii][0],
                                    x0p[ii][1],
                                    np.random.uniform(-vel_range, vel_range, 1)[0],
                                    np.random.uniform(-vel_range, vel_range, 1)[0]])

            x0 = x0.flatten()

            # Targets
            x02p = generate_distribution(dim, r, ntargets, target_swarm_formation)

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1],
                    0, 0])

            targets = x02.flatten()
            x0 = np.hstack((x0, targets))

        ###### LINEARIZED_QUADCOPTER ######
        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_2D.quadcopter_2D()

            # Agents
            x0p = generate_distribution(dim, r, nagents, agent_swarm_formation)

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
            x02p = generate_distribution(dim, r, ntargets, target_swarm_formation)

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1],
                    0, 0, 0, 0, 0, 0])

            targets = x02.flatten()
            x0 = np.hstack((x0, targets))



    if dim == 3:

        ###### DOUBLE_INTEGRATOR ######
        if agent_model == "Double_Integrator":

            A, B, C, D, dx, du, statespace = linear_models_3D.double_integrator_3D()

            # Agents
            x0p = generate_distribution(dim, r, nagents, agent_swarm_formation)

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
            x02p = generate_distribution(dim, r, ntargets, target_swarm_formation)

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1] + 500,
                    x02p[ii][2],
                    0, 0, 0])

            targets = x02.flatten()
            x0 = np.hstack((x0, targets))

        ###### LINEARIZED_QUADCOPTER ######
        if agent_model == "Linearized_Quadcopter":

            A, B, C, D, dx, du, statespace = linear_models_3D.quadcopter_3D()

            # Agents
            x0p = generate_distribution(dim, r, nagents, agent_swarm_formation)

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
            x02p = generate_distribution(dim, r, ntargets, target_swarm_formation)

            rot_x02p = np.random.uniform(-2*np.pi, 2*np.pi, (ntargets,dim)) # position spread
            vel_range = 50
            rot_vel_range = 25
            x02 = np.zeros((ntargets, dx))
            for ii, tt in enumerate(x02):
                x02[ii] = np.array([
                    x02p[ii][0],
                    x02p[ii][1],
                    x02p[ii][2],
                    0, 0, 0, 0, 0, 0, 0, 0, 0])

            targets = x02.flatten()
            x0 = np.hstack((x0, targets))

    return [x0, targets]

