
""" @file setup.py
"""

import numpy as np
import copy

# FormFlight
from FormFlight import assignments
from FormFlight import controls
from FormFlight import dynamics
from FormFlight import engine
from FormFlight import linear_models_2D
from FormFlight import linear_models_3D
from FormFlight import nonlinear_models
from FormFlight import run
from FormFlight import distributions
from FormFlight import agents
from FormFlight.scenarios import intercept_init

def setup_simulation(sim_profile):

    """ Returns dictionary of controls, dynamics, decision-making policy, and initial state parameters

    Input: Standard python dict containing descriptors outlining simulation requirements
    Output: Standard python dict containing controls, dynamics, assignment, etc. data structures

    NEW:
    creates actual dynamic objects (ie. Agents) and initializes controllers, assignment policies
    etc.

    """

    # params and initial conditions
    scenario = sim_profile['scenario']
    initial_conditions = sim_profile['initial_conditions']
    world_i = sim_profile['world']
    sim_params = sim_profile['sim_params']

    collisions = sim_params["collisions"]
    collision_tol = sim_params["collision_tol"]
    dim = sim_params["dim"]
    dt = sim_params["dt"]
    maxtime = sim_params["maxtime"]

    dynamic_indices = world_i.dynamic_multi_object_IDs
    static_indices = world_i.static_multi_object_IDs

    # NOTE assumes that there are grouped objects
    # populate dynamic objects with controllers, models, decision_makers
    dummy = agents.Agent('dummy_object')
    for group_ID in dynamic_indices:
        mas = world_i.multi_objects[group_ID]

        ### Decision-Maker
        if mas.decision_maker_type == 'DYN':
            apol = assignments.AssignmentDyn(0, 0) # constructor args deprecated/not used
            mas.decision_maker = apol
        elif mas.decision_maker_type == 'EMD':
            apol = assignments.AssignmentEMD(0, 0) # constructor args deprecated/not used
            mas.decision_maker = apol

        ### object dynamics
        for agent in mas.agent_list:
            statespace = None
            dx = None
            du = None
            dynamics_model = None
            controller = None

            dyn_type = agent.info['dyn_type']
            dyn_model = agent.info['dyn_model']
            control_pol = agent.info['control_pol']

            if dyn_type == 'linear':
                A = None
                B = None
                C = None
                D = None
                if dyn_model == 'Double_Integrator':
                    if dim == 2:
                        A, B, C, D, dx, du, statespace = linear_models_2D.double_integrator_2D()
                    elif dim == 3:
                        A, B, C, D, dx, du, statespace = linear_models_3D.double_integrator_3D()
                elif dyn_model == 'Linearized_Quadcopter':
                    if dim == 2:
                        A, B, C, D, dx, du, statespace = linear_models_2D.quadcopter_2D()
                    elif dim == 3:
                        A, B, C, D, dx, du, statespace = linear_models_3D.quadcopter_3D()

                dynamics_model = dynamics.LTIDyn(A, B, C, D)

                # controller
                dummy_state = np.zeros(dx)
                if control_pol == 'LQR':
                    Q = np.eye(dx)
                    R = np.eye(du)
                    # TODO match static states with 'target_MAS'
                    controller = controls.LinearFeedbackConstTracker(A, B, C, D, Q, R, dummy_state)
                elif control_pol == 'LQT':
                    Q = np.eye(dx)
                    R = np.eye(du)

                    # initialize LinearFeedbackAugmented by pre-assigning/augmenting this policy with
                    # dummy controller
                    dummy.info = agent.info
                    dummy_controller = controls.LinearFeedbackConstTracker(A, B, C, D, Q, R, dummy_state)
                    dummy.pol = dummy_controller
                    Acl = dummy_controller.get_closed_loop_A()
                    gcl = dummy_controller.get_closed_loop_g()

                    # NOTE hardcoded for now
                    if dyn_model == 'Linearized_Quadcopter':
                        if dim == 2:
                            Q[0, 0] = 1000
                            Q[1, 1] = 1000
                            Q[2,2] = 1000
                            Q[3,3] = 1000
                            Q[4,4] = 0.0
                            Q[5,5] = 0.0
                            Q[6, 6] = 0.0
                            Q[7, 7] = 0.0
                        elif dim == 3:
                            Q[0, 0] = 1000
                            Q[1, 1] = 1000
                            Q[2, 2] = 1000
                            Q[3,3] = 1000
                            Q[4,4] = 1000
                            Q[5,5] = 1000
                            Q[6,6] = 0.0
                            Q[7,7] = 0.0
                            Q[8,8] = 0.0
                            Q[9, 9] = 0.0
                            Q[10, 10] = 0.0
                            Q[11, 11] = 0.0
                    elif dyn_model == 'Double_Integrator':
                        if dim == 2:
                            Q[0, 0] = 1000
                            Q[1, 1] = 1000
                            Q[2,2] = 0.0
                            Q[3,3] = 0.0
                        elif dim == 3:
                            Q[0, 0] = 1000
                            Q[1, 1] = 1000
                            Q[2, 2] = 1000
                            Q[3,3] = 0.0
                            Q[4,4] = 0.0
                            Q[5,5] = 0.0

                    controller = controls.LinearFeedbackAugmented(A, B, C, D, Q, R, Acl, gcl) # initial

                agent.statespace = statespace
                agent.dx = dx
                agent.du = du
                agent.dim = dim
                agent.dyn = dynamics_model
                agent.pol = controller

            else: # nonlinear

                # dynamics
                if dyn_model == "NonlinearModel":
                    if dim == 2:
                        # retrieve the model
                        f, dx, du, statespace = nonlinear_models.NonlinearModel2D()
                        # generate a dynamics object based off the model
                        dynamics_model = dynamics.NonlinearDyn(f)

                    if dim == 3:
                        # retrieve the model
                        f, dx, du, statespace = nonlinear_models.NonlinearModel3D()
                        # generate a dynamics object based off the model
                        dynamics_model = dynamics.NonlinearDyn(f)

                # controller
                if control_pol == 'NonlinearController':
                    if dim == 2:
                        controller = controls.NonlinearController()

                    if dim == 3:
                        controller = controls.NonlinearController()

                if control_pol == 'NonlinearControllerTarget':
                    if dim == 2:
                        controller = controls.NonlinearControllerTarget()

                    if dim == 3:
                        controller = controls.NonlinearControllerTarget()


                agent.statespace = statespace
                agent.dx = dx
                agent.du = du
                agent.dim = dim
                agent.dyn = dynamics_model
                agent.pol = controller


    # populate static objects with models
    for group_ID in static_indices:
        mas = world_i.multi_objects[group_ID]

        ### object state information
        for point in mas.agent_list:
            statespace = None
            dx = None
            dynamics_model = None

            # TODO: bug but useful workaround for now
            dyn_type = agent.info['dyn_type']

            if dyn_type == 'linear':
                # TODO: bug but useful workaround for now
                if dyn_model == 'Double_Integrator':
                    if dim == 2:
                        _, _, _, _, dx, du, statespace = linear_models_2D.double_integrator_2D()
                    elif dim == 3:
                        _, _, _, _, dx, du, statespace = linear_models_3D.double_integrator_3D()
                elif dyn_model == 'Linearized_Quadcopter':
                    if dim == 2:
                        _, _, _, _, dx, du, statespace = linear_models_2D.quadcopter_2D()
                    elif dim == 3:
                        _, _, _, _, dx, du, statespace = linear_models_3D.quadcopter_3D()

                point.statespace = statespace
                point.dx = dx
                point.dim = dim

            else: # nonlinear
                # dynamics
                if dyn_model == "NonlinearModel":
                    if dim == 2:
                        # retrieve the model
                        f, dx, du, statespace = nonlinear_models.NonlinearModel2D()
                        # generate a dynamics object based off the model
                        dynamics_model = dynamics.NonlinearDyn(f)

                    if dim == 3:
                        # retrieve the model
                        f, dx, du, statespace = nonlinear_models.NonlinearModel3D()
                        # generate a dynamics object based off the model
                        dynamics_model = dynamics.NonlinearDyn(f)

                point.statespace = statespace
                point.dx = dx
                point.dim = dim



def generate_initial_conditions(dim, world_i):

    """ Returns initial states for agents, targets, and target terminal locations

    Input:
    - dim:                          integer dimension (2D/3D) for the simulation 
    - world_i:                        World datastructure

    Output:
    - x0:                           np.array representing initial states of time-varying
                                    agents
    - stationary_states:            np.array representing constant terminal states

    """

    initial_system_state = None

    # radius (circle, sphere)
    # r = 100
    space = 3500

    # TBD
    if not world_i.multi_objects:
        pass

    dynamic_indices = world_i.dynamic_multi_object_IDs
    static_indices = world_i.static_multi_object_IDs

    dynamic_formation_states = []
    static_formation_states = []
    for group_ID in dynamic_indices:
        multi_agent_system = world_i.multi_objects[group_ID]
        nagents = multi_agent_system.nagents
        formation = multi_agent_system.formation

        xyz_locations = generate_distribution(dim, space, nagents, formation)
        x0 = get_initial_object_states(dim, multi_agent_system, xyz_locations, True, vel_range=500)
        dynamic_formation_states.append(x0)

    for group_ID in static_indices:
        multi_agent_system = world_i.multi_objects[group_ID]
        nagents = multi_agent_system.nagents
        formation = multi_agent_system.formation

        xyz_locations = generate_distribution(dim, space, nagents, formation)
        x0 = get_initial_object_states(dim, multi_agent_system, xyz_locations, False)
        static_formation_states.append(x0)

    # compile all object states
    total_dynamic_object_states = np.hstack(dynamic_formation_states)
    total_static_object_states = np.hstack(static_formation_states)

    initial_system_state = np.hstack((total_dynamic_object_states, total_static_object_states))
    return initial_system_state

def get_initial_object_states(dim, multi_agent_system, object_positions, dynamic_states_flag, **kwargs):

    x0 = None
    nagents = multi_agent_system.nagents

    vel_range = 5
    if kwargs:
        vel_range = kwargs['vel_range']

    # get sum of statesize
    group_statesize = 0
    for agent in multi_agent_system.agent_list:
        dyn_model = agent.type

        dx = 0
        if dyn_model == "Double_Integrator":

            if dim == 2:
                _, _, _, _, dx, _, _ = linear_models_2D.double_integrator_2D()
            else:
                _, _, _, _, dx, _, _ = linear_models_3D.double_integrator_3D()

        elif dyn_model == "Linearized_Quadcopter":

            if dim == 2:
                _, _, _, _, dx, _, _ = linear_models_2D.quadcopter_2D()
            else:
                _, _, _, _, dx, _, _ = linear_models_3D.quadcopter_3D()

        elif dyn_model == "NonlinearModel":

            if dim == 2:
                 _, dx, _, _ = nonlinear_models.NonlinearModel2D()
            else:
                 _, dx, _, _ = nonlinear_models.NonlinearModel3D()

        group_statesize += dx

    # generate initial states
    x0 = []
    for i, agent in enumerate(multi_agent_system.agent_list):
        dyn_model = agent.type

        if dyn_model == "Double_Integrator":

            if dim == 2:
                if dynamic_states_flag:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0]])
                            )
                else:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        0, 0])
                        )

            elif dim == 3:
                if dynamic_states_flag:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0]])
                        )
                else:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        0, 0, 0])
                        )


        elif dyn_model == "Linearized_Quadcopter":

            if dim == 2:
                if dynamic_states_flag:
                    rot_x0p = np.random.uniform(-2*np.pi, 2*np.pi, (nagents,dim)) # rot position spread
                    rot_vel_range = 25
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        rot_x0p[i][0],
                        rot_x0p[i][1],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                        np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])
                        )
                else:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        0, 0, 0, 0, 0, 0])
                        )

            elif dim == 3:
                if dynamic_states_flag:
                    rot_x0p = np.random.uniform(-2*np.pi, 2*np.pi, (nagents,dim)) # position spread
                    rot_vel_range = 25
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        rot_x0p[i][0],
                        rot_x0p[i][1],
                        rot_x0p[i][2],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                        np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0],
                        np.random.uniform(-rot_vel_range, rot_vel_range, 1)[0]])
                        )
                else:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        0, 0, 0, 0, 0, 0, 0, 0, 0])
                        )

        elif dyn_model == "NonlinearModel":

            if dim == 2:
                if dynamic_states_flag:
                    theta = np.random.uniform(0, 2*np.pi, (nagents,dim)) # rot position spread

                    if multi_agent_system.name == 'Agent_MAS':
                        x0.append(np.array([
                            object_positions[i][0],
                            object_positions[i][1],
                            theta[i][0]])
                            )

                    elif multi_agent_system.name == 'Target_MAS':
                        x0.append(np.array([
                            object_positions[i][0] * 1.5,
                            object_positions[i][1] * 1.5,
                            theta[i][0] * 1.5])
                            )

                else: # static object
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        0])
                        )

# TODO: update for 3D -> static states match dynamic states for tracking
            elif dim == 3:
                if dynamic_states_flag:
                    theta = np.random.uniform(0, 2*np.pi, (nagents,dim)) # rot position spread
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0],
                        np.random.uniform(-vel_range, vel_range, 1)[0]])
                        )
                else:
                    x0.append(np.array([
                        object_positions[i][0],
                        object_positions[i][1],
                        object_positions[i][2],
                        0, 0, 0])
                        )

    # if dynamic_states_flag:
    #     x0 = np.hstack(x0)
    x0 = np.hstack(x0)

    return x0

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

    # np.random.seed(5)

    states = np.zeros(dim)
    if distribution == 'uniform_distribution':
        states = np.random.uniform(-space, space, (num_particles,dim))
    elif distribution == 'circle':
        radius = space
        states = [distributions.circle(dim, radius, num_particles, t) for t in range(num_particles)] # circle
    elif distribution == 'fibonacci_sphere':
        radius = space
        states = [distributions.fibonacci_sphere(radius, num_particles, t) for t in range(num_particles)] # sphere

    return states

def assign_decision_pol(world_i, mas_name, decision_maker_type, epoch):

    """
    Input:
    - world_i:                        World datastructure
    - mas_name:                     string naming the dynamic MultiAgentSystem to change
    - decision_maker_type:          string naming the decision-making algorithm for the
                                    MultiAgentSystem
    Output:
    """

    for group_ID in world_i.dynamic_multi_object_IDs:
        if world_i.multi_objects[group_ID].name == mas_name:
            world_i.multi_objects[group_ID].decision_maker_type = decision_maker_type
            world_i.multi_objects[group_ID].decision_epoch = epoch


