
import numpy as np
import copy

# TODO fix this import
from .. import linear_models_2D
from .. import linear_models_3D
from .. import dynamics
from .. import controls
from .. import agents
from .. import setup
from .. import schema
from .. import world

"""

Scenario Description:
    Agent multi-agent system tracks and performs an intercept of individual dynamic targets which
    are traveling to stationary terminal states

"""

# def create_world(ndynamic_objects, nstatic_objects):

#     """ Creates World datastructure containing the necessary objects to simulate a scenario
#     Input:
#     - ndynamic_objects:         integer representing the number of objects intended to evolve in
#                                 time
#     - nstatic_objects:          integer representing the number of objects inteded to remain static
#                                 in time
#     Output:
#     - world_i:                  World datastructure
#     """

#     ### SCENARIO SPECIFIC ###
#     nagents = int(ndynamic_objects / 2)
#     ntargets = int(ndynamic_objects / 2)
#     nterminal = nstatic_objects

# # TEST n < m case, n > m needs additional logic
#     # nagents = 2
#     # ntargets = 4
#     # nterminal = 4

#     # user defined (hard-coded for now)
#     agent_formation = 'uniform_distribution'
#     target_formation = 'uniform_distribution'
#     terminal_states_formation = 'circle'
#     agent_model = 'Linearized_Quadcopter'
#     target_model = 'Linearized_Quadcopter'

#     # homogeneous target models and control policies
#     target_dyn_model_list = [target_model for ii in range(ntargets)]
#     # target_dyn_model_list = ['Double_Integrator', 'Double_Integrator', 'Linearized_Quadcopter']
#     target_dyn_type_list = ['linear' for ii in range(ntargets)]
#     target_control_pol_list = ['LQR' for ii in range(ntargets)]

#     target_schema = schema.DynamicObjectSchema(ntargets, 'Targets')
#     target_schema.set_schema(target_dyn_model_list, target_dyn_type_list, target_control_pol_list)

#     # target terminal locations must follow the state descriptions of the target objects
#     static_points_schema = schema.StaticObjectSchema(nterminal, 'Target_terminal_locations')
#     static_points_schema.set_schema(target_dyn_model_list)

#     # homogeneous agent models and control policies
#     agent_dyn_model_list = [agent_model for ii in range(nagents)]
#     # agent_dyn_model_list = ['Double_Integrator', 'Double_Integrator', 'Linearized_Quadcopter']
#     agent_dyn_type_list = ['linear' for ii in range(nagents)]
#     agent_control_pol_list = ['LQT' for ii in range(nagents)]

#     agent_schema = schema.DynamicObjectSchema(nagents, 'Agents')
#     agent_schema.set_schema(agent_dyn_model_list, agent_dyn_type_list, agent_control_pol_list)

#     # # group up the agents, targets, and target terminal states, organized by ID
#     # scenario_params identifies the types of dynamic/static objects to inhabit the world
#     # the reason for placing them in a list is so you can organize how the dynamic_objects are
#     # ordered and ID'd internally - agents first, targets seconds, etc.
#     scenario_params = {'dynamic_objects': [agent_schema, target_schema],
#             'static_objects': [static_points_schema]}

#     # if we'd like to group them up and attribute additional information (ie. formation) we create
#     # MultiObjectSchemas
#     target_mas_schema = schema.MultiObjectSchema(ntargets, 'Target_MAS')
#     target_mas_schema.set_schema(target_formation)
#     agent_mas_schema = schema.MultiObjectSchema(nagents, 'Agent_MAS')
#     agent_mas_schema.set_schema(agent_formation)
#     static_points_group_schema = schema.MultiObjectSchema(ntargets, 'Region')
#     static_points_group_schema.set_schema(terminal_states_formation)

#     world_i = world.World(scenario_params)
#     world_i.generate(multi_object_schemas_list={'dynamic_objects': [agent_mas_schema,
#         target_mas_schema], 'static_objects': [static_points_group_schema]})

#     return world_i


def create_world(ndynamic_objects, nstatic_objects):

    """ Creates World datastructure containing the necessary objects to simulate a scenario
    Input:
    - ndynamic_objects:         integer representing the number of objects intended to evolve in
                                time
    - nstatic_objects:          integer representing the number of objects inteded to remain static
                                in time
    Output:
    - world_i:                  World datastructure
    """

    ### SCENARIO SPECIFIC ###
    nagents = int(ndynamic_objects / 2)
    ntargets = int(ndynamic_objects / 2)
    nterminal = nstatic_objects

# TEST n < m case, n > m needs additional logic
    # nagents = 2
    # ntargets = 4
    # nterminal = 4

    # user defined (hard-coded for now)
    agent_formation = 'uniform_distribution'
    target_formation = 'uniform_distribution'
    terminal_states_formation = 'circle'
    agent_model = 'NonlinearModel'
    target_model = 'NonlinearModel'

    # homogeneous target models and control policies
    target_dyn_model_list = [target_model for ii in range(ntargets)]
    # target_dyn_model_list = ['Double_Integrator', 'Double_Integrator', 'Linearized_Quadcopter']
    target_dyn_type_list = ['nonlinear' for ii in range(ntargets)]
    target_control_pol_list = ['NonlinearControllerTarget' for ii in range(ntargets)]

    target_schema = schema.DynamicObjectSchema(ntargets, 'Targets')
    target_schema.set_schema(target_dyn_model_list, target_dyn_type_list, target_control_pol_list)

    # target terminal locations must follow the state descriptions of the target objects
    static_points_schema = schema.StaticObjectSchema(nterminal, 'Target_terminal_locations')
    static_points_schema.set_schema(target_dyn_model_list)

    # homogeneous agent models and control policies
    agent_dyn_model_list = [agent_model for ii in range(nagents)]
    # agent_dyn_model_list = ['Double_Integrator', 'Double_Integrator', 'Linearized_Quadcopter']
    agent_dyn_type_list = ['nonlinear' for ii in range(nagents)]
    agent_control_pol_list = ['NonlinearController' for ii in range(nagents)]

    agent_schema = schema.DynamicObjectSchema(nagents, 'Agents')
    agent_schema.set_schema(agent_dyn_model_list, agent_dyn_type_list, agent_control_pol_list)

    # # group up the agents, targets, and target terminal states, organized by ID
    # scenario_params identifies the types of dynamic/static objects to inhabit the world
    # the reason for placing them in a list is so you can organize how the dynamic_objects are
    # ordered and ID'd internally - agents first, targets seconds, etc.
    scenario_params = {'dynamic_objects': [agent_schema, target_schema],
            'static_objects': [static_points_schema]}

    # if we'd like to group them up and attribute additional information (ie. formation) we create
    # MultiObjectSchemas
    target_mas_schema = schema.MultiObjectSchema(ntargets, 'Target_MAS')
    target_mas_schema.set_schema(target_formation)
    agent_mas_schema = schema.MultiObjectSchema(nagents, 'Agent_MAS')
    agent_mas_schema.set_schema(agent_formation)
    static_points_group_schema = schema.MultiObjectSchema(ntargets, 'Region')
    static_points_group_schema.set_schema(terminal_states_formation)

    world_i = world.World(scenario_params)
    world_i.generate(multi_object_schemas_list={'dynamic_objects': [agent_mas_schema,
        target_mas_schema], 'static_objects': [static_points_group_schema]})

    return world_i

