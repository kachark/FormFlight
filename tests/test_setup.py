
import pytest
import numpy as np

from DOT_assignment.setup import generate_initial_conditions, setup_simulation
import DOT_assignment.assignments
import DOT_assignment.controls
import DOT_assignment.run
import DOT_assignment.dynamics


@pytest.fixture
def sim_params_ex():

    # SIM PARAMETERS CONSTANT ACROSS ENSEMBLE
    dt = 0.01
    maxtime = 5
    dim = 2
    nagents = 5
    ntargets = 5
    # agent_model = "Double_Integrator"
    # target_model = "Double_Integrator"
    agent_model = "Linearized_Quadcopter"
    target_model = "Linearized_Quadcopter"
    collisions = True
    collision_tol = 1e-2
    agent_control_policy = "LQR"
    target_control_policy = "LQR"
    assignment_epoch = 10

    # Create directory for storage
    nensemble = 0

    # TODO ensemble should not default to 'identical'
    ensemble_name = 'TEST_ENSEMBLE'

    root_directory = '/Users/foo/my/project/'
    ensemble_directory = root_directory + ensemble_name

    # TODO assumes heterogeneous swarms
    # formations: uniform_distribution, circle, fibonacci_sphere
    initial_formation_params = {
            'nagents': nagents, 'agent_model': agent_model, 'agent_swarm_formation': 'uniform_distribution',
            'ntargets': ntargets, 'target_model': target_model, 'target_swarm_formation': 'fibonacci_sphere',
            'nstationary_states': ntargets, 'stationary_states_formation': 'circle'
            }

    sim_params = {'dt': dt, 'maxtime': maxtime, 'dim': dim, 'nagents': nagents, 'ntargets': ntargets, 'agent_model': agent_model, 'target_model': target_model, 'collisions': collisions, 'collision_tol': collision_tol, 'agent_control_policy': agent_control_policy, 'target_control_policy': target_control_policy, 'assignment_epoch': assignment_epoch, 'initial_formation_params': initial_formation_params}

    return sim_params

@pytest.fixture
def sim_profile_ex(sim_params_ex):

    sim_params = sim_params_ex

    dt = sim_params['dt']
    maxtime = sim_params['maxtime']
    dim = sim_params['dim']
    nagents = sim_params['nagents']
    ntargets = sim_params['ntargets']
    agent_model = sim_params['agent_model']
    target_model = sim_params['target_model']
    collisions = sim_params['collisions']
    collision_tol = sim_params['collision_tol']
    agent_control_policy = sim_params['agent_control_policy']
    target_control_policy = sim_params['target_control_policy']
    assignment_epoch = sim_params['assignment_epoch']

    initial_formation_params = sim_params['initial_formation_params']

    initial_conditions = generate_initial_conditions(dim, initial_formation_params)

    ###### DEFINE SIMULATION PROFILES ######
    sim_profiles = {}

    # EMD parameters
    dt = dt
    asst = 'AssignmentEMD'
    sim_profile_name = 'emd'
    sim_profiles = {'agent_model': agent_model, 'target_model': target_model,
        'agent_control_policy': agent_control_policy, 'target_control_policy': target_control_policy,
        'assignment_policy': asst, 'assignment_epoch': assignment_epoch, 'nagents': nagents, 'ntargets': ntargets,
        'collisions': collisions, 'collision_tol': collision_tol, 'dim': dim, 'dt': dt, 'maxtime': maxtime,
        'initial_conditions': initial_conditions}

    return sim_profiles

def test_ic_generation(sim_params_ex):
    """
    Test initial condition properly created for agent, target, and terminal locations

    Test 1: assert initial state length , x0, correct
    Test 2: assert the number of terminal states correct
    Test 3: assert the terminal states length correct

    """

    sim_params = sim_params_ex

    dim = sim_params['dim']
    nagents = sim_params['nagents']
    ntargets = sim_params['ntargets']
    agent_model = sim_params['agent_model']
    target_model = sim_params['target_model']

    if dim == 2:
        if agent_model == "Double_Integrator":
            dx = 4
        if agent_model == "Linearized_Quadcopter":
            dx = 8
    else:
        if agent_model == "Double_Integrator":
            dx = 6
        if agent_model == "Linearized_Quadcopter":
            dx = 12

    initial_formation_params = sim_params['initial_formation_params']

    initial_conditions = generate_initial_conditions(dim, initial_formation_params)
    x0 = initial_conditions[0]
    terminal_states = initial_conditions[1]

    len_term_states = 0
    for ts in terminal_states:
        len_term_states += len(ts)

    assert len(x0) == nagents*dx + ntargets*dx, 'test failed'
    assert len(terminal_states) == ntargets, 'test failed'
    assert len_term_states == ntargets*dx, 'test failed'

def test_sim_setup(sim_profile_ex):
    """
    Tests dimensions of initial conditions correct
    """

    sim = setup_simulation(sim_profile_ex)

    # Simulation data structures
    collisions = sim["collisions"]
    collision_tol = sim["collision_tol"]
    dt = sim["dt"]
    maxtime = sim["maxtime"]
    dx = sim["dx"]
    du = sim["du"]
    statespace = sim["statespace"]
    x0 = sim["x0"]
    ltidyn = sim["agent_dyn"]
    target_dyn = sim["target_dyns"]
    poltrack = sim["agent_pol"]
    poltargets = sim["target_pol"]
    assignment_pol = sim["asst_pol"]
    assignment_epoch = sim["asst_epoch"]
    nagents = sim["nagents"]
    ntargets = sim["ntargets"]
    runner = sim["runner"]

    # Other simulation information
    agent_model = sim["agent_model"]
    target_model = sim["target_model"]
    agent_control_policy = sim["agent_control_policy"]
    target_control_policy = sim["target_control_policy"]

    assert type(collisions) == bool, 'test failed'
    assert type(collision_tol) == float, 'test failed'
    assert type(dt) == float, 'test failed'
    assert type(maxtime) == int, 'test failed'
    assert type(dx) == int, 'test failed'
    assert type(du) == int, 'test failed'
    assert type(statespace) == dict, 'test failed'
    assert type(x0) == np.ndarray, 'test failed'
    assert type(ltidyn) == DOT_assignment.dynamics.LTIDyn, 'test failed'
    assert type(target_dyn) == DOT_assignment.dynamics.LTIDyn, 'test failed'
    assert type(poltrack) == DOT_assignment.controls.LinearFeedbackAugmented, 'test failed'
    assert type(poltargets) == list, 'test failed'
    assert type(assignment_pol) == DOT_assignment.assignments.AssignmentEMD, 'test failed'
    assert type(assignment_epoch) == int, 'test failed'
    assert type(nagents) == int, 'test failed'
    assert type(ntargets) == int, 'test failed'
    assert runner == DOT_assignment.run.run_identical_linearized_quadcopter_2D, 'test failed'
    assert type(agent_model) == str, 'test failed'
    assert type(target_model) == str, 'test failed'
    assert type(agent_control_policy) == str, 'test failed'
    assert type(target_control_policy) == str, 'test failed'

