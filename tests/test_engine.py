
import pytest
import numpy as np
import pandas as pd

import DOT_assignment.engine
import DOT_assignment.assignments
import DOT_assignment.controls
import DOT_assignment.run
import DOT_assignment.dynamics
import DOT_assignment.agents
import DOT_assignment.setup

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

    root_directory = '/Users/koray/Box Sync/test_results/'
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

    initial_conditions = DOT_assignment.setup.generate_initial_conditions(dim, initial_formation_params)

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

@pytest.fixture
def engine_ex(sim_profile_ex):

    sim = DOT_assignment.setup.setup_simulation(sim_profile_ex)

    dim = 2

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

    agents = [DOT_assignment.agents.TrackingAgent(dx, du, statespace, dim, ltidyn, poltrack) for ii in range(nagents)]
    targets = [DOT_assignment.agents.Agent(dx, du, statespace, dim, target_dyn, poltarget) for ii, poltarget in enumerate(poltargets)]

    sys = DOT_assignment.systems.OneVOne(agents, targets, assignment_pol, assignment_epoch)
    eng = DOT_assignment.engine.Engine(dim=dim, dt=dt, maxtime=maxtime, collisions=collisions, collision_tol=collision_tol)

    return [x0, sys, eng]

def test_log(engine_ex):
    """
    Tests the data logging within the engine
    """

    x0 = engine_ex[0]
    sys = engine_ex[1]
    eng = engine_ex[2]

    tout = np.ones((10,1))

    olddata = np.tile(x0, (tout.shape[0], 1))
    olddata_df = pd.DataFrame(olddata)
    eng.df = olddata_df

    new_x0 = x0 * 5
    newdata = np.tile(new_x0, (tout.shape[0], 1))
    newdata_df = pd.DataFrame(newdata)

    eng.log(newdata_df)

    assert len(eng.df) > len(olddata_df), 'test failed'

def test_apriori_collisions(engine_ex):
    """
    Tests the continuous/apriori collision detection
    """

    x0 = engine_ex[0]
    sys = engine_ex[1]
    eng = engine_ex[2]

    current_state = x0
    time = 1
    collisions, updated_state = eng.apriori_collisions(current_state, sys.agents, sys.targets, time)

    assert len(collisions) >= 0, 'test failed'


