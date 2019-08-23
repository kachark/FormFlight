

# Discrete Optimal Transport for Dynamic Decision-Making

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



## About The Project

This library provides a first-look at the potential that optimal transport has in the areas of task assignment, resource allocation, flight formation, and more by offering a simulation framework from which to test these scenarios. Packaged into this library are examples focused on task assignment, specifically centered around the canonical target assignment problem.

The examples provided generate a scenario where an agent swarm is tasked to rendezvous with a dynamically evolving
target swarm of equal size which is attempting to terminate at locations in space. A dynamic decision-maker leverages discrete optimal transport to perform the optimal assignment of agent members to target members. Comparisons are given with a standard nearest-neighbor method.

The following initial formations are availble for the agent, target, and terminal state distributions:
* Uniform distribution
* Fibonacci sphere
* Circle

The available agent and target swarm dynamic models availble:
* Double Integrator (2D/3D)
* Linearized Quadcoptor (2D/3D)

The available agent and target swarm controllers:
* Linearized Quadratic Tracker

Some demonstrations are available in the examples folder.

Documentation can be loaded by opening: docs/html/index.html


### Built With

* [POT Python Optimal Transport Library](https://github.com/rflamary/POT)


## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

DOT_assignment requires the following packages and subsequent dependencies in order to function.

* Python (>=3.6.7)
* Numpy (>=1.15.4)
* pandas (>=0.24.2)
* Matplotlib (>=3.0.3)

* Python Optimal Transport (>=0.5.1)
```sh
pip install POT
```

* pytest (>=5.1.0)
```sh
pip install pytest
```


### Installation

To download this package from the online git repository (currently not publically available):
 
1. Clone the repo
```sh
git clone git@bitbucket.org:goroda/targetingmdp.git
```


## Usage

### Running simulations

main.py is the primary entry point for tweaking simulation parameters. Simulations are organized together in batches that aim to keep constant initial states operating over different assignment policies. Multiple batches can be grouped together within an ensemble to perform Monte Carlo simulations.

Define number of batches
```python
nbatches = 1
```

Define simulation parameters that are constant across an ensemble of tests
```python
dt = 0.01
maxtime = 5
dim = 3
nagents = 5
ntargets = 5
agent_model = "Linearized_Quadcopter"
target_model = "Linearized_Quadcopter"
collisions = True
collision_tol = 1e-2
agent_control_policy = "LQR"
target_control_policy = "LQR"
assignment_epoch = 10
```
NOTE: the number of agents (nagents) and number of targets (ntargets) must be equal in release v0.1.0

Define simulation parameters constant across a batch, such as initial swarm formations
```python
# INITIAL SWARM DISTRIBUTIONS and TERMINAL LOCATION DISTRIBUTION
# formations: uniform_distribution, circle, fibonacci_sphere
initial_formation_params = {
        'nagents': nagents, 
        'agent_model': agent_model, 
        'agent_swarm_formation': 'uniform_distribution',
        'ntargets': ntargets, 
        'target_model': target_model, 
        'target_swarm_formation': 'fibonacci_sphere',
        'nstationary_states': ntargets, 
        'stationary_states_formation': 'circle'
        }
```

Create simulation profile to be run within a batch
```python

dt = dt
asst = 'AssignmentDyn'
sim_profile_name = 'dyn'
sim_profiles.update({sim_profile_name: {'agent_model': agent_model, 'target_model': target_model,
    'agent_control_policy': agent_control_policy, 'target_control_policy': target_control_policy,
    'assignment_policy': asst, 'assignment_epoch': assignment_epoch, 'nagents': nagents, 'ntargets': ntargets,
    'collisions': collisions, 'collision_tol': collision_tol, 'dim': dim, 'dt': dt, 'maxtime': maxtime, 'initial_conditions': initial_conditions}})

```

Run the ensemble of simulations
```sh
python main.py
```

NOTE: python commands are run from the root of the directory

### Loading and plotting data

load_sims.py will load saved test data and plot the results.

NOTE: the dimension of the test and agent and target dynamic models to correctly load the files. This information is readily available in each batch folder within the sim_info.txt.

load_sims.py will load single or all batches within an ensemble folder which is specified. In addition to the raw simulation data and post-processed results, simulation diagnostics are also plotted. 

load_sims.py
```python
# SETUP
#########################################################################
dim = 3

nagents = 5
ntargets = 5

agent_model = 'Double_Integrator'
target_model = 'Double_Integrator'
ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+\
         'identical_'+agent_model+'_LQR_LQR_2019_07_31_14_06_36'

root_directory = '/Users/koray/Documents/GradSchool/research/gorodetsky/draper/devspace/targetingmdp/'
ensemble_directory = root_directory + ensemble_name

#########################################################################

```

Load and plot simulations
```sh
python DOT_assignment/load_sims.py
```

Additional visualizations include:
* animate_trajectories.py : loads and plots a 3-dimensional animation of the agent and target swarms evolving over time.

* load_ensembles.py : loads and plots histograms using data from all ensembles in a given directory.
NOTE: must specify the 'agent'V'target' scenarios of the ensembles being loaded


_For more examples, please refer to the Examples page.

### Tests

In order to run tests:

```sh
python -m pytest -v tests
```


## Roadmap

Some immediate areas of improvement include the following additions to the target-assignment scenario:
* additional realistic dynamic models
* additional controllers
  * minimum-time intercept
  * fuel-optimal orbit injection
* additions to the target-assignment scenario, including waves of target swarms (ie. dynamic swarm size over time), stochastic dynamic conditions
* heterogeneous swarms




## License

Distributed under the MIT License. See `LICENSE` for more information.

TBD



## Contact

* Koray Kachar - [@linkedin](https://www.linkedin.com/in/koray-kachar/) - kkachar@umich.edu
* Alex Gorodetsky - https://www.alexgorodetsky.com - goroda@umich.edu


## Acknowledgements

* [Draper Laboratory](https://www.draper.com)
* [POT Python Optimal Transport Library](https://github.com/rflamary/POT)
* [University of Michigan](https://aero.engin.umich.edu)


