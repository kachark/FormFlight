

# FormFlight

![](https://github.com/kachark/FormFlight/blob/master/images/circle_formation.gif)
15 quadcopters with randomized initial states flying into formation

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

This library provides a first-look at the potential that optimal transport has in the areas of task assignment, resource allocation, flight formation, and more by offering a simulation framework from which to test these scenarios. Packaged into this library are examples focused on formation flight.

The examples provided generate a scenario where an agent swarm is tasked to maneuver into a stationary formation. A dynamic decision-maker leverages discrete optimal transport to perform the assignment of agent members to target members. Comparisons are given with a standard nearest-neighbor method.

The following formations are availble for the agent and terminal state distributions:
* Uniform distribution
* Fibonacci sphere
* Circle

The available agent swarm dynamic models:
* Double Integrator (2D/3D)
* Linearized Quadcoptor (2D/3D)

The available agent swarm controllers:
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

To download this package from the online git repository:
 
1. Clone the repo
```sh
git clone https://github.com/kachark/FormFlight.git
```


## Usage

The workflow for FormFlight is as follows
1. setup simulation parameters in main.py
2. run simulation
3. load and plot results

main.py creates the test data directory under the following format:
```sh
ensemble_0_<dimension>_<#agentsV#targets>_<agent model>_<agent start formation>_<agent controller>_<date/time>
```

Once a test is run, the user must provide the information of the new test folder into the load_sims.py (or equivalent) file to properly load and plot the results. 

### Simulation setup

main.py is the primary entry point for tweaking simulation parameters. 

Simulations are organized together in batches that aim to keep constant initial states operating over different assignment policies. Multiple batches can be grouped together within an ensemble to perform Monte Carlo simulations.

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
collisions = False
collision_tol = 1e-2
agent_control_policy = "LQR"
assignment_epoch = 10
```
NOTE: the number of agents (nagents) and number of targets (ntargets) must be equal in release v0.1.0

Define simulation parameters constant across a batch, such as initial swarm formations
```python
# INITIAL SWARM DISTRIBUTION and TERMINAL LOCATION DISTRIBUTION
# formations: uniform_distribution, circle, fibonacci_sphere
initial_formation_params = {
            'nagents': nagents,
            'agent_model': agent_model,
            'agent_swarm_formation': agent_formation,
            'ntargets': ntargets,
            'target_swarm_formation': target_formation
            }

```

Create simulation profile to be run within a batch

The available assignment algorithms are 'AssignmentEMD' and a template for creating custom
assignment policies. See 'AssignmentCustom'.

```python

dt = dt
asst = 'AssignmentEMD'
sim_profile_name = 'emd'
sim_profiles.update({sim_profile_name: {'agent_model': agent_model, 'agent_control_policy':
        agent_control_policy, 'agent_formation': agent_formation, 'target_formation':
          target_formation, 'assignment_policy': asst, 'assignment_epoch': assignment_epoch,
          'nagents': nagents, 'ntargets': ntargets, 'collisions': collisions, 'collision_tol':
          collision_tol, 'dim': dim, 'dt': dt, 'maxtime': maxtime, 'initial_conditions':
          initial_conditions}})

```

See the Examples page for example simulation setups

### Run simulation

By default, simulation batches are organized into ensembles and produce results that are stored in named folders at the root of the directory. Ensemble test folder names consist of the dimension of the simulation (2D/3D), scenario ('formation'), dynamics model used by the agents, the type of agent controllers and the date and time of the test. 

Within each ensemble test folder will be folders named by batch number, ordered sequentially by the time they were
performed. Each of these batch folders contain the individual simulation results (.csv) and diagnostics (.csv) for each
simulation profile that was used.

A sim_info.txt file is automatically provided in each ensemble folder which gives the details of all the ensemble-level test parameters used in the simulations along with general information.

Run the ensemble of simulations
```sh
python -O main.py
```

Debug Mode:
Debug mode allows you to run an ensemble of simulations without creating directories and saving
data. After completing the tests and post-processing, it will call plotting utilities to
allow users to immediately inspect results.

This logic can be altered as needed.

```sh
python main.py
```

NOTE: python commands are run from the root of the directory

### Loading and plotting data

load_sims.py will load saved test data and plot the results.

All loading files must first be edited with the desired test folder to load from (ensemble_directory) and
overall root directory (root_directory). Additionally, the number of agents, number of targets, agent model, target model, and dimension of the test must be editted in, similarly to was described above.

In the load_sims.py, make the following edits

#### Edit the simulation parameters for the folder to be loaded
```python
# SETUP
#########################################################################

# EDIT the following set of parameters used in the desired ensemble test folder
dim = 3

nagents = 5
ntargets = 5

agent_model = 'Double_Integrator'
```

#### Enter the correct date of the ensemble test folder
```python
# EDIT the date here to match the ensemble test folder, you would like to load 
ensemble_name = 'ensemble_' + str(nensemble) + '_' + (str(dim) + 'D') + '_' +\
        str(nagents) + '_' + str(ntargets) + '_' + agent_model + '_' + agent_formation + '_' + agent_controller +\
        '_' + 2019_07_31_14_06_36'

```

#### Enter the path of the root directory
```python
# EDIT the root directory path here to where the ensemble test folder is located
# DON'T FORGET THE '/' at the end!
root_directory = '/Users/foo/my/project/'


#########################################################################
```


load_sims.py will load single or all batches within an ensemble folder which is specified. In addition to the raw simulation data and post-processed results, simulation diagnostics are also plotted. 

Load and plot simulations
```sh
python load_sims.py
```

Additional possible visualizations include:
* 3-dimensional animation of the agent swarm evolving over time.

* Histograms using data from all ensembles in a given directory. 
NOTE: must specify the scenario of the ensembles being loaded

_For examples, please refer to the Examples page.

### Tests (In development)

In order to run tests:

```sh
python -m pytest -v tests
```

### Examples (In development)

The Examples page showcases some basic simulation configurations and loading files that can be used to guide
customization of main.py and the load__sims.py, load_ensembles.py, and animate_trajectory.py files to suit specific usecases. 

The simulation setups offered are:
- Double Integrator in 3D
- Linearized Quadcopter in 3D

The example loading and plotting utilities offered are:
- load_single_batch_sims.py
- load_ensembles.py
- animate_3D_trajectory.py

To run a simulation setup simply run
```python
python -O examples/double_integrator_3D/main.py
```

Load and plot an example simulation.
```python
python examples/double_integrator_3D/load_single_batch_sims.py
```

NOTE: the dimension of the test and agent and target dynamic models to correctly load the files. This information is readily available in each batch folder within the sim_info.txt. See 'Loading and plotting data'

NOTE: animate_3D_trajectory will only work with 3-Dimensional tests.



## Roadmap

Some immediate areas of improvement include the following additions to the target-assignment scenario:
* additional realistic dynamic models, stochastic models
* additional controllers
  * minimum-time intercept
  * fuel-optimal orbit injection
* additions to the flight formation scenario, including additional formations, moving formations
* heterogeneous swarms

## Publications
* Kachar, G., K., Gorodetsky, and A., A., “Dynamic multi-agent assignment via discrete optimal transport,” arXiv.org Available: https://arxiv.org/abs/1910.10748.


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

* Koray Kachar - [@linkedin](https://www.linkedin.com/in/koray-kachar/) - kkachar@umich.edu


## Acknowledgements

* Alex Gorodetsky - https://www.alexgorodetsky.com - goroda@umich.edu
* [Draper Laboratory](https://www.draper.com)
* [POT Python Optimal Transport Library](https://github.com/rflamary/POT)
* [University of Michigan](https://aero.engin.umich.edu)


