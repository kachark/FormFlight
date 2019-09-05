
## @include .
## @example ./examples/double_integrator_3D/load_ensembles.py
# An Example of how to load and plot results from multiple ensembles useful in generating histograms from mutiple Monte Carlo tests (ensembles).
# Additionally, simulation diagnostics are also loaded and plotted.
# See the SETUP section


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DOT_assignment.post_process import post_process
from DOT_assignment import log

if __name__ == "__main__":

# SETUP
# .########################################################################

    # loads all ensembles within a desired directory

    # EDIT the following set of parameters used in the desired ensemble test folder
    dim = 3

    # get list of ensemble tests

    root_directory = os.getcwd() + '/'
    # elements = [x[0] for x in os.walk(ensemble_directory)] # recursively get (root, dirs, files)

    dirs = next(os.walk(root_directory))[1]
    ensembles = []
    for d in dirs:
        if 'ensemble_0' in d:
            ensembles.append(d)

    # EDIT the type of ensemble you'd like to load
    # ensemble engagement scenarios that are loaded. loads 5v5, 10v10, 20v20 agent-target engagements
    # desired_conditions = ['5v5', '10v10', '20v20']
    # desired_conditions = ['5v5', '10v10']
    desired_conditions = ['10v10']

# .########################################################################

    ensembles_to_load = []
    for ensemble in ensembles:
        for condition in desired_conditions:
            if condition in ensemble:
                ensembles_to_load.append(ensemble)

    # load ensembles and plot
    sim_name_list = ['AssignmentDyn', 'AssignmentEMD']
    loaded_ensemble_metrics = {}
    loaded_ensemble_diagnostics = {}
    for ensemble_name in ensembles_to_load:

        ensemble_directory = root_directory + ensemble_name

        # get number of batches
        batch_dirs = [x[0] for x in os.walk(ensemble_directory)]
        nbatches = len(batch_dirs[1:])

        # metrics
        ensemble_metrics = []
        for ii in range(nbatches):
            batch_name = 'batch_{0}'.format(ii)
            loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)
            ensemble_metrics.append(loaded_batch)
        loaded_ensemble_metrics.update({ensemble_name: ensemble_metrics})

        # # diagnostics
        # ensemble_diagnostics = []
        # for ii in range(nbatches):
        #     batch_name = 'batch_{0}'.format(ii)
        #     loaded_batch_diagnostics = log.load_batch_diagnostics(ensemble_directory, batch_name, sim_name_list)
        #     ensemble_diagnostics.append(loaded_batch_diagnostics)
        # loaded_ensemble_diagnostics.update({ensemble_name: ensemble_diagnostics})

    post_process.plot_ensemble_metric_comparisons(loaded_ensemble_metrics)
    # post_process.plot_ensemble_diagnostic_comparison(loaded_ensemble_diagnostics)

    print('finished plotting!')

    plt.show()
    print('done!')


