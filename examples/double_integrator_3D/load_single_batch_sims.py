
## @include .
## @example ./examples/double_integrator_3D/load_single_batch_sims.py
# An Example of how to load and plot results from a single batch within an ensemble.
# Additionally, simulation diagnostics are also loaded and plotted.
# See the SETUP section
# Adjust 'batch_num' to select the desired batch to load within the ensemble folder


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DOT_assignment.post_process import post_process
from DOT_assignment import log

if __name__ == "__main__":

    # get list of ensemble tests
    # ensembles = [x[0] for x in os.walk('./test_results')]
    ensembles = [x[0] for x in os.walk('./')]

# SETUP
# .########################################################################

    # loads desired batches within a single ensemble which the user specifies

    # EDIT the following set of parameters used in the desired ensemble test folder
    dim = 3

    nagents = 10
    ntargets = 10

    agent_model = 'Double_Integrator'
    target_model = 'Double_Integrator'

    # EDIT the date here to match the ensemble test folder, you would like to load
    ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+\
            'identical_'+agent_model+'_LQR_LQR_DOUBLE_INTEGRATOR_3D_EXAMPLE'

    # EDIT the root directory path here to where the ensemble test folder is located
    root_directory = os.getcwd() + '/'
    ensemble_directory = root_directory + ensemble_name

# .########################################################################

    # get number of batches
    batch_dirs = [x[0] for x in os.walk(ensemble_directory)]
    nbatches = len(batch_dirs[1:])

    # load batches and plot
    sim_name_list = ['AssignmentDyn', 'AssignmentEMD']

    # # load and plot every batch within ensemble
    # for ii in range(nbatches):
    #     batch_name = 'batch_{0}'.format(ii)
    #     loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)
    #     post_process.plot_batch_performance_metrics(loaded_batch)

    # load and plot a specific batch
    batch_num = 0
    batch_name = 'batch_{0}'.format(batch_num)
    loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)
    post_process.plot_batch_performance_metrics(loaded_batch)

    # # cost histogram
    # ensemble_metrics = []
    # for ii in range(nbatches):
    #     batch_name = 'batch_{0}'.format(ii)
    #     loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)
    #     ensemble_metrics.append(loaded_batch)
    # post_process.plot_ensemble_histograms(ensemble_metrics)

    # # ensemble metrics 2 - save ensemble [dyn final costs, emd final_cost, switches]
    # ensemble_metrics = []
    # for ii in range(nbatches):
    #     batch_name = 'batch_{0}'.format(ii)
    #     loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)
    #     ensemble_metrics.append(loaded_batch)
    # post_process.save_ensemble_metrics(ensemble_metrics, ensemble_name)

    ##### LOAD DIAGNOSTICS

    # load and plot a specific batch
    batch_num = 0
    batch_name = 'batch_{0}'.format(batch_num)
    loaded_batch_diagnostics = log.load_batch_diagnostics(ensemble_directory, batch_name, sim_name_list)
    post_process.plot_batch_diagnostics(loaded_batch_diagnostics)

    # # diagnostics
    # ensemble_diagnostics = []
    # for ii in range(nbatches):
    #     batch_name = 'batch_{0}'.format(ii)
    #     loaded_batch_diagnostics = log.load_batch_diagnostics(ensemble_directory, batch_name, sim_name_list)
    #     ensemble_diagnostics.append(loaded_batch_diagnostics)
    # post_process.plot_ensemble_diagnostics(ensemble_diagnostics)

    print('finished plotting!')

    plt.show()
    print('done!')



