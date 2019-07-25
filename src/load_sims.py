
import os
import post_process
import log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # get list of ensemble tests
    ensembles = [x[0] for x in os.walk('./test_results')]

    dim = 3

    nagents = 20
    ntargets = 20
    ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+\
            'identical_Double_Integrator_LQR_LQR_2019_07_25_19_10_24'

    # # old
    # ensemble_name = 'ensemble_0_'+str(dim)+'D_'+'identical_Double_Integrator_LQR_LQR_2019_07_17_00_09_55'

    root_directory = '/Users/koray/Box Sync/test_results/'
    ensemble_directory = root_directory + ensemble_name

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



