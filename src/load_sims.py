
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
            'identical_Double_Integrator_LQR_LQR_2019_07_17_16_31_41'

    # old
    # ensemble_name = 'ensemble_0_'+str(dim)+'D_'+'identical_Double_Integrator_LQR_LQR_2019_07_16_16_49_02'

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

    print('finished plotting!')

    plt.show()
    print('done!')



