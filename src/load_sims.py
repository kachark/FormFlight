
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
    ensemble_name = 'ensemble_0_'+str(dim)+'D_'+'identical_Double_Integrator_LQR_LQR_2019_07_15_13_04_18'
    ensemble_directory = './test_results/' + ensemble_name

    # get number of batches
    batch_dirs = [x[0] for x in os.walk(ensemble_directory)]
    nbatches = len(batch_dirs[1:])

    # load batches and plot
    sim_name_list = ['AssignmentEMD', 'AssignmentDyn']
    for ii in range(nbatches):

        batch_name = 'batch_{0}'.format(ii)
        loaded_batch = log.load_batch_metrics(ensemble_directory, batch_name, sim_name_list)

        post_process.plot_batch_performance_metrics(loaded_batch)

    print('finished plotting!')

    plt.show()
    print('done!')



