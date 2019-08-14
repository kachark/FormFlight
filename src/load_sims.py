
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

    nagents = 5
    ntargets = 5

#     agent_model = 'Double_Integrator'
#     target_model = 'Double_Integrator'
#     ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+\
#             'identical_'+agent_model+'_LQR_LQR_2019_07_31_14_06_36'

    agent_model = 'Linearized_Quadcopter'
    target_model = 'Linearized_Quadcopter'
    ensemble_name = 'ensemble_0_'+str(dim)+'D_'+str(nagents)+'v'+str(ntargets)+'_'+\
            'identical_'+agent_model+'_LQR_LQR_2019_08_07_16_10_26'

    # # # old data - 100v100 ensemble PRIOR TO JULY 25, 2019
    # root_directory = '/Users/koray/Box Sync/TargetAssignment/draper_paper/raw_data/100v100_ensemble/'
    # ensemble_name = 'ensemble_0_'+str(dim)+'D_'+'identical_Double_Integrator_LQR_LQR_2019_07_17_00_09_55'

    # root_directory = '/Users/koray/Box Sync/TargetAssignment/draper_paper/raw_data/'
    root_directory = '/Users/koray/Box Sync/TargetAssignment/draper_paper/raw_data/5v5_quadcopter_ensemble/'
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
    batch_num = 3
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
    batch_num = 3
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



