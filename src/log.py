import glob
import os
import errno
import pandas as pd
from datetime import datetime

def save_batch_metrics_to_csv(batch_performance_metrics, ensemble_directory, batch_name):

    cwd = os.getcwd()
    directory = ensemble_directory + '/' + batch_name

    try:
        os.makedirs(directory)
    except FileExistsError:
        # directory already exists
        pass

    for sim_name, post_processed_results_df in batch_performance_metrics.items():
        file_name = batch_name + '_' + sim_name + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'
        path = directory + '/' + file_name
        post_processed_results_df.to_csv(path, index=False, header=False)

def load_batch_metrics(ensemble_directory, batch_name, sim_name_list):

    directory = ensemble_directory + '/' + batch_name
    sim_file_list = glob.glob(directory + '/' + '*.csv')

    batch_performance_metrics = {}
    for sim_file_name in sim_file_list:
        for sim_name in sim_name_list:
            if sim_name in sim_file_name:
                loaded_sim = pd.read_csv(sim_file_name, header=None)
                batch_performance_metrics.update({sim_name: loaded_sim})

    return batch_performance_metrics
