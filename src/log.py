import glob
import os
import errno
import pandas as pd
from datetime import datetime

def save_test_info_to_txt(test_name, test_conditions, test_directory, starttime, endtime, elapsedtime):

    # save info to text file
    with open(test_directory + '/' + 'sim_info.txt', 'w') as text_file:
        print('Test Name:', file=text_file)
        print(test_name, file=text_file)

        line = "*"*40
        print(line, file=text_file)
        for condition, value in test_conditions.items():
            print(condition, ': ', value, file=text_file)
        print(line, file=text_file)

        line = "="*40
        print(file=text_file)
        print(line, file=text_file)
        print(starttime, file=text_file)
        print(endtime, file=text_file)
        print("Elapsed time:", elapsedtime, file=text_file)
        print(line, file=text_file)
        print(file=text_file)

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

def save_batch_diagnostics_to_csv(packed_batch_diagnostics, ensemble_directory, batch_name):

    cwd = os.getcwd()
    directory = ensemble_directory + '/' + batch_name

    try:
        os.makedirs(directory)
    except FileExistsError:
        # directory already exists
        pass

    for sim_name, post_processed_diag_df in packed_batch_diagnostics.items():
        file_name = batch_name + '_' + sim_name + '_' + 'DIAGNOSTICS' + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'
        path = directory + '/' + file_name
        post_processed_diag_df.to_csv(path, index=False, header=False)

def load_batch_metrics(ensemble_directory, batch_name, sim_name_list):

    directory = ensemble_directory + '/' + batch_name

    # TODO fix conflicts of loading in diagnostics instead of performance metrics
    sim_file_list = glob.glob(directory + '/' + '*.csv')

    batch_performance_metrics = {}
    for sim_file_name in sim_file_list:
        for sim_name in sim_name_list:
            if sim_name in sim_file_name and 'DIAGNOSTICS' not in sim_file_name:
                loaded_sim = pd.read_csv(sim_file_name, header=None)
                batch_performance_metrics.update({sim_name: loaded_sim})

    return batch_performance_metrics

def load_batch_diagnostics(ensemble_directory, batch_name, sim_name_list):

    directory = ensemble_directory + '/' + batch_name

    # TODO fix conflicts of loading in diagnostics instead of performance metrics
    csv_file_list = glob.glob(directory + '/' + '*.csv')

    batch_diagnostics = {}
    for csv_file_name in csv_file_list:
        for sim_name in sim_name_list:
            if sim_name in csv_file_name and 'DIAGNOSTICS' in csv_file_name:
                loaded_diagnostics = pd.read_csv(csv_file_name, header=None)
                batch_diagnostics.update({sim_name: loaded_diagnostics})

    return batch_diagnostics
