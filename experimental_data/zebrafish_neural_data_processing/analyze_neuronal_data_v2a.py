
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from experimental_data.zebrafish_neural_data_processing import analyze_neuronal_data

def main():

    data_path = 'experimental_data/zebrafish_neural_data_processing'

    # V2a NEURONS
    filename = f'{data_path}/V2a and MN intrinsic properties and connectivity.xlsx'
    properties_df = np.array(
            pd.read_excel(
            filename,
            sheet_name = "V2a INs",
            usecols    = "A:C",
            skiprows   = []
            )
    )

    data_v2a = {
        "ap_thresh"           : properties_df[10:10+13,0],
        "bursting_probability": 0.846153846153846,
        "Rinp"                : properties_df[27:27+21,2]
    }

    # Plotting
    analyze_neuronal_data.plot_experimental_data_distributions(
        data_dict   = data_v2a,
        data_keys   = ["ap_thresh", "Rinp"],
        data_titles = ['AP threshold', 'Rinp']
    )
    plt.savefig(f'{data_path}/results/v2a_parameters_distribution.png')

    # Saving
    with open(f'{data_path}/v2a_slow_data.pickle', 'wb') as handle:
        pickle.dump(data_v2a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()

if __name__ == '__main__':
    main()