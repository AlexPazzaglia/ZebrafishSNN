
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

    # PROPRIOCEPTIVE NEURONS
    filename = f'{data_path}/Intrinsic_properties.xlsx'
    properties_df = np.array(
            pd.read_excel(
            filename,
            sheet_name = "Intraspinal proprioceptors",
            usecols    = "E:H",
            skiprows   = [1,2,3,4]
            )
    )

    data = {
        "ap_thresh": properties_df[:-1,0],
        "v_rest"   : properties_df[:-1,1],
        "Rinp"     : properties_df[:-1,2]
    }

    # Plotting
    analyze_neuronal_data.plot_experimental_data_distributions(
        data_dict   = data,
        data_keys   = ["ap_thresh", "v_rest", "Rinp"],
        data_titles = ['AP threshold', 'Resting potential', 'Rinp'],
    )
    plt.savefig(f'{data_path}/results/ps_parameters_distribution.png')

    # Saving
    with open(f'{data_path}/stretch_data.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()

if __name__ == '__main__':
    main()