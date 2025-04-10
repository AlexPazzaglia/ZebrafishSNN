''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd

def save_muscle_parameters_from_single_optimization(
    optimization_name: str,
    target_g0        : float,
    target_wn        : float,
    target_zc        : float,
    index_iteration  : int,
    original_alpha   : float,
    original_beta    : float,
    original_delta   : float,
):
    ''' Save the muscle parameters '''

    optimization_name = (
        f'{optimization_name}_'
        f'FN_{round(target_wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_zc*1e3)}_'
        f'G0_{round(target_g0*1e3)}'
    )

    # Load the optimization results
    folder_name = f'experimental_data/zebrafish_kinematics_muscles/results/{optimization_name}'
    file_name   = f'{folder_name}/performance_iteration_{index_iteration}.dill'

    with open(file_name, 'rb') as file:
        performance_iteration = dill.load(file)

    # Extract the parameters from the optimization results
    muscle_params = {
        'alpha': performance_iteration['gains_scalings_alpha'] * original_alpha,
        'beta' : performance_iteration['gains_scalings_beta']  * original_beta,
        'delta': performance_iteration['gains_scalings_delta'] * original_delta,
    }

    # Save the parameters
    muscle_params_file_name = (
        'experimental_data/zebrafish_kinematics_muscles/optimized_parameters/'
        f'{optimization_name}_gen_{index_iteration}.csv'
    )

    muscle_params_df = pd.DataFrame(muscle_params)
    muscle_params_df.to_csv(muscle_params_file_name)

    return

def save_muscle_parameters_from_multiple_optimization():
    ''' Save the muscle parameters '''

    # Parameters of the target optimizations
    optimization_name = 'muscle_parameters_optimization_all'


    # TARGET PARAMETERS
    target_G0 = 2 * np.pi/ 15
    target_Zc = np.array( [1.00] )
    target_Fn = np.arange(10.5, 20.1, 0.5)

    target_Zc_list = target_Zc
    target_Wn_list = target_Fn * 2 * np.pi

    index_iteration = 50

    original_alpha = 8.4e-9
    original_beta  = 1.0e-8
    original_delta = 1.0e-8

    # Save the parameters for each target optimization
    for target_Wn in target_Wn_list:
        for target_Zc in target_Zc_list:
            save_muscle_parameters_from_single_optimization(
                optimization_name = optimization_name,
                target_g0         = target_G0,
                target_wn         = target_Wn,
                target_zc         = target_Zc,
                index_iteration   = index_iteration,
                original_alpha    = original_alpha,
                original_beta     = original_beta,
                original_delta    = original_delta,
            )


if __name__ == '__main__':
    save_muscle_parameters_from_multiple_optimization()