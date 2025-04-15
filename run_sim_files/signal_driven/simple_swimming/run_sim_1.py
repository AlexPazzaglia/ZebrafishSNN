''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.zebrafish.signal_driven.default as default

N_JOINTS_AXIS = default.N_JOINTS_AXIS

FREQUENCY = 5.0
IPL_ARR   = np.linspace(0.0, 0.9, N_JOINTS_AXIS)
# AMP_ARR   = np.array(
#     [
#         0.262,    #0
#         0.270,    #0.5
#         0.283,    #1
#         0.325,    #1.5
#         0.370,    #2
#         0.448,    #2.5
#         0.552,    #3
#         0.668,    #3.5
#         0.780,    #4
#         0.848,    #4.5
#         0.798,    #5
#         0.735,    #5.5
#         0.675,    #6
#         0.668,    #6.5
#         0.755,    #7
#     ]
# ) * 0.5

AMP_ARR = np.array(
    [
        0.1291765607556779,
        0.057408768177580366,
        0.0005804666378778153,
        0.09277541893172517,
        0.1262787722177159,
        1.0948953510134658e-06,
        0.19830654709499893,
        0.22319632379330792,
        2.5562962349138677e-06,
        0.35684188862245075,
        0.42349868468413465,
        3.0404379481030445e-08,
        0.578776260481662,
        9.09494701385498e-14,
        9.09494701385498e-14
    ]
) * 100

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results_test'
    simulation_data_file_tag = 'simple_swimming'

    # Default parameters
    default_params = default.get_default_parameters(
        muscle_parameters_tag = 'FN_6000_ZC_1000_G0_419_gen_199',
    )

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
    }

    # Params runs
    motor_output_signal_pars = default.get_signal_driven_gait_params(
        frequency    = FREQUENCY,
        amp_arr      = AMP_ARR,
        ipl_arr      = IPL_ARR,
    )

    params_runs = [
        {
            'motor_output_signal_pars' : motor_output_signal_pars,

            'mech_sim_options' : {
                'video'      : True,
                'video_fps'  : 15,
                'video_speed': 1.0,
            }
        }
    ]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

if __name__ == '__main__':
    main()