''' Run the spinal cord model '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build
import network_experiments.default_parameters.zebrafish.open_loop.default as default

def main():
    ''' Run the spinal cord model '''

    results_path             = '/data/pazzagli/simulation_results'
    simulation_data_file_tag = 'swimming'

    # Default parameters
    default_params = default.get_default_parameters()

    default_params['modname']  = default.MODELS_OPENLOOP['zebrafish_fictive'][0]
    default_params['parsname'] = default.MODELS_OPENLOOP['zebrafish_fictive'][1]

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,
        'duration'                 : 10.0,
    }

    # Params runs
    params_runs = [
        {
            'stim_a_off': -5.0,
        }
    ]

    # Simulate
    results_runs = simulate_single_net_multi_run_open_loop_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Check the results
    ptcc           = results_runs['neur_ptcc_ax'][0]
    freq           = results_runs['neur_freq_ax'][0]
    total_wave_lag = results_runs['neur_twl'][0]

    body_length              = 0.018
    first_active_segment_pos = 0.003
    last_active_segment_pos  = 0.015

    wave_number = total_wave_lag * body_length / (last_active_segment_pos - first_active_segment_pos)

    print(f'PTCC           : {ptcc:.4f}')
    print(f'Frequency      : {freq:.4f}')
    print(f'Total wave lag : {total_wave_lag:.4f}')
    print(f'Wave number    : {wave_number:.4f}')


if __name__ == '__main__':
    main()