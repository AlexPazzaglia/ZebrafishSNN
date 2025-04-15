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

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'swimming'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': False,
        'duration'                 : 10.0,
    }

    # PTCC           : 1.4748
    # Frequency      : 3.5993
    # Total wave lag : 0.3528
    # Wave number    : 0.5292

    stim_amp = 0.0
    hemicord = 0

    if hemicord:
        stim_a_off  = -4.0
        stim_lr_off = +4.0 + stim_amp
    else:
        stim_a_off  = stim_amp
        stim_lr_off = 0.0


    # Params runs
    params_runs = [
        {
            'stim_a_off' : stim_a_off,
            'stim_lr_off': stim_lr_off,

            'ex2ex_weight' : 0.11739050225137393,
            'ex2in_weight' : 17.258175942362477,
            'in2ex_weight' : 11.399022085512307,
            'in2in_weight' : 9.819722125878327,
            'rs2ex_weight' : 9.668793262191324,
            'rs2in_weight' : 0.038516794359802514,
        }
    ]

    # Simulate
    results_runs = simulate_single_net_multi_run_open_loop_build(
        modname             = f'{CURRENTDIR}/net_openloop_zebrafish_scale_cpg_weights.py',
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