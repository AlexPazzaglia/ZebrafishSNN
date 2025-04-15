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
from network_experiments.snn_simulation import simulate_single_net_multi_run_open_loop_build

import network_experiments.default_parameters.zebrafish.open_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'scaled_ps_gains'

    # Default parameters
    default_params = default.get_default_parameters()

    ########################################################
    # CPG PARAMS ###########################################
    ########################################################
    connection_range   = 0.65

    ex_range_scaling = connection_range
    in_range_scaling = connection_range

    ########################################################
    # PROCESS PARAMS #######################################
    ########################################################
    params_process = default_params['params_process'] | {
        'duration'                 : 20.0,
        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': False,

        'connectivity_axial_newpars' : {
            'ax2ax': [
                default.get_scaled_ex_to_cpg_connections(ex_range_scaling),
                default.get_scaled_in_to_cpg_connections(in_range_scaling),
            ]
        }
    }

    ########################################################
    # RUN PARAMS ###########################################
    ########################################################

    # Run parameters
    params_runs = [
        {
            'ex2ex_weight': 3.83037,
            'ex2in_weight': 49.47920,
            'in2ex_weight': 0.84541,
            'in2in_weight': 0.10330,
            'rs2ex_weight': 8.74440,
            'rs2in_weight': 3.28338,

        }
    ]

    # Extra conditions
    hemicord = 0
    in_only  = 0
    ex_only  = 0

    new_pars_run = {
        'stim_a_off' : -4.0 if hemicord else 0.0,
        'stim_lr_off': +4.0 if hemicord else 0.0,
    }

    if in_only:
        new_pars_run['rs2ex_weight'] =  0.0

    if ex_only:
        new_pars_run['ex2in_weight'] =  0.0
        new_pars_run['rs2in_weight'] =  0.0

    params_runs[0].update(new_pars_run)

    # Process parameters
    params_process = get_params_processes(
        params_processes_shared = params_process,
        np_random_seed          = 100,
    )[0][0]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_open_loop_build(
        modname             = f'{CURRENTDIR}/net_openloop_zebrafish_simple_swimming.py',
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = False,
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Check the results
    ptcc           = metrics_runs['neur_ptcc_ax'][0]
    freq           = metrics_runs['neur_freq_ax'][0]
    ipls           = metrics_runs['neur_ipl_ax_a'][0]
    total_wave_lag = metrics_runs['neur_twl'][0]

    n_segments               = 32
    body_length              = 0.018
    first_active_segment_pos = 0.003
    last_active_segment_pos  = 0.015
    body_length_active       = last_active_segment_pos - first_active_segment_pos

    wave_number = total_wave_lag * body_length / body_length_active

    target_wave_number = 0.95
    target_neur_twl    = target_wave_number * body_length_active / body_length
    target_neur_ipl    = target_neur_twl / ( n_segments - 1 )

    print(f'PTCC           : {ptcc:.4f}')
    print(f'Frequency      : {freq:.4f}')
    print(f'IPL            : {ipls:.4f} (target: {target_neur_ipl:.4f})')
    print(f'Total wave lag : {total_wave_lag:.4f} (target: {target_neur_twl:.4f})')
    print(f'Wave number    : {wave_number:.4f} (target: {target_wave_number:.4f})')

    return metrics_runs

if __name__ == '__main__':
    main()
