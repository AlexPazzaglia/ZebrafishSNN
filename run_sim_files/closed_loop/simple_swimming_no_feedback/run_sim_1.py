''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.zebrafish.closed_loop.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'simple_swimming_open_loop'

    # Default parameters
    default_params = default.get_default_parameters()

    # Process parameters
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag' : simulation_data_file_tag,
        'load_connectivity_indices': True,


        'monitor_farmsim': {
            'active': True,
            'plotpars': {
                'joint_angles'    : True,
                'joint_velocities': False,
                'com_trajectory'  : True,
                'trajectory_fit'  : False,
                'animate'         : True,
            }
        }

    }

    # Params runs
    params_runs = [
        {
            'stim_a_off' : 0.0,

            'mech_sim_options' : {
                'video'      : False,
                'video_fps'  : 30,
                'video_speed': 1.0,
            }
        }
    ]

    # Simulate
    results_runs = simulate_single_net_multi_run_closed_loop_build(
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
    ipls           = results_runs['neur_ipl_ax_a'][0]
    total_wave_lag = results_runs['neur_twl'][0]

    n_segments               = 32
    body_length              = 0.018
    first_active_segment_pos = 0.003
    last_active_segment_pos  = 0.015
    body_length_active       = last_active_segment_pos - first_active_segment_pos

    wave_number = total_wave_lag * body_length / body_length_active

    target_wave_number = 0.95
    target_neur_twl    = target_wave_number * body_length_active / body_length
    target_neur_ipl    = target_neur_twl / ( n_segments - 1 )

    ipl_err = 100 * ( ipls - target_neur_ipl ) / target_neur_ipl
    twl_err = 100 * ( total_wave_lag - target_neur_twl ) / target_neur_twl
    wn_err  = 100 * ( wave_number - target_wave_number ) / target_wave_number

    print(f'PTCC           : {ptcc:.4f}')
    print(f'Frequency      : {freq:.4f}')
    print(f'IPL            : {ipls:.4f} (target: {target_neur_ipl:.4f}, error: {ipl_err:.4f}%)')
    print(f'Total wave lag : {total_wave_lag:.4f} (target: {target_neur_twl:.4f}, error: {twl_err:.4f}%)')
    print(f'Wave number    : {wave_number:.4f} (target: {target_wave_number:.4f}, error: {wn_err:.4f}%)')

    # Compare actual and commanded positions
    kinematics_data      = default_params['kinematics_data']
    desired_joints_displ = np.array( kinematics_data['joints_displ_amp'] )
    actual_joints_displ  = results_runs['mech_joints_disp_amp'][0]

    # Compute norm
    joint_displ_diff   = np.abs( actual_joints_displ - desired_joints_displ )
    joints_displ_error = np.mean( joint_displ_diff[:-2] / desired_joints_displ[:-2] )

    print(f'Mean error: {joints_displ_error * 100 :.2f}%')

    # Plot
    plt.plot( actual_joints_displ )
    plt.plot( desired_joints_displ )
    plt.show()

    return



if __name__ == '__main__':
    main()
