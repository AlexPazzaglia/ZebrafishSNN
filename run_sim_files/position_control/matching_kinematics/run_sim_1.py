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
from network_experiments.snn_simulation import simulate_single_net_multi_run_position_control_build
import network_experiments.default_parameters.zebrafish.position_control.default as default

DURATION      = 10
TIMESTEP      = 1e-4
N_JOINTS_AXIS = default.N_JOINTS_AXIS

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'matching_kinematics_position_control'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_data = default_params['kinematics_data']
    kinematics_file = (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        'kinematics/data_swimming_test.csv'
    )
    default.create_kinematics_file(
        filename  = kinematics_file,
        duration  = DURATION,
        timestep  = TIMESTEP,
        frequency = 4.0,        # kinematics_data['frequency'],
        amp_arr   = kinematics_data['joints_displ_amp'],
        ipl_arr   = np.linspace(0, kinematics_data['wave_number_ax'], N_JOINTS_AXIS),
    )

    # Params process
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag': simulation_data_file_tag,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    params_process['mech_sim_options'].update(
        {
            'video'      : False,
            'video_fps'  : 30,
            'video_speed': 1.0,

            'timestep'  : TIMESTEP,
        }
    )

    # Params runs
    params_runs = [
        {
            'mech_sim_options' : {
                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file,
                }

            }
        }
    ]

    # Simulate
    results_runs = simulate_single_net_multi_run_position_control_build(
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

    # Error with reference angles
    ref_joint_angles   = kinematics_data['joints_displ_amp']
    joint_angles       = results_runs['mech_joints_disp_amp'][0]
    joint_displ_diff   = np.abs( joint_angles - ref_joint_angles )
    joints_displ_error = np.mean( joint_displ_diff[:-2] / ref_joint_angles[:-2] )

    print(f'Tracking error= {joints_displ_error * 100 :.2f}%')
    # Angles = 8.55
    # Velocities = 13.39

    # Check the results
    freq           = results_runs['mech_freq_ax'][0]
    total_wave_lag = results_runs['mech_wave_number_a'][0]
    forward_speed  = results_runs['mech_speed_fwd_bl'][0]

    body_length              = 0.018
    first_active_segment_pos = 0.003
    last_active_segment_pos  = 0.015

    wave_number = total_wave_lag * body_length / (last_active_segment_pos - first_active_segment_pos)

    print(f'Frequency      : {freq:.4f}')
    print(f'Total wave lag : {total_wave_lag:.4f}')
    print(f'Wave number    : {wave_number:.4f}')
    print(f'Forward speed  : {forward_speed:.4f}')

    return

if __name__ == '__main__':
    main()