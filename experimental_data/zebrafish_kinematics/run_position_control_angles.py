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
from network_experiments.snn_simulation import simulate_single_net_multi_run_position_control_build
import network_experiments.default_parameters.zebrafish.position_control.default as default

DURATION      = 10
TIMESTEP      = 0.001
N_JOINTS_AXIS = default.N_JOINTS_AXIS

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'
    simulation_data_file_tag = 'matching_kinematics'

    # Default parameters
    default_params = default.get_default_parameters()

    # Create kinematics file
    kinematics_data = default_params['kinematics_data']
    kinematics_file = (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        'kinematics/data_swimming.csv'
    )
    default.create_kinematics_file(
        filename  = kinematics_file,
        duration  = DURATION,
        timestep  = TIMESTEP,
        frequency = kinematics_data['frequency'],
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
            'video_fps'  : 60,
            'video_speed': 0.2,
        }
    )

    # Params runs
    params_runs = [
        {
            'mech_sim_options' : {
                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file,

                    'position_control_gains' : default.get_position_control_options(
                        gains_p = 5.0e-03,
                        gains_d = 1.0e-06,
                    )
                }

            }
        }
    ]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_position_control_build(
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

    # Compare actual and commanded positions
    desired_joints_displ = np.array( kinematics_data['joints_displ_amp'] )
    actual_joints_displ  = metrics_runs['mech_joints_disp_amp'][0]

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