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
TIMESTEP      = 0.001
N_JOINTS_AXIS = default.N_JOINTS_AXIS

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results_test'
    simulation_data_file_tag = 'vary_drag_coefficients'

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
        frequency = kinematics_data['frequency'],
        amp_arr   = kinematics_data['joints_displ_amp'],
        ipl_arr   = np.linspace(0, kinematics_data['wave_number_ax'], N_JOINTS_AXIS),
    )

    # Drag coefficients
    coeff_x = np.array([ 0.0002292, 0.0001674, 0.0001936, 0.0002286, 0.0002286, 0.0002286, 0.0003197, 0.0001958, 0.0001924, 0.0001881, 0.0001846, 0.0000763, 0.0000497, 0.0000639, 0.0000953, 0.0002141 ])
    coeff_y = np.array([ 0.0103131, 0.0075350, 0.0087103, 0.0102864, 0.0102864, 0.0102864, 0.0143873, 0.0088091, 0.0086597, 0.0084647, 0.0083067, 0.0034322, 0.0022343, 0.0028773, 0.0042906, 0.0096333 ])
    coeff_z = np.array([ 0.0080213, 0.0058606, 0.0067746, 0.0080005, 0.0080005, 0.0080005, 0.0111901, 0.0068516, 0.0067353, 0.0065836, 0.0064608, 0.0026695, 0.0017378, 0.0022379, 0.0033371, 0.0074926 ])
    coeff_x[1:] = 0

    drag_coeffs = np.array( [ coeff_x, coeff_y, coeff_z ] ).T

    linear_drag_coefficents_options = default.get_linear_drag_coefficients_options(
        drag_coeffs = drag_coeffs,
    )

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    params_process['mech_sim_options'].update(
        {
            'video'                           : True,
            'video_fps'                       : 15,
            'video_speed'                     : 1.0,
            'linear_drag_coefficients_options': linear_drag_coefficents_options,
        }
    )

    # Params runs
    params_runs = [
        {
            'mech_sim_options' : {
                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file,
                },
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
        plot_figures        = True,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

if __name__ == '__main__':
    main()