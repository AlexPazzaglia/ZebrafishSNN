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

    # Create kinematics files
    n_runs      = 14
    frequencies = np.linspace(2.0, 15.0, n_runs)

    kinematics_data = default_params['kinematics_data']
    kinematics_files = [
        (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        f'kinematics/data_swimming_{ round(frequency * 10) }.csv'
        )
        for frequency in frequencies
    ]

    for kinematics_file, frequency in zip(kinematics_files, frequencies):
        default.create_kinematics_file(
            filename  = kinematics_file,
            duration  = DURATION,
            timestep  = TIMESTEP,
            frequency = frequency,
            amp_arr   = kinematics_data['joints_displ_amp'],
            ipl_arr   = np.linspace(0, kinematics_data['wave_number_ax'], N_JOINTS_AXIS),
        )

    # Params process
    params_process = default_params['params_process'] | {

        'simulation_data_file_tag': simulation_data_file_tag,
        'duration'                : DURATION,
        'timestep'                : TIMESTEP,
    }

    # DRAG COEFFICIENTS
    linear_coeff_x = 1.0 / 0.50  * np.array( [ 0.0008897, 0.0006142, 0.0007069, 0.0007854, 0.0007854, 0.0007854, 0.0010063, 0.0005195, 0.0004618, 0.0003752, 0.0002886, 0.0001728, 0.0000982, 0.0000982, 0.0000687, 0.0001237 ] )
    linear_coeff_y = 0.5 / 0.60  * np.array( [ 0.0014778, 0.0011095, 0.0012851, 0.0015551, 0.0015551, 0.0015551, 0.0022384, 0.0014270, 0.0014270, 0.0014270, 0.0014270, 0.0005702, 0.0003770, 0.0004948, 0.0007540, 0.0016965 ] )
    linear_coeff_z = 0.5 / 0.60  * np.array( [ 0.0010218, 0.0012009, 0.0014279, 0.0015551, 0.0015551, 0.0015551, 0.0018355, 0.0017474, 0.0015532, 0.0012620, 0.0009708, 0.0004147, 0.0001885, 0.0002474, 0.0002639, 0.0003299 ] )

    linear_drag_coeffs = np.array( [ linear_coeff_x, linear_coeff_y, linear_coeff_z ] ).T

    angular_coeff_x = 10 * np.array( [ 5.5217e-12, 1.6381e-12, 2.3030e-12, 3.1416e-12, 3.1416e-12, 3.1416e-12, 6.5745e-12, 1.0313e-12, 8.0122e-13, 5.3103e-13, 3.3573e-13, 8.7960e-14, 3.1063e-14, 3.1063e-14, 2.0381e-14, 2.3600e-13 ] )
    angular_coeff_y = np.array( [ 8.2108e-12, 5.4089e-12, 8.0146e-12, 1.1620e-11, 1.1620e-11, 1.1620e-11, 2.3778e-11, 1.2941e-11, 1.1503e-11, 9.3466e-12, 7.1897e-12, 6.9857e-13, 1.6824e-13, 3.4000e-13, 8.0191e-13, 2.9164e-12 ] )
    angular_coeff_z = np.array( [ 5.7211e-12, 9.2118e-12, 1.4282e-11, 1.9172e-11, 1.9172e-11, 1.9172e-11, 3.2769e-11, 3.7260e-11, 3.2475e-11, 2.6179e-11, 2.0847e-11, 1.3121e-12, 2.2912e-13, 7.0660e-13, 3.7422e-12, 1.0546e-11 ] )

    angular_drag_coeffs = np.array( [ angular_coeff_x, angular_coeff_y, angular_coeff_z ] ).T

    # Params runs
    params_runs = [
        {
            'mech_sim_options' : {
                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file,

                    'position_control_gains' : default.get_position_control_options(
                        gains_p = 5.0e-03,
                        gains_d = 1.0e-06,
                    ),

                },

                'linear_drag_coefficients_options': default.get_linear_drag_coefficients_options(
                    drag_coeffs = linear_drag_coeffs,
                ),

                'rotational_drag_coefficients_options': default.get_rotational_drag_coefficients_options(
                    drag_coeffs = angular_drag_coeffs,
                )

            }
        }
        for kinematics_file in kinematics_files
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

    # Actual and commanded positions
    joints_displ_error_arr = np.zeros( n_runs )

    for run_index in range(n_runs):

        desired_joints_displ = np.array( kinematics_data['joints_displ_amp'] )
        actual_joints_displ  = metrics_runs['mech_joints_disp_amp'][run_index]

        joint_displ_diff   = np.abs( actual_joints_displ - desired_joints_displ )
        joints_displ_error = np.mean( joint_displ_diff[:-2] / desired_joints_displ[:-2] )

        joints_displ_error_arr[run_index] = joints_displ_error
        print(f'Position error (freq = {frequencies[run_index]}Hz ) : {joints_displ_error * 100 :.2f}% \n')

    speed_fwd_bl = np.array( metrics_runs['mech_speed_fwd_bl'] )


    # Plot

    plt.figure('Tracking error')
    plt.plot( frequencies, joints_displ_error_arr )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Tracking error [%]')

    plt.figure('Speed')
    plt.plot( frequencies, speed_fwd_bl )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Speed [BL/s]')

    plt.show()



    return

if __name__ == '__main__':
    main()


