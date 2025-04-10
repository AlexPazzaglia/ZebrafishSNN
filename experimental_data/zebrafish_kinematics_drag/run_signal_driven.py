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
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.zebrafish.signal_driven.default as default

############################################################
# PARAMETERS ###############################################
############################################################

def get_sim_params(
    drive_amp_arr      : np.ndarray,
    linear_drag_coeffs : np.ndarray,
    angular_drag_coeffs: np.ndarray,
    video              : bool = False,
    frequency          : float = None,
):
    '''
    Get the parameters for the simulation
    '''

    # Default parameters
    simulation_data_file_tag = 'matching_kinematics_signal_driven'
    default_params           = default.get_default_parameters()
    kinematics_data          = default_params['kinematics_data']

    # Frequency
    if frequency is None:
        frequency = kinematics_data['frequency']

    # Params process
    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': simulation_data_file_tag,
    }

    # Drive signal parameters
    motor_output_signal_pars = default.get_signal_driven_gait_params(
        frequency    = frequency,
        ipl_arr      = kinematics_data['ipl_array'],
        amp_arr      = drive_amp_arr,
    )

    # Params runs
    params_runs = [
        {
            'motor_output_signal_pars' : motor_output_signal_pars,

            'mech_sim_options' : {
                # VIDEO
                'video'      : video,
                'video_fps'  : 30,
                'video_speed': 1.0,

                # DRAG COEFFICIENTS
                'linear_drag_coefficients_options': default.get_linear_drag_coefficients_options(
                    drag_coeffs = linear_drag_coeffs,
                ),
                'rotational_drag_coefficients_options': default.get_rotational_drag_coefficients_options(
                    drag_coeffs = angular_drag_coeffs,
                )
            }
        }
    ]

    return params_process, params_runs

############################################################
# METRICS ##################################################
############################################################

def joint_displacement_error(
    kinematics_data : dict,
    results_runs    : dict,
):
    ''' Get the joint displacement error '''

    joint_angles_ref = kinematics_data['joints_displ_amp']
    joint_angles     = results_runs['mech_joints_disp_amp'][0]

    # Compute norm
    joint_displ_diff   = np.abs( joint_angles - joint_angles_ref )
    joints_displ_error = np.mean( joint_displ_diff[:-2] / joint_angles_ref[:-2] )

    return joints_displ_error

def print_metrics_value(
    results_runs    : dict,
):
    ''' Print the metrics '''

    print(f"Frequency : { results_runs['mech_freq_ax'][0]          :.2e}")
    print(f"TWL       : { results_runs['mech_wave_number_a'][0]    :.2e}")
    print(f"Speed     : { results_runs['mech_speed_fwd_bl'][0]     :.2e}")
    print(f"Tail beat : { results_runs['mech_tail_beat_amp_bl'][0] :.2e}")

    return

def print_metrics_error(
    kinematics_data : dict,
    results_runs    : dict,
):
    ''' Compare the metrics between the actual and the desired kinematics '''

    target_freq      = kinematics_data['frequency']
    target_twl       = kinematics_data['wave_number_ax']
    target_speed     = kinematics_data['speed_fwd_bl']
    target_tail_beat = kinematics_data['tail_beat_bl']

    actual_freq      = results_runs['mech_freq_ax'][0]
    actual_twl       = results_runs['mech_wave_number_a'][0]
    actual_speed     = results_runs['mech_speed_fwd_bl'][0]
    actual_tail_beat = results_runs['mech_tail_beat_amp_bl'][0]

    print(f'Frequency error : { ( actual_freq      - target_freq      ) / target_freq      * 100:.2f}%')
    print(f'TWL error       : { ( actual_twl       - target_twl       ) / target_twl       * 100:.2f}%')
    print(f'Speed error     : { ( actual_speed     - target_speed     ) / target_speed     * 100:.2f}%')
    print(f'Tail beat error : { ( actual_tail_beat - target_tail_beat ) / target_tail_beat * 100:.2f}%')

    return

############################################################
# SIMULATION ###############################################
############################################################
def simulate_signal_driven(
    drive_amp_arr      : np.ndarray,
    linear_drag_coeffs : np.ndarray,
    angular_drag_coeffs: np.ndarray,
    results_path       : str,
    frequency          : float = None,
    video              : bool  = False,
    plot               : bool  = False,
    metrics_print      : bool  = True,
    metrics_error      : bool  = True,
):
    ''' Simulate the spinal cord model together with the mechanical simulator '''
    # Default parameters
    default_params  = default.get_default_parameters()
    kinematics_data = default_params['kinematics_data']

    # Get params
    params_process, params_runs = get_sim_params(
        drive_amp_arr       = drive_amp_arr,
        linear_drag_coeffs  = linear_drag_coeffs,
        angular_drag_coeffs = angular_drag_coeffs,
        video               = video,
        frequency           = frequency,
    )

    # Simulate
    results_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = plot,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Print
    if metrics_print:
        print_metrics_value(
            results_runs = results_runs,
        )

    if metrics_error:
        print_metrics_error(
            kinematics_data = kinematics_data,
            results_runs    = results_runs,
        )

    return results_runs

############################################################
# OPTIMIZATION #############################################
############################################################
def _update_drive_amp_arr(
    drive_amp_arr   : np.ndarray,
    kinematics_data : dict,
    results_runs    : dict,
):
    ''' Update the drive amplitudes '''

    joint_angles_ref = kinematics_data['joints_displ_amp']
    joint_angles     = results_runs['mech_joints_disp_amp'][0]

    # Update joints_amps_scalings
    drive_amp_arr[:-2] *= ( 1 + 0.5 * (-1 + joint_angles_ref[:-2] / joint_angles[:-2]) )

    # Print
    print(
        'mc_gain_axial: ['
        + ', '.join( [f'{joint_amp:.5f}' for joint_amp in drive_amp_arr] )
        + ']\n'
    )

    disp_error = joint_displacement_error(
        kinematics_data = kinematics_data,
        results_runs    = results_runs,
    )

    return drive_amp_arr, disp_error

def optimize_drive_amp(
    drive_amp_arr      : np.ndarray,
    linear_drag_coeffs : np.ndarray,
    angular_drag_coeffs: np.ndarray,
    n_trials           : int,
    results_path       : str,
    frequency          : float = None,
    tolerance          : float = 0.05,
    metrics_print      : bool  = True,
    metrics_error      : bool  = False,
):
    ''' Optimize the drive amplitudes '''
    default_params   = default.get_default_parameters()
    kinematics_data  = default_params['kinematics_data']

    if not n_trials:
        return drive_amp_arr

    for trial in range(n_trials):
        print(f'Trial {trial}')

        results_runs = simulate_signal_driven(
            drive_amp_arr      = drive_amp_arr,
            linear_drag_coeffs = linear_drag_coeffs,
            angular_drag_coeffs= angular_drag_coeffs,
            results_path       = results_path,
            frequency          = frequency,
            metrics_print      = metrics_print,
            metrics_error      = metrics_error,
        )

        # Compare actual and commanded positions
        drive_amp_arr, disp_error = _update_drive_amp_arr(
            drive_amp_arr   = drive_amp_arr,
            kinematics_data = kinematics_data,
            results_runs    = results_runs,
        )

        # Check convergence
        print(f'Position error  : {disp_error * 100 :.2f}%')

        if disp_error < tolerance:
            print('Optimization converged')
            break

    print('Optimization finished\n')

    return drive_amp_arr

############################################################
############################################################
############################################################

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path = '/data/pazzagli/simulation_results'

    ########################################################
    # DRAG COEFFICIENTS ####################################
    ########################################################

    gain_x = 0.05 / 0.05
    gain_y = 0.70 / 0.70
    gain_z = 0.70 / 0.70

    linear_coeff_x = gain_x * np.array( [ 8.8970e-05, 1.8425e-05, 2.1206e-05, 2.3562e-05, 2.3562e-05, 2.3562e-05, 3.0189e-05, 1.5586e-05, 1.3854e-05, 1.1257e-05, 8.6590e-06, 5.1836e-06, 2.9452e-06, 2.9452e-06, 2.0617e-06, 3.7110e-06 ] )
    linear_coeff_y = gain_y * np.array( [ 1.7241e-03, 1.2945e-03, 1.4992e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.6114e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 6.6523e-04, 4.3982e-04, 5.7727e-04, 8.7965e-04, 1.9792e-03 ] )
    linear_coeff_z = gain_z * np.array( [ 1.1921e-03, 1.4011e-03, 1.6658e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.1414e-03, 2.0386e-03, 1.8121e-03, 1.4723e-03, 1.1325e-03, 4.8381e-04, 2.1991e-04, 2.8863e-04, 3.0788e-04, 3.8485e-04 ] )

    linear_drag_coeffs = np.array( [ linear_coeff_x, linear_coeff_y, linear_coeff_z ] ).T

    angular_coeff_x = gain_x * np.array( [ 5.5217e-11, 1.6381e-11, 2.3030e-11, 3.1416e-11, 3.1416e-11, 3.1416e-11, 6.5745e-11, 1.0313e-11, 8.0122e-12, 5.3103e-12, 3.3573e-12, 8.7960e-13, 3.1063e-13, 3.1063e-13, 2.0381e-13, 2.3600e-12 ] )
    angular_coeff_y = gain_y * np.array( [ 9.5793e-12, 6.3104e-12, 9.3503e-12, 1.3556e-11, 1.3556e-11, 1.3556e-11, 2.7741e-11, 1.5098e-11, 1.3421e-11, 1.0904e-11, 8.3879e-12, 8.1499e-13, 1.9628e-13, 3.9667e-13, 9.3556e-13, 3.4024e-12 ] )
    angular_coeff_z = gain_z * np.array( [ 6.6746e-12, 1.0747e-11, 1.6662e-11, 2.2368e-11, 2.2368e-11, 2.2368e-11, 3.8231e-11, 4.3470e-11, 3.7887e-11, 3.0542e-11, 2.4321e-11, 1.5308e-12, 2.6730e-13, 8.2437e-13, 4.3659e-12, 1.2303e-11 ] )

    angular_drag_coeffs = np.array( [ angular_coeff_x, angular_coeff_y, angular_coeff_z ] ).T

    ########################################################
    # OPTIMIZE DRIVE AMP ###################################
    ########################################################
    n_trials  = 5
    tolerance = 0.05

    # frequency = kinematics_data['frequency']
    # drive_amp_arr_0 = kinematics_data['joints_signals_amps']

    # F = 15.0 Hz
    frequency       = 15.0
    drive_amp_arr_0 = np.array( [0.18195, 0.07197, 0.06235, 0.06190, 0.06905, 0.07548, 0.08525, 0.09064, 0.10876, 0.14818, 0.16362, 0.17330, 0.10699, 0.00000, 0.00000])

    # F = 4.0 Hz
    # frequency       = 4.0
    # drive_amp_arr_0 = np.array( [0.09524, 0.03797, 0.03688, 0.04004, 0.04756, 0.05420, 0.06407, 0.07108, 0.08667, 0.12230, 0.14146, 0.16214, 0.18635, 0.00000, 0.00000])


    drive_amp_arr = optimize_drive_amp(
        drive_amp_arr       = drive_amp_arr_0,
        linear_drag_coeffs  = linear_drag_coeffs,
        angular_drag_coeffs = angular_drag_coeffs,
        n_trials            = n_trials,
        results_path        = results_path,
        tolerance           = tolerance,
        frequency           = frequency,
        metrics_print       = True,
        metrics_error       = False,
    )

    ########################################################
    # SIMULATE #############################################
    ########################################################

    # Simulate
    metrics_runs = simulate_signal_driven(
        drive_amp_arr      = drive_amp_arr,
        linear_drag_coeffs = linear_drag_coeffs,
        angular_drag_coeffs= angular_drag_coeffs,
        results_path       = results_path,
        frequency          = frequency,
        video              = True,
        plot               = False,
        metrics_print      = True,
        metrics_error      = True,
    )

    ########################################################
    # PLOT #################################################
    ########################################################
    # plt.plot( metrics_runs['mech_links_disp_amp_bl'][0] )

    # plt.plot( actual_joints_displ )
    # plt.plot( desired_joints_displ )
    # plt.show()

    return

if __name__ == '__main__':
    main()


