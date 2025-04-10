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

import run_signal_driven
import network_experiments.default_parameters.zebrafish.signal_driven.default as default

DURATION      = 10
TIMESTEP      = 0.001
N_JOINTS_AXIS = default.N_JOINTS_AXIS

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path    = '/data/pazzagli/simulation_results'
    default_params  = default.get_default_parameters()
    kinematics_data = default_params['kinematics_data']

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

    angular_coeff_x = gain_x * np.array( [ 5.5217e-13, 1.6381e-13, 2.3030e-13, 3.1416e-13, 3.1416e-13, 3.1416e-13, 6.5745e-13, 1.0313e-13, 8.0122e-14, 5.3103e-14, 3.3573e-14, 8.7960e-15, 3.1063e-15, 3.1063e-15, 2.0381e-15, 2.3600e-14 ] )
    angular_coeff_y = gain_y * np.array( [ 9.5793e-12, 6.3104e-12, 9.3503e-12, 1.3556e-11, 1.3556e-11, 1.3556e-11, 2.7741e-11, 1.5098e-11, 1.3421e-11, 1.0904e-11, 8.3879e-12, 8.1499e-13, 1.9628e-13, 3.9667e-13, 9.3556e-13, 3.4024e-12 ] )
    angular_coeff_z = gain_z * np.array( [ 6.6746e-12, 1.0747e-11, 1.6662e-11, 2.2368e-11, 2.2368e-11, 2.2368e-11, 3.8231e-11, 4.3470e-11, 3.7887e-11, 3.0542e-11, 2.4321e-11, 1.5308e-12, 2.6730e-13, 8.2437e-13, 4.3659e-12, 1.2303e-11 ] )

    angular_drag_coeffs = np.array( [ angular_coeff_x, angular_coeff_y, angular_coeff_z ] ).T

    ########################################################
    # OPTIMIZE DRIVE AMP ###################################
    ########################################################
    n_opt_runs    = 10
    tolerance     = 0.01
    n_frequencies = 14
    frequencies   = np.linspace(2.0, 15.0, n_frequencies)

    drive_amp_array_freq    = np.array(
        [
            [0.08255, 0.03606, 0.03519, 0.03812, 0.04500, 0.05110, 0.05992, 0.06610, 0.07928, 0.10877, 0.12443, 0.14114, 0.18118, 0.00000, 0.00000], #  2.00 Hz
            [0.08718, 0.03902, 0.03753, 0.04024, 0.04722, 0.05355, 0.06251, 0.06853, 0.08164, 0.11160, 0.12702, 0.14270, 0.18556, 0.00000, 0.00000], #  3.00 Hz
            [0.09360, 0.04347, 0.04093, 0.04326, 0.05038, 0.05712, 0.06638, 0.07235, 0.08553, 0.11635, 0.13125, 0.14577, 0.19230, 0.00000, 0.00000], #  4.00 Hz
            [0.10167, 0.04946, 0.04549, 0.04727, 0.05453, 0.06185, 0.07159, 0.07763, 0.09109, 0.12338, 0.13765, 0.15102, 0.20193, 0.00000, 0.00000], #  5.00 Hz
            [0.11080, 0.05663, 0.05095, 0.05206, 0.05947, 0.06746, 0.07781, 0.08403, 0.09801, 0.13229, 0.14595, 0.15808, 0.21421, 0.00000, 0.00000], #  6.00 Hz
            [0.12127, 0.06509, 0.05745, 0.05779, 0.06540, 0.07424, 0.08536, 0.09191, 0.10665, 0.14363, 0.15670, 0.16743, 0.22991, 0.00000, 0.00000], #  7.00 Hz
            [0.13232, 0.07431, 0.06456, 0.06407, 0.07190, 0.08169, 0.09371, 0.10067, 0.11635, 0.15654, 0.16910, 0.17876, 0.24803, 0.00000, 0.00000], #  8.00 Hz
            [0.14401, 0.08424, 0.07227, 0.07090, 0.07896, 0.08981, 0.10281, 0.11029, 0.12708, 0.17088, 0.18297, 0.19136, 0.26840, 0.00000, 0.00000], #  9.00 Hz
            [0.15583, 0.09454, 0.08028, 0.07799, 0.08627, 0.09826, 0.11230, 0.12036, 0.13833, 0.18599, 0.19765, 0.20585, 0.28994, 0.00000, 0.00000], # 10.00 Hz
            [0.16851, 0.10557, 0.08892, 0.08566, 0.09422, 0.10741, 0.12257, 0.13123, 0.15051, 0.20243, 0.21360, 0.21934, 0.31359, 0.00000, 0.00000], # 11.00 Hz
            [0.18170, 0.11718, 0.09798, 0.09371, 0.10251, 0.11693, 0.13322, 0.14250, 0.16313, 0.21937, 0.23004, 0.23090, 0.33833, 0.00000, 0.00000], # 12.00 Hz
            [0.19514, 0.12927, 0.10734, 0.10193, 0.11090, 0.12649, 0.14382, 0.15362, 0.17544, 0.23570, 0.24583, 0.23683, 0.36271, 0.00000, 0.00000], # 13.00 Hz
            [0.20957, 0.14272, 0.11744, 0.11061, 0.11960, 0.13614, 0.15426, 0.16425, 0.18692, 0.25047, 0.25996, 0.21888, 0.38616, 0.00000, 0.00000], # 14.00 Hz
            [0.22551, 0.15905, 0.12908, 0.12016, 0.12876, 0.14578, 0.16411, 0.17359, 0.19635, 0.26163, 0.27035, 0.11081, 0.40655, 0.00000, 0.00000], # 15.00 Hz
        ]
    )

    for freq_ind, frequency in enumerate(frequencies):

        if n_opt_runs == 0:
            break

        # Initial drive amplitudes
        # ref_ind           = max(0, freq_ind - 1)
        ref_ind           = freq_ind
        drive_amp_array_0 = np.copy( drive_amp_array_freq[ref_ind] )

        # Optimize drive amplitudes
        drive_amp_array_freq[freq_ind] = run_signal_driven.optimize_drive_amp(
            drive_amp_arr       = drive_amp_array_0,
            linear_drag_coeffs  = linear_drag_coeffs,
            angular_drag_coeffs = angular_drag_coeffs,
            n_trials            = n_opt_runs,
            results_path        = results_path,
            tolerance           = tolerance,
            frequency           = frequency,
            metrics_print       = True,
            metrics_error       = False,
        )

    for drive_amp in drive_amp_array_freq:
        drive_str = ', '.join([f'{amp:.5f}' for amp in drive_amp])
        print(f'[{drive_str}],')

    ########################################################
    # SIMULATE #############################################
    ########################################################
    metrics_runs = [None] * n_frequencies

    for freq_ind, frequency in enumerate(frequencies):

        metrics_runs[freq_ind] = run_signal_driven.simulate_signal_driven(
            drive_amp_arr       = drive_amp_array_freq[freq_ind],
            linear_drag_coeffs  = linear_drag_coeffs,
            angular_drag_coeffs = angular_drag_coeffs,
            results_path        = results_path,
            frequency           = frequency,
            video               = False,
            plot                = False,
            metrics_print       = True,
            metrics_error       = False,
        )

    ########################################################
    # RESULTS ##############################################
    ########################################################

    # Frequency = [ 2.00e+00, 3.00e+00, 4.00e+00, 5.00e+00, 6.00e+00, 7.00e+00, 8.00e+00, 9.00e+00, 1.00e+01, 1.10e+01, 1.20e+01, 1.30e+01, 1.40e+01, 1.50e+01 ]
    # TWL       = [ 1.02e+00, 9.91e-01, 9.98e-01, 1.04e+00, 1.05e+00, 1.03e+00, 1.04e+00, 1.09e+00, 1.05e+00, 1.12e+00, 1.08e+00, 1.12e+00, 1.13e+00, 1.14e+00 ]
    # Speed     = [ 4.50e-01, 7.61e-01, 1.03e+00, 1.08e+00, 1.38e+00, 1.89e+00, 2.30e+00, 2.56e+00, 2.85e+00, 3.12e+00, 3.38e+00, 3.63e+00, 3.87e+00, 4.11e+00 ]
    # Tail_beat = [ 3.78e-02, 3.90e-02, 8.63e-02, 2.46e-02, 6.57e-02, 8.55e-02, 8.54e-02, 8.34e-02, 8.21e-02, 8.05e-02, 7.88e-02, 7.71e-02, 7.55e-02, 7.38e-02 ]

    actual_joints_displ_freq = np.array(
        [
            metrics_run['mech_joints_disp_amp'][0]
            for metrics_run in metrics_runs
        ]
    )

    speed_fwd_bl_freq = np.array(
        [
            metrics_run['mech_speed_fwd_bl'][0]
            for metrics_run in metrics_runs
        ]
    )

    # Actual and commanded positions
    desired_joints_displ    = kinematics_data['joints_displ_amp']
    joints_displ_error_freq = np.zeros( n_frequencies )

    for run_index in range(n_frequencies):

        actual_joints_displ = actual_joints_displ_freq[run_index]
        joint_displ_diff    = np.abs( actual_joints_displ - desired_joints_displ )
        joints_displ_error  = np.mean( joint_displ_diff[:-2] / desired_joints_displ[:-2] )

        joints_displ_error_freq[run_index] = joints_displ_error
        print(f'Position error (freq = {frequencies[run_index]}Hz ) : {joints_displ_error * 100 :.2f}% \n')

    ########################################################
    # SAVE #################################################
    ########################################################
    np.savetxt(
        os.path.join(CURRENTDIR, 'drive_amp_array_freq.txt'),
        drive_amp_array_freq,
    )

    ########################################################
    # PLOT #################################################
    ########################################################

    plt.figure('Tracking error')
    plt.plot( frequencies, joints_displ_error_freq * 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Tracking error [%]')

    plt.figure('Speed')
    plt.plot( frequencies, speed_fwd_bl_freq )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Speed [BL/s]')

    plt.figure('Drive amplitudes')
    colors = plt.cm.jet(np.linspace(0, 1, N_JOINTS_AXIS))
    for joint_index in range(N_JOINTS_AXIS):
        plt.plot(
            frequencies,
            drive_amp_array_freq[:, joint_index],
            label = f'Joint {joint_index}',
            color = colors[joint_index],
        )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        prop={'size': 6},
    )
    plt.tight_layout()

    plt.show()

    return

if __name__ == '__main__':
    main()


