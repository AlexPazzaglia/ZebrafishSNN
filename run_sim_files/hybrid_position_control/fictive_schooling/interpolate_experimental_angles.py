import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from experimental_angles import *

from network_experiments.snn_signals_kinematics import kinematics_output_signal_save

def interpolate_signal(times, signal, times_sampled):
    ''' Interpolates using cubic spline interpolation. '''
    func_interpolate    = CubicSpline(times, signal)
    interpolated_values = func_interpolate(times_sampled)
    return interpolated_values

def transform_to_sinusoidal_signal(
    angles_original: np.ndarray,
    times_original : np.ndarray,
    times_interp   : np.ndarray,
    amplitude      : float = np.pi / 6,
    plot           : bool = False,
    max_pos_peaks  : int  = 10,
    max_neg_peaks  : int  = 9,
):
    ''' Make sinusoidal signal. '''

    # NOTE: Signal starts with a positive peak and ends with a negative peak

    sampling_dt = times_interp[1] - times_interp[0]

    # Get positive and negative peaks
    pos_peaks = find_peaks(+angles_original)[0][:max_pos_peaks]
    neg_peaks = find_peaks(-angles_original)[0][:max_neg_peaks]

    all_peaks_original = np.concatenate((pos_peaks, neg_peaks))
    all_peaks_original = np.sort(all_peaks_original)

    print(f'Signal has {pos_peaks.size} positive peaks and {neg_peaks.size} negative peaks.')

    # Convert peaks to indices in the interpolated signal
    all_peaks_interp = np.array(
        [
            np.argmin(np.abs(times_interp - times_original[peak]))
            for peak in all_peaks_original
        ]
    )

    # Create a sinusoidal signal with variable frequency
    sinusoidal_signal = np.zeros_like(times_interp)
    for i in range(len(all_peaks_interp) - 1):
        peak0     = all_peaks_interp[i]
        peak1     = all_peaks_interp[i + 1]

        v0_original = angles_original[all_peaks_original[i]]
        v1_original = angles_original[all_peaks_original[i + 1]]

        t_half    = times_interp[peak1] - times_interp[peak0]
        frequency = 0.5 / t_half

        t0   = times_interp[peak0]
        idx0 = peak0
        idx1 = peak1

        if i == 0:
            prev_t0       = times_interp[peak0]
            prev_crossing = times_interp[peak0] - 0.5 * t_half
            prev_sign     = np.sign(v0_original)
            prev_freq     = frequency

        if i == len(all_peaks_interp) - 2:
            next_t0       = times_interp[peak1]
            next_crossing = times_interp[peak1] + 0.5 * t_half
            next_sign     = np.sign(v1_original)
            next_freq     = frequency

        t_segment                    = times_interp[idx0:idx1]
        sinusoidal_signal[idx0:idx1] = np.cos(2 * np.pi * frequency * (t_segment - t0)) * np.sign(v0_original)

    # Generate time vectors for extensions
    prev_crossing_times = np.arange( prev_crossing,       prev_t0, sampling_dt)
    next_crossing_times = np.arange(       next_t0, next_crossing, sampling_dt)

    # Trim the sinusoidal signal
    max_prev_time = prev_crossing_times[-1]
    min_next_time = next_crossing_times[0]

    trim_prev = len(times_interp[times_interp <= max_prev_time])
    trim_next = len(times_interp[times_interp >= min_next_time])

    sampling_times_trim    = times_interp[trim_prev:-trim_next]
    sinusoidal_signal_trim = sinusoidal_signal[trim_prev:-trim_next]

    # Generate signals for extensions
    prev_crossing_signal = np.cos(2 * np.pi * prev_freq * (prev_crossing_times - prev_t0) ) * prev_sign
    next_crossing_signal = np.cos(2 * np.pi * next_freq * (next_crossing_times - next_t0) ) * next_sign

    # Concatenate time and signal arrays
    full_times  = np.concatenate((prev_crossing_times, sampling_times_trim, next_crossing_times))
    full_signal = np.concatenate((prev_crossing_signal, sinusoidal_signal_trim, next_crossing_signal))

    # Scale the signal
    full_signal *= amplitude

    # Plot the sinusoidal signal
    if plot:
        x_min = min(full_times[0], times_interp[0])
        x_max = max(full_times[-1], times_interp[-1])

        plt.figure(figsize=(10, 5))
        plt.plot(times_original, angles_original, label='Original Signal')
        plt.plot(full_times, full_signal, label='Full Signal')
        plt.plot([x_min, x_max], [0, 0], 'k--', lw=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(-1, +1)
        for peak in all_peaks_interp:
            plt.axvline(x=times_interp[peak], c='k', ls='--', alpha=0.5, lw=0.5)
        plt.legend()

    # Remove time offset
    full_times  -= full_times[0]

    return full_times, full_signal


def get_scaled_signal(
    timestep,
    total_duration,
    freq_scaling,
    t_offset       = 0.0,
    signal_repeats = 1,
    make_sinusoidal = True,
):
    ''' Get scaled signal. '''

    # INTERPOLATE
    joint_angles_rad   = np.deg2rad(JOINT_ANGLES)
    duration_scaling   = 1 / freq_scaling
    sampling_times     = duration_scaling * SAMPLING_TIMES
    t_start            = duration_scaling * ( SAMPLING_TIMES[0] )
    t_end              = duration_scaling * ( SAMPLING_TIMES[-1] + SAMPLING_DT )
    times_interpolated = np.arange( t_start, t_end, timestep)

    # MAKE SINUSOIDAL
    if make_sinusoidal:
        times_interpolated, angles_interpolated = transform_to_sinusoidal_signal(
            angles_original = joint_angles_rad,
            times_original  = sampling_times,
            times_interp    = times_interpolated,
            amplitude       = np.pi / 6,
            plot            = False,
            max_pos_peaks   = 9,
            max_neg_peaks   = 9,
        )

    # INTERPOLATE
    else:
        angles_interpolated = interpolate_signal(
            times         = sampling_times,
            signal        = joint_angles_rad,
            times_sampled = times_interpolated,
        )

    # APPEND ZEROS
    signal_duration = times_interpolated[-1] * signal_repeats
    angles_length   = len(angles_interpolated)

    signal_onset_time = (total_duration - signal_duration) / 2 + t_offset
    signal_onset_index = round( signal_onset_time / timestep )

    print(f'Signal  onset time: {signal_onset_index * timestep:.3f} s')
    print(f'Signal offset time: {(signal_onset_index + angles_length) * timestep:.3f} s')

    times_angles = np.arange(0, total_duration + timestep, timestep)
    joint_angles = np.zeros_like(times_angles)

    for repeat in range(signal_repeats):
        i_start = signal_onset_index + repeat * angles_length
        i_end   = i_start + angles_length

        joint_angles[i_start: i_end] = angles_interpolated


    return joint_angles, times_angles

def create_fictive_schooling_file(
    timestep         = 0.001,
    total_duration   = 10.0,
    freq_scaling     = 1.0,
    time_offset      = 0.0,
    signal_repeats   = 1,
    make_sinusoidal  = True,
    plot             = False,
):
    ''' Create fictive schooling file. '''

    # INTERPOLATE
    joint_angles, times_angles = get_scaled_signal(
        timestep        = timestep,
        total_duration  = total_duration,
        freq_scaling    = freq_scaling,
        t_offset        = time_offset,
        signal_repeats  = signal_repeats,
        make_sinusoidal = make_sinusoidal,
    )

    n_times_angles = len(times_angles)

    # MAP TO MULTI-JOINT MODEL
    joints_bend     = np.ones(N_JOINTS_AXIS, dtype=bool)
    joints_bend[0]  = False
    joints_bend[-1] = False
    joints_bend[-2] = False

    joint_angle_evolution              = np.zeros((N_JOINTS_AXIS, n_times_angles))
    joint_angle_evolution[joints_bend] = joint_angles / np.sum(joints_bend)
    joint_angle_evolution              = joint_angle_evolution.transpose()

    ########################################################
    # CREATE KINEMATICS FILE ###############################
    ########################################################

    kinematics_file = (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        'kinematics/data_fictive_schooling.csv'
    )

    kinematics_output_signal_save(
        times                = times_angles,
        motor_output_signals = joint_angle_evolution,
        save_file            = kinematics_file,
    )

    ########################################################
    # PLOT #################################################
    ########################################################

    if not plot:
        return kinematics_file

    plt.figure()
    plt.plot(
        times_angles,
        np.rad2deg(joint_angles)
    )
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.show()

    return kinematics_file




if __name__ == '__main__':
    create_fictive_schooling_file(
        freq_scaling = 0.30,
        plot         = True
    )
