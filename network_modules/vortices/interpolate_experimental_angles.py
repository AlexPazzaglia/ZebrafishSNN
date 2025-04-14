
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
from network_modules.vortices.data.experimental_angles import *

def interpolate_signal(times, signal, times_sampled):
    ''' Interpolates using cubic spline interpolation. '''
    func_interpolate    = CubicSpline(times, signal)
    # func_interpolate    = Akima1DInterpolator(times, signal)
    # func_interpolate    = interp1d(times, signal, kind='linear')
    interpolated_values = func_interpolate(times_sampled)
    return interpolated_values

def filter_signal(
    signal    : np.ndarray,
    signal_dt : float,
    fcut_hp   : float = None,
    fcut_lp   : float = None,
    filt_order: int = 5,
    pad_type  : str = 'odd',
):
    ''' Butterwort, zero-phase filtering '''

    # Nyquist frequency
    fnyq = 0.5 / signal_dt

    # Filters
    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp/fnyq,  btype= 'highpass')
        signal  = filtfilt(num, den, signal, padtype= pad_type)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp/fnyq, btype= 'lowpass' )
        signal  = filtfilt(num, den, signal, padtype= pad_type)

    return signal

def get_signal_peaks(signal):
    ''' Get signal peaks. '''
    pos_peaks  = find_peaks(signal)[0]
    neg_peaks  = find_peaks(-signal)[0]
    all_peaks  = np.sort( np.concatenate((pos_peaks, neg_peaks)) )
    return all_peaks

def pad_signal(
    times       : np.ndarray,
    signal      : np.ndarray,
    padding_time: float,
):
    ''' Pad signal so that it loops. '''

    # Compute padding
    sampling_dt    = times[1] - times[0]
    padding_length = round(padding_time / sampling_dt)

    val0          = signal[-1]
    val1          = signal[0]
    padding_sig   = np.linspace(val0, val1, padding_length + 2)[1:-1]
    padding_times = np.arange(padding_length) * sampling_dt + times[-1] + sampling_dt

    # Extend the signal
    padded_times  = np.concatenate((times, padding_times))
    padded_signal = np.concatenate((signal, padding_sig))

    return padded_times, padded_signal

def link_signal_limits(
    times     : np.ndarray,
    signal    : np.ndarray,
    peaks_inds: np.ndarray,
    plot_data : bool = False,
):
    ''' Smooth signal loop. '''

    n_signal = len(signal)
    timestep = times[1] - times[0]

    # Looped signal
    signal_repeat = np.concatenate((signal, signal))
    times_repeat  = np.arange(2 * n_signal) * timestep

    # Fit a cosine signal to the loop half-period
    ind0  = peaks_inds[-1]
    ind1  = peaks_inds[0] + n_signal

    time0 = times_repeat[ind0]
    time1 = times_repeat[ind1]

    val0 = signal_repeat[ind0]
    val1 = signal_repeat[ind1]

    fit_mean = np.mean((val0, val1))
    fit_amp  = (val1 - val0) / 2
    fit_freq = 0.5 / (time1 - time0)
    fit_sign = np.sign(val0)

    fit_times = times_repeat[ind0 : ind1+1]
    fit_vals  = fit_mean + fit_sign * fit_amp * np.cos(2 * np.pi * fit_freq * (fit_times - time0))

    # Substitute
    n_sub0   = ind1 - n_signal + 2
    n_sub1   = n_signal - ind0
    signal_f = np.copy(signal)

    signal_f[:n_sub0] = fit_vals[-n_sub0:]
    signal_f[-n_sub1:] = fit_vals[:n_sub1]

    if plot_data:
        plt.figure(figsize=(10, 5))
        plt.plot(times_repeat, signal_repeat, label='Original signal')
        plt.plot(times, signal_f, label='Smoothed signal', lw=3)
        plt.plot(fit_times, fit_vals, '-o', label='Fitted signal', lw=1)

    return signal_f

def adjust_signal_for_loop(
    times    : np.ndarray,
    signal   : np.ndarray,
    plot_data: bool = False,
):
    ''' Adjust signal for loop. '''

    timestep = times[1] - times[0]

    # Compute peak intervals and frequency
    all_peaks           = get_signal_peaks(signal)
    all_peaks_times     = times[all_peaks]
    all_peaks_intervals = np.diff(all_peaks_times)

    first_interval = all_peaks_intervals[0]
    last_interval  = all_peaks_intervals[-1]

    # Compute ideal loop interval
    ideal_loop_interval  = np.mean((first_interval, last_interval))
    actual_loop_interval = all_peaks_times[0] + (times[-1] - all_peaks_times[-1])
    padding_time         = ideal_loop_interval - actual_loop_interval

    # Compute padding
    (padded_times, padded_signal) = pad_signal(
        times        = times,
        signal       = signal,
        padding_time = padding_time,
    )

    # Smooth the signal loop
    padded_signal = link_signal_limits(
        times      = padded_times,
        signal     = padded_signal,
        peaks_inds = all_peaks,
        plot_data  = plot_data,
    )

    if plot_data:
        plt.figure(figsize=(10, 5))
        plt.plot(times, signal, label='Original signal')
        plt.plot(times[all_peaks], signal[all_peaks], 'o')

        plt.plot(padded_times, padded_signal, label='Padded signal')
        plt.plot(padded_times + padded_times[-1] + timestep, padded_signal)

        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.legend()

    return padded_times, padded_signal

def create_fictive_schooling_trace(
    timestep      : float,
    total_duration: float,
    freq_scaling  : float,
    time_offset   : float = 0.0,
    signal_repeats: int   = None,
    signal_only   : bool  = False,
    plot          : bool  = False,
    verbose       : bool  = False,
):
    ''' Get scaled signal. '''

    # SCALE TIMES
    joint_angles_rad = np.deg2rad(JOINT_ANGLES)
    duration_scaling = 1 / freq_scaling
    sampling_times   = duration_scaling * SAMPLING_TIMES

    # APPLY LOOP ADJUSTMENT
    (sampling_times, joint_angles_rad) = adjust_signal_for_loop(
        times    = sampling_times,
        signal   = joint_angles_rad,
        plot_data= plot,
    )

    # INTERPOLATE SIGNAL
    t_start = sampling_times[0]
    t_end   = sampling_times[-1]

    times_interpolated  = np.arange( t_start, t_end, timestep)
    angles_interpolated = interpolate_signal(
        times         = sampling_times,
        signal        = joint_angles_rad,
        times_sampled = times_interpolated,
    )
    n_steps = len(times_interpolated)

    if verbose:
        original_chunk_duration = SAMPLING_TIMES[-1] - SAMPLING_TIMES[0]
        scaled_chunk_duration   = times_interpolated[-1] - times_interpolated[0]
        print(f"Original chunk duration: {original_chunk_duration:.2f} s")
        print(f"Scaled chunk duration  : {  scaled_chunk_duration:.2f} s")

    # SIGNAL ONLY
    if signal_only:
        signal_repeats     = int( np.ceil(total_duration / times_interpolated[-1]) )
        signal_onset_time  = 0.0 + time_offset
    else:
        signal_repeats = 1 if not signal_repeats else signal_repeats
        signal_duration   = times_interpolated[-1] * signal_repeats
        signal_onset_time = (total_duration - signal_duration) / 2 + time_offset

    signal_onset_index = round( signal_onset_time / timestep )

    # APPEND ZEROS
    times_angles = np.arange(0, total_duration + timestep, timestep)
    joint_angles = np.zeros_like(times_angles)

    for repeat in range(signal_repeats):
        i_start = signal_onset_index + repeat * n_steps
        i_end   = i_start + n_steps

        diff_start = max(0, -i_start)
        diff_end   = max(0, i_end - len(joint_angles))

        i_start += diff_start
        i_end   -= diff_end

        joint_angles[i_start: i_end] = angles_interpolated[diff_start: n_steps - diff_end]

    # PLOT
    if not plot:
        return joint_angles, times_angles

    plt.figure()
    plt.plot(
        times_angles,
        np.rad2deg(joint_angles),
        label     = 'Interpolated signal',
    )
    plt.plot(
        sampling_times,
        np.rad2deg(joint_angles_rad),
        label = 'Original signal',
        marker    = 'o',
        linestyle = '--',
        linewidth = 1.0,
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.show()

    return joint_angles, times_angles


if __name__ == '__main__':

    timestep       = 0.001
    total_duration = 30.0
    freq_scaling   = 0.30
    time_offset    = 0.0
    signal_repeats = None
    signal_only    = True
    plot           = True

    create_fictive_schooling_trace(
        timestep       = timestep,
        total_duration = total_duration,
        freq_scaling   = freq_scaling,
        time_offset    = time_offset,
        signal_repeats = signal_repeats,
        signal_only    = signal_only,
        plot           = plot,
    )
