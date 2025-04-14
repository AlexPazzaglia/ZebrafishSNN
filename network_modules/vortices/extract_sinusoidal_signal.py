"""
Extract the DLC data in world coordinates
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from scipy.signal import find_peaks, hilbert

import network_modules.vortices.load_data as load_data
import network_modules.vortices.interpolate_experimental_angles as interpolate_experimental_angles

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_signal_from_DLC_data(
    folder_name     : str,
    file_name       : str,
    target_fish     : str,
    sig_name        : str,
    start_recording : int,
    end_recording   : int,
    timestep        : float,
    total_duration  : float,
    freq_scaling    : float,
    verbose         : bool = False,
):
    ''' Load the signal from DLC data. '''

    scaled_signals_df = load_data.get_experimental_signal(
        folder_name     = folder_name,
        file_name       = file_name,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        timestep        = timestep,
        total_duration  = total_duration,
        freq_scaling    = freq_scaling,
        save_data       = False,
        plot_data       = False,
        detrend_data    = True,
        verbose         = verbose,
    )

    # Get the signal from a specific column
    times  = scaled_signals_df['time'].values
    signal = scaled_signals_df[sig_name].values

    # Get the signal from the sum of the angles
    # x_signals_names = [name for name in scaled_signals_df.columns if 'x_' in name]
    # y_signals_names = [name for name in scaled_signals_df.columns if 'y_' in name]

    # x_signals = scaled_signals_df[x_signals_names].values.T
    # y_signals = scaled_signals_df[y_signals_names].values.T

    # signal_original = compute_angles_sum(
    #     x_signals = x_signals,
    #     y_signals = y_signals,
    # )

    if verbose:
        print('\n--- LOADED THE TARGET SIGNAL FROM DLC DATA ---\n')

    return times, signal

def load_signal_from_experimental_angles(
    timestep       : float,
    total_duration : float,
    freq_scaling   : float,
    verbose        : bool = False,
):
    ''' Load the signal from experimental angles. '''
    signal, times = interpolate_experimental_angles.create_fictive_schooling_trace(
        timestep       = timestep,
        total_duration = total_duration,
        freq_scaling   = freq_scaling,
        time_offset    = 0.0,
        signal_repeats = None,
        signal_only    = True,
        plot           = False,
        verbose        = verbose,
    )
    if verbose:
        print('\n--- LOADED THE TARGET SIGNAL FROM REFERENCE SCHOOLING ANGLES ---\n')
    return times, signal

def load_reference_signal(
    folder_name     : str,
    file_name       : str,
    target_fish     : str,
    start_recording : int,
    end_recording   : int,
    timestep        : float,
    total_duration  : float,
    freq_scaling    : float,
    verbose         : bool = False,
    sig_name        : str = 'y_SC 8',
):
    ''' Load the reference signal. '''

    if (
        start_recording == 12104 and
        end_recording   == 12264 and
        target_fish     == 'Fish3'
    ):
        # Load the signal from exmperimental angles
        times_original, signal_original = load_signal_from_experimental_angles(
            timestep       = timestep,
            total_duration = total_duration,
            freq_scaling   = freq_scaling,
            verbose        = verbose,
        )
    else:
        # Load the signal from DLC data
        times_original, signal_original = load_signal_from_DLC_data(
            folder_name     = folder_name,
            file_name       = file_name,
            target_fish     = target_fish,
            sig_name        = sig_name,
            start_recording = start_recording,
            end_recording   = end_recording,
            timestep        = timestep,
            total_duration  = total_duration,
            freq_scaling    = freq_scaling,
            verbose         = verbose,
        )

    return times_original, signal_original

###############################################################################
# DATA PROCESSING #############################################################
###############################################################################

def study_signal_fft(
    signal : np.ndarray,
    times  : np.ndarray,
    verbose: bool = False,
    plot   : bool = False,
):
    ''' Study the FFT of the signal '''

    n_steps  = len(signal)
    timestep = times[1] - times[0]

    # Rremove the mean
    signal  = signal - np.mean(signal)

    # Compute the FFT
    point_fft       = np.fft.fft(signal)
    point_fft_abs   = np.abs(point_fft)
    point_fft_abs   = point_fft_abs[:n_steps//2]
    point_fft_freqs = np.fft.fftfreq(len(signal), d=timestep)
    point_fft_freqs = point_fft_freqs[:n_steps//2]

    # Compute dominant frequency
    tail_freq_ind = np.argmax(point_fft_abs)
    max_fft      = point_fft_abs[tail_freq_ind]
    max_freq     = point_fft_freqs[tail_freq_ind]

    if verbose:
        print(f"Max FFT frequency: {max_freq:.2f} Hz")

    # Plot the FFT
    if not plot:
        return max_freq

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(times, signal)
    ax[0].plot(times, signal,'r')
    ax[0].set_xlim([times[0], times[-1]])
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Signal')
    ax[0].set_title('Signal')

    ax[1].plot(point_fft_freqs, point_fft_abs)
    ax[1].plot(max_freq, max_fft, 'ro')
    ax[1].plot([max_freq, max_freq], [0, max_fft], 'r--')
    ax[1].text(
        max_freq + 5 * (point_fft_freqs[1] - point_fft_freqs[0]),
        max_fft,
        f"{max_freq:.2f} Hz", fontsize=12
    )
    ax[1].set_xlim(point_fft_freqs[0], 3 * max_freq)
    ax[1].set_ylim(0, 1.1 * max_fft)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('FFT(signal)')
    ax[1].set_title('Fourier transform of the signal')

    return max_freq

def get_inds_from_times(
    times    : np.ndarray,
    timestep : float,
):
    ''' Get indices from times. '''
    return np.array([round(t / timestep) for t in times])

def get_cycle_by_cycle_frequency(
    timestep  : float,
    peak_times: np.ndarray,
    verbose   : bool = False,
    plot_data : bool = False,
):
    ''' Get the real time frequency. '''

    n_peaks    = len(peak_times)
    peak_times = np.sort(peak_times)
    peak_inds  = get_inds_from_times(peak_times, timestep)
    times      = np.arange(0, peak_times[-1] + timestep, timestep)

    freqs = np.zeros_like(times)
    for i in range(n_peaks - 1):
        peak0 = peak_inds[i]
        peak1 = peak_inds[i+1]
        freq  = 0.5 / (times[peak1] - times[peak0])
        freqs[peak0:peak1] = freq
        if i == 0:
            freqs[:peak0] = freq
        if i == n_peaks - 2:
            freqs[peak1:] = freq

    mean_frequency = np.mean(freqs)
    std_frequency  = np.std(freqs)

    if verbose:
        print(
            f'Mean frequency: {mean_frequency:.2f} Hz +/- {std_frequency:.2f} Hz '
            f'({np.min(freqs):.2f} Hz - {np.max(freqs):.2f} Hz)'
        )

    if plot_data:
        plt.figure(figsize=(10, 5))
        plt.plot(freqs)
        plt.axhline(mean_frequency - std_frequency, lw=2, c='r')
        plt.axhline(mean_frequency + std_frequency, lw=2, c='r')

    return freqs

def get_real_time_phase_fun(
    times  : np.ndarray,
    signal : np.ndarray,
) -> Callable:
    ''' Get the real time phase. '''
    signal_hilb  = hilbert(signal)
    signal_phase = np.unwrap(np.angle(signal_hilb))
    phase_fun    = lambda t: np.interp(t, times, signal_phase)
    return phase_fun

def get_real_time_amp_fun(
    times    : np.ndarray,
    signal   : np.ndarray,
    normalize: bool = False,
) -> Callable:
    ''' Get the real time amplitude. '''
    signal_hilb  = hilbert(signal)
    signal_amp   = np.abs(signal_hilb)
    if normalize:
        signal_amp /= np.max(signal_amp)
    amp_fun = lambda t: np.interp(t, times, signal_amp)
    return amp_fun

def get_real_time_frequency_fun(
    times  : np.ndarray,
    signal : np.ndarray,
) -> Callable:
    ''' Get the real time frequency. '''
    dt_sig          = times[1] - times[0]
    signal_hilb     = hilbert(signal)
    signal_phase    = np.unwrap(np.angle(signal_hilb))
    signal_freq     = np.zeros_like(times)
    signal_freq[1:] = np.diff(signal_phase) / (2.0*np.pi) / dt_sig
    signal_freq[0]  = signal_freq[1]
    freq_fun        = lambda t: np.interp(t, times, signal_freq)
    return freq_fun

def get_signal_peaks(
    signal : np.ndarray,
):
    ''' Get signal peaks. '''

    mean_angle = np.mean(signal)
    pos_peaks  = find_peaks(+signal, height= mean_angle)[0]
    neg_peaks  = find_peaks(-signal, height= mean_angle)[0]

    median_pos_cycle = np.median(np.diff(pos_peaks))
    median_neg_cycle = np.median(np.diff(neg_peaks))
    median_cycle     = np.mean([median_pos_cycle, median_neg_cycle])

    pos_peaks = find_peaks(+signal, height= mean_angle, width=0.1 * median_cycle)[0]
    neg_peaks = find_peaks(-signal, height= mean_angle, width=0.1 * median_cycle)[0]

    all_peaks_original = np.concatenate((pos_peaks, neg_peaks))
    all_peaks_original = np.sort(all_peaks_original)

    return all_peaks_original

def get_bounded_frequencies(
    peak_times : np.ndarray,
    min_freq   : float = None,
    max_freq   : float = None,
    verbose    : bool  = False,
):
    ''' Get rescaled frequencies. '''

    n_peaks      = len(peak_times)
    freqs        = 0.5 / np.diff(peak_times)
    min_freq_sig = np.min(freqs)
    max_freq_sig = np.max(freqs)

    if min_freq is None:
        min_freq = min_freq_sig
    if max_freq is None:
        max_freq = max_freq_sig

    if verbose:
        print(f'Original frequency range: {min_freq_sig:.2f} - {max_freq_sig:.2f} Hz')
        print(f'Rescaled frequency range: {min_freq:.2f} - {max_freq:.2f} Hz')

    freqs_scaled        = np.clip(freqs, min_freq, max_freq)
    half_periods_scaled = 0.5 / freqs_scaled

    peak_times_scaled = np.array(
        [peak_times[0]] +
        [
            np.cumsum(half_periods_scaled)[i] + peak_times[0]
            for i in range(n_peaks - 1)
        ]
    )
    return peak_times_scaled

def get_signal_with_variable_frequency(
    peak_times  : np.ndarray,
    first_sign  : int   = +1,
    amplitude   : float = 1.0,
    sampling_dt : float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Make a signal with variable frequency. '''

    peak_inds      = get_inds_from_times(peak_times, sampling_dt)
    half_periods   = np.diff(peak_times)
    half_periods_n = np.diff(peak_inds)

    n_peaks  = len(peak_times)
    n_signal = peak_inds[-1] + half_periods_n[-1]
    signal   = np.zeros(n_signal)
    times    = np.arange(n_signal) * sampling_dt

    for i in range(n_peaks - 1):
        peak0 = peak_inds[i]
        peak1 = peak_inds[i + 1]

        sign0 = first_sign * (-1) ** i
        sign1 = - sign0

        t0        = times[peak0]
        t_half    = half_periods[i]
        frequency = 0.5 / t_half

        if i == 0:
            prev_t0       = times[peak0]
            prev_crossing = times[peak0] - 0.5 * t_half
            prev_sign     = sign0
            prev_freq     = frequency

        if i == n_peaks - 2:
            next_t0       = times[peak1]
            next_crossing = times[peak1] + 0.5 * t_half
            next_sign     = sign1
            next_freq     = frequency

        t_segment                    = times[peak0:peak1]
        signal[peak0:peak1] = np.cos(2 * np.pi * frequency * (t_segment - t0)) * sign0

    # Generate time vectors for extensions
    prev_crossing_times = np.arange( prev_crossing,       prev_t0, sampling_dt)
    next_crossing_times = np.arange(       next_t0, next_crossing, sampling_dt)

    # Trim the sinusoidal signal
    max_prev_time = prev_crossing_times[-1]
    min_next_time = next_crossing_times[0]

    trim_prev = len(times[times <= max_prev_time])
    trim_next = len(times[times >= min_next_time])

    times_trim  = times[trim_prev:-trim_next]
    signal_trim = signal[trim_prev:-trim_next]

    # Generate signals for extensions
    prev_crossing_signal = np.cos(2 * np.pi * prev_freq * (prev_crossing_times - prev_t0) ) * prev_sign
    next_crossing_signal = np.cos(2 * np.pi * next_freq * (next_crossing_times - next_t0) ) * next_sign

    # Concatenate time and signal arrays
    full_times  = np.concatenate((prev_crossing_times, times_trim, next_crossing_times))
    full_signal = np.concatenate((prev_crossing_signal, signal_trim, next_crossing_signal))

    # Scale the signal
    full_signal *= amplitude

    return full_times, full_signal

def transform_to_sinusoidal_signal(
    signal_original : np.ndarray,
    times_original  : np.ndarray,
    times_interp    : np.ndarray,
    amplitude       : float = 1.0,
    modulate_amp    : bool  = False,
    min_freq        : float = None,
    max_freq        : float = None,
    plot_data       : bool = False,
    save_data       : bool = False,
    verbose         : bool = False,

):
    ''' Make sinusoidal signal. '''

    sampling_dt = times_interp[1] - times_interp[0]

    # Get signal peaks
    peak_inds  = get_signal_peaks(signal_original)
    peak_times = times_original[peak_inds]
    peak_sign  = np.sign(signal_original[peak_inds[0]])

    # Apply frequency scaling
    peak_times = get_bounded_frequencies(
        peak_times = peak_times,
        min_freq   = min_freq,
        max_freq   = max_freq,
        verbose    = verbose,
    )

    # Create a sinusoidal signal with variable frequency
    full_times, full_signal = get_signal_with_variable_frequency(
        peak_times  = peak_times,
        first_sign  = peak_sign,
        amplitude   = amplitude,
        sampling_dt = sampling_dt,
    )

    # Modulate amplitude
    if modulate_amp:
        sig_max      = np.max(np.abs(signal_original))
        sig_norm     = signal_original / sig_max
        amp_fun      = get_real_time_amp_fun( times_original, sig_norm )
        full_signal *= amp_fun(full_times)

    # Study signal frequency
    get_cycle_by_cycle_frequency(
        timestep   = sampling_dt,
        peak_times = peak_times,
        verbose    = verbose,
        plot_data  = plot_data,
    )

    # Plot the sinusoidal signal
    if plot_data:

        x_min = min(full_times[0], times_interp[0])
        x_max = max(full_times[-1], times_interp[-1])

        # y_min = np.amin(signal_original)
        # y_max = np.amax(signal_original)
        # signal_original = 2 * (signal_original - y_min) / (y_max - y_min) - 1

        plt.figure(figsize=(10, 5))
        plt.plot(times_original, signal_original, label='Original Signal')
        plt.plot(times_original[peak_inds], signal_original[peak_inds], 'ro')
        plt.plot(full_times, full_signal, label='Full Signal')
        plt.plot([x_min, x_max], [0, 0], 'k--', lw=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(-1.1 * amplitude, +1.1 * amplitude)
        plt.legend()

    # Remove time offset
    full_times  -= full_times[0]

    # Save the signal
    if save_data:
        folder_name = 'network_modules/vortices/data'
        signal_dict = { 'time': full_times, 'signal': full_signal }
        signal_df   = pd.DataFrame(signal_dict)
        signal_df.to_csv(
            os.path.join(folder_name, 'kinematics_signal_sinusoidal.csv'),
            index=False,
        )

    return full_times, full_signal

def compute_angles_sum(
    x_signals   : np.ndarray,
    y_signals   : np.ndarray,
):
    ''' Compute the evolution of the sum of all angles '''

    # Vectors
    vects_x = x_signals[1:, :] - x_signals[:-1, :]
    vects_y = y_signals[1:, :] - y_signals[:-1, :]

    # Dot products
    dot_products = (
        vects_x[1:] * vects_x[:-1] +
        vects_y[1:] * vects_y[:-1]
    )

    # Cross products
    cross_products = (
        vects_x[1:] * vects_y[:-1] -
        vects_y[1:] * vects_x[:-1]
    )

    # Angles
    angles     = np.arctan2(cross_products, dot_products)
    angles_sum = np.sum(angles, axis=0)

    return angles_sum

def get_sinusoidal_signal(
    folder_name     : str,
    file_name       : str,
    target_fish     : str,
    start_recording : int,
    end_recording   : int,
    timestep        : float,
    total_duration  : float,
    freq_scaling    : float,
    save_data       : bool,
    plot_data       : bool,
    sig_name        : str   = 'y_SC 8',
    sig_amp         : float = 1.0,
    modulate_amp    : bool  = False,
    min_freq        : float = None,
    max_freq        : float = None,
    verbose         : bool  = True,

):
    ''' Get the sinusoidal signal. '''

    # Extend loading time
    total_duration_load = total_duration * 1.05
    times_interp        = np.arange(0, total_duration_load + timestep, timestep)

    # Load the signal from exmperimental angles
    times_original, signal_original = load_reference_signal(
        folder_name     = folder_name,
        file_name       = file_name,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        timestep        = timestep,
        total_duration  = total_duration_load,
        freq_scaling    = freq_scaling,
        verbose         = verbose,
        sig_name        = sig_name,
    )

    # Transform to sinusoidal
    (
        signal_times,
        signal_sinusoidal,
    ) = transform_to_sinusoidal_signal(
        signal_original  = signal_original,
        times_original   = times_original,
        times_interp     = times_interp,
        amplitude        = sig_amp,
        modulate_amp     = modulate_amp,
        min_freq         = min_freq,
        max_freq         = max_freq,
        plot_data        = plot_data,
        save_data        = save_data,
        verbose          = verbose,
    )

    # Trim the signal
    n_signal          = round(total_duration / timestep) + 1
    signal_times      = signal_times[:n_signal]
    signal_sinusoidal = signal_sinusoidal[:n_signal]

    return signal_times, signal_sinusoidal

###############################################################################
# COORDINATES EVOLUTION #######################################################
###############################################################################

def get_coordinates_evolution_sinusoidal(
    folder_name    : str,
    file_name      : str,
    target_fish    : str,
    signal_name    : str,
    start_recording: int,
    end_recording  : int,
    timestep       : float,
    total_duration : float,
    freq_scaling   : float,
    amp_scaling    : float,
    amp_modulation : bool,
    freq_min       : float,
    freq_max       : float,
    save_data      : bool = False,
    plot_data      : bool = False,
    **kwargs
):
    ''' Get the fish coordinates evolution from the sinusoidal signal. '''

    # Get the signal
    (
        signal_times,
        signal_sinusoidal,
    ) = get_sinusoidal_signal(
        folder_name     = folder_name,
        file_name       = file_name,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        timestep        = timestep,
        total_duration  = total_duration,
        freq_scaling    = freq_scaling,
        save_data       = save_data,
        plot_data       = plot_data,
        sig_name        = signal_name,
        verbose         = True,

        # AMPLITUDE MODULATION
        modulate_amp    = amp_modulation,

        # MANUAL FREQUENCY RANGE
        min_freq        = freq_min,
        max_freq        = freq_max,
    )

    # KINEMATICS (from Di Santo et al. 2021)
    wave_number = 0.95
    n_steps     = len(signal_times)
    phase_fun   = get_real_time_phase_fun(signal_times, signal_sinusoidal)

    if amp_modulation:
        amp_fun = get_real_time_amp_fun(signal_times, signal_sinusoidal)
    else:
        amp_fun = lambda t: 1.0

    # ref_points_x = np.array([0.0, 44.5, 66.5, 110.0]) / 110.0
    # ref_points_y = np.array([0.04, 0.05, 0.16, 0.24]) * 0.5
    # envelope_fun = CubicSpline(ref_points_x, ref_points_y, bc_type=((1, 0.1), (1, 0.1)))

    c1, c2, c3   = +0.05, -0.13, +0.28
    envelope_fun = lambda s: 0.6 * ( c1 + c2 * s + c3 * s**2 ) * amp_scaling

    body_length  = 0.018
    positions_x  = np.linspace(0, 1, 100) * body_length
    positions_y  = np.zeros_like(positions_x)
    s_vals       = positions_x / body_length
    amplitudes_y = body_length * envelope_fun(s_vals)

    # KINEMATIC WAVE
    positions_x_evolution = np.array([positions_x] * n_steps)
    positions_y_evolution = np.array(
        [
            amp * amp_fun(signal_times) * np.cos(
                phase_fun(signal_times) - 2*np.pi * wave_number * s_val
            )
            for s_val, amp in zip(s_vals, amplitudes_y)
        ]
    ).T

    return (
        signal_times,
        positions_x_evolution,
        positions_y_evolution,
        amp_fun,
        phase_fun,
    )

###############################################################################
# MAIN ########################################################################
###############################################################################

def main():

    folder_name = 'network_modules/vortices/data'
    file_name   = 'kinematics_recording.csv'

    save_data = False
    plot_data = True

    # KI reference: 12100 (Containing chunk: 11300)
    # KI reference: 12260 (Containing chunk: 12400)

    target_fish     = 'Fish3'
    start_recording = 11300 # 12100
    end_recording   = 12400 # 12260
    sig_name        = 'y_SC 8'

    # Get the frequency-scaled signal
    timestep       = 0.001
    total_duration = 30
    freq_scaling   = 0.25

    # Load the signal
    (
        signal_times,
        signal_sinusoidal,
        signal_freqs,
    ) = get_sinusoidal_signal(
        folder_name     = folder_name,
        file_name       = file_name,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        timestep        = timestep,
        total_duration  = total_duration,
        freq_scaling    = freq_scaling,
        plot_data       = plot_data,
        save_data       = save_data,
        sig_name        = sig_name,
    )

    plt.show()

    return


if __name__ == '__main__':
    main()