'''
Loading of the processed results of single vortex matching simulations.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_paths import RESULTS_PATH as DEFAULT_RESULTS_PATH

import dill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH    = DEFAULT_RESULTS_PATH
MODULE_NAME     = 'net_farms_zebrafish'
SIMULATION_TAG = 'dynamic_water_vortices_fixed_head_experimental'

from scipy.signal import find_peaks


###############################################################################
# UTILS #######################################################################
###############################################################################

def _linearize_phases(
    times  : np.ndarray,
    phases : np.ndarray,
    plot   : bool = False,
):
    ''' Linearize phases to cycles '''

    # Find crossings 2pi
    phases_wrapped   = phases % (2 * np.pi)
    cycle_crossings  = np.where(np.diff(phases_wrapped) < -np.pi)[0].astype(int)

    # Create linearized phases
    phases_lin = np.copy(phases)

    for i in range(len(cycle_crossings) - 1):

        # Get start and end indices
        idx0 = cycle_crossings[i]
        idx1 = cycle_crossings[i + 1]

        v0 = phases[idx0]
        v1 = phases[idx1]

        # Linear interpolation for each cycle
        # phases_lin[idx0:idx1] = (
        #     phases[idx0] +
        #     np.linspace(0, 2 * np.pi, idx1 - idx0, endpoint=False)
        # )

        phases_lin[idx0:idx1 + 1] = (
            v0 + (v1 - v0) * np.linspace(0, 1, idx1 - idx0 + 1)
        )


    if not plot:
        return phases_lin

    # Plot
    phases_lin_wrapped = phases_lin % (2 * np.pi)

    plt.figure()
    plt.plot(times, phases_wrapped, label='phases')
    plt.plot(times, phases_lin_wrapped, label='phases_lin')
    plt.plot(
        times[cycle_crossings],
        phases_wrapped[cycle_crossings],
        'o',
        color = 'red',
        label = 'crossings'
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (rad)')
    plt.title('Phase Unwrapping Check')
    plt.legend()

    return phases_lin

def _get_linearized_phase_difference(
    data_df: pd.DataFrame,
):
    ''' Get the linearized phase difference between experimental and reference '''

    # Linearize phases
    times = data_df['times'].values
    p_exp = data_df['joint_phases'].values
    p_ref = data_df['joint_phases_ref'].values

    p_exp = _linearize_phases( times=times, phases=p_exp, plot=False )
    p_ref = _linearize_phases( times=times, phases=p_ref, plot=False )

    # Compute difference
    p_diff = p_ref - p_exp

    # Normalize phases
    p_diff = np.mod(p_diff, 2*np.pi)
    p_diff[p_diff > np.pi] -= 2*np.pi

    # Update dataframe
    data_df['joint_phases_diff'] = p_diff

    return data_df

def _get_signal_amps(
    signal : np.ndarray,
) -> np.ndarray:
    ''' Calculate the amplitude using the peaks of the signal '''
    signal_osc   = signal - np.mean(signal)
    signal_abs   = np.abs(signal_osc)
    signal_peaks = find_peaks(signal_abs)[0]
    signal_amps  = signal_abs[signal_peaks]
    return signal_amps, signal_peaks

def _compute_signal_amps_stats(
    signal : np.ndarray,
) -> np.ndarray:
    ''' Calculate the amplitude using the peaks of the signal '''

    signal_amps, _ = _get_signal_amps(signal)

    if len(signal_amps) == 0:
        return 0.0, 0.0

    amp_mean = np.mean(signal_amps)
    amp_std  = np.std(signal_amps)

    return amp_mean, amp_std

def match_peaks_to_nearest(inds_1, inds_2):
    """Match each index in inds_1 to the nearest index in inds_2."""
    matches = []

    for p in inds_1:
        i = np.searchsorted(inds_2, p)

        # Previous + Next candidate in inds_2
        prev = inds_2[i-1] if i > 0 else None
        next = inds_2[i]   if i < len(inds_2) else None

        # choose nearest
        if prev is None:
            matches.append(next)
        elif next is None:
            matches.append(prev)
        else:
            matches.append(prev if abs(prev - p) < abs(next - p) else next)

    return np.array(matches)

###############################################################################
# FILE NAMES ##################################################################
###############################################################################

def get_folder_name_from_seed(
    np_random_seed: int,
    mod_name      : str,
    sim_tag       : str,
) -> str:
    ''' Get the folder name '''
    return f'{mod_name}_{sim_tag}_{np_random_seed}_SIM'

def get_seed_from_folder_name( folder_name: str ):
    ''' Get the seed from the folder name '''
    parts = folder_name.split('_')
    seed  = int(parts[-1])
    return seed

def get_ps_tag(
    stim_a_off    : float,
    feedback_mode : bool,
) -> str:
    ''' Get the ps tag '''
    if stim_a_off <= -3.5:
        return 'passive_body'
    if feedback_mode:
        return 'closed_loop'
    return 'open_loop'

def get_vx_tag(
    vortex_mode: int,
) -> str:
    ''' Get the vx tag '''
    if vortex_mode:
        return 'with_vortices'
    return 'pure_flow'

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_parameters_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
):
    ''' Load the parameters for a single simulation '''

    # Get the target folder
    results_folder = os.path.join( results_root_folder, folder_name, process_folder)

    # Load parameters
    with open(os.path.join(results_folder, 'snn_parameters_process.dill'), 'rb') as f:
        simulation_parameters = dill.load(f)

    return simulation_parameters

def load_performance_data_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
):
    ''' Load the performance data for a single simulation '''

    # Get the target folder
    results_folder = os.path.join( results_root_folder, folder_name, process_folder)

    # Load performance data
    with open(os.path.join(results_folder, 'snn_performance_process.dill'), 'rb') as f:
        simulation_performance = dill.load(f)

    return simulation_performance

def load_com_vs_angle_data_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
    target_quantities  : list[str] = None,
) -> pd.DataFrame:
    ''' Load the data '''

    if target_quantities is None:
        target_quantities = [
            'times',
            'com_positions_diff',
            'joint_phases_diff',
        ]
    if 'times' not in target_quantities:
        target_quantities.insert(0, 'times')
    if 'joint_positions' not in target_quantities:
        target_quantities.append('joint_positions')
    if 'coherence_vel_vel' not in target_quantities:
        target_quantities.append('coherence_vel_vel')
    if 'coherence_ang_vor' not in target_quantities:
        target_quantities.append('coherence_ang_vor')

    if 'ex_firing_rate' not in target_quantities:
        target_quantities.append('ex_firing_rate')
    if 'in_firing_rate' not in target_quantities:
        target_quantities.append('in_firing_rate')
    if 'mn_firing_rate' not in target_quantities:
        target_quantities.append('mn_firing_rate')
    if 'ps_firing_rate' not in target_quantities:
        target_quantities.append('ps_firing_rate')


    # NOTE: Data was saved in the following format:
    #     current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    #     file_name    = f'com_vs_angle_data_{current_time}.csv'


    # Get the target folder
    results_folder = os.path.join( results_root_folder, folder_name, process_folder)
    farms_folder   = os.path.join( results_folder, 'run_0/farms')

    # Check that there is one .csv file
    csv_files = [ f for f in os.listdir(farms_folder) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {farms_folder}")

    # Sort by modification time, descending
    csv_files.sort(
        key     = lambda f: os.path.getmtime( os.path.join(farms_folder, f) ),
        reverse = True
    )

    # Select the most recent CSV file
    csv_path = os.path.join(farms_folder, csv_files[0])

    # Load the data
    data_df = pd.read_csv(csv_path)

    # LINEARIZE PHASES DIFFERENCE
    # NOTE: Considers linearized phase evolution between peaks
    data_df = _get_linearized_phase_difference( data_df )

    # Only keep the target quantities
    data_df = data_df[target_quantities]

    return data_df

def _add_power_metrics(
    simulation_performance: dict,
    times                 : np.ndarray,
):
    '''
    Add power metrics if not present
    NOTE: 2.0 seconds are discarded in the energy computation
    '''
    duration_energy = times[-1] - times[0] - 2.0

    for metric_type in ['total', 'active']:
        metric = f'mech_power_{metric_type}'
        if metric in simulation_performance:
            continue
        simulation_performance[metric] = (
            simulation_performance[f'mech_energy_{metric_type}'] /
            duration_energy
        )
    return simulation_performance

def _add_tail_angle_metrics(
    simulation_performance: dict,
    times                 : np.ndarray,
    tail_angles           : np.ndarray,
):
    ''' Add tail angle metrics '''

    # Remove first 5 seconds
    n_skip      = np.sum( times < 5.0 )
    tail_angles = tail_angles[n_skip:]

    # Mean and Std of tail angle amplitude
    (
        tail_angle_amp_mean,
        tail_angle_amp_std
    ) = _compute_signal_amps_stats(tail_angles)

    simulation_performance['tail_angle_amp_mean'] = np.array([tail_angle_amp_mean])
    simulation_performance['tail_angle_amp_std']  = np.array([tail_angle_amp_std])

    return simulation_performance

def _add_tail_angle_distribution_metrics(
    simulation_performance: dict,
    times                 : np.ndarray,
    tail_angles           : np.ndarray,
    coherence_lin         : np.ndarray,
    coherence_rot         : np.ndarray,
    ex_rates              : np.ndarray,
    in_rates              : np.ndarray,
    mn_rates              : np.ndarray,
    ps_rates              : np.ndarray,
):
    ''' Add tail angle metrics '''

    # Remove first 5 seconds
    n_skip   = np.sum( times < 5.0 )
    times    = times[n_skip:]
    timestep = times[1] - times[0]
    duration = times[-1] - times[0]

    tail_angles   = tail_angles[n_skip:]
    coherence_lin = coherence_lin[n_skip:]
    coherence_rot = coherence_rot[n_skip:]

    ex_rates = ex_rates[n_skip:]
    in_rates = in_rates[n_skip:]
    mn_rates = mn_rates[n_skip:]
    ps_rates = ps_rates[n_skip:]

    # Tail angles amplitudes
    _, tail_angle_inds = _get_signal_amps(tail_angles)
    _, ex_rate_inds    = _get_signal_amps(ex_rates)
    _, in_rate_inds    = _get_signal_amps(in_rates)
    _, mn_rate_inds    = _get_signal_amps(mn_rates)
    _, ps_rate_inds    = _get_signal_amps(ps_rates)

    # Nearest-peak matching
    tail_to_ex_matches = match_peaks_to_nearest(tail_angle_inds, ex_rate_inds)
    tail_to_in_matches = match_peaks_to_nearest(tail_angle_inds, in_rate_inds)
    tail_to_mn_matches = match_peaks_to_nearest(tail_angle_inds, mn_rate_inds)
    tail_to_ps_matches = match_peaks_to_nearest(tail_angle_inds, ps_rate_inds)

    # Get amps distribution
    tail_angle_times   = times[tail_angle_inds]
    tail_angle_amps    = np.abs(tail_angles[tail_angle_inds])

    # Helper: average signal in a time window around each peak
    def _avg_around_peaks(signal, times, peak_inds, window_s=0.2):
        half = window_s / 2.0
        vals = []
        for idx in peak_inds:
            t0   = times[idx]
            mask = (times >= t0 - half) & (times <= t0 + half)
            vals.append(signal[mask].mean())
        return np.asarray(vals)

    # Coherence averaged over 200 ms around each tail-angle peak
    coherence_lin_vals = _avg_around_peaks(coherence_lin, times, tail_angle_inds)
    coherence_rot_vals = _avg_around_peaks(coherence_rot, times, tail_angle_inds)

    # Add to performance
    simulation_performance['tail_angle_times_distribution'] = np.array([tail_angle_times])
    simulation_performance['tail_angle_amps_distribution']  = np.array([tail_angle_amps])
    simulation_performance['coherence_lin_distribution']    = np.array([coherence_lin_vals])
    simulation_performance['coherence_rot_distribution']    = np.array([coherence_rot_vals])

    # Rates distributions
    def add_rate_metrics(rates, inds, inds_to_tail, tag):
        sim_perf   = simulation_performance
        rates_abs  = np.abs(rates)

        # Whole signal
        sig_mean  = np.mean(rates_abs)
        sig_power = np.sum(rates_abs**2) * timestep / duration

        sim_perf[f'{tag}_rate_mean']  = np.array([sig_mean])
        sim_perf[f'{tag}_rate_power'] = np.array([sig_power])

        # Peaks
        if len(inds) > 0:
            rates_amps = rates_abs[inds]
            amp_mean   = np.mean(rates_amps)
            amp_std    = np.std(rates_amps)

            sim_perf[f'{tag}_rate_amp_mean'] = np.array([amp_mean])
            sim_perf[f'{tag}_rate_amp_std']  = np.array([amp_std])
        else:
            sim_perf[f'{tag}_rate_amp_mean'] = np.array([0.0])
            sim_perf[f'{tag}_rate_amp_std']  = np.array([0.0])

        # Peaks mapped to tail angle peaks
        if len(inds_to_tail) > 0:
            rates_amps = [
                rates_abs[i] if i is not None else np.nan
                for i in inds_to_tail
            ]
            rates_amps = np.array(rates_amps)

            sim_perf[f'{tag}_rate_amps_distribution'] = rates_amps
        else:
            sim_perf[f'{tag}_rate_amps_distribution'] = np.array([0.0])

    add_rate_metrics(ex_rates, ex_rate_inds, tail_to_ex_matches, 'ex')
    add_rate_metrics(in_rates, in_rate_inds, tail_to_in_matches, 'in')
    add_rate_metrics(mn_rates, mn_rate_inds, tail_to_mn_matches, 'mn')
    add_rate_metrics(ps_rates, ps_rate_inds, tail_to_ps_matches, 'ps')

    return simulation_performance

def load_results_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
    target_quantities  : list[str] = None,
) -> pd.DataFrame:
    ''' Load the data '''

    # LOAD PARAMETERS
    simulation_parameters = load_parameters_single_simulation(
        results_root_folder = results_root_folder,
        folder_name         = folder_name,
        process_folder      = process_folder,
    )

    simulation_metadata = {
        'seed_value': np.array([ simulation_parameters['seed_value'] ]),
        'file_tag'  : np.array([ simulation_parameters['simulation_data_file_tag'] ]),
        'stim_a_off': np.array([ simulation_parameters['stim_a_off'] ]),
    }

    # LOAD PERFORMANCE DATA
    simulation_performance = load_performance_data_single_simulation(
        results_root_folder = results_root_folder,
        folder_name         = folder_name,
        process_folder      = process_folder,
    )

    # LOAD COM VS ANGLE DATA
    data_df = load_com_vs_angle_data_single_simulation(
        results_root_folder = results_root_folder,
        folder_name         = folder_name,
        process_folder      = process_folder,
        target_quantities   = target_quantities,
    )

    # Add power metrics if not present
    simulation_performance = _add_power_metrics(
        simulation_performance = simulation_performance,
        times                 = data_df['times'].values
    )

    # Add tail angle metrics
    simulation_performance = _add_tail_angle_metrics(
        simulation_performance = simulation_performance,
        times                 = data_df['times'].values,
        tail_angles            = data_df['joint_positions'].values,
    )

    # Add tail angle vs ps firing metrics
    simulation_performance = _add_tail_angle_distribution_metrics(
        simulation_performance = simulation_performance,
        times                  = data_df['times'].values,
        tail_angles            = data_df['joint_positions'].values,
        coherence_lin          = data_df['coherence_vel_vel'].values,
        coherence_rot          = data_df['coherence_ang_vor'].values,
        ex_rates               = data_df['ex_firing_rate'].values,
        in_rates               = data_df['in_firing_rate'].values,
        mn_rates               = data_df['mn_firing_rate'].values,
        ps_rates               = data_df['ps_firing_rate'].values,
    )

    # Add metadata
    simulation_performance = simulation_metadata | simulation_performance

    return data_df, simulation_performance

def load_results_multi_simulation(
    results_root_folder: str,
    process_folders    : list[list[str]],
    np_random_seed_vals: np.ndarray,
    module_name        : str = MODULE_NAME,
    simulation_tag     : str = SIMULATION_TAG,
) -> list[list[list[pd.DataFrame]]]:
    ''' Load the data for multiple simulations '''

    # Folders:   net_farms_zebrafish_dynamic_water_vortices_fixed_head_100_SIM
    # Processes: process_030Hz_at_010x_distance_050BL_seed_100_closed_loop
    # CSV files: run_0/farms/com_vs_angle_data_2024-12-13_16-34-01.csv

    # Folder names
    folder_names = [
        get_folder_name_from_seed(
            np_random_seed = seed,
            mod_name       = module_name,
            sim_tag        = simulation_tag,
        )
        for seed in np_random_seed_vals
    ]

    # Load the data
    results_list = [
        [
            [
                load_results_single_simulation(
                    results_root_folder = results_root_folder,
                    folder_name         = folder_names[seed_ind],
                    process_folder      = process_folder,
                )
                for seed_ind, process_folder in enumerate(process_folders_multi_seed)
            ]
            for process_folders_multi_seed in process_folders_fb_mode
        ]
        for process_folders_fb_mode in process_folders
    ]

    data_df_list = [
        [ [ rseed[0] for rseed in rprs ] for rprs in rfb ]
        for rfb in results_list
    ]

    performance_list = [
        [ [ rseed[1] for rseed in rprs ] for rprs in rfb ]
        for rfb in results_list
    ]

    return data_df_list, performance_list
