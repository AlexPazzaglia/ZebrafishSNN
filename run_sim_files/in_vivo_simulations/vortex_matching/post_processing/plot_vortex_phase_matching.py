'''
Post-processing of the vortex phase matching simulations (Fig. 4A-F).

For each PS weight (20: closed loop, 0: open loop) the script loads the results
collected by move_results_phase_matching.py, computes the phase difference
between leader and follower as a function of their relative position, and
plots the phase-distance distributions and the performance metrics (e.g.
mechanical power) with and without vortices.

The difference analysis then compares closed and open loop, with and without
vortices, and saves the corresponding figures and statistics in
ANALYSIS_PATH/comparing_vortices/comparisons.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_paths import ANALYSIS_PATH

import re
import json
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from scipy.signal import find_peaks
from scipy.stats import (
    ttest_rel, wilcoxon,        # Paired tests
    ttest_ind, mannwhitneyu,    # Independent tests
)

import histogram_fitting

from run_sim_files.in_vivo_simulations.\
    vortex_matching.sim_runner import get_water_dynamics_file_name

RESULTS_PATH   = f'{ANALYSIS_PATH}/'
MODULE_NAME    = 'net_farms_zebrafish'
SIMULATION_TAG = 'dynamic_water_vortices_free_head_muscle_FN_5500'

LEADER_FREQUENCY = 3.5
BODY_MASS        = 5.022e-05
COM_POS_REL      = 0.4321805865303308
PHASE_OFFSET     = 0.0 * np.pi

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

def _remove_date_from_string(
    s: str,
):
    ''' Remove date from string '''
    return re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+', '', s)

def _shorten_tag(tag: str) -> str:
    ''' Shorten the tag '''

    tag = tag.replace('process_', '')
    tag = tag.replace('passive_body_', '')
    tag = tag.replace('continuous_', '')
    tag = tag.replace('bounded_NaN_NaN_Hz_', '')
    tag = tag.replace('30s_', '')
    tag = tag.replace('15s_', '')
    tag = tag.replace('10s_', '')
    tag = _remove_date_from_string(tag)
    tag = tag.replace('__', '_')

    return tag

def get_process_folder_name_from_parameters( parameters: dict ):
    ''' Get the simulation tag '''

    leader_signal_str : str   = parameters['leader_signal_str']
    spawn_x           : float = parameters['spawn_x']
    spawn_y           : float = parameters['spawn_y']
    np_random_seed    : int   = parameters['np_random_seed']
    vx_tag            : int   = parameters['vx_tag']
    ps_tag            : str   = parameters['ps_tag']

    # Remove the date
    leader_signal_str = _remove_date_from_string(leader_signal_str)

    # Transform to strings
    spawn_x_str    = str(round( spawn_x * 100)).zfill(3)  # E.g. 0.86 -> 086BL
    spawn_y_str    = str(round( spawn_y * 100)).zfill(3)  # E.g. 0.00 -> 000BL

    process_folder = (
        f'free_body_'
        f'{leader_signal_str}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}_'
        f'{vx_tag}'
    )

    process_folder = _shorten_tag(process_folder)

    return process_folder

def get_process_folder_nickname_from_parameters( parameters: dict ):
    ''' Get the simulation tag '''

    leader_signal_str : str   = parameters['leader_signal_str']
    vx_tag            : int   = parameters['vx_tag']
    ps_tag            : str   = parameters['ps_tag']

    # Remove the date
    leader_signal_str = _remove_date_from_string(leader_signal_str)

    process_folder = (
        f'{leader_signal_str}_'
        f'{ps_tag}_'
        f'{vx_tag}'
    )

    process_folder = _shorten_tag(process_folder)

    return process_folder

def get_folder_name_from_seed(
    np_random_seed: int,
    mod_name      : str,
    sim_tag       : str,
) -> str:
    ''' Get the folder name '''
    return f'{mod_name}_{sim_tag}_{np_random_seed}_SIM'

def get_seed_from_folder_name(
    folder_name: str
) -> int:
    ''' Get the seed from the folder name '''
    parts = folder_name.split('_')
    seed  = int(parts[-1])
    return seed

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
    data_df                : pd.DataFrame,
):
    ''' Add tail angle metrics '''

    times         = data_df['times'].values
    tail_angles   = data_df['joint_positions'].values
    coherence_lin = data_df['coherence_vel_vel'].values
    coherence_rot = data_df['coherence_ang_vor'].values

    ex_rates_d      = data_df['ex_firing_rate_diff'].values
    in_rates_d      = data_df['in_firing_rate_diff'].values
    mn_rates_d      = data_df['mn_firing_rate_diff'].values
    ps_rates_d      = data_df['ps_firing_rate_diff'].values

    ex_rates_s      = data_df['ex_firing_rate_sum'].values
    in_rates_s      = data_df['in_firing_rate_sum'].values
    mn_rates_s      = data_df['mn_firing_rate_sum'].values
    ps_rates_s      = data_df['ps_firing_rate_sum'].values

    # Remove first 5 seconds
    n_skip   = np.sum( times < 5.0 )
    times    = times[n_skip:]
    timestep = times[1] - times[0]
    duration = times[-1] - times[0]

    tail_angles   = tail_angles[n_skip:]
    coherence_lin = coherence_lin[n_skip:]
    coherence_rot = coherence_rot[n_skip:]

    ex_rates_d = ex_rates_d[n_skip:]
    in_rates_d = in_rates_d[n_skip:]
    mn_rates_d = mn_rates_d[n_skip:]
    ps_rates_d = ps_rates_d[n_skip:]

    ex_rates_s = ex_rates_s[n_skip:]
    in_rates_s = in_rates_s[n_skip:]
    mn_rates_s = mn_rates_s[n_skip:]
    ps_rates_s = ps_rates_s[n_skip:]

    # Signals amplitudes
    _, tail_angle_inds = _get_signal_amps(tail_angles)
    _, ex_rate_inds    = _get_signal_amps(ex_rates_d)
    _, in_rate_inds    = _get_signal_amps(in_rates_d)
    _, mn_rate_inds    = _get_signal_amps(mn_rates_d)
    _, ps_rate_inds    = _get_signal_amps(ps_rates_d)


    # Helper: average signal in a time window around each peak
    # def _avg_around_peaks(signal, times, peak_inds, window_s=0.2):
    #     half = window_s / 2.0
    #     vals = []
    #     for idx in peak_inds:
    #         t0   = times[idx]
    #         mask = (times >= t0 - half) & (times <= t0 + half)
    #         vals.append(signal[mask].mean())
    #     return np.asarray(vals)

    # Coherence averaged over 200 ms around each tail-angle peak
    # coherence_lin_vals = _avg_around_peaks(coherence_lin, times, tail_angle_inds)
    # coherence_rot_vals = _avg_around_peaks(coherence_rot, times, tail_angle_inds)


    # Rates distributions
    def add_global_rate_metrics(rates_d, rates_s, inds, tag):
        sim_perf    = simulation_performance
        rates_d_abs = np.abs(rates_d)

        # Rates difference
        abs_diff_mean  = np.mean(rates_d_abs)
        abs_diff_power = np.sum(rates_d_abs**2) * timestep / duration
        abs_diff_rms   = np.sqrt( abs_diff_power )

        sim_perf[f'{tag}_diff_rate_mean']  = np.array([abs_diff_mean])
        sim_perf[f'{tag}_diff_rate_power'] = np.array([abs_diff_power])
        sim_perf[f'{tag}_diff_rate_rms']   = np.array([abs_diff_rms])

        # Rates sum
        sum_mean  = np.mean(rates_s)
        sum_power = np.sum(rates_s**2) * timestep / duration

        sim_perf[f'{tag}_sum_rate_mean']  = np.array([sum_mean])
        sim_perf[f'{tag}_sum_rate_power'] = np.array([sum_power])

        # Peaks
        if len(inds) > 0:
            rates_amps = rates_d_abs[inds]
            amp_mean   = np.mean(rates_amps)
            amp_std    = np.std(rates_amps)

            sim_perf[f'{tag}_diff_rate_amp_mean'] = np.array([amp_mean])
            sim_perf[f'{tag}_diff_rate_amp_std']  = np.array([amp_std])
        else:
            sim_perf[f'{tag}_diff_rate_amp_mean'] = np.array([0.0])
            sim_perf[f'{tag}_diff_rate_amp_std']  = np.array([0.0])

    add_global_rate_metrics(ex_rates_d, ex_rates_s, ex_rate_inds, 'ex')
    add_global_rate_metrics(in_rates_d, in_rates_s, in_rate_inds, 'in')
    add_global_rate_metrics(mn_rates_d, mn_rates_s, mn_rate_inds, 'mn')
    add_global_rate_metrics(ps_rates_d, ps_rates_s, ps_rate_inds, 'ps')

    # Segments rates
    def add_segment_rate_metrics(rate_l, rate_r, tag, seg):

        tag        = f'{tag}_seg_{seg}'
        sim_perf   = simulation_performance
        rate_s     = rate_l + rate_r
        rate_d_abs = np.abs(rate_l - rate_r)

        # Sides
        mean_l  = np.mean(rate_l)
        mean_r  = np.mean(rate_r)
        power_l = np.sum(rate_l**2) * timestep / duration
        power_r = np.sum(rate_r**2) * timestep / duration

        sim_perf[f'{tag}_mean_rate_l'] = np.array([mean_l])
        sim_perf[f'{tag}_mean_rate_r'] = np.array([mean_r])
        sim_perf[f'{tag}_mean_rate']   = np.array([mean_l + mean_r]) / 2

        sim_perf[f'{tag}_power_rate_l'] = np.array([power_l])
        sim_perf[f'{tag}_power_rate_r'] = np.array([power_r])
        sim_perf[f'{tag}_power_rate']   = np.array([power_l + power_r]) / 2

        # Rates difference
        abs_diff_mean  = np.mean(rate_d_abs)
        abs_diff_power = np.sum(rate_d_abs**2) * timestep / duration
        abs_diff_rms   = np.sqrt( abs_diff_power )

        sim_perf[f'{tag}_diff_rate_mean']  = np.array([abs_diff_mean])
        sim_perf[f'{tag}_diff_rate_power'] = np.array([abs_diff_power])
        sim_perf[f'{tag}_diff_rate_rms']   = np.array([abs_diff_rms])

        # Rates sum
        sum_mean  = np.mean(rate_s)
        sum_power = np.sum(rate_s**2) * timestep / duration

        sim_perf[f'{tag}_sum_rate_mean']  = np.array([sum_mean])
        sim_perf[f'{tag}_sum_rate_power'] = np.array([sum_power])

        # Peaks
        rate_amps_l = rate_l[find_peaks(rate_l)[0]]
        rate_amps_r = rate_r[find_peaks(rate_r)[0]]
        rate_amps   = np.concatenate([rate_amps_l, rate_amps_r])

        amp_mean_l = np.mean(rate_amps_l) if len(rate_amps_l) > 0 else 0.0
        amp_mean_r = np.mean(rate_amps_r) if len(rate_amps_r) > 0 else 0.0
        amp_mean   = np.mean(rate_amps)   if len(rate_amps) > 0   else 0.0

        sim_perf[f'{tag}_rate_amp_mean']   = np.array([amp_mean])
        sim_perf[f'{tag}_rate_amp_mean_l'] = np.array([amp_mean_l])
        sim_perf[f'{tag}_rate_amp_mean_r'] = np.array([amp_mean_r])

        return


    n_seg = 32
    for ner_type in ['ex', 'in', 'mn', 'ps']:
        for i in range(n_seg):

            counts_l = f'{ner_type}_firing_rate_{2*i + 0}'
            counts_r = f'{ner_type}_firing_rate_{2*i + 1}'

            rate_l = data_df[counts_l].values
            rate_r = data_df[counts_r].values

            rate_l = rate_l[n_skip:]
            rate_r = rate_r[n_skip:]

            add_segment_rate_metrics(rate_l, rate_r, ner_type, i)

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
        times                  = data_df['times'].values,
        tail_angles            = data_df['joint_positions'].values,
    )

    # Add firing rate metrics
    n_seg = 32

    for ner_type in ['ex', 'in', 'mn', 'ps']:

        counts_l = [ f'{ner_type}_firing_rate_{2*i + 0}' for i in range(n_seg) ]
        counts_r = [ f'{ner_type}_firing_rate_{2*i + 1}' for i in range(n_seg) ]

        rates_l = data_df[counts_l].values
        rates_r = data_df[counts_r].values

        rates_diff = rates_l - rates_r
        rates_sum  = rates_l + rates_r

        data_df[f'{ner_type}_firing_rate_diff'] = np.mean(rates_diff, axis=1)
        data_df[f'{ner_type}_firing_rate_sum']  = np.mean(rates_sum, axis=1)

        data_df[f'{ner_type}_firing_rate_l'] = np.mean(rates_l, axis=1)
        data_df[f'{ner_type}_firing_rate_r'] = np.mean(rates_r, axis=1)

        # Drop the individual segment rates to save memory
        # data_df.drop(columns=counts_l + counts_r, inplace=True)

    # Add tail angle vs ps firing metrics
    simulation_performance = _add_tail_angle_distribution_metrics(
        simulation_performance = simulation_performance,
        data_df                = data_df,
    )

    # Add metadata
    simulation_performance = simulation_metadata | simulation_performance

    return data_df, simulation_performance

###############################################################################
# HISTOGRAM ###################################################################
###############################################################################

class SchoolingPlotting:

    def __init__(
        self,
        results_root_folder: str,
        stim_a_off         : float,
        vortex_mode        : int,
        feedback_mode      : int,
        leader_signal_str  : str,
        spawn_x_vals       : np.ndarray,
        spawn_y_vals       : np.ndarray,
        np_random_seed_vals: np.ndarray,
        leader_freq        : float = LEADER_FREQUENCY,
        use_actual_com_pos : bool = True,
        use_actual_phases  : bool = True,
        module_name        : str = MODULE_NAME,
        simulation_tag     : str = SIMULATION_TAG,
        skip_histogram     : bool = False,
        excluded_condition : callable = None,
    ):
        """ Initialize an incremental 2D histogram."""

        # Parameters
        self.module_name         = module_name
        self.simulation_tag      = simulation_tag
        self.results_root_folder = results_root_folder

        self.stim_a_off          = stim_a_off

        self.vortex_mode         = vortex_mode
        self.feedback_mode       = feedback_mode

        self.leader_signal_str   = leader_signal_str
        self.leader_freq         = leader_freq

        self.spawn_x_vals        = spawn_x_vals
        self.spawn_y_vals        = spawn_y_vals
        self.np_random_seed_vals = np_random_seed_vals

        self.use_actual_com_pos  = use_actual_com_pos
        self.use_actual_phases   = use_actual_phases
        self.skip_histogram      = skip_histogram

        self.n_spawn_x = len(spawn_x_vals)
        self.n_spawn_y = len(spawn_y_vals)
        self.n_seeds   = len(np_random_seed_vals)
        self.n_sim     = self.n_spawn_x * self.n_spawn_y * self.n_seeds
        self.sim_shape = (self.n_spawn_x, self.n_spawn_y, self.n_seeds)

        self.vx_tag    = self._get_vx_tag()
        self.ps_tag    = self._get_ps_tag()

        # Excluded simulations
        if excluded_condition is None:
            # All included
            excl_map = np.zeros(self.n_sim, dtype=bool)
        else:
            # Set exclusion condition
            excl_map = np.array(
                [
                    excluded_condition(spawn_x, spawn_y, seed)
                    for spawn_x in spawn_x_vals
                    for spawn_y in spawn_y_vals
                    for seed in np_random_seed_vals
                ]
            )

        self.excl_map_flat = excl_map[:]
        self.excl_map_matr = excl_map[:].reshape(self.sim_shape)

        # Define parameters
        self._define_parameters()

        # Folders names
        self._get_folder_names()

        # Initialize the histogram counts
        self._initialize_histogram()

        # Processes names
        self._get_processes_names()

        return

    ###########################################################################
    # PARAMETERS ##############################################################
    ###########################################################################

    def _define_parameters(self):
        ''' Define the parameters for the simulation '''

        self.process_parameters = {
            'leader_signal_str': self.leader_signal_str,
            'feedback_mode'    : self.feedback_mode,
            'vx_tag'           : self.vx_tag,
            'ps_tag'           : self.ps_tag,
        }

        self.all_parameters = self.process_parameters |{
            'spawn_x_vals'       : self.spawn_x_vals,
            'spawn_y_vals'       : self.spawn_y_vals,
            'np_random_seed_vals': self.np_random_seed_vals,
        }

        return

    def _get_processes_names(self):
        ''' Get the folder names for the simulations '''

        # Processes
        self.process_folders = [
            [
                get_process_folder_name_from_parameters(
                    self.process_parameters | {
                        'spawn_x'        : spawn_x,
                        'spawn_y'        : spawn_y,
                        'np_random_seed' : seed,
                    }
                )
                for seed in self.np_random_seed_vals
            ]
            for spawn_x in self.spawn_x_vals
            for spawn_y in self.spawn_y_vals
        ]

        # Save folder and file
        self.save_file_root = get_process_folder_nickname_from_parameters(
            parameters = self.process_parameters,
        )
        self.save_folder = f'{self.results_root_folder}/figures/{self.save_file_root}'

        os.makedirs(self.save_folder, exist_ok=True)

        return

    def _get_folder_names(self):
        ''' Get the folder names for the simulations '''

        self.folder_names = [
            get_folder_name_from_seed(
                np_random_seed = seed,
                mod_name       = self.module_name,
                sim_tag        = self.simulation_tag,
            )
            for seed in self.np_random_seed_vals
        ]

        self.save_file_root = None
        self.save_folder    = None

        return

    def _get_ps_tag(self) -> str:
        ''' Get the ps tag '''
        return get_ps_tag(
            stim_a_off    = self.stim_a_off,
            feedback_mode = self.feedback_mode,
        )

    def _get_vx_tag(self) -> str:
        ''' Get the vx tag '''
        return get_vx_tag(
            vortex_mode = self.vortex_mode,
        )

    ###########################################################################
    # DATA COLLECTION ##########################################################
    ###########################################################################

    # SAVED DATA

    def _load_saved_parameters(self):
        '''
        Load the collected data if it exists
        Return False if the loaded parameters match the expected ones.
        '''

        # Load the parameters
        save_folder        = self.save_folder
        save_file_root     = self.save_file_root
        process_parameters = self.process_parameters

        parameters_file = f'{save_folder}/parameters.csv'

        if not os.path.exists(parameters_file):
            return False

        with open(parameters_file, 'r') as f:
            # Read every line and compare the values
            for line in f:
                parts = line.strip().split(',')
                key   = parts[0]
                value = parts[1:]

                # Get the correct value
                if key in process_parameters:
                    par_value = process_parameters[key]
                elif hasattr(self, key):
                    par_value = getattr(self, key)
                else:
                    raise ValueError(f'Unknown key {key}')

                # Get the type of the value
                par_is_array = isinstance(par_value, (list, tuple, np.ndarray))
                par_type     = (
                    type(par_value)
                    if not par_is_array
                    else type(par_value[0])
                )

                # Handle parentheses
                if par_is_array:
                    value = [ val.replace('[', '').replace(']', '') for val in value ]

                # Convert the values to the correct type
                if not par_is_array:
                    value = par_type(value[0])
                else:
                    value = np.array([par_type(val) for val in value])

                # Compare the values
                if not par_is_array and ( par_value != value ):
                    return False

                if par_is_array and not np.allclose(par_value, value):
                    return False

        return True

    def _load_saved_histogram(self):
        ''' Load the histogram data if it exists '''


        if self.skip_histogram:
            print(f'Skipping histogram loading ...')
            return True

        save_folder = self.save_folder

        #####################
        # LOAD THE HISTOGRAM
        #####################

        histogram_file = f'{save_folder}/histogram.csv'

        if not os.path.exists(histogram_file):
            return False

        histogram = pd.read_csv(histogram_file, index_col=0).astype(int)
        vals_1    = np.array(histogram.columns, dtype=float)
        vals_2    = np.array(histogram.index, dtype=float)

        if len(vals_1) == self.x_bins and len(vals_2) == self.phi_bins:
            x_vals    = vals_1
            phi_vals  = vals_2
            hist_vals = histogram.values.T
        elif len(vals_1) == self.phi_bins and len(vals_2) == self.x_bins:
            x_vals    = vals_2
            phi_vals  = vals_1
            hist_vals = histogram.values
        else:
            print(f'ERROR: Histogram dimensions do not match expected values.')
            return False

        # Check bins values
        if not (
            np.allclose(  x_vals,   self.x_vals) and
            np.allclose(phi_vals, self.phi_vals)
        ):
            print(f'ERROR: Histogram values do not match expected values.')
            return False

        # Build histogram analysis object
        self.hist_analysis = histogram_fitting.HistogramPlotFitting(
            histogram = hist_vals,
            x_vals    = x_vals,
            phi_vals  = phi_vals,
            verbose   = True,
        )
        self.fit_results  = self.hist_analysis.fit_results
        self.corr_results = self.hist_analysis.corr_results

        # Assign histogram data
        self.histogram   = hist_vals
        self.loaded_hist = True

        return True

    def _load_saved_performance_data(self):
        ''' Load the performance data if it exists '''

        save_folder    = self.save_folder
        save_file_root = self.save_file_root

        # Load the performance data
        performance_file = f'{save_folder}/performance.csv'

        if not os.path.exists(performance_file):
            return False

        performance_df = pd.read_csv(performance_file)

        self.performance_data = {
            key : performance_df[key].values
            for key in performance_df.columns
        }

        return True

    def load_saved_data(self):
        ''' Load the collected data if it exists '''

        print(f'Searching existing save for {self.save_file_root} ... ')
        fail_str = (
            'FAILED \n'
            'Loading the data from scratch ...'
        )

        # Load the parameters
        if not self._load_saved_parameters():
            print(f'Matching parameters not found ... {fail_str}')
            return False


        # Load the performance data
        if not self._load_saved_performance_data():
            print(f'Matching performance data not found ... {fail_str}')
            return False

        # Load the histogram
        if not self._load_saved_histogram():
            print(f'Matching histogram not found ... {fail_str}')
            return False

        # Print message
        print(f'Found existing save for {self.save_file_root} ... LOADED')

        return True

    # NEW DATA

    def _collect_time_and_metrics_data(
        self,
        folder_name   : str,
        process_folder: str,
    ):
        ''' Collect the time data from the first simulation '''

        time_data, performance_data = load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
            target_quantities   = ['times'],
        )

        # Metrics data
        self.performance_data = {
            key : []
            for key, value in performance_data.items()
            if (
                not isinstance(value[0], np.ndarray) and
                not value[0] == np.nan
            )
        }

        # Time data
        self.times        = np.array( time_data['times'] )
        self.time_step    = self.times[1] - self.times[0]
        self.duration     = self.times[-1] - self.times[0]
        self.n_iterations = len(self.times)

        self.discard_time  = 5.0
        self.discard_ratio = self.discard_time / self.times[-1]
        self.n_iter_cons   = round(self.n_iterations * (1 - self.discard_ratio))

        return

    def _update_histogram_counts(
        self,
        x_values  : np.ndarray,
        phi_values: np.ndarray,
    ):
        """ Update the histogram with new x_values and phi_values. """

        # Compute the histogram for the new data
        hist, x_edges, phi_edges = np.histogram2d(
            x     = x_values,
            y     = phi_values,
            bins  = [ self.x_bins,  self.phi_bins],
            range = [self.x_range, self.phi_range],
        )

        # Add the new histogram counts to the existing one
        self.histogram += hist

        # Checks
        assert np.allclose(self.x_edges, x_edges), \
            f'Error: x_edges {self.x_edges} != {x_edges}'
        assert np.allclose(self.phi_edges, phi_edges), \
            f'Error: phi_edges {self.phi_edges} != {phi_edges}'

        return

    def _get_stacked_com_pos(
        self,
        data_df: pd.DataFrame,
        spawn_x: float,
        n_iter : int,
    ) -> np.ndarray:
        ''' Get stacked com positions for histogram '''

        # Positions evolution
        if self.use_actual_com_pos:
            com_pos_diff = data_df.com_positions_diff[-n_iter:] - COM_POS_REL
        else:
            com_pos_diff = spawn_x * np.ones(n_iter)

        # Stack
        com_pos_stack = np.concatenate( [ com_pos_diff ] * 5 )

        return com_pos_stack

    def _get_stacked_phases(
        self,
        data_df: pd.DataFrame,
        n_iter : int,
    ):
        ''' Get stacked phases for histogram '''

        # Phases evolution
        if self.use_actual_phases:
            phases_diff = data_df.joint_phases_diff[-n_iter:]
        else:
            phases_diff = data_df.joint_phases_diff[-n_iter:] + PHASE_OFFSET

        phases_diff = histogram_fitting.wrap_phase_to_pi_pi(phases_diff)

        # Stack
        phases_stack = np.concatenate(
            [
                phases_diff - 2*np.pi,   # 1. Mostly < range
                phases_diff,             # 2.
                phases_diff + 2*np.pi,   # 3.
                phases_diff + 4*np.pi,   # 4.
                phases_diff + 6*np.pi,   # 5. Mostly > range

            ]
        )
        return phases_stack

    def collect_single_simulation_data(
        self,
        folder_name   : str,
        process_folder: str,
        spawn_x       : float = None,
        excluded      : bool = False,
    ):
        ''' Collect data for a single simulation '''

        n_iter = self.n_iter_cons

        # Load data
        target_quantities = [
            'com_positions_diff',
            'joint_phases_diff',
            'coherence_pos_vel',
            'coherence_vel_vel',
            'coherence_acc_vel',
            'propulsion_fx',
            'propulsion_fy',
        ]

        data_df, performance_dict = load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
            target_quantities   = target_quantities,
        )

        ########################
        ### PERFORMANCE DATA ###
        ########################

        # Add performance data
        n_start = len( self.times[self.times < 5.0] )

        coherence_pos_vel = data_df['coherence_pos_vel'][n_start:].mean()
        coherence_vel_vel = data_df['coherence_vel_vel'][n_start:].mean()
        coherence_acc_vel = data_df['coherence_acc_vel'][n_start:].mean()
        propulsion_fx     = data_df['propulsion_fx'][n_start:].mean()
        propulsion_fy     = data_df['propulsion_fy'][n_start:].mean()

        position_diff    = (
            data_df['com_positions_diff'][n_start:] -
            data_df['com_positions_diff'][0]
        ).mean()

        performance_dict : dict = performance_dict | {
            'coherence_pos_vel' : np.array([coherence_pos_vel]),
            'coherence_vel_vel' : np.array([coherence_vel_vel]),
            'coherence_acc_vel' : np.array([coherence_acc_vel]),
            'propulsion_fx'     : np.array([propulsion_fx]),
            'propulsion_fy'     : np.array([propulsion_fy]),
            'position_diff'     : np.array([position_diff]),
        }

        # Store performance data
        for key in performance_dict.keys():
            if key not in self.performance_data:
                self.performance_data[key] = []
            self.performance_data[key].append(performance_dict[key][0])

        ########################
        ### HISTOGRAM DATA   ###
        ########################

        if excluded or self.skip_histogram:
            return

        # STACK POSITIONS
        com_positions_stack = self._get_stacked_com_pos(
            data_df = data_df,
            spawn_x = spawn_x,
            n_iter  = n_iter,
        )

        # STACK PHASES
        joints_phases_stack = self._get_stacked_phases(
            data_df = data_df,
            n_iter  = n_iter,
        )

        # Update histogram with new data
        self._update_histogram_counts(
            x_values   = com_positions_stack,
            phi_values = joints_phases_stack,
        )

        return

    def collect_all_simulation_data(self):
        """ Collect data from all simulations and update the histogram. """

        # List of all files
        folder_names_list = [
            self.folder_names[seed_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)

        ]

        process_folders_list = [
            self.process_folders[spawn_x_ind * self.n_spawn_y + spawn_y_ind][seed_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)
        ]

        spawn_x_vals_list = [
            self.spawn_x_vals[spawn_x_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)
        ]

        # Collect the time and metrics data
        self._collect_time_and_metrics_data(
            folder_name    = folder_names_list[0],
            process_folder = process_folders_list[0],
        )

        # Load the saved data if it exists
        if self.load_saved_data():
            return

        # Verify number of simulations
        n_sim = self.n_sim
        assert n_sim == len(process_folders_list) == len(spawn_x_vals_list), \
            f'Error: {n_sim} simulations found'

        # Collect data
        n_print = max(n_sim // 20, 1)
        print(f'Collecting data for {self.save_file_root}', end=' ... ', flush=True)

        for sim_ind in range(n_sim):

            if sim_ind % n_print == 0:
                print(f'{ 100 * sim_ind / n_sim :.1f} %', end=' ... ', flush=True)

            self.collect_single_simulation_data(
                folder_name    = folder_names_list[sim_ind],
                process_folder = process_folders_list[sim_ind],
                spawn_x        = spawn_x_vals_list[sim_ind],
                excluded       = self.excl_map_flat[sim_ind],
            )

        # Prune non-informative performance data
        prune_keys = []
        for key in self.performance_data.keys():
            if isinstance(self.performance_data[key][0], (str)):
                continue
            if isinstance(self.performance_data[key][0], (np.ndarray)):
                continue
            if (
                np.all(np.isnan(self.performance_data[key]))
            ):
                prune_keys.append(key)

        for key in prune_keys:
            del self.performance_data[key]

        # Build histogram analysis object
        if self.skip_histogram:
            print(f'Skipping histogram analysis ... ')
            return

        self.hist_analysis = histogram_fitting.HistogramPlotFitting(
            histogram = self.histogram,
            x_vals    = self.x_vals,
            phi_vals  = self.phi_vals,
            verbose   = True,
        )
        self.fit_results  = self.hist_analysis.fit_results
        self.corr_results = self.hist_analysis.corr_results

        print('DONE')
        return

    ###########################################################################
    # PLOTS ####################################################################
    ###########################################################################

    def _initialize_histogram(self):

        # Initialize the histogram counts
        self.x_bins  = len(self.spawn_x_vals)
        if self.x_bins >= 2:
            self.x_step  = self.spawn_x_vals[1] - self.spawn_x_vals[0]
        else:
            self.x_step  = 0.10
        self.x_range = (
            self.spawn_x_vals[0]  - self.x_step / 2,
            self.spawn_x_vals[-1] + self.x_step / 2
        )
        self.x_edges = np.arange(self.x_bins + 1) * self.x_step + self.x_range[0]
        self.x_vals  = self.x_edges[:-1] + self.x_step / 2

        # Phi bins
        phi_min = -np.pi
        phi_max = 5 * np.pi

        self.phi_bins  = 40
        self.phi_step  = (phi_max - phi_min) / ( self.phi_bins - 1 )
        self.phi_range = (
            phi_min - self.phi_step / 2,
            phi_max + self.phi_step / 2,
        )
        self.phi_edges = np.arange(self.phi_bins + 1) * self.phi_step + self.phi_range[0]
        self.phi_vals  = self.phi_edges[:-1] + self.phi_step / 2

        # Histogram
        self.histogram   = np.zeros((self.x_bins, self.phi_bins))
        self.loaded_hist = False

        return

    def plot_performance(self):
        ''' Plot the performance data '''

        y_range = (
            np.amin( self.spawn_y_vals ),
            np.amax( self.spawn_y_vals ),
        )
        vals_shape = ( self.n_spawn_x, self.n_spawn_y, self.n_seeds )

        target_quantities = [
            "mech_energy_total",
            "mech_energy_active",

            "mech_power_total",
            "mech_power_active",

            'mech_tail_beat_amp_bl',

            'coherence_pos_vel',
            'coherence_vel_vel',
            'coherence_acc_vel',

            'ex_sum_rate_mean',
            'mn_sum_rate_mean',
            'ps_sum_rate_mean',

            'ex_diff_rate_power',
            'mn_diff_rate_power',
            'ps_diff_rate_power',

            'ex_diff_rate_amp_mean',
            'mn_diff_rate_amp_mean',
            'ps_diff_rate_amp_mean',

        ]

        for target_quantity in target_quantities:
            fig = plt.figure(figsize=(8, 6))


            quantity_values = np.array(self.performance_data[target_quantity])
            quantity_values = quantity_values.reshape(vals_shape)

            # Excluded conditions
            quantity_values[self.excl_map_matr] = np.nan

            # Mean and std across seeds
            mean_values = np.nanmean(quantity_values, axis=2)
            std_values  = np.nanstd(quantity_values, axis=2)

            if len(self.spawn_y_vals) < 2:
                continue
            if len(self.spawn_x_vals) < 2:
                continue

            plt.contourf(
                self.spawn_x_vals,
                self.spawn_y_vals,
                mean_values.T,
                levels = 20,
                cmap   = 'viridis',
            )
            plt.colorbar(label=target_quantity)

            plt.xlim(self.x_range)
            plt.ylim(y_range)

            plt.title(f'{target_quantity} vs FB position')
            plt.xlabel('FB distance (bl)')
            plt.ylabel('FB height (bl)')
            plt.ylabel(target_quantity)

            self.figures_dict[target_quantity] = fig

        return

    def plot_histogram(
        self,
        max_density: float = 0.10,
        **kwargs
    ):
        """Plot the current histogram as a heatmap."""

        # Plot standard histogram
        fig, _ax = self.hist_analysis.plot_histogram(
            max_density = max_density,
        )
        self.figures_dict['histogram'] = fig

        # Plot non-uniform histogram
        fig, _ax = self.hist_analysis.plot_histogram_non_uniform_image(
            max_density = max_density,
        )
        self.figures_dict['histogram_non_uniform'] = fig

        # Plot 3D histogram
        fig, _ax = self.hist_analysis.plot_histogram_3d(
            max_density = max_density,
        )
        self.figures_dict['histogram_3d'] = fig

        # Plot polar histogram
        fig, _ax = self.hist_analysis.plot_histogram_polar(
            max_density = max_density,
        )
        self.figures_dict['histogram_polar'] = fig

        return

    def plots(
        self,
        performance : bool = True,
        histogram   : bool = True,
    ):
        ''' Plot the performance data and the histogram '''

        self.figures_dict : dict[str, plt.Figure] = {}

        if performance:
            self.plot_performance()

        if histogram and not self.skip_histogram:
            self.plot_histogram()

        return

    ###########################################################################
    # SAVE #####################################################################
    ###########################################################################

    def _load_fit_results(self):
        ''' Load histogram fitting results '''

        save_folder      = self.save_folder
        save_file_root   = self.save_file_root
        fit_results_file = f'{save_folder}/histogram_fit.json'

        if not os.path.exists(fit_results_file):
            return False

        with open(fit_results_file, 'r') as f:
            histogram_fit = json.load(f)

        # Assign results
        self.fit_results  = {
            'params' : [
                histogram_fit['fit_results']['slope'],
                histogram_fit['fit_results']['intercept'],
            ],
            'loss'   : histogram_fit['fit_results']['loss'],
            'r2'     : histogram_fit['fit_results']['r2'],
        }
        self.corr_results = histogram_fit['corr_results']

        return True

    def _save_fit_results(self):
        ''' Save histogram fitting results 00'''

        save_folder      = self.save_folder
        save_file_root   = self.save_file_root
        fit_results_file = f'{save_folder}/histogram_fit.json'

        fit_results = {
            'fit_results' : {
                'slope'    : self.fit_results['params'][0],
                'intercept': self.fit_results['params'][1],
                'loss'     : self.fit_results['loss'],
                'r2'       : self.fit_results['r2'],
            },
            'corr_results': {
                'rho_mean' : self.corr_results['rho_mean'],
                'pval_mean': self.corr_results['pval_mean'],
                'rho_dist' : self.corr_results['rho_dist'],
                'pval_dist': self.corr_results['pval_dist'],
            },
        }

        with open(fit_results_file, 'w') as f:
            json.dump(fit_results, f, indent=4)

        print(f'Saved histogram fit results to {fit_results_file}')
        return

    def save_all_simulation_data(self):
        ''' Save the collected data '''

        # Save the parameters as csv (without pandas)
        pars_file  = f'{self.save_folder}/parameters.csv'
        with open(pars_file, 'w') as f:
            for key, value in self.all_parameters.items():
                if isinstance(value, np.ndarray):
                    value = ','.join([str(val) for val in value])
                f.write(f'{key},{value}\n')
        print(f'Saved parameters to {pars_file}')

        # Save the performance data (convert to DataFrame and save as csv)
        performance_df = pd.DataFrame(self.performance_data)
        performance_df.to_csv(f'{self.save_folder}/performance.csv', index=False)

        # Save histogram data
        if self.skip_histogram:
            print(f'Skipping histogram data saving ... ')
            return

        # Save the histogram counts (convert to DataFrame and save as csv)
        x_edges   = ( self.x_edges + (self.x_edges[1] - self.x_edges[0]) / 2 )[:-1]
        phi_edges = ( self.phi_edges + (self.phi_edges[1] - self.phi_edges[0]) / 2 )[:-1]

        histogram_df = pd.DataFrame(
            self.histogram.T,
            columns = x_edges,
            index   = phi_edges,
        )

        histogram_file = f'{self.save_folder}/histogram.csv'
        histogram_df.to_csv(histogram_file)
        print(f'Saved histogram to {histogram_file}')

        # Save histogram fitting results
        self._save_fit_results()

        return

    def save_plots(self):
        """Save the current histogram as a heatmap."""

        # Create the save folder if it does not exist
        os.makedirs(self.save_folder, exist_ok=True)

        # Save the plots
        for name, figure in self.figures_dict.items():
            fig_path = f'{self.save_folder}/{name}'
            print(f'Saving {name} to {fig_path}.pdf')
            try:
                figure.savefig(f'{fig_path}.pdf', format='pdf')
            except Exception as e:
                print(f'Error saving {name} to {fig_path}.pdf: {e}')

        return


###############################################################################
# PLOT PARAMETERS COMBINATIONS ################################################
###############################################################################

def plot_parameters_combination(
    results_root_folder: str,
    stim_a_off         : float,
    vortex_mode        : int,
    feedback_mode      : int,
    leader_signal_str  : str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    use_actual_com_pos : bool = False,
    save               : bool = True,
    show               : bool = False,
    simulation_tag     : str = None,
    skip_histogram     : bool = False,
    excluded_condition : callable = None,
):
    ''' Plot the histogram for a given set of parameters '''

    # Initialize the incremental 2D histogram
    plotting = SchoolingPlotting(
        results_root_folder = results_root_folder,
        stim_a_off          = stim_a_off,
        vortex_mode         = vortex_mode,
        feedback_mode       = feedback_mode,
        leader_signal_str   = leader_signal_str,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
        simulation_tag      = simulation_tag,
        skip_histogram      = skip_histogram,
        excluded_condition  = excluded_condition,
    )

    # Collect all simulation data
    plotting.collect_all_simulation_data()

    # Save collected data
    plotting.save_all_simulation_data()

    # Plot
    plotting.plots(
        performance = True,
        histogram   = plotting.skip_histogram,
    )

    # Save
    if save:
        plotting.save_plots()

    # Show the plot
    if show:
        plt.show()

    plt.close('all')

    return

###############################################################################
# DIFFERENCE PLOTTING #########################################################
###############################################################################

def run_statistical_tests(
    q_values_0 : np.ndarray,
    q_values_1 : np.ndarray,
    q_target   : str,
    diff_type  : str,
):
    ''' Run paired statistical tests (t-test and Wilcoxon) '''

    if q_values_0.shape == q_values_1.shape:
        # Paired tests
        t_stat, p_val_t = ttest_rel(
            q_values_1,
            q_values_0,
            alternative='two-sided'
        )
        w_stat, p_val_w = wilcoxon(
            q_values_1,
            q_values_0,
            alternative='two-sided'
        )
    else:
        # Unpaired tests
        t_stat, p_val_t = ttest_ind(
            q_values_1,
            q_values_0,
            alternative='two-sided'
        )
        w_stat, p_val_w = mannwhitneyu(
            q_values_1,
            q_values_0,
            alternative='two-sided'
        )

    t_significance = p_val_t < 0.05
    w_significance = p_val_w < 0.05
    significance   = t_significance or w_significance

    print(
        f'{q_target} {diff_type.upper()} difference: \n'
        f't-statistic = {t_stat:.3f}, p-value (t-test)   = {p_val_t:.3e}, '
        f'significant = {t_significance}, \n'
        f'w-statistic = {w_stat:.3f}, p-value (Wilcoxon) = {p_val_w:.3e}, '
        f'significant = {w_significance}  \n'
    )

    return significance

def _plot_violinplot(
    performance_df_list: list[pd.DataFrame],
    modes_list         : list[str],
    target_quantity    : str,
    save_folder        : str,
):
    ''' Plot a paired boxplot for the given quantities '''

    assert len(performance_df_list) == len(modes_list)

    # -----------------------
    # Long dataframe for seaborn
    # -----------------------

    df_long = pd.concat(
        [
            pd.DataFrame({
                "Mode" : mode,
                "Value": perf_df[target_quantity].to_numpy()
            })
            for mode, perf_df in zip(modes_list, performance_df_list)
        ],
        ignore_index=True,
    )

    # -----------------------
    # Plot violins
    # -----------------------

    fig = plt.figure(figsize=(8, 6))

    sns.violinplot(
        data  = df_long,
        x     = 'Mode',
        y     = 'Value',
        inner = None,
        color = 'lightgray'
    )

    sns.boxplot(
        data     = df_long,
        x        = 'Mode',
        y        = 'Value',
        width    = 0.2,
        showcaps = True,
        boxprops = {'zorder': 2}
    )

    # Decorate
    plt.xticks(rotation=45)
    plt.title(f'Paired Comparison: {target_quantity}')
    plt.tight_layout()

    # Save the figure
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(
        f'{save_folder}/{target_quantity}_violinplot.png',
        format = 'png',
    )
    plt.close(fig)
    return

def _plot_difference_distribution(
    quantity_values_0 : np.ndarray,
    quantity_values_1 : np.ndarray,
    target_quantity    : str,
    save_folder        : str,
    relative_difference: bool = False,
):
    ''' Plot the distribution of paired differences (CL - OL) '''

    # Compute paired differences
    tag         = 'Absolute Difference'
    differences = quantity_values_1 - quantity_values_0
    if relative_difference:
        tag         = 'Relative Difference (%)'
        differences = differences / np.abs(quantity_values_0) * 100.0

    df = pd.DataFrame({tag: differences})

    # Plot
    fig = plt.figure(f'Distribution of Differences: {target_quantity}', figsize=(8, 6))

    # Violin and boxplot of the differences
    sns.violinplot(y=tag, data=df, inner=None, color='lightgray')
    sns.boxplot(y=tag, data=df, width=0.2, boxprops={'zorder': 2})

    # Add a zero-line for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.title(f'Distribution of Paired Differences\n({target_quantity})')
    plt.tight_layout()

    # Save the figure
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(
        f'{save_folder}/{target_quantity}_difference_distribution.png',
        format='png',
    )
    plt.close(fig)
    return

def _plot_contourf(
    spawn_x_vals   : np.ndarray,
    spawn_y_vals   : np.ndarray,
    diff_values    : np.ndarray,
    target_quantity: str,
    diff_type      : str,
    save_folder    : str,
    x_range        : tuple[float, float],
    y_range        : tuple[float, float],
):

    if len(spawn_y_vals) < 2:
        print('Not enough spawn_y_vals for contourf plot. Skipping.')
        return
    if len(spawn_x_vals) < 2:
        print('Not enough spawn_x_vals for contourf plot. Skipping.')
        return

    fig = plt.figure(f'DIFFERENCE Contourf: {target_quantity}', figsize=(8, 6))

    # Mean across seeds
    valid_counts = np.sum(~np.isnan(diff_values), axis=2)

    mean_diff_values = np.divide(
        np.nansum(diff_values, axis=2),
        valid_counts,
        where=valid_counts > 0
    )
    mean_diff_values[valid_counts == 0] = np.nan

    # Standard deviation across seeds
    std_diff_values = np.sqrt(
        np.divide(
            np.nansum((diff_values - mean_diff_values[..., None])**2, axis=2),
            valid_counts,
            where=valid_counts > 0
        )
    )
    std_diff_values[valid_counts <= 1] = np.nan

    # Max absolute difference
    max_abs_diff_value = np.nanmax(np.abs(mean_diff_values))

    # Plot contourf
    levels  = np.linspace(-max_abs_diff_value, max_abs_diff_value, 21)
    norm    = Normalize(vmin=-max_abs_diff_value, vmax=+max_abs_diff_value)
    contour = plt.contourf(
        spawn_x_vals,
        spawn_y_vals,
        mean_diff_values.T,
        levels = levels,
        cmap   = 'viridis',
        norm   = norm,
    )

    # Decorate
    plt.colorbar(contour, label=f'{diff_type.upper()} Difference in {target_quantity}')

    plt.xlim(x_range)
    plt.ylim(y_range)

    plt.title(f'{target_quantity} {diff_type.upper()} Difference vs FB position')
    plt.xlabel('FB distance (bl)')
    plt.ylabel('FB height (bl)')

    # Save the figure
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(
        f'{save_folder}/{target_quantity}_contourf.png',
        format = 'png',
    )
    plt.close(fig)
    return

def save_combined_performance(
    file_paths         : list[str],
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    seed_vals          : np.ndarray,
    save_folder        : str,
    excluded_condition : callable = None,
):
    ''' Save the performance comparison data in an xlsx file '''

    exclude = excluded_condition
    if exclude is None:
        exclude = lambda sx, sy, sd: False

    perf_df_list = [
        pd.read_csv(f_path)
        for f_path in file_paths
    ]
    n_files = len(file_paths)

    # Add metadata columns
    meta_list = [
        [
            {
                'spawn_x'      : x,
                'spawn_y'      : y,
                'np_seed'      : s,
                'vortex_mode'  : ('pure_flow' not in f_path),
                'feedback_mode': ('ps_0_'     not in f_path),
            }
            for x in ( spawn_x_vals if 'pure_flow' not in f_path else [0.0] )
            for y in ( spawn_y_vals if 'pure_flow' not in f_path else [0.0] )
            for s in seed_vals
        ]
        for f_path in file_paths
    ]

    modes_list = [
        ( 'CL'    if meta[0]['feedback_mode'] else  'OL' ) +
        ( '_Vort' if meta[0]['vortex_mode']   else '_Flow' )
        for meta in meta_list
    ]

    # Convert metadata to DataFrames
    meta_df_list = [pd.DataFrame(meta) for meta in meta_list]

    # Checks
    for perf_df, meta_df in zip(perf_df_list, meta_df_list):
        assert len(perf_df) == len(meta_df)

    # Drop rows with excluded conditions
    for i, meta_df in enumerate(meta_df_list):
        excl_mask = meta_df.apply(
            lambda r: exclude(r['spawn_x'], r['spawn_y'], r['np_seed']),
            axis=1
        )
        meta_df_list[i] = meta_df[~excl_mask]

    # Add metadata columns
    for i, (perf_df, meta_df) in enumerate(zip(perf_df_list, meta_df_list)):
        perf_df_list[i] = pd.concat(
            [
                meta_df.reset_index(drop=True),
                perf_df.reset_index(drop=True),
            ],
            axis=1
        )

    # Only keep target columns
    target_meta_columns = [
        'spawn_x',
        'spawn_y',
        'np_seed',
        'vortex_mode',
        'feedback_mode',
    ]
    target_perf_columns = [
        'mech_power_total',
        'mech_torque_total',
        'mech_tail_beat_amp_bl',
        'coherence_vel_vel',

        'ex_diff_rate_power',
        'mn_diff_rate_power',
        'ps_diff_rate_power',

        'ex_diff_rate_amp_mean',
        'mn_diff_rate_amp_mean',
        'ps_diff_rate_amp_mean',
    ]

    # Add summed columns
    for perf_ind, perf_df in enumerate(perf_df_list):
        for ner in ['ex', 'mn']:
            for key in [
                'mean_rate',
                'power_rate',
                'diff_rate_power',
                'sum_rate_power'
            ]:

                col_name = f'{ner}_{key}_head'
                sum_head = perf_df[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(8)]
                ].sum(axis=1)
                perf_df[col_name] = sum_head
                target_perf_columns.append(col_name)

                col_name = f'{ner}_{key}_trunk'
                sum_trunk = perf_df[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(8, 16)]
                ].sum(axis=1)
                perf_df[col_name] = sum_trunk
                target_perf_columns.append(col_name)

                col_name = f'{ner}_{key}_caud'
                sum_caud = perf_df[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(16, 24)]
                ].sum(axis=1)
                perf_df[col_name] = sum_caud
                target_perf_columns.append(col_name)

                col_name = f'{ner}_{key}_tail'
                sum_tail = perf_df[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(24, 32)]
                ].sum(axis=1)
                perf_df[col_name] = sum_tail
                target_perf_columns.append(col_name)

                perf_df_list[perf_ind] = perf_df

    # Combine into one overall dataframe
    perf_data_all = pd.concat(
        perf_df_list,
        ignore_index = True
    )

    # Select target columns
    target_columns = target_meta_columns + target_perf_columns
    perf_data_all  = perf_data_all[target_columns]

    # Save as xlsx
    os.makedirs(save_folder, exist_ok=True)

    perf_data_all.to_excel(
        f'{save_folder}/Performance_comparison.xlsx',
        index = False
    )

    # Plot violinplots
    for q_target in target_perf_columns:

        _plot_violinplot(
            performance_df_list = perf_df_list,
            modes_list          = modes_list,
            target_quantity     = q_target,
            save_folder         = save_folder,
        )

    return perf_data_all

def plot_performance_difference(
    spawn_x_vals      : np.ndarray,
    spawn_y_vals      : np.ndarray,
    seed_vals         : np.ndarray,
    file_path_0       : str,
    file_path_1       : str,
    mode_0            : str,
    mode_1            : str,
    save_folder       : str,
    target_quantities : list[tuple[str, str]] = None,
    excluded_condition: callable = None,
):
    ''' Plot the performance data '''

    if excluded_condition is None:
        excluded_condition = lambda sx, sy, sd: False

    # Parameters
    n_spawn_x = len(spawn_x_vals)
    n_spawn_y = len(spawn_y_vals)
    n_seeds   = len(seed_vals)
    q_shape   = ( n_spawn_x, n_spawn_y, n_seeds )
    x_range   = (0.4, 1.6)
    y_range   = (
        np.amin( spawn_y_vals ),
        np.amax( spawn_y_vals ),
    )

    q_shape = (n_spawn_x, n_spawn_y, n_seeds)
    q_size  = n_spawn_x * n_spawn_y * n_seeds

    # Excluded conditions mask
    excl_mask = np.array(
        [
            excluded_condition(spawn_x, spawn_y, seed)
            for spawn_x in spawn_x_vals
            for spawn_y in spawn_y_vals
            for seed in seed_vals
        ]
    )

    # Load performance data
    # NOTE: f'{self.save_folder}/performance.csv'

    performance_data_0 = pd.read_csv(file_path_0)
    performance_data_1 = pd.read_csv(file_path_1)

    # Add summed columns
    for perf_data in [performance_data_0, performance_data_1]:
        for ner in ['ex', 'mn']:
            for key in [
                'mean_rate',
                'power_rate',
                'diff_rate_power',
                'sum_rate_power'
            ]:

                col_name = f'{ner}_{key}_head'
                sum_head = perf_data[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(8)]
                ].sum(axis=1)
                perf_data[col_name] = sum_head
                target_quantities.append( (col_name, 'rel') )

                col_name = f'{ner}_{key}_trunk'
                sum_trunk = perf_data[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(8, 16)]
                ].sum(axis=1)
                perf_data[col_name] = sum_trunk
                target_quantities.append( (col_name, 'rel') )

                col_name = f'{ner}_{key}_caud'
                sum_caud = perf_data[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(16, 24)]
                ].sum(axis=1)
                perf_data[col_name] = sum_caud
                target_quantities.append( (col_name, 'rel') )

                col_name = f'{ner}_{key}_tail'
                sum_tail = perf_data[
                    [f"{ner}_seg_{seg}_{key}" for seg in range(24, 32)]
                ].sum(axis=1)
                perf_data[col_name] = sum_tail
                target_quantities.append( (col_name, 'rel') )

    # Plot the performance difference
    performance_df_list = [
        performance_data_0,
        performance_data_1,
    ]

    for q_target, diff_type in target_quantities:

        # Paired boxplot
        _plot_violinplot(
            performance_df_list = performance_df_list,
            modes_list          = [mode_0, mode_1],
            target_quantity     = q_target,
            save_folder         = save_folder,
        )

        # Get values
        q_values_0 = np.array(performance_data_0[q_target])
        q_values_1 = np.array(performance_data_1[q_target])

        # Run statistical tests
        significance = run_statistical_tests(
            q_values_0 = q_values_0,
            q_values_1 = q_values_1,
            q_target   = q_target,
            diff_type  = diff_type,
        )

        if not significance:
            print(f'{q_target} has no significant difference. Skipping ...')
            continue

        # Reshape to (n_spawn_x, n_spawn_y, n_seeds)
        if q_values_0.size != q_size:
            q_values_0 = q_values_0.reshape(1, 1, n_seeds)
            q_values_0 = np.broadcast_to(q_values_0, q_shape).flatten()

        if q_values_1.size != q_size:
            q_values_1 = q_values_1.reshape(1, 1, n_seeds)
            q_values_1 = np.broadcast_to(q_values_1, q_shape).flatten()

        # Apply exclusion mask
        q_values_0[excl_mask] = np.nan
        q_values_1[excl_mask] = np.nan

        # Compute the difference
        diff = ( q_values_1 - q_values_0 )

        if np.allclose(diff, 0, equal_nan=True, atol=1e-12):
            print(f'{q_target} has no difference. Skipping ...')
            continue

        if diff_type == 'abs':
            diff_values_flat = diff

        if diff_type == 'rel':
            diff_values_flat = diff / np.abs(q_values_0) * 100.0

        diff_values = diff_values_flat.reshape(q_shape)

        # Difference distribution
        _plot_difference_distribution(
            quantity_values_0   = q_values_0[~excl_mask],
            quantity_values_1   = q_values_1[~excl_mask],
            target_quantity     = q_target,
            save_folder         = save_folder,
            relative_difference = (diff_type == 'rel'),
        )

        # Plot contourf
        _plot_contourf(
            spawn_x_vals    = spawn_x_vals,
            spawn_y_vals    = spawn_y_vals,
            diff_values     = diff_values,
            target_quantity = q_target,
            diff_type       = diff_type,
            save_folder     = save_folder,
            x_range         = x_range,
            y_range         = y_range,
        )

        plt.close('all')

    return

###############################################################################
# POST PROCESSING #############################################################
###############################################################################

def run_analyses_processing(
    results_root_folder   : str,
    stim_a_off            : float,
    feedback_mode_vals    : list[int],
    vortex_mode_vals      : list[int],
    leader_signal_str     : str,
    spawn_x_vals          : np.ndarray,
    spawn_y_vals          : np.ndarray,
    np_random_seed_vals   : np.ndarray,
    use_actual_com_pos    : bool,
    save                  : bool = True,
    show                  : bool = False,
    simulation_tag        : str = None,
    skip_histogram        : bool = False,
    excluded_condition    : callable = None,
):
    ''' Run the sequential processing '''

    for vortex_mode in vortex_mode_vals:

        x_vals = (
            spawn_x_vals
            if vortex_mode
            else [0.0]
        )
        y_vals = (
            spawn_y_vals
            if vortex_mode
            else [0.0]
        )


        for feedback_mode in feedback_mode_vals:
            plot_parameters_combination(
                results_root_folder = results_root_folder,
                stim_a_off          = stim_a_off,
                feedback_mode       = feedback_mode,
                vortex_mode         = vortex_mode,
                leader_signal_str   = leader_signal_str,
                spawn_x_vals        = x_vals,
                spawn_y_vals        = y_vals,
                np_random_seed_vals = np_random_seed_vals,
                use_actual_com_pos  = use_actual_com_pos,
                save                = save,
                show                = show,
                simulation_tag      = simulation_tag,
                skip_histogram      = skip_histogram,
                excluded_condition  = excluded_condition,
            )

    return

def _get_sim_tag(
    muscle_fn : float = 5500.0,
    ps_weight : float = 15.0,
):
    ''' Get the simulation tag based on the parameters '''

    muscle_str = f'FN_{muscle_fn:.0f}'
    ps_str     = f'ps_{ps_weight:.0f}'

    sim_tag = f'{muscle_str}_{ps_str}'

    return sim_tag

def _get_results_root_folder(
    muscle_fn  : float = 5500.0,
    ps_weight  : float = 15.0,
    speed      : str = 'fast',
    dst_tag    : str = 'active_body',
):
    ''' Get the root folder for the results based on the parameters '''

    sim_tag = _get_sim_tag(muscle_fn=muscle_fn, ps_weight=ps_weight)

    # Root folder
    results_root_folder = (
        f'{ANALYSIS_PATH}/'
        'comparing_vortices/'
        f'schooling_experimental_xy_grid_free_head_{speed}_speed_'
        f'{sim_tag}_'
        f'{dst_tag}'
    )

    return results_root_folder

def _get_performance_file_path(
    folder_root: str,
    muscle_fn  : float,
    ps_weight  : float,
    speed      : str,
    dst_tag    : str,
    vx_mode    : str,
):

    results_root_folder = _get_results_root_folder(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight,
        speed     = speed,
        dst_tag   = dst_tag,
    )
    file_root = f'{results_root_folder}/figures/{folder_root}'
    file_tag  = f'closed_loop_{vx_mode}'
    file_path = f'{file_root}_{file_tag}/performance.csv'

    return file_path

def run_difference_analysis(
    ps_weight_0        : float,
    vx_mode_0          : str,
    ps_weight_1        : float,
    vx_mode_1          : str,
    dst_tag            : str,
    folder_root        : str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    exclude_cond       : callable = None,
    muscle_fn          : float = 5500.0,
    speed              : str = 'fast',
):
    ''' Run the difference analysis between two conditions '''

    save_folder_root = (
        f'{ANALYSIS_PATH}/'
        'comparing_vortices/comparisons/'
        f'{dst_tag}/'
    )

    target_quantities = [
        ("mech_torque_total", "rel"),
        ("mech_power_total",  "rel"),
        ("mech_energy_total", "rel"),

        ("mech_torque_active", "rel"),
        ("mech_power_active",  "rel"),
        ("mech_energy_active", "rel"),


        (         'mech_freq_ax', "abs"),
        ('mech_tail_beat_amp_bl', "rel"),


        ("ps_sum_rate_mean",  "rel"),
        ("ex_sum_rate_mean",  "rel"),
        ("mn_sum_rate_mean",  "rel"),

        ("ps_diff_rate_power",  "rel"),
        ("ex_diff_rate_power",  "rel"),
        ("mn_diff_rate_power",  "rel"),

        ("ps_diff_rate_amp_mean",  "rel"),
        ("ex_diff_rate_amp_mean",  "rel"),
        ("mn_diff_rate_amp_mean",  "rel"),

        ( "coherence_pos_vel",  "abs"),
        ( "coherence_vel_vel",  "abs"),
        ( "coherence_acc_vel",  "abs"),
        (     "position_diff",  "abs"),
    ]

    ####################
    # REFERENCE
    ####################


    results_root_folder_0 = _get_results_root_folder(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight_0,
        speed     = speed,
        dst_tag   = dst_tag,
    )
    sim_tag_0   = _get_sim_tag(muscle_fn=muscle_fn, ps_weight=ps_weight_0)
    file_root_0 = f'{results_root_folder_0}/figures/{folder_root}'
    file_tag_0  = f'closed_loop_{vx_mode_0}'

    ####################
    # COMPARISON
    ####################

    results_root_folder_1 = _get_results_root_folder(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight_1,
        speed     = speed,
        dst_tag   = dst_tag,
    )
    sim_tag_1   = _get_sim_tag(muscle_fn=muscle_fn, ps_weight=ps_weight_1)
    file_root_1 = f'{results_root_folder_1}/figures/{folder_root}'
    file_tag_1  = f'closed_loop_{vx_mode_1}'

    # Save combined performance data
    file_path_0 = f'{file_root_0}_{file_tag_0}/performance.csv'
    file_path_1 = f'{file_root_1}_{file_tag_1}/performance.csv'

    ####################
    # PLOT DIFFERENCE
    ####################

    # Mode tags
    mode_0 = f'{file_tag_0}_{sim_tag_0}'
    mode_1 = f'{file_tag_1}_{sim_tag_1}'

    if 'ps_0' in mode_0:
        mode_0 = mode_0.replace('ps_0', '')
        mode_0 = mode_0.replace('closed_loop_', 'open_loop_')
    if 'ps_0' in mode_1:
        mode_1 = mode_1.replace('ps_0', '')
        mode_1 = mode_1.replace('closed_loop_', 'open_loop_')

    muscle_fn_str = f'FN_{muscle_fn:.0f}_'
    mode_0        = mode_0.replace(muscle_fn_str, '')
    mode_1        = mode_1.replace(muscle_fn_str, '')

    mode_0 = mode_0.replace('__', '_')
    mode_1 = mode_1.replace('__', '_')

    mode_0 = mode_0.strip('_')
    mode_1 = mode_1.strip('_')

    # Save folder
    save_folder = (
        f'{save_folder_root}/'
        f'{mode_0}_VS_{mode_1}'
    )

    plot_performance_difference(
        spawn_x_vals = spawn_x_vals,
        spawn_y_vals = spawn_y_vals,
        seed_vals    = np_random_seed_vals,
        file_path_0  = file_path_0,
        file_path_1  = file_path_1,
        mode_0       = mode_0,
        mode_1       = mode_1,
        save_folder  = save_folder,

        target_quantities  = target_quantities,
        excluded_condition = exclude_cond,
    )
    return

def _repeat_post_processing(ps_weight: float):
    ''' Process the results of the simulations for a given PS weight '''

    # Studied parameters
    # 'intermediate', 'empirical', 'theoretical', 'fast'

    muscle_fn = 5500.0

    # Set to True to only recompute the performance metrics
    skip_histogram = False

    speed = 'fast'
    amp   = 1.00

    leader_signal_str = get_water_dynamics_file_name(
        amp_scaling        = amp,            # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,           # [0.23,  0.25 ]
        integration_method = 'abdquickest',  # ['abdquickest',  'implicit' ]
        speed_type         = speed,          # ['empirical',  'theoretical', 'intermediate'] ]
        amp_type           = 'constant',     # [None, 'constant', 'modulated']
        step_bounds        = [12104, 12264], # [ [12104, 12264], 11300_12400 ]
        freq_bounds        = (None, None),   # [None, (2.5, 4.5)]
        matching_string    = 'sketch_body',  # ['sketch_body', None]
    )

    spawn_x_vals = np.linspace(0.4, 1.6, 41)
    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)

    dst_tag             = 'active_body'
    stim_a_off          = 0.0
    vortex_mode_vals    = [0, 1]
    feedback_mode_vals  = [1]
    np_random_seed_vals = [101, 102, 103, 104, 105]

    # Processing
    use_actual_com_pos = True
    save               = True
    show               = False

    # Simulation tag
    sim_tag = _get_sim_tag(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight
    )
    simulation_tag = (
        f'dynamic_water_vortices_free_head_muscle_{sim_tag}'
    )

    # Root folder
    results_root_folder = _get_results_root_folder(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight,
        speed     = speed,
        dst_tag   = dst_tag,
    )

    # RUN ANALYSIS
    run_analyses_processing(
        results_root_folder = results_root_folder,
        stim_a_off          = stim_a_off,
        vortex_mode_vals    = vortex_mode_vals,
        feedback_mode_vals  = feedback_mode_vals,
        leader_signal_str   = leader_signal_str,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
        save                = save,
        show                = show,
        simulation_tag      = simulation_tag,
        skip_histogram      = skip_histogram,
        excluded_condition  = None,
    )

    return

def _repeat_difference_analysis():
    ''' Compare closed and open loop, with and without vortices '''

    # Studied parameters
    # 'intermediate', 'empirical', 'theoretical', 'fast'

    muscle_fn = 5500.0
    speed = 'fast'
    amp   = 1.00

    leader_signal_str = get_water_dynamics_file_name(
        amp_scaling        = amp,            # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,           # [0.23,  0.25 ]
        integration_method = 'abdquickest',  # ['abdquickest',  'implicit' ]
        speed_type         = speed,          # ['empirical',  'theoretical', 'intermediate'] ]
        amp_type           = 'constant',     # [None, 'constant', 'modulated']
        step_bounds        = [12104, 12264], # [ [12104, 12264], 11300_12400 ]
        freq_bounds        = (None, None),   # [None, (2.5, 4.5)]
        matching_string    = 'sketch_body',  # ['sketch_body', None]
    )

    spawn_x_vals        = np.linspace(0.4, 1.6, 41)
    spawn_y_vals        = np.arange(-0.20, 0.25, 0.05)
    np_random_seed_vals = [101, 102, 103, 104, 105]

    dst_tag = 'active_body'

    # Root folder
    folder_root = _shorten_tag(leader_signal_str).strip('_')

    # Exclusion condition
    exclude_nothing_cond = lambda sx, sy, sd: False

    exclude_seed_val_cond = lambda sx, sy, sd: (
        ( sd in [100] )               # Exclude specific seed values
    )

    exclude_too_1ose_cond = lambda sx, sy, sd: (
        abs(sx) < 0.80 + 0.01         # Too close longitudinally
    )

    exclude_not_aligned_cond = lambda sx, sy, sd: (
        abs(sy) > 0.00 + 0.01         # Not aligned longitudinally
    )

    exclude_far_lateral_cond = lambda sx, sy, sd: (
        abs(sy) > 0.15 + 0.01         # Too far laterally
    )

    exclude_superposition_cond = lambda sx, sy, sd: (
        (abs(sx) < 0.80 + 0.01 )  or    # Too close longitudinally
        (
        (abs(sx) < 1.00 - 0.01)  and    # Too close longitudinally
        (abs(sy) < 0.15 - 0.01)         # Too close laterally
        )
    )

    exclude_cond = exclude_seed_val_cond


    ####################
    # SAVE PERFORMANCE
    ####################

    save_folder_root = (
        f'{ANALYSIS_PATH}/'
        'comparing_vortices/comparisons/'
        f'{dst_tag}/'
    )

    file_paths = [
        _get_performance_file_path(
            folder_root = folder_root,
            muscle_fn   = muscle_fn,
            speed       = speed,
            dst_tag     = dst_tag,

            ps_weight   = ps_weight,
            vx_mode     = vx_mode,
        )
        for ps_weight, vx_mode in [
            ( 0.0, 'pure_flow'     ),
            (20.0, 'pure_flow'     ),
            ( 0.0, 'with_vortices' ),
            (20.0, 'with_vortices' ),
        ]
    ]

    save_combined_performance(
        file_paths         = file_paths,
        spawn_x_vals       = spawn_x_vals,
        spawn_y_vals       = spawn_y_vals,
        seed_vals          = np_random_seed_vals,
        save_folder        = f'{save_folder_root}/combined',
        excluded_condition = exclude_cond,
    )

    ####################
    # DIFFERENCE ANALYSIS
    ####################

    combinations = [
        [
            (20.0, 'pure_flow'     ),
            (20.0, 'with_vortices' ),
        ],
        [
            ( 0.0, 'with_vortices' ),
            (20.0, 'with_vortices' ),
        ],
        [
            ( 0.0, 'pure_flow'     ),
            ( 0.0, 'with_vortices' ),
        ],
        [
            ( 0.0, 'pure_flow'     ),
            (20.0, 'pure_flow'     ),
        ],

    ]

    for (ps_weight_0, vx_mode_0), (ps_weight_1, vx_mode_1) in combinations:
        run_difference_analysis(
            ps_weight_0         = ps_weight_0,
            vx_mode_0           = vx_mode_0,
            ps_weight_1         = ps_weight_1,
            vx_mode_1           = vx_mode_1,
            dst_tag             = dst_tag,
            folder_root         = folder_root,
            spawn_x_vals        = spawn_x_vals,
            spawn_y_vals        = spawn_y_vals,
            np_random_seed_vals = np_random_seed_vals,
            exclude_cond        = exclude_cond,
            muscle_fn           = muscle_fn,
            speed               = speed,
        )

    return

def main():
    ''' Run the post-processing and the difference analysis '''

    repeat_post_processing     = True
    repeat_difference_analysis = True

    # Closed loop (PS weight 20) and open loop (PS weight 0)
    if repeat_post_processing:
        for ps_weight in [20.0, 0.0]:
            _repeat_post_processing(ps_weight)

    if repeat_difference_analysis:
        _repeat_difference_analysis()


if __name__ == '__main__':
    main()
