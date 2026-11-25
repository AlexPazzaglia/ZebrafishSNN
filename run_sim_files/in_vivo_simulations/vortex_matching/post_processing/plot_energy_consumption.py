'''
Post-processing of the energy consumption simulations (Fig. 4L-M).

The script loads the results collected by move_results_energy_consumption.py
(50 networks, follower 1.0 BL behind the leader), computes the mechanical
power consumption (positive part of the joint power) and the phase relation
between leader and follower, and compares the simulations with and without
vortices, in closed loop (PS weight 20) and open loop (PS weight 0).
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
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import (
    ttest_rel, wilcoxon,        # Paired tests
    ttest_ind, mannwhitneyu,    # Independent tests
)

from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import circmean

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

def _get_signal_phase(
    signal : np.ndarray,
):

    # Convert joint angles to phases
    signal_hilb  = hilbert(signal)
    signal_phase = np.unwrap(np.angle(signal_hilb))

    # Find crossings 2pi
    phases_wrap     = signal_phase % (2 * np.pi)
    phases_diff     = np.diff(phases_wrap)
    cycle_crossings = np.where(phases_diff < -np.pi)[0]
    cycle_crossings = cycle_crossings.astype(int)

    # Linearize phases
    for i in range(len(cycle_crossings) - 1):

        # Get start and end indices
        idx0 = cycle_crossings[i]
        idx1 = cycle_crossings[i + 1]

        # Get values at crossings
        v0 = signal_phase[idx0]
        v1 = signal_phase[idx1]

        # Linearize between crossings
        signal_phase[idx0:idx1 + 1] = (
            v0 + (v1 - v0) * np.linspace(0, 1, idx1 - idx0 + 1)
        )

    return signal_phase

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

###############################
# POWER PLOTTING ###
###############################

def _filter_joint_position_for_phase(
    signal : np.ndarray,
    dt     : float,
    f_low  : float = 0.2,
    f_high : float = 5.0,
    order  : int   = 2,
):
    """
    Gently band-pass filter joint position before Hilbert phase estimation.

    The goal is not to change the mechanics, only to obtain a cleaner phase
    signal for cycle segmentation.
    """

    signal = np.asarray(signal)

    # Remove slow offset/drift before filtering
    signal_centered = signal - np.mean(signal)

    fs   = 1.0 / dt
    fnyq = 0.5 * fs

    # Safety: avoid invalid cutoff frequencies
    f_low_norm  = max(f_low / fnyq, 1e-6)
    f_high_norm = min(f_high / fnyq, 0.99)

    if f_low_norm >= f_high_norm:
        raise ValueError(
            f"Invalid filter frequencies: f_low={f_low}, f_high={f_high}, fs={fs}"
        )

    b, a = butter(
        order,
        [f_low_norm, f_high_norm],
        btype="bandpass",
    )

    return filtfilt(b, a, signal_centered, padtype="odd")
def _get_or_create_named_axes(
    figure_name: str,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (9, 5),
    sharex: bool = False,
):
    """
    Get an existing named matplotlib figure, or create it if needed.

    The figure is not cleared when it already exists, so repeated calls from
    different simulations overlay curves in the same figure.
    """

    fig = plt.figure(figure_name, figsize=figsize)

    n_axes_expected = nrows * ncols

    if len(fig.axes) == 0:
        axs = fig.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
        )

        if n_axes_expected == 1:
            axs = np.asarray([axs])
        else:
            axs = np.asarray(axs).ravel()

    elif len(fig.axes) == n_axes_expected:
        axs = np.asarray(fig.axes)

    else:
        # Only clear if the existing figure has the wrong axis structure.
        fig.clear()

        axs = fig.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
        )

        if n_axes_expected == 1:
            axs = np.asarray([axs])
        else:
            axs = np.asarray(axs).ravel()

    return fig, axs


def _extract_cycle_plot_tag(
    save_path: str | None,
    title_tag: str = "",
):
    """
    Extract the comparison tag from a path whose basename is expected to be:

        cycle_plot_{ps_str}_{vx_str}

    Examples:
        cycle_plot_0.0_pure_flow.pdf      -> 0.0_pure_flow
        cycle_plot_20.0_with_vortices.pdf -> 20.0_with_vortices
    """

    if save_path is None:
        return title_tag.strip() if title_tag.strip() else "simulation"

    basename = os.path.basename(save_path)
    basename = os.path.splitext(basename)[0]

    if basename.startswith("cycle_plot_"):
        return basename.replace("cycle_plot_", "", 1)

    return basename


def plot_cycle_averaged_joint_signals_single_simulation(
    data_df     : pd.DataFrame,
    n_bins      : int = 100,
    t_skip      : float = 5.0,
    n_joints_tot: int = 15,
    n_joints_pas: int = 2,
    title_tag   : str = "",
    save_path   : str | None = None,
    make_comparison_figures: bool = True,
):
    """
    Plot cycle-averaged joint power, torque, and speed components
    for one simulation.

    Each signal is segmented using the phase of the corresponding filtered
    joint position signal:
        joint_0 signals -> phase(joint_0_pos)
        joint_1 signals -> phase(joint_1_pos)
        etc.

    Required columns:
        times

        joint_{j}_pos
        joint_{j}_vel

        joint_{j}_trq_tot
        joint_{j}_trq_act_sum
        joint_{j}_trq_act_dif
        joint_{j}_trq_spring
        joint_{j}_trq_damping

    Plotted torque components:
        trq_tot
        trq_act_sum
        trq_act_dif
        trq_act = trq_act_sum + trq_act_dif
        trq_spring
        trq_damping

    Plotted power components:
        P_tot     = vel * trq_tot
        P_act_sum = vel * trq_act_sum
        P_act_dif = vel * trq_act_dif
        P_act     = vel * (trq_act_sum + trq_act_dif)
        P_spring  = vel * trq_spring
        P_damping = vel * trq_damping

    Additional plots:
        - Global cycle-averaged power components
        - Global cycle-averaged torque components
        - Negative total-power fraction
        - One stacked figure per joint:
            1. power components
            2. torque components
            3. joint speed

    Additional named comparison figures:
        Curves are added to persistent figures named with plt.figure("...").
        The label is extracted from save_path, assuming save_path is:

            cycle_plot_{ps_str}_{vx_str}

        The comparison label becomes:

            {ps_str}_{vx_str}
    """

    # ---------------------------------------------------------------------
    # Save directory and comparison tag
    # ---------------------------------------------------------------------

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = "."

    condition_tag = _extract_cycle_plot_tag(
        save_path=save_path,
        title_tag=title_tag,
    )

    # ---------------------------------------------------------------------
    # Time and phase grid
    # ---------------------------------------------------------------------

    times = data_df["times"].values

    if len(times) < 2:
        raise ValueError("The dataframe must contain at least two time samples.")

    timestep = times[1] - times[0]
    n_skip   = np.sum(times < t_skip)

    n_joints_act = n_joints_tot - n_joints_pas
    phase_grid   = np.linspace(0.0, 1.0, n_bins)

    # ---------------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------------

    required_columns = ["times"]

    for j in range(n_joints_act):
        required_columns.extend([
            f"joint_{j}_pos",
            f"joint_{j}_vel",
            f"joint_{j}_trq_tot",
            f"joint_{j}_trq_act_sum",
            f"joint_{j}_trq_act_dif",
            f"joint_{j}_trq_spring",
            f"joint_{j}_trq_damping",
        ])

    missing_columns = [
        col for col in required_columns
        if col not in data_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ---------------------------------------------------------------------
    # Signal keys
    # ---------------------------------------------------------------------

    power_keys = [
        "P_tot",
        "P_act_sum",
        "P_act_dif",
        "P_act",
        "P_spring",
        "P_damping",
    ]

    torque_keys = [
        "trq_tot",
        "trq_act_sum",
        "trq_act_dif",
        "trq_act",
        "trq_spring",
        "trq_damping",
    ]

    # ---------------------------------------------------------------------
    # Global containers: all joint-cycle curves pooled together
    # ---------------------------------------------------------------------

    power_curves = {
        key: [] for key in power_keys
    }

    torque_curves = {
        key: [] for key in torque_keys
    }

    velocity_curves = []

    # ---------------------------------------------------------------------
    # Per-joint containers: one mean curve per joint
    # ---------------------------------------------------------------------

    joint_power_mean_curves = {
        key: [] for key in power_keys
    }

    joint_torque_mean_curves = {
        key: [] for key in torque_keys
    }

    joint_velocity_mean_curves = []

    joint_cycle_counts = []

    # ---------------------------------------------------------------------
    # Cycle segmentation and interpolation
    # ---------------------------------------------------------------------

    for j in range(n_joints_act):

        j_pos = data_df[f"joint_{j}_pos"].values[n_skip:]
        j_vel = data_df[f"joint_{j}_vel"].values[n_skip:]

        j_trq_tot     = data_df[f"joint_{j}_trq_tot"].values[n_skip:]
        j_trq_act_sum = data_df[f"joint_{j}_trq_act_sum"].values[n_skip:]
        j_trq_act_dif = data_df[f"joint_{j}_trq_act_dif"].values[n_skip:]
        j_trq_spring  = data_df[f"joint_{j}_trq_spring"].values[n_skip:]
        j_trq_damping = data_df[f"joint_{j}_trq_damping"].values[n_skip:]

        j_trq_act = j_trq_act_sum + j_trq_act_dif

        local_torque_signals = {
            "trq_tot"    : j_trq_tot,
            "trq_act_sum": j_trq_act_sum,
            "trq_act_dif": j_trq_act_dif,
            "trq_act"    : j_trq_act,
            "trq_spring" : j_trq_spring,
            "trq_damping": j_trq_damping,
        }

        local_power_signals = {
            "P_tot"    : j_vel * j_trq_tot,
            "P_act_sum": j_vel * j_trq_act_sum,
            "P_act_dif": j_vel * j_trq_act_dif,
            "P_act"    : j_vel * j_trq_act,
            "P_spring" : j_vel * j_trq_spring,
            "P_damping": j_vel * j_trq_damping,
        }

        # -----------------------------------------------------------------
        # Segment using filtered joint-position phase
        # -----------------------------------------------------------------

        j_pos_filtered = _filter_joint_position_for_phase(j_pos, timestep)
        j_phase         = _get_signal_phase(j_pos_filtered)
        j_phase_wrapped = j_phase % (2 * np.pi)

        cycle_starts = np.where(np.diff(j_phase_wrapped) < -np.pi)[0] + 1

        cycle_intervals = [
            (i0, i1)
            for i0, i1 in zip(cycle_starts[:-1], cycle_starts[1:])
            if i1 > i0 + 10
        ]

        joint_cycle_counts.append(len(cycle_intervals))

        local_joint_power_curves = {
            key: [] for key in power_keys
        }

        local_joint_torque_curves = {
            key: [] for key in torque_keys
        }

        local_joint_velocity_curves = []

        for i0, i1 in cycle_intervals:

            cycle_len = i1 - i0

            if cycle_len < 2:
                continue

            x_cycle = np.linspace(0.0, 1.0, cycle_len)

            # -------------------------------------------------------------
            # Power components
            # -------------------------------------------------------------

            for key, signal in local_power_signals.items():
                curve = np.interp(
                    phase_grid,
                    x_cycle,
                    signal[i0:i1],
                )

                power_curves[key].append(curve)
                local_joint_power_curves[key].append(curve)

            # -------------------------------------------------------------
            # Torque components
            # -------------------------------------------------------------

            for key, signal in local_torque_signals.items():
                curve = np.interp(
                    phase_grid,
                    x_cycle,
                    signal[i0:i1],
                )

                torque_curves[key].append(curve)
                local_joint_torque_curves[key].append(curve)

            # -------------------------------------------------------------
            # Joint speed
            # -------------------------------------------------------------

            vel_curve = np.interp(
                phase_grid,
                x_cycle,
                j_vel[i0:i1],
            )

            velocity_curves.append(vel_curve)
            local_joint_velocity_curves.append(vel_curve)

        # -----------------------------------------------------------------
        # Store one mean curve per joint
        # -----------------------------------------------------------------

        for key in power_keys:
            if len(local_joint_power_curves[key]) > 0:
                joint_power_mean_curves[key].append(
                    np.mean(local_joint_power_curves[key], axis=0)
                )

        for key in torque_keys:
            if len(local_joint_torque_curves[key]) > 0:
                joint_torque_mean_curves[key].append(
                    np.mean(local_joint_torque_curves[key], axis=0)
                )

        if len(local_joint_velocity_curves) > 0:
            joint_velocity_mean_curves.append(
                np.mean(local_joint_velocity_curves, axis=0)
            )

    # ---------------------------------------------------------------------
    # Convert to arrays
    # ---------------------------------------------------------------------

    for key in power_keys:
        power_curves[key] = np.asarray(power_curves[key])
        joint_power_mean_curves[key] = np.asarray(joint_power_mean_curves[key])

    for key in torque_keys:
        torque_curves[key] = np.asarray(torque_curves[key])
        joint_torque_mean_curves[key] = np.asarray(joint_torque_mean_curves[key])

    velocity_curves = np.asarray(velocity_curves)
    joint_velocity_mean_curves = np.asarray(joint_velocity_mean_curves)

    if power_curves["P_tot"].size == 0:
        raise ValueError("No valid joint cycles found.")

    # ---------------------------------------------------------------------
    # Means and standard deviations
    # ---------------------------------------------------------------------

    power_mean = {
        key: np.mean(value, axis=0)
        for key, value in power_curves.items()
    }

    power_std = {
        key: np.std(value, axis=0)
        for key, value in power_curves.items()
    }

    torque_mean = {
        key: np.mean(value, axis=0)
        for key, value in torque_curves.items()
    }

    torque_std = {
        key: np.std(value, axis=0)
        for key, value in torque_curves.items()
    }

    velocity_mean = np.mean(velocity_curves, axis=0)
    velocity_std  = np.std(velocity_curves, axis=0)

    neg_fraction = np.mean(power_curves["P_tot"] <= 0.0, axis=0)

    # ---------------------------------------------------------------------
    # Peak detection on total signed power
    # ---------------------------------------------------------------------

    peak_inds, _ = find_peaks(power_mean["P_tot"])
    peak_inds = peak_inds[power_mean["P_tot"][peak_inds] > 0.0]

    if len(peak_inds) >= 2:
        top_two = peak_inds[np.argsort(power_mean["P_tot"][peak_inds])[-2:]]
        top_two = top_two[np.argsort(phase_grid[top_two])]

    elif len(peak_inds) == 1:
        top_two = peak_inds

    else:
        top_two = np.array([np.argmax(power_mean["P_tot"])])

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    print("\nCycle-averaged joint signal summary")
    print("-----------------------------------")
    print(f"Condition tag: {condition_tag}")
    print(f"Number of joints: {n_joints_act}")
    print(f"Mean cycles per joint: {np.mean(joint_cycle_counts):.2f}")
    print(
        f"Min/max cycles per joint: "
        f"{np.min(joint_cycle_counts)} / {np.max(joint_cycle_counts)}"
    )
    print(f"Number of joint-cycle curves: {power_curves['P_tot'].shape[0]}")

    print("\nPower components")
    for key in power_keys:
        print(
            f"{key}: "
            f"mean = {np.mean(power_curves[key]):.6g}, "
            f"peak abs mean = {np.max(np.abs(power_mean[key])):.6g}"
        )

    print("\nTorque components")
    for key in torque_keys:
        print(
            f"{key}: "
            f"mean = {np.mean(torque_curves[key]):.6g}, "
            f"peak abs mean = {np.max(np.abs(torque_mean[key])):.6g}"
        )

    print("\nJoint speed")
    print(f"Mean speed: {np.mean(velocity_curves):.6g}")
    print(f"Peak absolute mean speed: {np.max(np.abs(velocity_mean)):.6g}")

    for peak_id, peak_idx in enumerate(top_two, start=1):
        print(
            f"P_tot peak {peak_id}: "
            f"x = {phase_grid[peak_idx]:.3f}, "
            f"power = {power_mean['P_tot'][peak_idx]:.6g}"
        )

    # ---------------------------------------------------------------------
    # Plot 1: global power components
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    for key in power_keys:
        ax.plot(
            phase_grid,
            power_mean[key],
            linewidth=2.0,
            label=key,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)

    for peak_id, peak_idx in enumerate(top_two, start=1):

        x_peak = phase_grid[peak_idx]
        y_peak = power_mean["P_tot"][peak_idx]

        ax.plot(
            x_peak,
            y_peak,
            marker="o",
            markersize=8,
            color="red",
        )

        x_offset = 0.04 if peak_id == 1 else -0.20
        y_offset = 0.8e-9 if peak_id == 1 else 1.0e-9

        ax.annotate(
            f"P_tot peak {peak_id} = {y_peak:.2e}",
            xy=(x_peak, y_peak),
            xytext=(x_peak + x_offset, y_peak + y_offset),
            arrowprops=dict(arrowstyle="->", linewidth=1),
            fontsize=10,
        )

    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Normalized local joint cycle")
    ax.set_ylabel("Joint power")
    ax.set_title(f"Cycle-averaged joint power components {title_tag}")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path.replace("cycle_plot", "power_components"),
            dpi=300,
        )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 2: global torque components
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    for key in torque_keys:
        ax.plot(
            phase_grid,
            torque_mean[key],
            linewidth=2.0,
            label=key,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Normalized local joint cycle")
    ax.set_ylabel("Joint torque")
    ax.set_title(f"Cycle-averaged joint torque components {title_tag}")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path.replace("cycle_plot", "torque_components"),
            dpi=300,
        )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 3: global joint speed
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        phase_grid,
        velocity_mean,
        linewidth=2.0,
        label="joint speed",
    )

    ax.fill_between(
        phase_grid,
        velocity_mean - velocity_std,
        velocity_mean + velocity_std,
        alpha=0.25,
        label="±1 SD",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Normalized local joint cycle")
    ax.set_ylabel("Joint speed")
    ax.set_title(f"Cycle-averaged joint speed {title_tag}")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path.replace("cycle_plot", "joint_speed"),
            dpi=300,
        )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 4: negative total-power fraction
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        phase_grid,
        neg_fraction,
        linewidth=2.0,
        label="P_tot ≤ 0 fraction",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Normalized local joint cycle")
    ax.set_ylabel("Fraction of joint-cycle samples ≤ 0")
    ax.set_title(f"Negative total-power occupancy {title_tag}")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path.replace("cycle_plot", "negative_power_fraction"),
            dpi=300,
        )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 5: one stacked figure per joint
    # ---------------------------------------------------------------------

    n_valid_joints = len(joint_velocity_mean_curves)

    for j in range(n_valid_joints):

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(10, 9),
            sharex=True,
        )

        # -------------------------------------------------------------
        # Subplot 1: joint power components
        # -------------------------------------------------------------

        ax = axs[0]

        for key in power_keys:

            if j >= len(joint_power_mean_curves[key]):
                continue

            ax.plot(
                phase_grid,
                joint_power_mean_curves[key][j],
                linewidth=1.8,
                label=key,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax.set_ylabel("Joint power")
        ax.set_title(f"Joint {j}: cycle-averaged components {title_tag}")
        ax.legend(loc="best", fontsize=9)

        # -------------------------------------------------------------
        # Subplot 2: joint torque components
        # -------------------------------------------------------------

        ax = axs[1]

        for key in torque_keys:

            if j >= len(joint_torque_mean_curves[key]):
                continue

            ax.plot(
                phase_grid,
                joint_torque_mean_curves[key][j],
                linewidth=1.8,
                label=key,
            )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax.set_ylabel("Joint torque")
        ax.legend(loc="best", fontsize=9)

        # -------------------------------------------------------------
        # Subplot 3: joint speed
        # -------------------------------------------------------------

        ax = axs[2]

        ax.plot(
            phase_grid,
            joint_velocity_mean_curves[j],
            linewidth=2.0,
            label="joint speed",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Joint speed")
        ax.legend(loc="best", fontsize=9)

        fig.tight_layout()

        if save_path is not None:

            save_dir_joint = f"{save_dir}/joint_{j:02d}"
            os.makedirs(save_dir_joint, exist_ok=True)

            save_path_joint = save_path.replace(
                save_dir,
                save_dir_joint,
            )

            fig.savefig(
                save_path_joint.replace("cycle_plot", "stacked_components"),
                dpi=300,
            )

        plt.close(fig)

    # ---------------------------------------------------------------------
    # Named comparison figures
    # ---------------------------------------------------------------------
    # These figures are intentionally not cleared. Calling this function
    # several times overlays different conditions in the same figures.

    if make_comparison_figures:

        # -----------------------------------------------------------------
        # Comparison: active torque only
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Active Torque Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            torque_mean["trq_act"],
            linewidth=2.0,
            label=f"{condition_tag} - trq_act",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Active joint torque")
        ax.set_title("Active Torque Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()

        # -----------------------------------------------------------------
        # Comparison: spring and damping torque only
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Spring Damping Torque Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            torque_mean["trq_spring"],
            linewidth=2.0,
            label=f"{condition_tag} - trq_spring",
        )

        ax.plot(
            phase_grid,
            torque_mean["trq_damping"],
            linewidth=2.0,
            label=f"{condition_tag} - trq_damping",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Passive joint torque")
        ax.set_title("Spring/Damping Torque Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()

        # -----------------------------------------------------------------
        # Comparison: active and passive torque
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Active vs Passive Torque Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            torque_mean["trq_act"],
            linewidth=2.0,
            label=f"{condition_tag} - trq_act",
        )

        ax.plot(
            phase_grid,
            torque_mean["trq_spring"] + torque_mean["trq_damping"],
            linewidth=2.0,
            label=f"{condition_tag} - trq_passive",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Passive joint torque")
        ax.set_title("Active vs Passive Torque Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()

        # -----------------------------------------------------------------
        # Comparison: total power only
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Total Power Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            power_mean["P_tot"],
            linewidth=2.0,
            label=f"{condition_tag} - P_tot",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Total joint power")
        ax.set_title("Total Power Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()

        # -----------------------------------------------------------------
        # Comparison: active power only
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Active Power Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            power_mean["P_act"],
            linewidth=2.0,
            label=f"{condition_tag} - P_act",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Active joint power")
        ax.set_title("Active Power Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()

        # -----------------------------------------------------------------
        # Comparison: joint speed
        # -----------------------------------------------------------------

        fig, axs = _get_or_create_named_axes(
            figure_name="Joint Speed Comparison",
            nrows=1,
            ncols=1,
            figsize=(10, 5),
        )

        ax = axs[0]

        ax.plot(
            phase_grid,
            velocity_mean,
            linewidth=2.0,
            label=f"{condition_tag} - joint speed",
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Normalized local joint cycle")
        ax.set_ylabel("Joint speed")
        ax.set_title("Joint Speed Comparison")
        ax.legend(fontsize=8)

        fig.tight_layout()


    return {
        "condition_tag": condition_tag,
        "phase_grid": phase_grid,

        "power_curves": power_curves,
        "power_mean": power_mean,
        "power_std": power_std,

        "torque_curves": torque_curves,
        "torque_mean": torque_mean,
        "torque_std": torque_std,

        "velocity_curves": velocity_curves,
        "velocity_mean": velocity_mean,
        "velocity_std": velocity_std,

        "neg_fraction": neg_fraction,

        "joint_cycle_counts": joint_cycle_counts,

        "joint_power_mean_curves": joint_power_mean_curves,
        "joint_torque_mean_curves": joint_torque_mean_curves,
        "joint_velocity_mean_curves": joint_velocity_mean_curves,

        "peak_indices": top_two,
        "peak_phases": phase_grid[top_two],
        "peak_values": power_mean["P_tot"][top_two],
    }

###############################
# POWER METRICS ###
###############################

# def _add_power_explanation_metrics(
#     simulation_performance: dict,
#     data_df               : pd.DataFrame,
# ):
#     '''
#     Add mechanistic metrics explaining changes in positive mechanical power.

#     This function does not require logged spring/damping torques.
#     It computes:
#     - joint velocity amplitude
#     - joint position amplitude
#     - total torque amplitude
#     - active torque amplitude
#     - torque-velocity alignment
#     - torque-position alignment
#     - active/total torque ratios
#     - active-vs-total cancellation
#     - stiffness-like kinematic proxy based on -position
#     - damping-like kinematic proxy based on -velocity

#     NOTE:
#     The stiffness and damping quantities are proxies, not physical torques,
#     unless multiplied by the corresponding mechanical coefficients.
#     They are useful for relative comparisons across conditions.
#     '''


#     # -------------------------------------------------------------------------
#     # Helper functions
#     # -------------------------------------------------------------------------

#     def rms(x):
#         return np.sqrt(np.mean(x**2))

#     def abs_mean(x):
#         return np.mean(np.abs(x))

#     def abs_p95(x):
#         return np.percentile(np.abs(x), 95)

#     def safe_ratio(num, den, eps=1e-12):
#         return num / (den + eps)

#     def signed_alignment(x, y, eps=1e-12):
#         '''
#         Normalized signed alignment between two signals.

#         +1  = strongly aligned
#          0  = weak/no signed relation
#         -1  = opposite sign/alignment
#         '''
#         return np.mean(x * y) / (rms(x) * rms(y) + eps)

#     def abs_alignment(x, y, eps=1e-12):
#         '''
#         Magnitude of coupling irrespective of sign.
#         '''
#         return np.mean(np.abs(x * y)) / (rms(x) * rms(y) + eps)


#         if np.std(sig_a_osc) < 1e-12 or np.std(sig_b_osc) < 1e-12:
#             return 0.0


#         return circmean(phase_diff)


#         if np.std(sig_a_osc) < 1e-12 or np.std(sig_b_osc) < 1e-12:
#             return 0.0


#         return np.abs(np.mean(np.exp(1j * phase_diff)))


#     def signed_power_mean(torque_like, velocity):
#         '''
#         Mean signed power from a torque-like signal.
#         '''
#         return np.mean(torque_like * velocity)


#     # -------------------------------------------------------------------------
#     # Allocate metrics
#     # -------------------------------------------------------------------------


#     # -------------------------------------------------------------------------
#     # Per-joint computation
#     # -------------------------------------------------------------------------

#     for j in range(n_joints_act):


#         # ---------------------------------------------------------------------
#         # Kinematic passive proxies
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Position / velocity amplitudes
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Torque amplitudes
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Torque-velocity alignment
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Torque-position alignment
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Phase relation between total torque and velocity
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Active/total ratios
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Active vs total cancellation
#         # ---------------------------------------------------------------------
#         # Larger values mean active torque is large relative to net total torque.
#         # This indicates compensation by non-active effects.


#         # ---------------------------------------------------------------------
#         # Passive proxy amplitudes
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Passive proxy powers
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Proxy/total ratios
#         # ---------------------------------------------------------------------


#         # ---------------------------------------------------------------------
#         # Cancellation proxies
#         # ---------------------------------------------------------------------
#         # These indices are dimensionless but proxy-based.
#         # They should be interpreted only comparatively across conditions.


#     # -------------------------------------------------------------------------
#     # Store joint-averaged metrics
#     # -------------------------------------------------------------------------


#     return perf

def _add_power_metrics(
    simulation_performance: dict,
    data_df               : pd.DataFrame,
    plot_cycle_power      : bool = False,
    plot_cycle_path       : str = None,
):
    '''
    Add power metrics if not present
    NOTE: 5.0 seconds are discarded in the energy computation
    '''

    perf    = simulation_performance
    times   = data_df['times'].values
    n_steps = len(times)
    n_skip  = np.sum(times < 5.0)

    n_joints_tot = 15
    n_joints_pas = 2
    n_joints_act = n_joints_tot - n_joints_pas
    n_steps_eval = n_steps - n_skip

    # --------------------------------------------------------------------------
    # Plot cycle-averaged power for one simulation
    # --------------------------------------------------------------------------

    if plot_cycle_power:
        plot_cycle_averaged_joint_signals_single_simulation(
            data_df    = data_df,
            save_path  = plot_cycle_path,
        )

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def neg_fract(p):
        """Fraction of power samples that are negative."""
        return np.mean(p <= 0)

    def pow_mean(p):
        """Mean of power amplitude."""
        return np.mean(p)

    def pow_p95(p):
        """95th percentile of positive power amplitude."""
        p_pos = p[p > 0]
        return np.percentile(p_pos, 95) if len(p_pos) > 0 else 0.0

    # -------------------------------------------------------------------------
    # Per-joint sign and amplitude metrics
    # -------------------------------------------------------------------------

    neg_fract_tot     = np.zeros(n_joints_act)
    neg_fract_act_sum = np.zeros(n_joints_act)
    neg_fract_act_dif = np.zeros(n_joints_act)
    neg_fract_act     = np.zeros(n_joints_act)

    pow_mean_tot     = np.zeros(n_joints_act)
    pow_mean_act_sum = np.zeros(n_joints_act)
    pow_mean_act_dif = np.zeros(n_joints_act)
    pow_mean_act     = np.zeros(n_joints_act)

    pow_p95_tot       = np.zeros(n_joints_act)
    pow_p95_act_sum   = np.zeros(n_joints_act)
    pow_p95_act_dif   = np.zeros(n_joints_act)
    pow_p95_act       = np.zeros(n_joints_act)

    # -------------------------------------------------------------------------
    # Time-resolved positive power, summed across joints
    # -------------------------------------------------------------------------

    glob_pow_tot     = np.zeros(n_steps_eval)
    glob_pow_act_sum = np.zeros(n_steps_eval)
    glob_pow_act_dif = np.zeros(n_steps_eval)
    glob_pow_act     = np.zeros(n_steps_eval)

    for j in range(n_joints_act):

        j_pos         = data_df[f'joint_{j}_pos'].values[n_skip:]
        j_vel         = data_df[f'joint_{j}_vel'].values[n_skip:]
        j_trq_tot     = data_df[f'joint_{j}_trq_tot'].values[n_skip:]
        j_trq_act_sum = data_df[f'joint_{j}_trq_act_sum'].values[n_skip:]
        j_trq_act_dif = data_df[f'joint_{j}_trq_act_dif'].values[n_skip:]

        # Compute powers
        j_pow_tot     = j_vel * j_trq_tot
        j_pow_act_sum = j_vel * j_trq_act_sum
        j_pow_act_dif = j_vel * j_trq_act_dif
        j_pow_act     = j_pow_act_dif + j_pow_act_sum

        # Negative Fractions
        neg_fract_tot[j]     = neg_fract(j_pow_tot)
        neg_fract_act_sum[j] = neg_fract(j_pow_act_sum)
        neg_fract_act_dif[j] = neg_fract(j_pow_act_dif)
        neg_fract_act[j]     = neg_fract(j_pow_act)

        # Cap negative
        j_pow_tot     = np.clip(j_pow_tot,     a_min=0, a_max=None)
        j_pow_act_sum = np.clip(j_pow_act_sum, a_min=0, a_max=None)
        j_pow_act_dif = np.clip(j_pow_act_dif, a_min=0, a_max=None)
        j_pow_act     = np.clip(j_pow_act,     a_min=0, a_max=None)

        # Means
        pow_mean_tot[j]     = pow_mean(j_pow_tot)
        pow_mean_act_sum[j] = pow_mean(j_pow_act_sum)
        pow_mean_act_dif[j] = pow_mean(j_pow_act_dif)
        pow_mean_act[j]     = pow_mean(j_pow_act)

        # Amplitude
        pow_p95_tot[j]       = pow_p95(j_pow_tot)
        pow_p95_act_sum[j]   = pow_p95(j_pow_act_sum)
        pow_p95_act_dif[j]   = pow_p95(j_pow_act_dif)
        pow_p95_act[j]       = pow_p95(j_pow_act)

        # Total across joints
        glob_pow_tot     += j_pow_tot
        glob_pow_act_sum += j_pow_act_sum
        glob_pow_act_dif += j_pow_act_dif
        glob_pow_act     += j_pow_act

    # -------------------------------------------------------------------------
    # Joint Averaged Metrics
    # -------------------------------------------------------------------------
    arr_mean = lambda p: np.array([np.mean(p)])

    # Negative fraction
    perf['pow_neg_fract_tot']     = arr_mean(neg_fract_tot)
    perf['pow_neg_fract_act_sum'] = arr_mean(neg_fract_act_sum)
    perf['pow_neg_fract_act_dif'] = arr_mean(neg_fract_act_dif)
    perf['pow_neg_fract_act']     = arr_mean(neg_fract_act)

    # Positive mean
    perf['pow_mean_tot']     = arr_mean(pow_mean_tot)
    perf['pow_mean_act_sum'] = arr_mean(pow_mean_act_sum)
    perf['pow_mean_act_dif'] = arr_mean(pow_mean_act_dif)
    perf['pow_mean_act']     = arr_mean(pow_mean_act)

    # Positive amplitude
    perf['pow_p95_tot']       = arr_mean(pow_p95_tot)
    perf['pow_p95_act_sum']   = arr_mean(pow_p95_act_sum)
    perf['pow_p95_act_dif']   = arr_mean(pow_p95_act_dif)
    perf['pow_p95_act']       = arr_mean(pow_p95_act)

    # -------------------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------------------

    # Global Mean
    perf['glob_pow_mean_tot']     = np.array([ pow_mean(glob_pow_tot) ])
    perf['glob_pow_mean_act_sum'] = np.array([ pow_mean(glob_pow_act_sum) ])
    perf['glob_pow_mean_act_dif'] = np.array([ pow_mean(glob_pow_act_dif) ])
    perf['glob_pow_mean_act']     = np.array([ pow_mean(glob_pow_act) ])

    # Global Amplitude
    perf['glob_pow_p95_tot']       = np.array([ pow_p95(glob_pow_tot) ])
    perf['glob_pow_p95_act_sum']   = np.array([ pow_p95(glob_pow_act_sum) ])
    perf['glob_pow_p95_act_dif']   = np.array([ pow_p95(glob_pow_act_dif) ])
    perf['glob_pow_p95_act']       = np.array([ pow_p95(glob_pow_act) ])

    return perf

###############################
# OTHER METRICS ###
###############################

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

def _add_spike_count_metrics(
    simulation_performance: dict,
    data_df                : pd.DataFrame,
):
    ''' Add tail angle metrics '''

    times      = data_df['times'].values
    body_phase = data_df['joint_phases'].values

    # Remove first 5 seconds
    n_skip     = np.sum( times < 5.0 )
    times      = times[n_skip:]
    body_phase = body_phase[n_skip:]

    # Cycles
    timestep  = times[1] - times[0]
    duration  = times[-1] - times[0]
    frequency = simulation_performance['neur_freq_ax'][0]
    n_cycles  = duration * frequency

    ###########################################################################
    # SPIKE COUNTS METRICS (GLOBAL)
    ###########################################################################

    count_key = lambda pop, seg, side: f'{pop}_spike_count_{2*seg + side}'

    n_segments   = 32
    n_pools_dict = {
        'ex': 32,
        'in': 48,
        'mn': 60,
        'ps': 4,
    }


    def _get_filtered_count(count: np.ndarray):
        ''' Apply a low-pass filter around the signal frequency '''

        # Continuous counts
        sigma_t = 0.050
        sigma_n = sigma_t / timestep
        count_f = gaussian_filter1d(count, sigma=sigma_n)

        # Butter filtering
        fnyq = 0.5 / timestep
        ford = 2

        # Low-pass filter
        f_high   = 10.0
        num, den = butter(ford, 2*f_high/fnyq, btype= 'lowpass' )
        count_f = filtfilt(num, den, count_f, padtype= 'odd')

        # High-pass filter
        f_high   = 0.5
        num, den = butter(ford, 2*f_high/fnyq, btype= 'high' )
        count_f = filtfilt(num, den, count_f, padtype= 'odd')

        return count_f


    def add_global_count_metrics(counts_l, counts_r, pop):
        sim_perf = simulation_performance

        n_pool = n_pools_dict[pop] * n_segments
        n_hemi = n_pool / 2

        # Filter counts
        counts_l_filt = _get_filtered_count(counts_l)
        counts_r_filt = _get_filtered_count(counts_r)

        # Count phases
        count_phase_l = _get_signal_phase(counts_l_filt)
        count_phase_r = _get_signal_phase(counts_r_filt)

        phase_diff_lr = count_phase_l - count_phase_r
        phase_diff_lb = count_phase_l - body_phase
        phase_diff_rb = count_phase_r - body_phase

        phase_diff_lr = np.angle(np.exp(1j * phase_diff_lr))[100:-100]
        phase_diff_lb = np.angle(np.exp(1j * phase_diff_lb))[100:-100]
        phase_diff_rb = np.angle(np.exp(1j * phase_diff_rb))[100:-100]

        phase_diff_lr_mean = circmean(phase_diff_lr)
        phase_diff_lb_mean = circmean(phase_diff_lb)
        phase_diff_rb_mean = circmean(phase_diff_rb)

        sim_perf[f'{pop}_phase_diff_lr_mean'] = np.array([phase_diff_lr_mean])
        sim_perf[f'{pop}_phase_diff_lb_mean'] = np.array([phase_diff_lb_mean])
        sim_perf[f'{pop}_phase_diff_rb_mean'] = np.array([phase_diff_rb_mean])

        # Total spike counts
        tot_count_l = np.sum(counts_l)
        tot_count_r = np.sum(counts_r)
        tot_count   = tot_count_l + tot_count_r

        sim_perf[f'{pop}_tot_count_l'] = np.array([tot_count_l])
        sim_perf[f'{pop}_tot_count_r'] = np.array([tot_count_r])
        sim_perf[f'{pop}_tot_count']   = np.array([tot_count])

        # Count per neuron
        count_n_l = tot_count_l / n_hemi
        count_n_r = tot_count_r / n_hemi
        count_n   = tot_count / n_pool

        sim_perf[f'{pop}_count_per_neuron_l'] = np.array([count_n_l])
        sim_perf[f'{pop}_count_per_neuron_r'] = np.array([count_n_r])
        sim_perf[f'{pop}_count_per_neuron']   = np.array([count_n])

        # Count per neuron per cycle
        count_nc_l = count_n_l / n_cycles
        count_nc_r = count_n_r / n_cycles
        count_nc   = count_n   / n_cycles

        sim_perf[f'{pop}_count_per_neuron_per_cycle_l'] = np.array([count_nc_l])
        sim_perf[f'{pop}_count_per_neuron_per_cycle_r'] = np.array([count_nc_r])
        sim_perf[f'{pop}_count_per_neuron_per_cycle']   = np.array([count_nc])

    # Add spike count metrics
    for pop in ['ex', 'in', 'mn', 'ps']:

        keys_l = [ count_key(pop, seg, 0) for seg in range(n_segments) ]
        keys_r = [ count_key(pop, seg, 1) for seg in range(n_segments) ]

        counts_l = np.sum( data_df[keys_l].values, axis=1 )[n_skip:]
        counts_r = np.sum( data_df[keys_r].values, axis=1 )[n_skip:]

        add_global_count_metrics(counts_l, counts_r, pop)

    ###########################################################################
    # SPIKE COUNTS METRICS (SEGMENTS)
    ###########################################################################

    def add_segment_count_metrics(count_l, count_r, pop, seg):

        tag        = f'{pop}_seg_{seg}'
        sim_perf   = simulation_performance

        n_pool = n_pools_dict[pop]
        n_hemi = n_pool / 2

        # Total spike counts
        tot_count_l = np.sum(count_l)
        tot_count_r = np.sum(count_r)
        tot_count   = tot_count_l + tot_count_r

        sim_perf[f'{tag}_tot_count_l'] = np.array([tot_count_l])
        sim_perf[f'{tag}_tot_count_r'] = np.array([tot_count_r])
        sim_perf[f'{tag}_tot_count']   = np.array([tot_count])

        # Count per neuron
        count_n_l = tot_count_l / n_hemi
        count_n_r = tot_count_r / n_hemi
        count_n   = tot_count / n_pool

        sim_perf[f'{tag}_count_per_neuron_l'] = np.array([count_n_l])
        sim_perf[f'{tag}_count_per_neuron_r'] = np.array([count_n_r])
        sim_perf[f'{tag}_count_per_neuron']   = np.array([count_n])

        # Count per neuron per cycle
        count_nc_l = count_n_l / n_cycles
        count_nc_r = count_n_r / n_cycles
        count_nc   = count_n   / n_cycles

        sim_perf[f'{tag}_count_per_neuron_per_cycle_l'] = np.array([count_nc_l])
        sim_perf[f'{tag}_count_per_neuron_per_cycle_r'] = np.array([count_nc_r])
        sim_perf[f'{tag}_count_per_neuron_per_cycle']   = np.array([count_nc])

    for pop in ['ex', 'in', 'mn', 'ps']:
        for seg in range(n_segments):

            keys_l = count_key(pop, seg, 0)
            keys_r = count_key(pop, seg, 1)

            count_l = data_df[keys_l].values[n_skip:]
            count_r = data_df[keys_r].values[n_skip:]

            add_segment_count_metrics(count_l, count_r, pop, seg)

    return simulation_performance

def load_results_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
    target_quantities  : list[str] = None,
    plot_cycle_power   : bool = False,
    plot_cycle_path    : str = None,
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
        data_df                = data_df,
        plot_cycle_power       = plot_cycle_power,
        plot_cycle_path        = plot_cycle_path,
    )

    # Add tail angle metrics
    simulation_performance = _add_tail_angle_metrics(
        simulation_performance = simulation_performance,
        times                  = data_df['times'].values,
        tail_angles            = data_df['joint_positions'].values,
    )

    # Add tail angle vs ps firing metrics
    simulation_performance = _add_spike_count_metrics(
        simulation_performance = simulation_performance,
        data_df                = data_df,
    )


    # Add metadata
    simulation_performance = simulation_metadata | simulation_performance

    return data_df, simulation_performance

###############################################################################
# PLOTTING ####################################################################
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
        excluded_condition : callable = None,
    ):
        """ Initialize an incremental plotting """

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

        # Initialize the binning
        self._initialize_binning()

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
            plot_cycle_power    = False,
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

    def collect_single_simulation_data(
        self,
        folder_name   : str,
        process_folder: str,
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

        # Example:
        # prc = 'free_body_sketch_body_distance_000BL_height_000BL_seed_106_closed_loop_pure_flow'
        # fld = 'net_farms_zebrafish_dynamic_water_vortices_free_head_muscle_FN_5500_ps_0_coupled_106_SIM'

        ps_str   = f"ps_{ folder_name.split('ps_')[1].split('_')[0] }"
        vx_str   = self.vx_tag
        pos_str  = process_folder.split('sketch_body_')[1].split('_seed')[0]
        seed_str = f"seed_{ folder_name.split('coupled_')[1].split('_')[0] }"

        plot_cycle_path   = (
            f'{ANALYSIS_PATH}/'
            'comparing_vortices_coupled/comparisons/'
            f'{seed_str}/'
            f'cycle_plot_{ps_str}_{vx_str}.pdf'
        )

        data_df, performance_dict = load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
            target_quantities   = target_quantities,
            plot_cycle_power    = True,
            plot_cycle_path     = plot_cycle_path,
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


        return

    def collect_all_simulation_data(self):
        """ Collect data from all simulations. """

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
        # if self.load_saved_data():
        #     return

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


        print('DONE')
        return

    ###########################################################################
    # PLOTS ####################################################################
    ###########################################################################

    def _initialize_binning(self):

        # Initialize the binning for the plots
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

        return


    ###########################################################################
    # SAVE #####################################################################
    ###########################################################################

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
    simulation_tag     : str = None,
    excluded_condition : callable = None,
):
    ''' Plot the results for a given set of parameters '''

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
        excluded_condition  = excluded_condition,
    )

    # Collect all simulation data
    plotting.collect_all_simulation_data()

    # Save collected data
    plotting.save_all_simulation_data()


    return

###############################################################################
# DIFFERENCE PLOTTING #########################################################
###############################################################################

def run_statistical_tests(
    q_values_0 : np.ndarray,
    q_values_1 : np.ndarray,
    q_target   : str,
):
    ''' Run paired statistical tests (t-test and Wilcoxon) '''

    if np.allclose(q_values_0, q_values_1, atol=1e-12):
        print(f'{q_target} difference:')
        print('No difference (all values are close)')
        return False

    def _print_test(
        q_key : str,
        t_stat: float,
        t_val : float,
        w_stat: float,
        w_val : float,
        paired: bool,
    ):
        t_sig = t_val < 0.05
        w_sig = w_val < 0.05
        print(
            f'{q_key} difference: \n'
            f'{"PAIRED" if paired else "UNPAIRED"} TESTS: \n'
            f't-statistic = {t_stat:.3f}, p-value (t-test)   = {t_val:.3e}, '
            f'significant = {t_sig}, \n'
            f'w-statistic = {w_stat:.3f}, p-value (Wilcoxon) = {w_val:.3e}, '
            f'significant = {w_sig}  \n'
        )

    significance = False

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
    significance = significance or (p_val_t < 0.05) or (p_val_w < 0.05)
    _print_test(q_target, t_stat, p_val_t, w_stat, p_val_w, paired=True)

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
    significance = significance or (p_val_t < 0.05) or (p_val_w < 0.05)
    _print_test(q_target, t_stat, p_val_t, w_stat, p_val_w, paired=False)

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

    # Individual points
    sns.swarmplot(
        data   = df_long,
        x      = 'Mode',
        y      = 'Value',
        color  = 'black',
        size   = 3,
        alpha  = 0.7,
        zorder = 4,
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
    sns.violinplot(
        y     = tag,
        data  = df,
        inner = None,
        color = 'lightgray'
    )
    sns.boxplot(
        y        = tag,
        data     = df,
        width    = 0.2,
        boxprops = {'zorder': 2}
    )
    # Individual points
    sns.swarmplot(
        y    = tag,
        data = df,
        color = 'black',
        size  = 3,
        alpha = 0.7,
        zorder = 4,
    )

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
        # Existing mechanical performance metrics
        'mech_power_total',
        'mech_torque_total',

        'mech_power_active',
        'mech_torque_active',

        # Neural activity metrics
        'ex_count_per_neuron_per_cycle',
        'mn_count_per_neuron_per_cycle',
        'ps_count_per_neuron_per_cycle',

        # Negative fraction
        'pow_neg_fract_tot',
        'pow_neg_fract_act_sum',
        'pow_neg_fract_act_dif',
        'pow_neg_fract_act',

        # Positive mean
        'pow_mean_tot',
        'pow_mean_act_sum',
        'pow_mean_act_dif',
        'pow_mean_act',

        # Positive amplitude
        'pow_p95_tot',
        'pow_p95_act_sum',
        'pow_p95_act_dif',
        'pow_p95_act',

        # Global Mean
        'glob_pow_mean_tot',
        'glob_pow_mean_act_sum',
        'glob_pow_mean_act_dif',
        'glob_pow_mean_act',

        # Global Amplitude
        'glob_pow_p95_tot',
        'glob_pow_p95_act_sum',
        'glob_pow_p95_act_dif',
        'glob_pow_p95_act',

    ]

    target_perf_columns += [
        # Position / velocity amplitudes
        'pos_rms_tot',
        'pos_abs_mean_tot',
        'pos_p95_tot',

        'vel_rms_tot',
        'vel_abs_mean_tot',
        'vel_p95_tot',

        # Total torque amplitudes
        'trq_rms_tot',
        'trq_abs_mean_tot',
        'trq_p95_tot',

        # Active torque amplitudes
        'trq_rms_act',
        'trq_abs_mean_act',
        'trq_p95_act',

        # Torque-velocity relation
        'tau_vel_align_tot',
        'tau_vel_abs_align_tot',
        'tau_vel_phase_diff_tot',
        'tau_vel_phase_rval_tot',

        # Torque-position relation
        'tau_pos_align_tot',
        'tau_pos_abs_align_tot',

        # Active contribution
        'active_to_total_trq_rms_ratio',
        'active_to_total_pow_ratio',
        'torque_cancellation_act_total',

        # Kinematic passive proxies
        'spring_proxy_rms',
        'spring_proxy_abs_mean',
        'spring_proxy_p95',

        'damping_proxy_rms',
        'damping_proxy_abs_mean',
        'damping_proxy_p95',

        'signed_pow_spring_proxy',
        'signed_pow_damping_proxy',

        'pow_mean_spring_proxy',
        'pow_mean_damping_proxy',

        'pow_neg_fract_spring_proxy',
        'pow_neg_fract_damping_proxy',

        'spring_proxy_to_total_trq_rms_ratio',
        'damping_proxy_to_total_trq_rms_ratio',

        'spring_proxy_to_total_pow_ratio',
        'damping_proxy_to_total_pow_ratio',

        'active_spring_proxy_cancellation',
        'active_damping_proxy_cancellation',
        'active_passive_proxy_cancellation',
    ]


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
        f'{save_folder}/Performance_comparison_paired_analysis.xlsx',
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
    # get_col  = lambda ner, seg,  key: f"{ner}_seg_{seg}_{key}"
    # get_cols = lambda ner, seg0, seg1, key: [
    #     get_col(ner, seg, key) for seg in range(seg0, seg1)
    # ]
    # get_mean = lambda df, ner, seg0, seg1, key: (
    #     df[ get_cols(ner, seg0, seg1, key) ].mean(axis=1)
    # )

    # for perf_data in [performance_data_0, performance_data_1]:
    #     for ner in ['ex', 'mn']:
    #         for key in [
    #             'count_per_neuron_per_cycle',
    #         ]:


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
    simulation_tag        : str = None,
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
                simulation_tag      = simulation_tag,
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
        'comparing_vortices_coupled/'
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
        'comparing_vortices_coupled/comparisons/'
        f'{dst_tag}/'
    )

    target_quantities = [
        ("mech_torque_total", "rel"),
        ("mech_power_total",  "rel"),
        ("mech_energy_total", "rel"),

        ("mech_torque_active", "rel"),
        ("mech_power_active",  "rel"),
        ("mech_energy_active", "rel"),

        ("mech_freq_ax",          "abs"),
        ("mech_tail_beat_amp_bl", "rel"),

        ("coherence_pos_vel", "abs"),
        ("coherence_vel_vel", "abs"),
        ("coherence_acc_vel", "abs"),
        ("position_diff",     "abs"),

        ("ps_tot_count", "rel"),
        ("ex_tot_count", "rel"),
        ("mn_tot_count", "rel"),

        ("ps_count_per_neuron_per_cycle", "rel"),
        ("ex_count_per_neuron_per_cycle", "rel"),
        ("mn_count_per_neuron_per_cycle", "rel"),

        # Negative fraction
        ( 'pow_neg_fract_tot',     "abs" ),
        ( 'pow_neg_fract_act_sum', "abs" ),
        ( 'pow_neg_fract_act_dif', "abs" ),
        ( 'pow_neg_fract_act',     "abs" ),

        # Positive mean
        ( 'pow_mean_tot',          "rel" ),
        ( 'pow_mean_act_sum',      "rel" ),
        ( 'pow_mean_act_dif',      "rel" ),
        ( 'pow_mean_act',          "rel" ),

        # Positive amplitude
        ( 'pow_p95_tot',           "rel" ),
        ( 'pow_p95_act_sum',       "rel" ),
        ( 'pow_p95_act_dif',       "rel" ),
        ( 'pow_p95_act',           "rel" ),

        # Global Mean
        ( 'glob_pow_mean_tot',     "rel" ),
        ( 'glob_pow_mean_act_sum', "rel" ),
        ( 'glob_pow_mean_act_dif', "rel" ),
        ( 'glob_pow_mean_act',     "rel" ),

        # Global Amplitude
        ( 'glob_pow_p95_tot',      "rel" ),
        ( 'glob_pow_p95_act_sum',  "rel" ),
        ( 'glob_pow_p95_act_dif',  "rel" ),
        ( 'glob_pow_p95_act',      "rel" ),
    ]

    target_quantities += [

        # Position amplitude
        ("pos_rms_tot",      "rel"),
        ("pos_abs_mean_tot", "rel"),
        ("pos_p95_tot",      "rel"),

        # Velocity amplitude
        ("vel_rms_tot",      "rel"),
        ("vel_abs_mean_tot", "rel"),
        ("vel_p95_tot",      "rel"),

        # Total torque amplitude
        ("trq_rms_tot",      "rel"),
        ("trq_abs_mean_tot", "rel"),
        ("trq_p95_tot",      "rel"),

        # Active torque amplitude
        ("trq_rms_act",      "rel"),
        ("trq_abs_mean_act", "rel"),
        ("trq_p95_act",      "rel"),

        # Torque-velocity relation
        ("tau_vel_align_tot",     "abs"),
        ("tau_vel_abs_align_tot", "abs"),
        ("tau_vel_align_act",     "abs"),
        ("tau_vel_abs_align_act", "abs"),

        # Torque-position relation
        ("tau_pos_align_tot",     "abs"),
        ("tau_pos_abs_align_tot", "abs"),

        # Phase relation
        ("tau_vel_phase_diff_tot", "abs"),
        ("tau_vel_phase_rval_tot", "abs"),

        # Active contribution ratios
        ("active_to_total_trq_rms_ratio", "abs"),
        ("active_to_total_pow_ratio",     "abs"),

        # Active vs total cancellation
        ("torque_cancellation_act_total", "abs"),

        # Passive spring-like proxy amplitude
        ("spring_proxy_rms",      "rel"),
        ("spring_proxy_abs_mean", "rel"),
        ("spring_proxy_p95",      "rel"),

        # Passive damping-like proxy amplitude
        ("damping_proxy_rms",      "rel"),
        ("damping_proxy_abs_mean", "rel"),
        ("damping_proxy_p95",      "rel"),

        # Passive proxy signed powers
        ("signed_pow_spring_proxy",  "abs"),
        ("signed_pow_damping_proxy", "abs"),

        # Passive proxy positive powers
        ("pow_mean_spring_proxy",  "rel"),
        ("pow_mean_damping_proxy", "rel"),

        # Passive proxy non-positive fractions
        ("pow_neg_fract_spring_proxy",  "abs"),
        ("pow_neg_fract_damping_proxy", "abs"),

        # Passive proxy ratios
        ("spring_proxy_to_total_trq_rms_ratio",  "abs"),
        ("damping_proxy_to_total_trq_rms_ratio", "abs"),

        ("spring_proxy_to_total_pow_ratio",  "abs"),
        ("damping_proxy_to_total_pow_ratio", "abs"),

        # Proxy cancellation metrics
        ("active_spring_proxy_cancellation",  "abs"),
        ("active_damping_proxy_cancellation", "abs"),
        ("active_passive_proxy_cancellation", "abs"),
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

def _repeat_post_processing(
    ps_weight : float = 20.0,
):
    ''' Process the results of the simulations for a given PS weight '''

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

    spawn_x_vals = [1.0]
    spawn_y_vals = [0.0]

    dst_tag             = 'active_body'
    stim_a_off          = 0.0
    vortex_mode_vals    = [0, 1]
    feedback_mode_vals  = [1]

    np_random_seed_vals = list(range(101, 101 + 50))

    # Processing
    use_actual_com_pos = True

    # Simulation tag
    sim_tag = _get_sim_tag(
        muscle_fn = muscle_fn,
        ps_weight = ps_weight
    )
    simulation_tag = (
        f'dynamic_water_vortices_free_head_muscle_{sim_tag}_coupled'
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
        simulation_tag      = simulation_tag,
        excluded_condition  = None,
    )

    return

def _repeat_difference_analysis(
    ps_weight_0 : float = 0.0,
    ps_weight_1 : float = 20.0,
):
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

    spawn_x_vals        = [1.0]
    spawn_y_vals        = [0.0]
    np_random_seed_vals = list(range(101, 101 + 50))

    dst_tag = 'active_body'

    # Root folder
    folder_root = _shorten_tag(leader_signal_str).strip('_')

    # Exclusion condition
    exclude_nothing_cond = lambda sx, sy, sd: False
    exclude_cond         = exclude_nothing_cond

    ####################
    # SAVE PERFORMANCE
    ####################

    save_folder_root = (
        f'{ANALYSIS_PATH}/'
        'comparing_vortices_coupled/comparisons/'
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
            (ps_weight_0, 'pure_flow'     ),
            (ps_weight_1, 'pure_flow'     ),
            (ps_weight_0, 'with_vortices' ),
            (ps_weight_1, 'with_vortices' ),
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
            (ps_weight_1, 'pure_flow'     ),
            (ps_weight_1, 'with_vortices' ),
        ],
        [
            (ps_weight_0, 'with_vortices' ),
            (ps_weight_1, 'with_vortices' ),
        ],
        [
            (ps_weight_0, 'pure_flow'     ),
            (ps_weight_0, 'with_vortices' ),
        ],
        [
            (ps_weight_0, 'pure_flow'     ),
            (ps_weight_1, 'pure_flow'     ),
        ],

    ]

    for (w_ps_0, vx_mode_0), (w_ps_1, vx_mode_1) in combinations:

        ps_mode_0 = 'open_loop' if w_ps_0 == 0.0 else 'closed_loop'
        ps_mode_1 = 'open_loop' if w_ps_1 == 0.0 else 'closed_loop'

        mode_0 = f'{ps_mode_0}_{vx_mode_0}'
        mode_1 = f'{ps_mode_1}_{vx_mode_1}'

        print('#' * 50)
        print(f'COMPARING {mode_0} VS {mode_1}')
        print('#' * 50)

        run_difference_analysis(
            ps_weight_0         = w_ps_0,
            vx_mode_0           = vx_mode_0,
            ps_weight_1         = w_ps_1,
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
            _repeat_post_processing( ps_weight = ps_weight )

    if repeat_difference_analysis:
        _repeat_difference_analysis(
            ps_weight_0 = 0.0,
            ps_weight_1 = 20.0,
        )


if __name__ == '__main__':
    main()
