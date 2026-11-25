'''
Post-processing of the muscle stiffness sweep (Supplementary Material).

For every muscle natural frequency, the script loads the results collected by
move_results_muscle_stiffness_sweep.py and plots the phase-distance
distributions in closed and open loop, together with the performance
differences between the two conditions.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_paths import ANALYSIS_PATH

import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from run_sim_files.in_vivo_simulations.vortex_matching.sim_runner import get_water_dynamics_file_name
from run_sim_files.in_vivo_simulations.vortex_matching.run_muscle_stiffness_sweep import MUSCLE_TAGS

from scipy.stats import ttest_rel, wilcoxon

import muscle_sweep_data_plotting as data_plotting

def sequential_processing(
    results_root_folder   : str,
    muscle_tag            : str,
    stim_a_off            : float,
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : np.ndarray,
    spawn_y_vals          : np.ndarray,
    np_random_seed_vals   : np.ndarray,
    use_actual_com_pos    : bool,
    save                  : bool = True,
    show                  : bool = False,
):
    ''' Run the sequential processing '''

    for feedback_mode in feedback_mode_vals:
        for leader_signal_str in leader_signal_str_list:
            data_plotting.plot_parameters_combination(
                results_root_folder = results_root_folder,
                muscle_tag          = muscle_tag,
                stim_a_off          = stim_a_off,
                feedback_mode       = feedback_mode,
                leader_signal_str   = leader_signal_str,
                spawn_x_vals        = spawn_x_vals,
                spawn_y_vals        = spawn_y_vals,
                np_random_seed_vals = np_random_seed_vals,
                use_actual_com_pos  = use_actual_com_pos,
                save                = save,
                show                = show,
            )

    return

def parallel_processing(
    results_root_folder   : str,
    muscle_tag            : str,
    stim_a_off            : float,
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : np.ndarray,
    spawn_y_vals          : np.ndarray,
    np_random_seed_vals   : np.ndarray,
    use_actual_com_pos    : bool,
    save                  : bool = True,
    show                  : bool = False,
):
    ''' Run the parallel processing '''

    pool = multiprocessing.Pool()
    tasks = [
        (
            results_root_folder,        # Results root folder
            muscle_tag,                 # Muscle tag
            stim_a_off,                 # Stimulus amplitude offset
            feedback_mode,              # Feedback mode
            leader_signal_str,          # Leader signal string
            spawn_x_vals,               # Spawn x values
            spawn_y_vals,               # Spawn y values
            np_random_seed_vals,        # Random seed values
            use_actual_com_pos,         # Use actual center of mass position
            save,                       # Save
            show,                       # Show
        )
        for feedback_mode in feedback_mode_vals
        for leader_signal_str in leader_signal_str_list
    ]

    pool = multiprocessing.Pool(processes=min(5, len(tasks)))
    pool.starmap(data_plotting.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def _plot_violinplot(
    quantity_values_ol : np.ndarray,
    quantity_values_cl : np.ndarray,
    target_quantity    : str,
    save_folder        : str,
    folder_root        : str,
):
    ''' Plot a paired boxplot for the given quantities '''

    df = pd.DataFrame({
        'Open Loop'  : quantity_values_ol,
        'Closed Loop': quantity_values_cl
    })
    df_long = df.melt(var_name='Mode', value_name='Value')
    df_long['Index'] = np.tile(np.arange(len(quantity_values_ol)), 2)

    # Plot
    fig = plt.figure(f'Paired Violinplot: {target_quantity}', figsize=(8, 6))
    sns.violinplot(data=df_long, x='Mode', y='Value', inner=None, color='lightgray')
    sns.boxplot(data=df_long, x='Mode', y='Value', width=0.2, showcaps=True, boxprops={'zorder': 2})

    # Draw paired lines
    # Randomly sample 100 indices
    num_samples    = min(100, len(quantity_values_ol))
    sample_indices = np.random.choice(len(quantity_values_ol), num_samples, replace=False)

    # Draw lines for the sampled indices
    for i in sample_indices:
        plt.plot(
            ['Open Loop', 'Closed Loop'],
            [quantity_values_ol[i], quantity_values_cl[i]],
            color='gray', alpha=0.5, linewidth=1
        )

    plt.title(f'Paired Comparison: {target_quantity}')
    plt.tight_layout()

    # Save the figure
    diff_folder = f'{save_folder}/{folder_root}_DIFFERENCE'
    os.makedirs(diff_folder, exist_ok=True)

    fig.savefig(
        f'{diff_folder}/{target_quantity}_violinplot.pdf',
        format = 'pdf',
    )

    return

def _plot_difference_distribution(
    quantity_values_ol : np.ndarray,
    quantity_values_cl : np.ndarray,
    target_quantity    : str,
    save_folder        : str,
    folder_root        : str,
    relative_difference: bool = False,
):
    ''' Plot the distribution of paired differences (CL - OL) '''

    # Compute paired differences
    tag         = 'Absolute Difference'
    differences = quantity_values_cl - quantity_values_ol
    if relative_difference:
        tag         = 'Relative Difference (%)'
        differences = differences / np.abs(quantity_values_ol) * 100.0

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
    diff_folder = f'{save_folder}/{folder_root}_DIFFERENCE'
    os.makedirs(diff_folder, exist_ok=True)

    fig.savefig(
        f'{diff_folder}/{target_quantity}_difference_distribution.pdf',
        format='pdf',
    )
    return

def _plot_contourf(
    spawn_x_vals   : np.ndarray,
    spawn_y_vals   : np.ndarray,
    diff_values    : np.ndarray,
    target_quantity: str,
    diff_type      : str,
    save_folder    : str,
    folder_root    : str,
    x_range        : tuple[float, float],
    y_range        : tuple[float, float],
):

    fig = plt.figure(f'DIFFERENCE Contourf: {target_quantity}', figsize=(8, 6))

    # Mean and std across seeds
    mean_diff_values   = np.mean(diff_values, axis=2)
    std_diff_values    = np.std(diff_values, axis=2)
    max_abs_diff_value = np.max(np.abs(mean_diff_values))

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
    diff_folder = f'{save_folder}/{folder_root}_DIFFERENCE'
    os.makedirs(diff_folder, exist_ok=True)

    fig.savefig(
        f'{diff_folder}/{target_quantity}_contourf.pdf',
        format = 'pdf',
    )
    return

def plot_performance_difference(
    spawn_x_vals: np.ndarray,
    spawn_y_vals: np.ndarray,
    seed_vals   : np.ndarray,
    save_folder : str,
    folder_root : str,
):
    ''' Plot the performance data '''

    # Parameters
    n_spawn_x = len(spawn_x_vals)
    n_spawn_y = len(spawn_y_vals)
    n_seeds   = len(seed_vals)
    x_range   = (0.4, 1.6)
    y_range   = (
        np.amin( spawn_y_vals ),
        np.amax( spawn_y_vals ),
    )

    target_quantities = [
        [ 'mech_torque_total',     'rel' ],
        [ 'mech_energy_total',     'rel' ],
        [ 'mech_power_total',      'rel' ],
        [ 'mech_tail_beat_amp_bl', 'rel' ],
        [ 'coherence_pos_vel',     'abs' ],
        [ 'coherence_vel_vel',     'abs' ],
        [ 'coherence_acc_vel',     'abs' ],
        [ 'propulsion_fx',         'rel' ],
    ]

    # Load performance data
    # NOTE: Was saved as performance_df.to_csv(f'{self.save_folder}/{self.save_file_root}_performance.csv', index=False)
    file_ol = f'{save_folder}/{folder_root}_open_loop/{folder_root}_open_loop_performance.csv'
    file_cl = f'{save_folder}/{folder_root}_closed_loop/{folder_root}_closed_loop_performance.csv'

    performance_data_ol = pd.read_csv(file_ol)
    performance_data_cl = pd.read_csv(file_cl)

    # Plot the performance difference
    for target_quantity, diff_type in target_quantities:

        # Get values
        quantity_values_ol = np.array(performance_data_ol[target_quantity])
        quantity_values_cl = np.array(performance_data_cl[target_quantity])

        # Compute the difference
        diff =  ( quantity_values_cl - quantity_values_ol )

        if diff_type == 'abs':
            diff_values_flat = diff

        if diff_type == 'rel':
            diff_values_flat = diff / np.abs(quantity_values_ol) * 100.0

        diff_values = diff_values_flat.reshape(
            (n_spawn_x, n_spawn_y, n_seeds)
        )

        # Run statistical tests
        t_stat, p_val_t = ttest_rel(
            quantity_values_cl,
            quantity_values_ol,
            alternative='two-sided'
        )
        w_stat, p_val_w = wilcoxon(
            quantity_values_cl,
            quantity_values_ol,
            alternative='two-sided'
        )

        t_significance = p_val_t < 0.05
        w_significance = p_val_w < 0.05

        print(
            f'{target_quantity} {diff_type.upper()} difference: \n'
            f't-statistic = {t_stat:.3f}, p-value (t-test)   = {p_val_t:.3e}, significant = {t_significance}, \n'
            f'w-statistic = {w_stat:.3f}, p-value (Wilcoxon) = {p_val_w:.3e}, significant = {w_significance}  \n'
        )


        # Paired boxplot
        _plot_violinplot(
            quantity_values_ol = quantity_values_ol,
            quantity_values_cl = quantity_values_cl,
            target_quantity    = target_quantity,
            save_folder        = save_folder,
            folder_root        = folder_root,
        )

        # Difference distribution
        _plot_difference_distribution(
            quantity_values_ol = quantity_values_ol,
            quantity_values_cl = quantity_values_cl,
            target_quantity    = target_quantity,
            save_folder        = save_folder,
            folder_root        = folder_root,
            relative_difference= (diff_type == 'rel'),
        )

        # Plot contourf
        _plot_contourf(
            spawn_x_vals   = spawn_x_vals,
            spawn_y_vals   = spawn_y_vals,
            diff_values    = diff_values,
            target_quantity= target_quantity,
            diff_type      = diff_type,
            save_folder    = save_folder,
            folder_root    = folder_root,
            x_range        = x_range,
            y_range        = y_range,
        )

        plt.close('all')

    return

def muscle_analysis(
    muscle_tag   : str,
    analysis_type: str = 'active',
):

    # Studied parameters
    # 'intermediate', 'empirical', 'theoretical', 'fast'

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

    leader_signal_str_list = [
        leader_signal_str,
    ]

    spawn_x_vals = np.linspace(0.4, 1.6, 41)

    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)

    varying_parameters = {
        # ACTIVE NETWORK
        'active': {
            'dst_tag'             : 'active_body',
            'stim_a_off'          : 0.0,
            'feedback_mode_vals'  : [1, 0],
            'np_random_seed_vals' : [100, 101, 102, 103, 104],
        },

        # PASSIVE NETWORK
        'passive': {
            'dst_tag'             : 'passive_body',
            'stim_a_off'          : -4.0,
            'feedback_mode_vals'  : [0],
            'np_random_seed_vals' : [100],
        }
    }


    dst_tag             = varying_parameters[analysis_type]['dst_tag']
    stim_a_off          = varying_parameters[analysis_type]['stim_a_off']
    feedback_mode_vals  = varying_parameters[analysis_type]['feedback_mode_vals']
    np_random_seed_vals = varying_parameters[analysis_type]['np_random_seed_vals']

    # Processing
    use_actual_com_pos = True

    parallel = False
    save     = True
    show     = False

    # Common
    results_root_folder = (
        f'{ANALYSIS_PATH}/'
        'comparing_muscles/'
        f'schooling_experimental_xy_grid_free_head_{speed}_speed_continuous_'
        f'FN_{muscle_tag}_'
        f'{dst_tag}'
    )

    # Processing
    processing_kwargs = {
        'results_root_folder'   : results_root_folder,
        'muscle_tag'            : muscle_tag,
        'stim_a_off'            : stim_a_off,
        'feedback_mode_vals'    : feedback_mode_vals,
        'leader_signal_str_list': leader_signal_str_list,
        'spawn_x_vals'          : spawn_x_vals,
        'spawn_y_vals'          : spawn_y_vals,
        'np_random_seed_vals'   : np_random_seed_vals,

        'use_actual_com_pos' : use_actual_com_pos,
        'save'               : save,
        'show'               : show,
    }

    if parallel:
        parallel_processing(**processing_kwargs)
    else:
        sequential_processing(**processing_kwargs)


    if (
        analysis_type != 'active' or
        0 not in feedback_mode_vals or
        1 not in feedback_mode_vals
    ):
        return

    # Close open figures
    plt.close('all')

    # Plot performance difference
    plot_performance_difference(
        spawn_x_vals = spawn_x_vals,
        spawn_y_vals = spawn_y_vals,
        seed_vals    = np_random_seed_vals,
        save_folder  = f'{results_root_folder}/figures',
        folder_root  = 'amp_100_constant_034Hz_abdquickest_exp_12104_12264_scaled_025_fast_speed_sketch_body',
    )

    # Close all figures
    plt.close('all')

    return

def main():
    ''' Run the post processing for all the muscle natural frequencies '''

    for muscle_tag in MUSCLE_TAGS:
        print(f'RUNNING ANALYSIS FOR MUSCLE TAG: {muscle_tag}')
        muscle_analysis(muscle_tag, analysis_type='active')

    return

if __name__ == '__main__':
    main()
