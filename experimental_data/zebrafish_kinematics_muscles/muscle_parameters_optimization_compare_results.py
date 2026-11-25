''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import muscle_parameters_optimization_sim_params as opt_params

def _load_optimization_results_single(
    optimization_path  : str,
    optimization_name  : str,
):
    ''' Load the optimization results '''

    # Get the last iteration
    target_iteration = opt_params._get_last_iteration(
        optimization_path = optimization_path,
        optimization_name = optimization_name
    )

    # Scalings from optimization results
    folder_name = f'{optimization_path}/{optimization_name}'
    file_name   = f'{folder_name}/performance_iteration_{target_iteration}.dill'
    performance = dill.load(open(file_name, 'rb'))

    return performance

def load_optimization_results_all(
    targets_list: list,
    name_prefix : str,
):
    ''' Load all optimization results '''
    # Load all optimization results
    opt_results_list = []

    for (target_g0, target_zc, target_fn) in targets_list:

        optimization_path = opt_params.OPT_RESULTS_DIR
        optimization_name = opt_params.get_optimization_name(
            target_wn   = 2*np.pi *target_fn,
            target_zc   = target_zc,
            target_g0   = target_g0,
            name_prefix = name_prefix
        )

        # Load the optimization results
        opt_results = _load_optimization_results_single(
            optimization_path  = optimization_path,
            optimization_name  = optimization_name,
        )
        opt_results['target_g0'] = target_g0
        opt_results['target_zc'] = target_zc
        opt_results['target_fn'] = target_fn
        opt_results_list.append( opt_results )

    # Convert to pandas DataFrame
    opt_results_df = pd.DataFrame(opt_results_list)

    return opt_results_df

def _get_parameter_across_joints(
    opt_results_df : pd.DataFrame,
    n_joints_active : int,
    parameter_name  : str,
):
    ''' Get parameter across joints '''
    parameter_values = np.array(
        [
            [ alpha_vals[joint_ind] for joint_ind in range(n_joints_active) ]
            for alpha_vals in opt_results_df[parameter_name]
        ]
    )
    return parameter_values


def _plot_opt_gains(
    opt_results_df  : pd.DataFrame,
    target_metric   : str,
    n_joints_active : int,
):
    ''' Plot the gains '''

    metric_vals  = opt_results_df[target_metric].values
    joint_colors = plt.cm.jet(np.linspace(0, 1, n_joints_active))

    # Get gains across joints
    alpha_vals = _get_parameter_across_joints(
        opt_results_df  = opt_results_df,
        n_joints_active = n_joints_active,
        parameter_name  = 'alpha',
    )
    beta_vals = _get_parameter_across_joints(
        opt_results_df  = opt_results_df,
        n_joints_active = n_joints_active,
        parameter_name  = 'beta',
    )
    delta_vals = _get_parameter_across_joints(
        opt_results_df  = opt_results_df,
        n_joints_active = n_joints_active,
        parameter_name  = 'delta',
    )

    # Plot the gains
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for joint in range(n_joints_active):
        axes[0].plot(
            metric_vals,
            alpha_vals[:, joint],
            linestyle = '--',
            marker    = 'o',
            color     = joint_colors[joint],
            label     = f'Joint {joint}'
        )
        axes[1].plot(
            metric_vals,
            beta_vals[:, joint],
            linestyle = '--',
            marker    = 'o',
            color     = joint_colors[joint],
            label     = f'Joint {joint}'
        )
        axes[2].plot(
            metric_vals,
            delta_vals[:, joint],
            linestyle = '--',
            marker    = 'o',
            color     = joint_colors[joint],
            label     = f'Joint {joint}'
        )

    label_size  = 16
    tick_size   = 14
    legend_size = 8

    axes[0].set_yscale('log')
    axes[0].set_xlabel(target_metric, fontsize=label_size)
    axes[0].set_ylabel('Alpha [log]', fontsize=label_size)
    axes[0].tick_params(axis='both', labelsize=tick_size)
    axes[0].grid(which='both')
    axes[0].legend(fontsize=legend_size)

    axes[1].set_yscale('log')
    axes[1].set_xlabel(target_metric, fontsize=label_size)
    axes[1].set_ylabel('Beta [log]', fontsize=label_size)
    axes[1].tick_params(axis='both', labelsize=tick_size)
    axes[1].grid(which='both')
    axes[1].legend(fontsize=legend_size)

    axes[2].set_yscale('log')
    axes[2].set_xlabel(target_metric, fontsize=label_size)
    axes[2].set_ylabel('Delta [log]', fontsize=label_size)
    axes[2].tick_params(axis='both', labelsize=tick_size)
    axes[2].grid(which='both')
    axes[2].legend(fontsize=legend_size)

    return

def plot_optimization(
    name_prefix   : str,
    target_g0_list: list,
    target_zc_list: list,
    target_fn_list: list,
    n_joints_axis : int = 15,
    n_joints_tail : int = 2,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Targets list
    targets_list = [
        (target_g0, target_zc, target_fn)
        for target_g0 in target_g0_list
        for target_zc in target_zc_list
        for target_fn in target_fn_list
    ]

    # Load all optimization results
    opt_results_df = load_optimization_results_all(
        targets_list = targets_list,
        name_prefix  = name_prefix
    )

    alpha_arr = np.array(
        [
            [
                alpha_vals[joint_ind]
                for joint_ind in range(n_joints_axis - n_joints_tail)
            ]
            for alpha_vals in opt_results_df['alpha']
        ]
    )

    # Plot the gains
    _plot_opt_gains(
        opt_results_df  = opt_results_df,
        target_metric   = 'target_fn',
        n_joints_active = n_joints_axis - n_joints_tail,
    )

    # plt.tight_layout()
    plt.show()
    return


def main():
    # TOPOLOGY
    n_joints_axis   = 15
    n_joints_tail   = 2

    # TARGET PARAMETERS
    name_prefix    = 'muscle_parameters_optimization_all'
    target_g0_list = np.array( [2*np.pi] ) / n_joints_axis
    target_zc_list = np.array( [1.00] )
    target_fn_list = np.concatenate(
        (
            np.arange( 4.0, 10.0, 0.25,),
            np.arange(10.0, 20.0, 0.50,),
            [25.0, 50.0, 75.0, 100.0],
        )
    )

    # OPTIMIZATION
    plot_optimization(
        name_prefix    = name_prefix,
        target_g0_list = target_g0_list,
        target_zc_list = target_zc_list,
        target_fn_list = target_fn_list,
        n_joints_axis  = n_joints_axis,
        n_joints_tail  = n_joints_tail,
    )



if __name__ == '__main__':
    main()