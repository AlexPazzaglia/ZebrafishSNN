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
import matplotlib.pyplot as plt

import muscle_parameters_optimization_sim_params as opt_params

def load_optimization_results(
    optimization_path  : str,
    optimization_name  : str,
    starting_iteration : int,
    n_iterations       : int,
):
    ''' Load the optimization results '''

    # Scalings from optimization results
    folder_name = f'{optimization_path}/{optimization_name}'
    file_names  = [
        f'{folder_name}/performance_iteration_{iteration}.dill'
        for iteration in range(starting_iteration, starting_iteration + n_iterations)
    ]

    iterations_performance = [
        dill.load(open(file_name, 'rb'))
        for file_name in file_names
    ]

    # Extract the parameters from the optimization results
    WN_hat = [ performance['WN_hat'] for performance in iterations_performance ]
    WN_trg = [ performance['WN_target'] for performance in iterations_performance ]

    ZC_hat = [ performance['ZC_hat'] for performance in iterations_performance ]
    ZC_trg = [ performance['ZC_target'] for performance in iterations_performance ]

    gains_alpha = [ performance['gains_scalings_alpha'] for performance in iterations_performance ]
    gains_beta  = [ performance['gains_scalings_beta']  for performance in iterations_performance ]
    gains_delta = [ performance['gains_scalings_delta'] for performance in iterations_performance ]

    opt_results = {
        'WN_hat'            : WN_hat,
        'WN_target'         : WN_trg,
        'ZC_hat'            : ZC_hat,
        'ZC_target'         : ZC_trg,
        'gains_scalings_alpha' : gains_alpha,
        'gains_scalings_beta'  : gains_beta,
        'gains_scalings_delta' : gains_delta,
    }

    return opt_results

def _plot_opt_objectives(
    n_iterations       : int,
    n_joints_active    : int,
    WN_hat             : list,
    WN_trg             : list,
    ZC_hat             : list,
    ZC_trg             : list,
):
    ''' Plot the optimization objectives '''
    joint_colors = plt.cm.jet(np.linspace(0, 1, n_joints_active))

    # Plot WN and ZC
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for joint in range(n_joints_active):
        axes[0].plot(
            [ WN_hat[it][joint] - WN_trg[it][joint] for it in range(n_iterations) ],
            linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )
        axes[1].plot(
            [ ZC_hat[it][joint] - ZC_trg[it][joint] for it in range(n_iterations) ],
            linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('WN Error')
    axes[0].grid(which='both')
    axes[0].legend()

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('ZC Error')
    axes[1].grid(which='both')
    axes[1].legend()

    plt.tight_layout()
    return

def _plot_opt_gains(
    n_iterations       : int,
    n_joints_active    : int,
    gains_alpha        : list,
    gains_beta         : list,
    gains_delta        : list,
):
    ''' Plot the gains '''
    joint_colors = plt.cm.jet(np.linspace(0, 1, n_joints_active))

    # Plot the gains
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    for joint in range(n_joints_active):
        axes[0].plot(
            [ gains_alpha[it][joint] for it in range(n_iterations) ],
            linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )
        axes[1].plot(
            [ gains_beta[it][joint] for it in range(n_iterations) ],
            linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )
        axes[2].plot(
            [ gains_delta[it][joint] for it in range(n_iterations) ],
            linestyle= '--',
            color= joint_colors[joint],
            label= f'Joint {joint}'
        )

    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gains Alpha [log]')
    axes[0].grid(which='both')
    axes[0].legend()

    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Gains Beta [log]')
    axes[1].grid(which='both')
    axes[1].legend()

    axes[2].set_yscale('log')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Gains Delta [log]')
    axes[2].grid(which='both')
    axes[2].legend()

    return

def plot_optimization(
    optimization_path  : str,
    optimization_name  : str,
    starting_iteration = None,
    n_iterations       = None,
    n_joints_axis      = 15,
    n_joints_tail      = 2,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    if starting_iteration is None:
        starting_iteration = opt_params._get_first_iteration(
            optimization_path = optimization_path,
            optimization_name = optimization_name
        )

    if n_iterations is None:
        final_iteration = opt_params._get_last_iteration(
            optimization_path = optimization_path,
            optimization_name = optimization_name
        )
        n_iterations = final_iteration - starting_iteration + 1

    print(f'Starting iteration: {starting_iteration}')
    print(f'Number of iterations: {n_iterations}')

    n_joints_active = n_joints_axis - n_joints_tail

    # Load the optimization results
    opt_results = load_optimization_results(
        optimization_path  = optimization_path,
        optimization_name  = optimization_name,
        starting_iteration = starting_iteration,
        n_iterations       = n_iterations,
    )

    # PLOTTING
    _plot_opt_objectives(
        n_iterations    = n_iterations,
        n_joints_active = n_joints_active,
        WN_hat          = opt_results['WN_hat'],
        WN_trg          = opt_results['WN_target'],
        ZC_hat          = opt_results['ZC_hat'],
        ZC_trg          = opt_results['ZC_target'],
    )

    # Plot the gains
    _plot_opt_gains(
        n_iterations    = n_iterations,
        n_joints_active = n_joints_active,
        gains_alpha     = opt_results['gains_scalings_alpha'],
        gains_beta      = opt_results['gains_scalings_beta'],
        gains_delta     = opt_results['gains_scalings_delta'],
    )

    # plt.tight_layout()
    plt.show()
    return


def main():
    # TOPOLOGY
    n_joints_axis   = 15
    n_joints_tail   = 2

    # TARGET PARAMETERS
    target_g0 = 2*np.pi/ n_joints_axis
    target_zc = 1.00
    target_fn = 20.0

    # OPTIMIZATION
    starting_iteration = None
    n_iterations       = None

    optimization_path = opt_params.OPT_RESULTS_DIR
    optimization_name = opt_params.get_optimization_name(
        target_wn   = 2*np.pi *target_fn,
        target_zc   = target_zc,
        target_g0   = target_g0,
        name_prefix = 'muscle_parameters_optimization_all'
    )

    plot_optimization(
        optimization_path  = optimization_path,
        optimization_name  = optimization_name,
        starting_iteration = starting_iteration,
        n_iterations       = n_iterations,
        n_joints_axis      = n_joints_axis,
        n_joints_tail      = n_joints_tail,
    )


if __name__ == '__main__':
    main()