import os
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COM_POS_REL = 0.4321805865303308

###############################################################################
# PLOTTING ####################################################################
###############################################################################

def _decorate_plot(
    axis       : plt.Axes,
):
    ''' Decorate the plot '''

    # Add dashed lines at multiples of pi
    for i in range(-1, 6):
        plt.axhline(i * np.pi, color='white', linestyle='--', linewidth=0.8)

    # Put ylabels and ticks at multiples of pi
    axis.set_yticks(np.arange(-np.pi, 5*np.pi+1, np.pi))
    axis.set_yticklabels(
        [
            r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'
        ]
    )

    # Label axes
    axis.set_xlabel("FB distance (bl)")
    axis.set_ylabel("Phase difference $\\Delta \\Phi$")
    axis.set_title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$")

    # Set limits
    axis.set_xlim(0.4, 1.6)
    axis.set_ylim(-np.pi, 5*np.pi)

    return

def plot_experimental_density_map_1(
    fb_distance        : np.ndarray,
    joints_phases_array: np.ndarray,
):
    ''' Plot the experimental density map '''

    fig_com_vs_angle = plt.figure('COM Position vs Joint Phase Difference 1', figsize=(8, 6))
    axis             = fig_com_vs_angle.add_subplot(111)

    # Create 2D histogram (density map)
    bins_fb           = np.linspace(0.4, 1.6, 50)  # Bins for fb_distance
    bins_phase        = np.linspace(-np.pi, 5 * np.pi, 50)  # Bins for joints_phases_array
    H, xedges, yedges = np.histogram2d(fb_distance, joints_phases_array, bins=[bins_fb, bins_phase])

    # Transpose H for correct orientation in imshow
    H = H.T

    # Plot the density map
    plt.imshow(
        H,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        aspect='auto',
        cmap=plt.cm.hot
    )

    # Decorate
    _decorate_plot(axis)

    # Add colorbar
    plt.colorbar(label="Density")

    plt.tight_layout()

    return

def plot_experimental_density_map_2(
    fb_distance        : np.ndarray,
    joints_phases_array: np.ndarray,
):
    ''' Plot the experimental density map '''

    fig_com_vs_angle = plt.figure('COM Position vs Joint Phase Difference 2', figsize=(8, 6))
    axis             = fig_com_vs_angle.add_subplot(111)

    # Plot hexbin
    hb = axis.hexbin(
        fb_distance,
        joints_phases_array,
        gridsize = 50,
        cmap     = plt.cm.hot,                       #'inferno'
        extent   = (0.4, 1.6, -np.pi, 5*np.pi),
    )

    # Decorate
    _decorate_plot(axis)

    plt.colorbar(hb, ax = axis)

    return

###############################################################################
# MAIN ########################################################################
###############################################################################
def plot_results():

    target_folder = (
        'simulation_results_test/data/'
        'net_farms_zebrafish_dynamic_water_vortices_closed_loop_fixed_head_100_SIM/'
        'process_0/run_0/farms/com_vs_angle_data_030Hz_open_loop'
    )

    # NOTE: Data was saved in the following format:
    #     current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    #     file_name    = f'com_vs_angle_data_{current_time}.csv'

    #     data = pd.DataFrame(
    #         {
    #             'times'              : times,
    #             'joint_positions'    : joint_positions,
    #             'joint_positions_ref': joint_positions_ref,
    #             'joint_phases'       : joint_phases,
    #             'joint_phases_ref'   : joint_phases_ref,
    #             'com_positions'      : com_positions,
    #             'com_positions_ref'  : com_positions_ref,
    #             'com_positions_diff' : com_positions_diff,
    #             'joint_phases_diff'  : joint_phases_diff,
    #         }
    #     )
    #     data.to_csv(f'{self.results_data_folder}/{file_name}')

    # Load the data (all csv files)
    data = [
        pd.read_csv(f'{target_folder}/{file}')
        for file in os.listdir(target_folder)
        if file.endswith('.csv')
    ]

    n_files = len(data)

    # Parameters
    timestep     = data[0].times[1] - data[0].times[0]
    n_iterations = len(data[0].times)
    times        = np.arange(0, timestep*n_iterations, timestep)

    discard_time = 5.0
    discard_ratio = discard_time / times[-1]

    n_iter_cons = round(n_iterations * (1 - discard_ratio))


    ########################################################
    # COM vs Joint Phase ###################################
    ########################################################

    # concatenate 3 copies of COM positions
    com_positions_stack = np.concatenate(
        [
            np.concatenate(
                [
                    com_positions_diff[-n_iter_cons:] - COM_POS_REL,
                    com_positions_diff[-n_iter_cons:] - COM_POS_REL,
                    com_positions_diff[-n_iter_cons:] - COM_POS_REL,
                ]
            )
            for com_positions_diff in [data[i].com_positions_diff for i in range(n_files)]
        ]
    )

    joints_pjases_stack = np.concatenate(
        [
            np.concatenate(
                [
                    joint_phases_diff[-n_iter_cons:],
                    joint_phases_diff[-n_iter_cons:] + 2*np.pi,
                    joint_phases_diff[-n_iter_cons:] + 4*np.pi,
                ]
            )
            for joint_phases_diff in [data[i].joint_phases_diff for i in range(n_files)]
        ]
    )

    # Plot the experimental density map
    plot_experimental_density_map_1(
        fb_distance         = com_positions_stack,
        joints_phases_array = joints_pjases_stack,
    )

    plot_experimental_density_map_2(
        fb_distance         = com_positions_stack,
        joints_phases_array = joints_pjases_stack,
    )

    plt.show()


if __name__ == '__main__':
    plot_results()