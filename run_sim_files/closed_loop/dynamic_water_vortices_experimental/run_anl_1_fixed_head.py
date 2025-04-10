''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import analysis_runner

def get_process_tag(
    leader_signal_str: str,
    spawn_x          : float,
    spawn_y          : float,
    np_random_seed   : int,
    ps_tag           : str,
    **kwargs,
):
    ''' Get the simulation tag '''

    # Remove the date
    leader_signal_str = leader_signal_str.split('_', 1)[1]

    # Transform to strings
    spawn_x_str    = str(round( spawn_x * 100)).zfill(3)  # E.g. 0.86 -> 086BL
    spawn_y_str    = str(round( spawn_y * 100)).zfill(3)  # E.g. 0.00 -> 000BL

    process_tag = (
        f'{leader_signal_str}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    return process_tag

def run_analysis(
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : list[float],
    spawn_y_vals       : list[float],
    np_random_seed_vals   : list[int],

):
    ''' Run the analysis '''

    sim_settings_list = []

    for feedback_mode in feedback_mode_vals:

        if feedback_mode:
            ps_tag                = 'closed_loop'
            ps_min_activation_deg = 5.0
            ps_weight             = 20.0
        else:
            ps_tag                = 'open_loop'
            ps_min_activation_deg = 360.0
            ps_weight             = 0.0

        sim_settings_list_mode = [
            analysis_runner.get_sim_settings(
                # Fluid
                leader_signal_str = leader_signal_str,
                spawn_x           = spawn_x,
                spawn_y           = spawn_y,

                # Random seed
                np_random_seed = np_random_seed,

                # Feedback
                ps_weight      = ps_weight,
                ps_act_deg     = ps_min_activation_deg,

                # Process tag
                process_tag    = get_process_tag(
                    leader_signal_str = leader_signal_str,
                    spawn_x           = spawn_x,
                    spawn_y           = spawn_y,
                    np_random_seed    = np_random_seed,
                    ps_tag            = ps_tag,
                ),
            )
            for leader_signal_str in leader_signal_str_list
            for spawn_x           in spawn_x_vals
            for spawn_y        in spawn_y_vals
            for np_random_seed    in np_random_seed_vals
        ]

        sim_settings_list += sim_settings_list_mode

    # Run the analysis
    analysis_runner.run_analysis(sim_settings_list)

    return

def main():

    # Wait to run the simulation
    # import time
    # wait_time = 12 * 3600
    # print(f"Waiting for {wait_time / 3600} hours before starting the simulation")
    # time.sleep(wait_time)

    # STUDIED PARAMETERS
    feedback_mode_vals  = [1, 0]
    spawn_x_vals        = np.linspace(0.4, 1.6, 41)
    spawn_y_vals        = [-0.1, -0.0, -0.1]
    np_random_seed_vals = np.arange(100, 110)

    leader_signal_str_list = [
        '30s_continuous_036Hz_abdquickest_exp_11300_12400_scaled_025_large_grid_empirical_speed_2025-02-14T17:17:41.188478',
        '30s_continuous_036Hz_abdquickest_exp_11300_12400_scaled_025_large_grid_theoretical_speed_2025-02-15T01:09:19.294857',
    ]

    # (41 * 3 * 2 * 2) * 10 = ( 246 * 2 ) * 10 = ( 492 ) * 10 = 4920 simulations

    run_analysis(
        feedback_mode_vals     = feedback_mode_vals,
        leader_signal_str_list = leader_signal_str_list,
        spawn_x_vals           = spawn_x_vals,
        spawn_y_vals           = spawn_y_vals,
        np_random_seed_vals    = np_random_seed_vals,
    )

    return


if __name__ == '__main__':
    main()