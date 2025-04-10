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
    leader_name   : float,
    spawn_x       : float,
    spawn_y       : float,
    np_random_seed: int,
    ps_tag        : str,
    **kwargs,
):
    ''' Get the simulation tag '''

    spawn_x_str = str(round( spawn_x * 100)).zfill(3)  # E.g. 0.86 -> 086BL
    spawn_y_str = str(round( spawn_y * 100)).zfill(3)  # E.g. 0.00 -> 000BL

    process_tag = (
        f'{leader_name}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    return process_tag

def run_analysis(
    feedback_mode_vals : list[int],
    spawn_x_vals       : list[float],
    spawn_y_vals       : list[float],
    np_random_seed_vals: list[int],
    leader_pars_vals   : list[list[float, str]],
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
                leader_freq    = leader_pars[0],
                leader_name    = leader_pars[1],
                spawn_x        = spawn_x,
                spawn_y        = spawn_y,

                # Random seed
                np_random_seed = np_random_seed,

                # Feedback
                ps_weight      = ps_weight,
                ps_act_deg     = ps_min_activation_deg,

                # Process tag
                process_tag    = get_process_tag(
                    leader_name   = leader_pars[1],
                    spawn_x       = spawn_x,
                    spawn_y       = spawn_y,
                    np_random_seed= np_random_seed,
                    ps_tag        = ps_tag,
                ),
            )
            for leader_pars    in leader_pars_vals
            for spawn_x        in spawn_x_vals
            for spawn_y        in spawn_y_vals
            for np_random_seed in np_random_seed_vals
        ]

        sim_settings_list += sim_settings_list_mode

    # Run the analysis
    analysis_runner.run_analysis(sim_settings_list)

    return

def main():

    # Wait to run the simulation
    import time
    wait_time = 6 * 3600
    print(f"Waiting for {wait_time / 3600} hours before starting the simulation")
    time.sleep(wait_time)

    ##############################################################
    # GRID ANALYSIS ##############################################
    ##############################################################
    print('Running simulations with theoretical speed')

    feedback_mode_vals  = [1, 0]
    spawn_x_vals        = np.linspace(0.4, 1.6, 61)
    spawn_y_vals        = np.linspace(0.0, 0.5, 11)
    np_random_seed_vals = np.arange(100, 120)

    # 61 * 11 * 2 * 20 = (1342) * 20 = 26840 simulations

    leader_pars_vals = [
        [ 3.5, '30s_035Hz_implicit_large_grid_theoretical_speed' ],
    ]

    run_analysis(
        feedback_mode_vals  = feedback_mode_vals,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        leader_pars_vals    = leader_pars_vals,
    )

    return



if __name__ == '__main__':
    main()