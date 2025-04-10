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
    leader_freq   : float,
    speed_mult_y  : float,
    spawn_x       : float,
    np_random_seed: int,
    ps_tag        : str,
):
    ''' Get the simulation tag '''

    leader_freq_str = str(round(leader_freq *  10)).zfill(3)  # E.g. 3.0  -> 030Hz
    speed_mult_str  = str(round(speed_mult_y * 10)).zfill(3)  # E.g. 1.5  -> 015x
    spawn_x_str     = str(round(spawn_x * 100)).zfill(3)      # E.g. 0.86 -> 086BL

    process_tag = (
        f'{leader_freq_str}Hz_at_{speed_mult_str}x_'
        f'distance_{spawn_x_str}BL_'
        f'seed_{np_random_seed}_{ps_tag}'
    )

    return process_tag

def run_analysis(
    feedback_mode_vals : list[int],
    leader_pars_vals   : list[list[float, str]],
    speed_mult_y_vals  : list[float],
    spawn_x_vals       : list[float],
    np_random_seed_vals: list[int],
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
                leader_freq  = leader_pars[0],
                leader_name  = leader_pars[1],
                speed_mult_y = speed_mult_y,
                spawn_x      = spawn_x,

                # Random seed
                np_random_seed = np_random_seed,

                # Feedback
                ps_weight      = ps_weight,
                ps_act_deg     = ps_min_activation_deg,

                # Process tag
                process_tag    = get_process_tag(
                    leader_freq   = leader_pars[0],
                    speed_mult_y  = speed_mult_y,
                    spawn_x       = spawn_x,
                    np_random_seed= np_random_seed,
                    ps_tag        = ps_tag,
                ),
            )
            for leader_pars    in leader_pars_vals
            for speed_mult_y   in speed_mult_y_vals
            for spawn_x        in spawn_x_vals
            for np_random_seed in np_random_seed_vals
        ]

        sim_settings_list += sim_settings_list_mode

    # Run the analysis
    analysis_runner.run_analysis(sim_settings_list)

    return



def main():

    # STUDIED PARAMETERS
    feedback_mode_vals  = [1, 0]
    speed_mult_y_vals   = [1.0, 1.5]
    spawn_x_vals        = np.linspace(0.5, 1.5, 51)
    np_random_seed_vals = np.arange(100, 120)

    leader_pars_vals = [
        [ 3.0, '3.0_small_grid'],
        [ 3.5, '3.5_small_grid'],
        [ 4.0, '4.0_small_grid'],
    ]

    run_analysis(
        feedback_mode_vals  = feedback_mode_vals,
        leader_pars_vals    = leader_pars_vals,
        speed_mult_y_vals   = speed_mult_y_vals,
        spawn_x_vals        = spawn_x_vals,
        np_random_seed_vals = np_random_seed_vals,
    )

    return


if __name__ == '__main__':
    main()