''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import re
import numpy as np
import sim_runner
import analysis_runner


def remove_date_from_string(s):
    return re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+', '', s)

def _shorten_tag(tag: str) -> str:
    ''' Shorten the tag '''

    tag = tag.replace('process_', '')
    tag = tag.replace('passive_body_', '')
    tag = tag.replace('continuous_', '')
    tag = tag.replace('bounded_NaN_NaN_Hz_', '')
    tag = remove_date_from_string(tag)
    tag = tag.replace('__', '_')

    return tag

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
        'passive_body_'
        f'{leader_signal_str}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    process_tag = _shorten_tag(process_tag)

    return process_tag

def run_analysis(
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : list[float],
    spawn_y_vals          : list[float],
    np_random_seed_vals   : list[int],
    stim_a_off            : float,
    simulation_tag        : str,
):
    ''' Run the analysis '''

    sim_settings_list = []

    for feedback_mode in feedback_mode_vals:

        ps_settings           = analysis_runner.get_ps_settings(feedback_mode, stim_a_off)
        ps_tag                = ps_settings['ps_tag']
        ps_min_activation_deg = ps_settings['ps_min_activation_deg']
        ps_weight             = ps_settings['ps_weight']

        delay_start = 5.0 # 5.0
        duration    = 20.0 + delay_start

        sim_settings_list_mode = [
            analysis_runner.get_sim_settings(

                sim_file_tag = simulation_tag,
                delay_start  = delay_start,
                duration     = duration,

                # Drive
                stim_a_off = stim_a_off,

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
            for spawn_y           in spawn_y_vals
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

    simulation_tag = 'dynamic_water_vortices_fixed_head_experimental_continuous'

    speed = 'fast' # theoretical intermediate fast
    amp   = 1.00   # 1.00 1.25

    leader_signal_str = sim_runner.get_water_dynamics_file_name(
        amp_scaling        = amp,            # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,           # [0.23,  0.25 ]
        integration_method = 'abdquickest',  # ['abdquickest',  'implicit' ]
        speed_type         = speed,          # ['empirical',  'theoretical', 'intermediate'] ]
        amp_type           = 'constant',     # [None, 'constant', 'modulated']
        step_bounds        = [12104, 12264], # [ [12104, 12264], 11300_12400 ]
        freq_bounds        = (None, None),   # [None, (2.5, 4.5)]
    )

    leader_signal_str_list = [
        leader_signal_str,
    ]

    spawn_x_vals = np.linspace(0.4, 1.6, 41)
    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)
    # spawn_y_vals = [-0.20, -0.15, -0.05, 0.05, 0.15, 0.20]
    # spawn_y_vals = [-0.10, 0.00, 0.10]

    # ACTIVE NETWORK
    stim_a_off          = 0.0
    feedback_mode_vals  = [1, 0]
    np_random_seed_vals = [100, 101, 102, 103, 104]

    # 41 * 6 * 2 = 492
    # 41 * 6 * 2 * 5 = 2460

    run_analysis(
        feedback_mode_vals     = feedback_mode_vals,
        leader_signal_str_list = leader_signal_str_list,
        spawn_x_vals           = spawn_x_vals,
        spawn_y_vals           = spawn_y_vals,
        np_random_seed_vals    = np_random_seed_vals,
        stim_a_off             = stim_a_off,
        simulation_tag         = simulation_tag,
    )

    # PASSIVE NETWORK
    stim_a_off          = -4.0
    feedback_mode_vals  = [0]
    np_random_seed_vals = [100]

    # 41 * 6 * 1 = 246

    run_analysis(
        feedback_mode_vals     = feedback_mode_vals,
        leader_signal_str_list = leader_signal_str_list,
        spawn_x_vals           = spawn_x_vals,
        spawn_y_vals           = spawn_y_vals,
        np_random_seed_vals    = np_random_seed_vals,
        stim_a_off             = stim_a_off,
        simulation_tag         = simulation_tag,
    )

    return


if __name__ == '__main__':
    main()