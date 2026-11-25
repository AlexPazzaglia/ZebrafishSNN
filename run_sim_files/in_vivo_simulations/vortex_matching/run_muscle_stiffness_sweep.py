'''
Vortex phase matching for different muscle stiffness (Supplementary Material).

The vortex phase matching analysis is repeated for different natural
frequencies FN of the body muscles, from 2.5 Hz (very compliant body) to
15.0 Hz (very stiff body). The damping ratio and the zero-frequency gain are
kept constant, so that the muscle parameters scale with FN.

For each FN, the follower is placed on the spatial grid behind the leader
(0.40 to 1.60 BL longitudinally, -0.20 to +0.20 BL laterally), held by a
spring-damper acting on its third link, and simulated
    - with an active network (5 seeds) in closed and open loop
    - with a passive body (silenced network, 1 seed)

Pipeline:
    1. run_muscle_stiffness_sweep.py                  (this script)
    2. move_results_muscle_stiffness_sweep.py         (collect the results)
    3. post_processing/plot_muscle_stiffness_sweep.py (figures and statistics)
'''

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

from farms_core.model.options import SpawnMode

import network_experiments.default_parameters.zebrafish.closed_loop.default as default

# Natural frequency of the muscles (mHz), from 2.5 Hz to 15.0 Hz
MUSCLE_TAGS = [
    '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
    '6500', '7000', '7500', '8000', '8500', '9000', '9500', '15000',
]

def _get_spring_damper_options():
    ''' Get the spring damper options '''

    # Parameters
    body_length = default.LENGTH_AXIS_MODEL
    body_mass   = 5.022324522733332e-05

    target_link    = 2

    spring_factor  = np.array( [ 50.0, 10.0, 0.0] ) # 10.0 10.0 0.0
    damping_factor = np.array( [  1.0,  0.0, 0.0] )

    inactive_band  = np.array( [ 0.0,  0.0, 0.0] )

    delta_x_bl     = 0.00
    delta_y_bl     = 0.00
    delta_z_bl     = 0.00

    act_on_x       = True
    act_on_y       = True
    act_on_z       = False

    # Spring constant
    # Reference: x = 1BL --> a = 1 BL/s^2
    target_pos = 1.0 * body_length
    target_acc = 1.0 * body_length

    spring_constant  = ( body_mass * target_acc ) / target_pos * spring_factor

    # Damping constant
    # Reference: v = 1BL/s --> a = 1 BL/s^2
    target_vel = 1.0 * body_length
    target_acc = 1.0 * body_length

    damping_constant = ( body_mass * target_acc ) / target_vel * damping_factor

    # Position of the target link COM
    target_link_p0  : float = default.POINTS_POSITIONS_MODEL[target_link]
    target_link_p1  : float = default.POINTS_POSITIONS_MODEL[target_link + 1]
    target_link_pos : float = ( target_link_p0 + target_link_p1 ) / 2.0

    # Fixed point
    target_x = delta_x_bl * body_length - target_link_pos
    target_y = delta_y_bl * body_length
    target_z = delta_z_bl * body_length

    fixed_point = np.array(
        [
            target_x if act_on_x else np.nan,
            target_y if act_on_y else np.nan,
            target_z if act_on_z else np.nan,
        ]
    )

    # Inactive band
    inactive_band = inactive_band * body_length

    # Spring damper options
    spring_damper_options = {
        'spring_constant' : spring_constant,
        'damping_constant': damping_constant,
        'fixed_point'     : fixed_point,
        'target_link'     : target_link,
        'rest_length'     : 0.0,
        'inactive_band'   : inactive_band,
    }

    return spring_damper_options

def get_ps_settings(
    feedback_mode: int,
    stim_a_off   : float = 0.0,
) -> str:
    ''' Get the ps tag '''
    if stim_a_off <= -3.5:
        return {
            'ps_tag'                : 'passive_body',
            'ps_min_activation_deg' : 360.0,
            'ps_weight'             : 0.0,
        }
    if feedback_mode:
        return {
            'ps_tag'                : 'closed_loop',
            'ps_min_activation_deg' : 5.0,
            'ps_weight'             : 15.0,
        }

    return {
        'ps_tag'                : 'open_loop',
        'ps_min_activation_deg' : 360.0,
        'ps_weight'             : 0.0,
    }

def remove_date_from_string(s):
    return re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+', '', s)

def _shorten_tag(tag: str) -> str:
    ''' Shorten the tag '''

    tag = tag.replace('process_', '')
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
        'free_body_'
        f'{leader_signal_str}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    process_tag = _shorten_tag(process_tag)

    return process_tag

def _run_analysis(
    muscle_tag            : str,
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

    # Weights
    net_weights = {
        'ex2ex_weight': 3.83037,
        'ex2in_weight': 49.47920,
        'in2ex_weight': 0.84541,
        'in2in_weight': 0.10330,
        'rs2ex_weight': 8.74440,
        'rs2in_weight': 3.28338,

        # Non-saturated MN activation
        'ex2mn_weight': 5.0,
        'in2mn_weight': 0.0,
        'mn2mc_weight': 5.0,
    }

    # Muscles
    pre_tag = 'new'
    muscle_resonant_freq = None

    if int(muscle_tag) <= 3500:
        # Super compliant muscles
        gen_str = 'gen_10'
    elif int(muscle_tag) == 4000:
        # Very compliant muscles
        gen_str = 'gen_20'
    elif int(muscle_tag) <= 4750:
        # Compliant muscles
        gen_str = 'gen_50'

    elif int(muscle_tag) <= 7250:
        # Intermediate muscles
        gen_str = 'gen_100'

    elif int(muscle_tag) <= 9750:
        # Stiff muscles
        gen_str = 'gen_75'

    elif int(muscle_tag) <= 20_000:
        # Very stiff muscles
        pre_tag = 'all'
        gen_str = 'gen_50'

    elif int(muscle_tag) > 20_000:
        # Crazy stiff muscles
        pre_tag = 'all'
        gen_str = 'gen_100'

    else:
        raise ValueError(f'Unknown muscle tag: {muscle_tag}')

    muscle_tag = f'{pre_tag}_FN_{muscle_tag}_ZC_1000_G0_419_{gen_str}'

    # Collect simulation settings
    for feedback_mode in feedback_mode_vals:

        ps_settings           = get_ps_settings(feedback_mode, stim_a_off)
        ps_tag                = ps_settings['ps_tag']
        ps_min_activation_deg = ps_settings['ps_min_activation_deg']
        ps_weight             = ps_settings['ps_weight']

        delay_start = 5.0 # 5.0
        duration    = 20.0 + delay_start

        sim_settings_list_mode = [
            analysis_runner.get_sim_settings(

                # Simulation
                sim_file_tag = simulation_tag,
                delay_start  = delay_start,
                duration     = duration,

                # Drive
                stim_a_off = stim_a_off,

                # Constraints
                sim_spawn_mode = SpawnMode.TRANSVERSE,

                # Spring damper options
                spring_damper_options = _get_spring_damper_options(),

                # Fluid
                leader_signal_str = leader_signal_str,
                spawn_x           = spawn_x,
                spawn_y           = spawn_y,

                # Random seed
                np_random_seed = np_random_seed,

                # Weights
                net_weights    = net_weights,

                # Muscles
                muscle_tag           = muscle_tag,
                muscle_resonant_freq = muscle_resonant_freq,

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

def muscle_analysis(
    muscle_tag : str,
):
    ''' Run the simulations on the spatial grid for a given muscle FN (mHz) '''


    simulation_tag = f'dynamic_water_vortices_free_head_muscle_FN_{muscle_tag}'

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
        matching_string    = 'sketch_body',  # ['sketch_body', None]
    )

    leader_signal_str_list = [
        leader_signal_str,
    ]

    spawn_x_vals = np.linspace(0.4, 1.6, 41)
    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)

    # ACTIVE NETWORK
    stim_a_off          = 0.0
    feedback_mode_vals  = [1, 0]
    np_random_seed_vals = [100, 101, 102, 103, 104]


    _run_analysis(
        muscle_tag             = muscle_tag,
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


    _run_analysis(
        muscle_tag             = muscle_tag,
        feedback_mode_vals     = feedback_mode_vals,
        leader_signal_str_list = leader_signal_str_list,
        spawn_x_vals           = spawn_x_vals,
        spawn_y_vals           = spawn_y_vals,
        np_random_seed_vals    = np_random_seed_vals,
        stim_a_off             = stim_a_off,
        simulation_tag         = simulation_tag,
    )

    return


def main():
    ''' Main function to run the analysis '''

    # Run the analysis
    for muscle_tag in MUSCLE_TAGS:
        print(f'RUNNING ANALYSIS FOR MUSCLE TAG: {muscle_tag}')
        muscle_analysis(muscle_tag)

    return


if __name__ == '__main__':
    main()