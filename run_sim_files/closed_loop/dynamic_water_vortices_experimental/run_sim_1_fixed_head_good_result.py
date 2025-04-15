''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from farms_core.model.options import SpawnMode
from sim_runner import run_simulation, get_water_dynamics_file_name

def _get_pars(
    debug        = False,
    in_line      = True,
    passive_body = False,
    open_loop    = False,
):
    ''' Get parameters for the simulation '''

    pars = {}

    # DEBUG
    if debug:
        delay_start = 0.0
        duration    = 10.0 + delay_start
    else:
        delay_start = 5.0
        duration    = 20.0 + delay_start # 30.0

    pars.update({'delay_start': delay_start, 'duration': duration})

    # LEADER-FOLLOWER
    if in_line:
        spawn_x = 1.05 # 1.00
        spawn_y = 0.00
    else:
        spawn_x = 0.60
        spawn_y = -0.25

    pars.update({'spawn_x': spawn_x, 'spawn_y': spawn_y})

    # STIMULUS
    if passive_body:
        stim_a_off = -4.0
    else:
        stim_a_off = 0.0

    pars.update({'stim_a_off': stim_a_off})

    # FEEDBACK
    if open_loop:
        ps_min_activation_deg = 360.0
        ps_weight             = 0.0
    else:
        ps_min_activation_deg = 5.0
        ps_weight             = 20.0

    pars.update({'ps_min_activation_deg': ps_min_activation_deg, 'ps_weight': ps_weight})

    return pars


def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    results_path             = 'simulation_results'
    simulation_data_file_tag = 'dynamic_water_vortices_experimental_continuous'

    pars = _get_pars(
        debug        = False,
        in_line      = True,
        passive_body = False,
        open_loop    = False,
    )

    speed_mult_x = 1.0
    speed_mult_y = 1.0

    # Plotting
    video   = False
    plot    = True
    save    = True
    animate = True

    ###########################################################################
    # Simulation settings #####################################################
    ###########################################################################

    sim_settings = {
        'duration'            : pars['duration'],
        'results_path'        : results_path,
        'sim_file_tag'        : simulation_data_file_tag,
        'stim_a_off'          : pars['stim_a_off'],

        'process_tag'         : '0',
        'np_random_seed'      : 100,

        # Constraints
        'sim_spawn_mode'      : SpawnMode.ROTZ,
        # 'sim_spawn_mode'      : SpawnMode.FIXED,

        # Connections
        'load_connectivity'    : False,
        'ps_connections_range' : 0.50,
        'cpg_connections_range': 0.65,

        # Muscles
        'muscle_tag'          : 'new_FN_5500_ZC_1000_G0_419_gen_100',
        'muscle_resonant_freq': None,
        'mc_gain_axial'       : np.array(
            [
                # 5Hz
                0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
                0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
                0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
            ]
        ),

        # Weights
        'net_weights'         : {
            'ex2ex_weight': 3.83037,
            'ex2in_weight': 49.47920,
            'in2ex_weight': 0.84541,
            'in2in_weight': 0.10330,
            'rs2ex_weight': 8.74440,
            'rs2in_weight': 3.28338,
        },

        # Feedback
        'ps_weight'           : pars['ps_weight'],
        'ps_act_deg'          : pars['ps_min_activation_deg'],

        # Functionalities
        'video'               : video,
        'plot'                : plot,
        'save'                : save,
        'animate'             : animate,

        # Water dynamics
        'leader_signal_str'   : get_water_dynamics_file_name(
            amp_scaling        = 1.25,
            freq_scaling       = 0.25,
            integration_method = 'abdquickest',
            speed_type         = 'fast',
            amp_type           = 'constant',
            step_bounds        = [12104, 12264],
            freq_bounds        = (None, None),
        ),

        'spawn_x'             : pars['spawn_x'],
        'spawn_y'             : pars['spawn_y'],
        'speed_mult_x'        : speed_mult_x,
        'speed_mult_y'        : speed_mult_y,
        'speed_offset_x'      : 0.0,
        'speed_offset_y'      : 0.0,
        'delay_start'         : pars['delay_start'],
    }

    # Run the simulation
    run_simulation(sim_settings)

    return


if __name__ == '__main__':
    main()