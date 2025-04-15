''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Union
from farms_core.model.options import SpawnMode
from sim_runner import run_simulation, get_water_dynamics_file_name

import network_experiments.default_parameters.zebrafish.closed_loop.default as default

def _get_spring_damper_options(
    target_link   : int,
    spring_factor : Union[float, np.ndarray] = 10.0,
    damping_factor: Union[float, np.ndarray] = 0.0,
    inactive_band : float = 0.0,
    delta_x_bl    : float = 0.0,
    delta_y_bl    : float = 0.0,
    delta_z_bl    : float = 0.0,
    act_on_x      : bool  = True,
    act_on_y      : bool  = False,
    act_on_z      : bool  = False,
):
    ''' Get the spring damper options '''

    body_length = default.LENGTH_AXIS_MODEL
    body_mass   = 5.022324522733332e-05

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


def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    results_path             = 'simulation_results_test'
    simulation_data_file_tag = 'dynamic_water_vortices_closed_loop_free_body'

    delay_start = 5.0 # 5.0
    duration    = 30.0 + delay_start

    # Spawn
    spawn_x_bl = 1.00 # 0.75
    spawn_y_bl = 0.00

    # Vortex field
    leader_signal_str = get_water_dynamics_file_name(
        amp_scaling        = 1.25,          # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,          # [0.23,  0.25 ]
        integration_method = 'abdquickest', # ['abdquickest',  'implicit' ]
        speed_type         = 'theoretical', # ['empirical',  'theoretical' ]
    )

    speed_mult_x = 1.00
    speed_mult_y = 1.00

    speed_offset_x = 0.00   # 0.40
    speed_offset_y = 0.00

    # Soft spring constraint
    target_link    = 2

    spring_factor  = np.array( [ 1.0,  5.0, 0.0] ) # 10.0 10.0 0.0
    damping_factor = np.array( [ 0.0,  0.0, 0.0] )

    inactive_band  = np.array( [ 0.5,  0.0, 0.0] ) # 0.0 0.0 0.0

    delta_x_bl     = 0.00
    delta_y_bl     = 0.25 # 0.0
    act_on_x       = True
    act_on_y       = True

    spring_damper_options = _get_spring_damper_options(
        target_link    = target_link,
        spring_factor  = spring_factor,
        damping_factor = damping_factor,
        inactive_band  = inactive_band,
        delta_x_bl     = delta_x_bl,
        delta_y_bl     = delta_y_bl,
        act_on_x       = act_on_x,
        act_on_y       = act_on_y,
    )

    # Connections
    ps_connections_range  = 0.50
    cpg_connections_range = 0.65

    # Muscles
    muscle_resonant_freq = 5.0     # 7.0

    # muscle_tag    = 'new_FN_4500_ZC_1000_G0_419_gen_50'
    muscle_tag    = None

    mc_gain_axial = np.array(
        [
            # 5Hz
            0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
            0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
            0.13329, 0.13803, 0.20825, 0.00000, 0.00000,

            # 7Hz
            # 0.32395, 0.13874, 0.09938, 0.12272, 0.10301,
            # 0.10099, 0.11082, 0.10114, 0.05633, 0.12465,
            # 0.14184, 0.00270, 0.00108, 0.00000, 0.00000,

            # 4.5Hz
            # 0.28545, 0.08421, 0.08134, 0.08869, 0.09467,
            # 0.10110, 0.11570, 0.12595, 0.13671, 0.15351,
            # 0.21074, 0.23520, 0.15335, 0.00000, 0.00000,

        ]
    )

    # Weights
    net_weights = {
        'ex2ex_weight': 3.83037,
        'ex2in_weight': 49.47920,
        'in2ex_weight': 0.84541,
        'in2in_weight': 0.10330,
        'rs2ex_weight': 8.74440,
        'rs2in_weight': 3.28338,
    }

    # Feedback
    # ps_min_activation_deg = 360.0
    # ps_weight             = 0.0

    ps_min_activation_deg = 5.0
    ps_weight             = 20.0

    # Plotting
    video   = False
    plot    = True
    save    = True
    animate = True

    ###########################################################################
    # Simulation settings #####################################################
    ###########################################################################

    sim_settings = {
        'duration'            : duration,
        'results_path'        : results_path,
        'sim_file_tag'        : simulation_data_file_tag,

        # Constraints
        # NOTE:
        # SpawnMode.TRANSVERSE,     # Head in XY plane + rotation around Z
        # SpawnMode.TRANSVERSE0,    # Head in XY plane + no head rotation
        # SpawnMode.TRANSVERSE3,    # Head in XY plane + all head rotations

        # SpawnMode.SAGITTAL,       # Head in YZ plane + rotation around X
        # SpawnMode.SAGITTAL0,      # Head in YZ plane + no head rotation
        # SpawnMode.SAGITTAL3,      # Head in YZ plane + all head rotations

        # SpawnMode.CORONAL,        # Head in XZ plane + rotation around Y
        # SpawnMode.CORONAL0,       # Head in XZ plane + no head rotation
        # SpawnMode.CORONAL3,       # Head in XZ plane + all head rotations

        'sim_spawn_mode'       : SpawnMode.TRANSVERSE0,
        'spring_damper_options': spring_damper_options,

        # Connections
        'load_connectivity'    : False,
        'ps_connections_range' : ps_connections_range,
        'cpg_connections_range': cpg_connections_range,

        # Muscles
        'muscle_tag'          : muscle_tag,
        'muscle_resonant_freq': muscle_resonant_freq,
        'mc_gain_axial'       : mc_gain_axial,

        # Weights
        'net_weights'         : net_weights,

        # Feedback
        'ps_weight'           : ps_weight,
        'ps_act_deg'          : ps_min_activation_deg,

        # Functionalities
        'video'               : video,
        'plot'                : plot,
        'save'                : save,
        'animate'             : animate,

        # Water dynamics
        'leader_signal_str'   : leader_signal_str,
        'spawn_x'             : spawn_x_bl,
        'spawn_y'             : spawn_y_bl,
        'speed_mult_x'        : speed_mult_x,
        'speed_mult_y'        : speed_mult_y,
        'speed_offset_x'      : speed_offset_x,
        'speed_offset_y'      : speed_offset_y,
        'delay_start'         : delay_start,
    }


    # Run simulation
    run_simulation(**sim_settings)


if __name__ == '__main__':
    main()