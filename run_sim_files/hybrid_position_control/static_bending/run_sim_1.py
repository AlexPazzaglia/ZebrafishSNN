''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import sim_static_bending

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    stim_a_off     = 0.0
    simulation_tag = 'static_bending_all_data'

    # Kinematics
    # NOTE: Positive angles = Left side stretched

    tot_angle_deg = +30
    sig_funcion_kwargs = {
        'points_list' : np.array( [ 0.25, 0.75, 1.00 ] ),
        'tau'         : 0.1,
    }

    # Connections
    cpg_connections_range = 0.65
    ps_connections_range  = 0.50 # 0.50 - 1.00

    # Weights
    net_weights = {
        'ex2mn_weight' : 60.0,
        'in2mn_weight' : 1.0,

        # # Original
        # 'ex2ex_weight': 1.0,
        # 'ex2in_weight': 1.0,
        # 'in2ex_weight': 1.25,
        # 'in2in_weight': 1.25,
        # 'rs2ex_weight': 10.0,
        # 'rs2in_weight': 10.0,

        # # Official
        # 'ex2ex_weight': 3.83037,
        # 'ex2in_weight': 49.47920,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 3.28338,

        # # Test
        # 'ex2ex_weight': 3.0,
        # 'ex2in_weight': 20.0,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 4.0,

        # Test
        'ex2ex_weight': 3.83037,
        'ex2in_weight': 49.47920,
        'in2ex_weight': 0.84541,
        'in2in_weight': 0.10330,
        'rs2ex_weight': 8.74440,
        'rs2in_weight': 3.28338,


    }

    # Feedback
    ps_min_activation_deg = 15.0
    ps_weight             = 20.0  # 7.0  # 10.0 - 20.0 (8.5)

    # New parameters
    new_pars_prc = {}
    new_pars_run = None

    # Plotting
    video         = True
    plot          = True
    save          = True
    save_all_data = False

    # Run simulation
    sim_static_bending.run(
        duration              = duration,
        timestep              = timestep,
        stim_a_off            = stim_a_off,
        simulation_tag        = simulation_tag,
        tot_angle_deg         = tot_angle_deg,
        net_weights           = net_weights,
        ps_min_activation_deg = ps_min_activation_deg,
        ps_weight             = ps_weight,
        cpg_connections_range = cpg_connections_range,
        ps_connections_range  = ps_connections_range,
        video                 = video,
        plot                  = plot,
        save                  = save,
        save_all_data         = save_all_data,
        new_pars_prc          = new_pars_prc,
        new_pars_run          = new_pars_run,
        sig_funcion_kwargs    = sig_funcion_kwargs,
    )


if __name__ == '__main__':
    main()