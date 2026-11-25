'''
Static bending (in silico counterpart of the static tail-bend experiments).

The muscles are silenced and a constant bend is imposed in position control
to all joints except the head joint, while the network is active and receives
proprioceptive feedback from the position-controlled body. The body is kept
straight for the first 2.5 s, bent between 2.5 s and 7.5 s, and released
for the remaining 2.5 s.

Results are written to RESULTS_PATH (see network_experiments/snn_paths.py).
'''

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
    ''' Run the static bending simulation '''

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    stim_a_off     = 0.0
    simulation_tag = 'static_bending_all_data'

    # Kinematics
    # NOTE: Positive angles = Left side stretched
    # The bend is applied between 25% and 75% of the simulation duration
    tot_angle_deg = -30
    sig_funcion_kwargs = {
        'points_list' : np.array( [ 0.25, 0.75, 1.00 ] ),
        'tau'         : 0.2,
    }

    # Connections
    ps_connections_range  = 0.50
    cpg_connections_range = 0.65

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

    # Feedback
    ps_min_activation_deg = 5.0
    ps_weight             = 10.0

    # New parameters
    new_pars_prc = {
        'pars_synapses_filename' : 'pars_synapses_zebrafish_exp',

        'monitor_musclecells' : {
            'active'   : True,
            'save'     : True,
            'to_csv'   : False,
            'indices'  : True,
            'variables': ['v', 'I_tot'],
            'rate'     : 1,
            'plotpars' : {
                'showit'        : True,
                'filtering'     : False,
                'sampling_ratio': 0.50,
                'duty_cycle_ax' : {
                        'showit'    : True,
                        'filter'    : True,
                        'target_seg': [5, 6, 7, 8, 9],
                }
            }
        },
    }
    new_pars_run = None

    # Plotting
    video         = False
    plot          = True
    save          = True
    save_all_data = True

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
