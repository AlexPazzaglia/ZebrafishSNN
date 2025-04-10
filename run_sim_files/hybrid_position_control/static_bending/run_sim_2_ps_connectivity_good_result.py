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

    tot_angle_deg = -30
    sig_funcion_kwargs = {
        'points_list' : np.array( [ 0.25, 0.75, 1.00 ] ),
        'tau'         : 0.5,
    }

    # Connections
    cpg_connections_range = 0.70
    ps_connections_range  = 0.50 # 0.50 - 1.00

    # Weights
    net_weights = {
        'ex2mn_weight' : 30.0,
        'in2mn_weight' : 1.0,

        'ex2ex_weight': 3.85,
        'ex2in_weight': 50.0,
        'in2ex_weight': 0.85,
        'in2in_weight': 0.10,
        'rs2ex_weight': 8.25,
        'rs2in_weight': 3.30,
    }

    # Feedback
    ps_min_activation_deg = 15.0
    ps_weight             = 125.0  # 7.0  # 10.0 - 20.0 (8.5)

    # New parameters
    new_pars_prc = {
        'ps_connection_amp'      : 0.15,
        'pars_synapses_filename' : 'pars_synapses_zebrafish_exp_E_glyc',

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
                        'target_seg': [5, 6, 7, 8, 9], # [8, 9, 10, 11, 12]
                }
            }
        },
    }
    new_pars_run = None

    # Plotting
    video         = False
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