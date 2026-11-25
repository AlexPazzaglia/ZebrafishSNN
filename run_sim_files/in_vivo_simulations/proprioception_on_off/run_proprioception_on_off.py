'''
Proprioception ON vs OFF during self-driven swimming.

The network controls the mechanical body via the muscle models (closed loop).
The proprioceptive feedback is silenced during the first half of the
simulation (proprioception OFF) and activated during the second half
(proprioception ON), so that swimming frequency, V2a excitation window and
inhibitory currents can be compared within the same network
(see net_farms_zebrafish_proprioception_on_off.py).

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

import sim_closed_loop

def main(
    seed         : int   = 100,
    stim         : float = 0.0,
    default_save : bool  = False,
):
    ''' Run the proprioception ON/OFF simulation '''

    # Simulation
    duration       = 20.0
    timestep       = 0.001
    simulation_tag = 'scaled_cpg_rs_weights_and_topology_toggle_feedback'

    # Muscles
    muscle_parameters_tag = 'new_FN_9000_ZC_1000_G0_419_gen_75'

    mc_gain_axial = np.array(
        [
            0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
            0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
            0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
        ]
    )

    # Weights
    net_weights = {
        'ex2mn_weight' : 20.0,
        'in2mn_weight' : 1.0,

        'ex2ex_weight': 3.00,
        'ex2in_weight': 40.0,
        'in2ex_weight': 0.75,
        'in2in_weight': 1.00,
        'rs2ex_weight': 8.00,
        'rs2in_weight': 3.50,
    }

    # Drive
    stim_a_off = stim

    # Connections
    cpg_connections_range = 0.70
    ps_connections_range  = 0.50

    # Proprioceptive feedback (active in the second half of the simulation)
    ps_min_activation_deg = 10.0
    ps_weight             = 20.0

    # New parameters
    new_pars_process = {
        'ps_connection_amp'     : 0.50,
        'pars_synapses_filename': 'pars_synapses_zebrafish_exp_E_glyc',
        'save_by_default'       : default_save,

        # Record membrane potentials and inhibitory currents
        'monitor_states' : {
            'active' : True,
            'save'   : False,
            'to_csv' : False,
            'indices': True,
            'rate'   : 1,

            'variables': [
                'v',
                'I_glyc', 'g_glyc1_tot',
            ],

            'plotpars' : {
                'showit' : False,
                'figure' : False,
                'animate': False,

                'voltage_traces' : {
                    'modules' : ['cpg.axial.ex'],
                    'showit'  : False,
                    'save'    : False,
                    'close'   : False,
                }
            }
        },
    }

    new_pars_run = {
        'stim_a_off' : stim_a_off,
        'stim_lr_off': 0.0,
    }

    # Run simulation
    metrics_runs = sim_closed_loop.run(
        duration              = duration,
        timestep              = timestep,
        stim_a_off            = stim_a_off,
        simulation_tag        = simulation_tag,
        muscle_parameters_tag = muscle_parameters_tag,
        muscle_gains          = mc_gain_axial,
        net_weights           = net_weights,
        ps_min_activation_deg = ps_min_activation_deg,
        ps_weight             = ps_weight,
        cpg_connections_range = cpg_connections_range,
        ps_connections_range  = ps_connections_range,
        video                 = False,
        plot                  = True,
        save                  = True,
        save_all_data         = False,
        random_seed           = seed,
        new_pars_run          = new_pars_run,
        new_pars_process      = new_pars_process,
    )

    return metrics_runs


if __name__ == '__main__':
    main(
        seed         = 100,
        stim         = 0.0,
        default_save = False,
    )
