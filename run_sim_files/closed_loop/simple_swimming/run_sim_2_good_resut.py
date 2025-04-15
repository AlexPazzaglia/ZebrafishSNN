''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import scaled_weights_and_topology_sim

def main(
    seed         = 100,
    stim         = 0.0,
    default_save = False,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    duration       = 20.0
    timestep       = 0.001
    simulation_tag = 'scaled_cpg_rs_weights_and_topology_toggle_feedback'

    # Muscle gains
    muscle_parameters_tag = 'new_FN_9000_ZC_1000_G0_419_gen_75'

    mc_gain_axial = np.array(
        [
            # New
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

    # Feedback
    open_loop = False

    if open_loop:
        ps_min_activation_deg = 720.0
        ps_weight             = 0.0
    else:
        ps_min_activation_deg = 10.0
        ps_weight             = 20.0

    # Random seed
    # np_random_seed = 100
    np_random_seed = seed

    # New parameters
    new_pars_process = {
        'ps_connection_amp'     : 0.50,
        'pars_synapses_filename': 'pars_synapses_zebrafish_exp_E_glyc',
        'save_by_default'       : default_save,


        'monitor_states' : {
            'active' : True,
            'save'   : False,
            'to_csv' : False,
            'indices': True,
            'rate'   : 1,

            'variables': [
                'v',
                # w1', 'I_tot', 'I_ext',
                'I_glyc', 'g_glyc1_tot',
                # 'I_nmda', 'g_nmda1_tot',
                # 'I_ampa', 'g_ampa1_tot',
            ],

            'plotpars' : {
                'showit' : False,
                'figure' : False,
                'animate': False,

                'voltage_traces' : {
                    'modules' : ['cpg.axial.ex'],   # 'mn.axial.mn'
                    'showit'  : False,
                    'save'    : False,
                    'close'   : False,
                }
            }
        },
    }

    # Params runs
    hemicord     = 0
    in_only      = 0
    ex_only      = 0

    new_pars_run = {
        'stim_a_off' : -4.0 if hemicord else stim_a_off,
        'stim_lr_off': +4.0 if hemicord else 0.0,
    }

    if in_only:
        new_pars_run['rs2ex_weight'] =  0.0

    if ex_only:
        new_pars_run['ex2in_weight'] =  0.0
        new_pars_run['rs2in_weight'] =  0.0

    # Plotting
    video    = False
    plot     = True
    save     = True
    save_all = False

    # Run simulation
    metrics_runs = scaled_weights_and_topology_sim.run(
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
        video                 = video,
        plot                  = plot,
        save                  = save,
        save_all_data         = save_all,
        random_seed           = np_random_seed,
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

    # # Run simulations
    # import multiprocessing
    # default_save = True
    # for stim in [-0.25, 0.0, 0.25]:
    #     inputs = [ (seed, stim, default_save) for seed in [100, 101, 102] ]
    #     with multiprocessing.Pool(processes=5) as pool:
    #         pool.starmap(main, inputs)
