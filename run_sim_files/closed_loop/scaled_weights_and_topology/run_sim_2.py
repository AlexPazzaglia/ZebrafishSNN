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

def main(seed = 100):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    duration       = 20.0
    timestep       = 0.001
    simulation_tag = 'scaled_cpg_rs_weights_and_topology_all_data'

    # Muscle gains
    # muscle_parameters_tag = 'new_FN_5000_ZC_1000_G0_419_gen_100'
    # muscle_parameters_tag = 'new_FN_5500_ZC_1000_G0_419_gen_100'
    # muscle_parameters_tag = 'new_FN_7000_ZC_1000_G0_419_gen_100'
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

        # Original
        # Obtained with CPG2CPG = 0.65 and PS2CPG = 0.5
        # 'ex2ex_weight': 3.83037,
        # 'ex2in_weight': 49.47920,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 3.28338,

        # Decent frequency increase, good mid-cycle inhibition, no speed effect
        # Obtained with CPG2CPG = 1.0 and PS2CPG = 0.5
        # stim_a_off =  0.00 --> 3.65 Hz
        # stim_a_off = -0.25 --> 3.35 Hz
        # stim_a_off = -0.50 --> 3.15 Hz
        # stim_a_off = -0.75 --> 2.90 Hz

        # 'ex2ex_weight': 3.13859,
        # 'ex2in_weight': 29.21290,
        # 'in2ex_weight': 0.53887,
        # 'in2in_weight': 0.11511,
        # 'rs2ex_weight': 8.96591,
        # 'rs2in_weight': 1.22565,

        # Ugly-looking open-loop raster plot, very good frequency increase
        # stim_a_off =  0.00 --> 3.00 Hz

        # 'ex2ex_weight': 0.19724,
        # 'ex2in_weight': 32.18933,
        # 'in2ex_weight': 2.05110,
        # 'in2in_weight': 0.18614,
        # 'rs2ex_weight': 9.88040,
        # 'rs2in_weight': 0.38959,

        # A bit too perfect open-loop raster plot, too many V2a spikes, good frequency increase
        # No mid-cycle inhibition effect
        # stim_a_off =  0.00 --> 3.20 Hz

        # 'ex2ex_weight': 0.56278,
        # 'ex2in_weight': 15.49187,
        # 'in2ex_weight': 16.75322,
        # 'in2in_weight': 4.64773,
        # 'rs2ex_weight': 9.24310,
        # 'rs2in_weight': 0.21175,

        # Good looking open-loop raster plot, small frequency increase
        # stim_a_off =  0.00 --> 3.15 Hz
        # stim_a_off = -0.50 --> 2.85 Hz
        # stim_a_off = -0.75 --> 2.55 Hz but unstable

        # 'ex2ex_weight': 3.28478,
        # 'ex2in_weight': 38.65171,
        # 'in2ex_weight': 1.20133,
        # 'in2in_weight': 0.13263,
        # 'rs2ex_weight': 8.03325,
        # 'rs2in_weight': 0.42233,

        # Good looking open-loop raster plot, good frequency increase
        # Very high mid-cycle inhibition alredy in open-loop
        # stim_a_off =  0.50 --> 3.05 Hz
        # stim_a_off =  0.00 --> 2.80 Hz
        # stim_a_off = -0.50 --> 2.60 Hz
        # stim_a_off = -0.60 --> 2.55 Hz
        # stim_a_off = -0.75 --> 2.40 Hz but unstable

        # 'ex2ex_weight': 2.01391,
        # 'ex2in_weight': 25.15506,
        # 'in2ex_weight': 8.46539,
        # 'in2in_weight': 0.64199,
        # 'rs2ex_weight': 7.83521,
        # 'rs2in_weight': 2.33902,

        # Good looking open-loop raster plot, bad frequency increase
        # Very high mid-cycle inhibition alredy in open-loop
        # stim_a_off = 0.50 --> 3.05 Hz
        # stim_a_off = 0.00 --> 2.85 Hz

        # 'ex2ex_weight': 3.18108,
        # 'ex2in_weight': 38.96006,
        # 'in2ex_weight': 1.56519 * 0.5,
        # 'in2in_weight': 0.10107,
        # 'rs2ex_weight': 7.00274,
        # 'rs2in_weight': 1.25088,

        # Good looking open-loop raster plot, decent frequency increase
        # Not super high mid-cycle inhibition in open-loop
        # stim_a_off   =  0.00 --> 3.19 Hz
        # stim_a_off   = -0.50 --> 2.80 Hz
        # rs2ex_weight =  7.00 --> 2.85 Hz
        # rs2ex_weight =  7.50 --> 3.00 Hz
        # rs2ex_weight =  9.00 --> 3.60 Hz

        # 'ex2ex_weight': 3.85,
        # 'ex2in_weight': 30.0,
        # 'in2ex_weight': 0.65,
        # 'in2in_weight': 0.25,
        # 'rs2ex_weight': 8.00,
        # 'rs2in_weight': 1.25,

        # Good looking open-loop raster plot, moderate frequency increase

        # stim_a_off =  0.00 --> 3.35 Hz
        # stim_a_off = -0.25 --> 3.15 Hz
        # stim_a_off = -0.40 --> 3.00 Hz
        # stim_a_off = -0.50 --> 2.85 Hz

        # 'ex2ex_weight': 3.85,
        # 'ex2in_weight': 50.0,
        # 'in2ex_weight': 0.85,
        # 'in2in_weight': 0.10,
        # 'rs2ex_weight': 8.75,
        # 'rs2in_weight': 3.30,

        # CPG2CPG = 0.9 and PS2CPG = 0.5
        # Decent mid cycle inhibition
        # stim_a_off = +0.50 --> 3.80 Hz
        # stim_a_off =  0.00 --> 3.43 Hz
        # stim_a_off = -0.50 --> 3.00 Hz

        # 'ex2ex_weight': 3.85,
        # 'ex2in_weight': 50.0,
        # 'in2ex_weight': 0.85,
        # 'in2in_weight': 0.10,
        # 'rs2ex_weight': 9.25,
        # 'rs2in_weight': 3.30,

        # CPG2CPG = 0.7 and PS2CPG = 0.5
        # Decent mid cycle inhibition
        # stim_a_off = +0.25 --> 4.00 Hz
        # stim_a_off = +0.00 --> 3.80 Hz
        # stim_a_off = -0.25 --> 3.50 Hz
        # stim_a_off = -0.50 --> 3.20 Hz

        # 'ex2ex_weight': 3.85,
        # 'ex2in_weight': 50.0,
        # 'in2ex_weight': 0.85,
        # 'in2in_weight': 0.10,
        # 'rs2ex_weight': 9.25,
        # 'rs2in_weight': 3.30,

        # CPG2CPG = 0.7 and PS2CPG = 0.5
        # Good mid cycle inhibition, good frequency increase
        # 'ex2ex_weight': 3.00,  # Not too low to avoid sparse spikes, not too high to avoid point bursts
        # 'ex2in_weight': 35.0,  # High to dominate V0d activity
        # 'in2ex_weight': 0.75,  # Low to ensure mid-cycle inhibition effect
        # 'in2in_weight': 0.75,  # Frequency control
        # 'rs2ex_weight': 8.00,  # Frequency control, similar to drive
        # 'rs2in_weight': 3.50,  # Not too high to keep V0d activity dominated

        # Test
        'ex2ex_weight': 3.00,
        'ex2in_weight': 40.0,
        'in2ex_weight': 0.75,
        'in2in_weight': 1.00,
        'rs2ex_weight': 8.00,
        'rs2in_weight': 3.50,


    }

    # Drive
    stim_a_off = 0.0

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
        # 'save_by_default'       : True
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
    save_all = True

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

    main(100)

    # inputs = [
    #     (seed,)
    #     for seed in np.arange(100, 110)
    # ]

    # # Run simulations
    # import multiprocessing
    # with multiprocessing.Pool(processes=10) as pool:
    #     pool.starmap(main, inputs)
