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

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    simulation_tag = 'scaled_cpg_rs_weights_and_topology_ramp_current'

    # Muscle parameters
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

        # 'ex2ex_weight': 3.83037,
        # 'ex2in_weight': 49.47920,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 3.28338,

        # Original
        #
        # 'ex2ex_weight': 1.0,
        # 'ex2in_weight': 1.0,
        # 'in2ex_weight': 1.25,
        # 'in2in_weight': 1.25,
        # 'rs2ex_weight': 10.0,
        # 'rs2in_weight': 10.0

        # Good frequency increase, no late inhibition effect, small speed effect
        #
        # 'ex2ex_weight': 11.03748,
        # 'ex2in_weight': 8.37228,
        # 'in2ex_weight': 4.36458,
        # 'in2in_weight': 0.23006,
        # 'rs2ex_weight': 9.97614,
        # 'rs2in_weight': 3.09240,

        # Good late inhibition effect, small frequency increase, no speed effect
        #
        # 'ex2ex_weight': 3.52687,
        # 'ex2in_weight': 15.15590,
        # 'in2ex_weight': 2.79414,
        # 'in2in_weight': 0.80484,
        # 'rs2ex_weight': 8.16861,
        # 'rs2in_weight': 3.10099,

        # Good late inhibition effect, small frequency increase, no speed effect
        #
        # 'ex2ex_weight': 3.44786,
        # 'ex2in_weight': 11.10495,
        # 'in2ex_weight': 4.20985,
        # 'in2in_weight': 1.12677,
        # 'rs2ex_weight': 8.20428,
        # 'rs2in_weight': 3.05511,

        # Good late inhibition effect, good frequency increase, no speed effect
        #
        # 'ex2ex_weight': 5.47346,
        # 'ex2in_weight': 8.37838,
        # 'in2ex_weight': 8.29167,
        # 'in2in_weight': 0.26544,
        # 'rs2ex_weight': 9.97657,
        # 'rs2in_weight': 3.15468,

        # Good late inhibition effect, good frequency increase, no speed effect
        #
        # 'ex2ex_weight': 5.75903,
        # 'ex2in_weight': 8.29064,
        # 'in2ex_weight': 7.62601,
        # 'in2in_weight': 0.12872,
        # 'rs2ex_weight': 9.94795,
        # 'rs2in_weight': 3.16978,

        # Test
        #
        # 'ex2ex_weight': 3.13859,
        # 'ex2in_weight': 29.21290,
        # 'in2ex_weight': 0.53887,
        # 'in2in_weight': 0.11511,
        # 'rs2ex_weight': 8.96591,
        # 'rs2in_weight': 1.22565,

        # Decent frequency increase, good mid-cycle inhibition, no speed effect
        # Obtained with CPG2CPG = 1.0 and PS2CPG = 0.5
        # 'ex2ex_weight': 3.13859,
        # 'ex2in_weight': 29.21290,
        # 'in2ex_weight': 0.53887,
        # 'in2in_weight': 0.11511,
        # 'rs2ex_weight': 8.96591,
        # 'rs2in_weight': 1.22565,

        # CPG2CPG = 0.7 and PS2CPG = 0.5
        # Good mid cycle inhibition, good frequency increase
        'ex2ex_weight': 3.00,  # Not too low to avoid sparse spikes, not too high to avoid point bursts
        'ex2in_weight': 35.0,  # High to dominate V0d activity
        'in2ex_weight': 0.75,  # Low to ensure mid-cycle inhibition effect
        'in2in_weight': 0.75,  # Frequency control
        'rs2ex_weight': 8.00,  # Frequency control, similar to drive
        'rs2in_weight': 3.50,  # Not too high to keep V0d activity dominated

    }

    # Drive
    stim_a_off = 0.00

    # Connections
    cpg_connections_range = 0.70
    ps_connections_range  = 0.50

    # Random seed
    np_random_seed = 100

    # New parameters
    new_pars_process = {
        'pars_synapses_filename' : 'pars_synapses_zebrafish_exp_E_glyc',
    }

    # Params runs
    hemicord = 0
    in_only  = 0
    ex_only  = 0

    new_pars_run = {
        'stim_a_off' : -4.0 if hemicord else stim_a_off,
        'stim_lr_off': +4.0 if hemicord else 0.0,
    }

    if in_only:
        new_pars_run['rs2ex_weight'] =  0.0

    if ex_only:
        new_pars_run['ex2in_weight'] =  0.0
        new_pars_run['rs2in_weight'] =  0.0

    # Feedback
    open_loop = False

    if open_loop:
        ps_min_activation_deg = 720.0
        ps_weight             = 0.0
    else:
        ps_min_activation_deg = 10.0
        ps_weight             = 20.0

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
    main()
