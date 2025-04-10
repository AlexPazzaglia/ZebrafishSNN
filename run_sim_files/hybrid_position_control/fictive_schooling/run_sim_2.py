''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import multiprocessing
import numpy as np
import sim_fictive_schooling

def main(
    random_seed: int    = 100,
    stim_a_off  : float = 0.0,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    stim_str    = str(round( stim_a_off * 1000)).zfill(4)  # E.g. 0.255 -> 0255

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    simulation_tag = f'fictive_schooling_stim_020_{stim_str}'

    # Kinematics
    frequency_scaling = 0.20
    time_offset       = 0.0
    signal_repeats    = 1
    make_sinusoidal   = True

    # Connections
    cpg_connections_range = 0.70 # 0.65
    ps_connections_range  = 0.50 # 0.50

    # Weights
    net_weights = {

        'ex2mn_weight' : 20.0,
        'in2mn_weight' : 1.0,


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

        # Test
        'ex2ex_weight': 3.00,
        'ex2in_weight': 40.0,
        'in2ex_weight': 0.75,
        'in2in_weight': 1.00,
        'rs2ex_weight': 8.00,
        'rs2in_weight': 3.50,
    }

    stim_a_off  = stim_a_off
    random_seed = random_seed
    noise_term  = False

    # New parameters
    new_pars_process = {
        'pars_synapses_filename': 'pars_synapses_zebrafish_exp_E_glyc',
        'save_by_default'       : True,
        'noise_term'            : noise_term,
    }


    new_pars_run = {}

    if noise_term:
        # v2a_inds = np.arange(0, 1024)
        # v0d_inds = np.arange(1024, 2560)
        indices_cpg = np.arange(2560)
        noise_level = 5.0

        new_pars_run['scalings'] = {
            'neural_params' : [
                {
                    'neuron_group_ind': 0,
                    'var_name'        : 'sigma',
                    'indices'         : indices_cpg,
                    'nominal_value'   : [1.0, 'mV'],
                    'scaling'         : noise_level,
                }
            ],
        }


    # Feedback
    open_loop = False

    if open_loop:
        ps_min_activation_deg = 720.0
        ps_weight             = 0.0

    else:
        ps_min_activation_deg = 10.0
        ps_weight             = 20.0

    # Plotting
    video         = False
    plot          = True
    save          = True
    save_all_data = True

    # Run simulation
    sim_fictive_schooling.run(
        duration              = duration,
        timestep              = timestep,
        stim_a_off            = stim_a_off,
        simulation_tag        = simulation_tag,
        frequency_scaling     = frequency_scaling,
        make_sinusoidal       = make_sinusoidal,
        time_offset           = time_offset,
        signal_repeats        = signal_repeats,
        net_weights           = net_weights,
        ps_min_activation_deg = ps_min_activation_deg,
        ps_weight             = ps_weight,
        cpg_connections_range = cpg_connections_range,
        ps_connections_range  = ps_connections_range,
        video                 = video,
        plot                  = plot,
        save                  = save,
        save_all_data         = save_all_data,
        random_seed           = random_seed,
        new_pars_prc          = new_pars_process,
        new_pars_run          = new_pars_run,
    )


if __name__ == '__main__':

    # main()

    # seeds = [100]
    # stims = [0.0]

    seeds = np.arange(100, 105)
    stims = [-0.50, -0.25,  +0.00, +0.25]

    # NOTE: Avoid using the same seed at the same time
    for seed in seeds:
        inputs = [
            (seed, stim)
            for stim in stims
        ]

        # Run simulations
        with multiprocessing.Pool(processes=11) as pool:
            pool.starmap(main, inputs)
