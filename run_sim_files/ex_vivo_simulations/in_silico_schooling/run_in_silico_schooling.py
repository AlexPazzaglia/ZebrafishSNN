'''
In silico schooling (Fig. 3I-J).

The muscles of the model are silenced and the body is bent in position control
following the tail kinematics of a leader fish recorded during a close-pair
interaction (the same sequence used for the ex vivo schooling experiments).
The network is active and receives proprioceptive feedback from the imposed
movement.

The experiment is repeated for 5 networks (random seeds) and 4 levels of
descending drive (i.e. different baseline swimming frequencies).

Results are written to RESULTS_PATH (see network_experiments/snn_paths.py)
and the EMG figures are collected by move_results.py.
'''

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

SEEDS = np.arange(100, 105)
STIMS = [-0.50, -0.25, +0.00, +0.25]

def get_simulation_tag(stim_a_off: float) -> str:
    ''' Get the simulation tag for a given descending drive offset '''
    stim_str = str(round( stim_a_off * 1000)).zfill(4)  # E.g. 0.255 -> 0255
    return f'fictive_schooling_stim_020_{stim_str}'

def main(
    random_seed: int   = 100,
    stim_a_off : float = 0.0,
):
    ''' Run a single in silico schooling simulation '''

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    simulation_tag = get_simulation_tag(stim_a_off)

    # Kinematics (leader tail-beat sequence)
    frequency_scaling = 0.20
    time_offset       = 0.0
    signal_repeats    = 1
    make_sinusoidal   = True

    # Connections
    cpg_connections_range = 0.70
    ps_connections_range  = 0.50

    # Weights
    net_weights = {
        'ex2mn_weight': 20.0,
        'in2mn_weight': 1.0,

        'ex2ex_weight': 3.00,
        'ex2in_weight': 40.0,
        'in2ex_weight': 0.75,
        'in2in_weight': 1.00,
        'rs2ex_weight': 8.00,
        'rs2in_weight': 3.50,
    }

    # New parameters
    new_pars_process = {
        'pars_synapses_filename': 'pars_synapses_zebrafish_exp_E_glyc',
        'save_by_default'       : True,
        'noise_term'            : False,
    }
    new_pars_run = {}

    # Proprioceptive feedback
    ps_min_activation_deg = 10.0
    ps_weight             = 20.0

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
        video                 = False,
        plot                  = True,
        save                  = True,
        save_all_data         = True,
        random_seed           = random_seed,
        new_pars_prc          = new_pars_process,
        new_pars_run          = new_pars_run,
    )


if __name__ == '__main__':

    # NOTE: Simulations sharing the same seed are not run in parallel
    for seed in SEEDS:
        inputs = [ (seed, stim) for stim in STIMS ]

        with multiprocessing.Pool(processes=len(STIMS)) as pool:
            pool.starmap(main, inputs)
