'''
In silico pacing (Fig. 3F-H).

The muscles of the model are silenced and the body is bent rhythmically in
position control (all joints except the head joint), while the spiking network
is active and receives proprioceptive feedback from the imposed movement.

For each baseline network frequency (4, 5 and 6 Hz) the body is paced at
2, 4, 6, 8 and 10 Hz. Each simulation lasts 10 s: the body is kept straight
for the first and last part of the simulation and is bent rhythmically in
between (see sim_rhythmic_bending.py).

Results are written to RESULTS_PATH (see network_experiments/snn_paths.py)
and the main figures are collected by move_results.py.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import multiprocessing

import sim_rhythmic_bending
from move_results import move_files

# Synaptic weights obtained from the calibration of the isolated network
SLOW_WEIGHTS = {
    'ex2ex_weight': 3.83037,
    'ex2in_weight': 49.47920,
    'in2ex_weight': 0.84541,
    'in2in_weight': 0.10330,
    'rs2ex_weight': 8.74440,
    'rs2in_weight': 3.28338,
}

FAST_WEIGHTS = {
    'ex2ex_weight': 4.36998,
    'ex2in_weight': 44.12017,
    'in2ex_weight': 0.31253,
    'in2in_weight': 0.21465,
    'rs2ex_weight': 9.96475,
    'rs2in_weight': 3.08688,
}

# Baseline network frequency -> network parameters
# 'stim_a_off'       : offset of the descending drive
# 'neural_pars_gain' : scaling of the adaptation time constant of the CPG neurons
NETWORK_FREQ_TO_PARS = {
    '4.00' : FAST_WEIGHTS | {'stim_a_off': 0.00, 'neural_pars_gain': 1.00},
    '5.00' : SLOW_WEIGHTS | {'stim_a_off': 0.50, 'neural_pars_gain': 1.70},
    '6.00' : FAST_WEIGHTS | {'stim_a_off': 0.00, 'neural_pars_gain': 1.50},
}

def run_rhythmic_bending_experiment( frequency_args : tuple[float, float] ):
    ''' Run a single pacing experiment (network frequency, stimulus frequency) '''

    network_frequency, stimulus_frequency = frequency_args

    # Simulation tag
    net_f_str      = str(round(  network_frequency * 100)).zfill(3)  # E.g. 2.50 -> 250
    stm_f_str      = str(round( stimulus_frequency * 100)).zfill(3)  # E.g. 3.75 -> 375
    simulation_tag = f'rhythmic_bending_from_{net_f_str}_to_{stm_f_str}'

    print(f'Rhythmic bending simulation: {simulation_tag}')

    # Simulation
    duration = 10.0
    timestep = 0.001

    # Kinematics (total head-to-tail bending angle)
    tot_angle_deg = 30

    # Connections
    cpg_connections_range = 0.65
    ps_connections_range  = 0.50

    # Network parameters
    net_weights = dict( NETWORK_FREQ_TO_PARS[f'{network_frequency:.2f}'] )
    stim_a_off  = net_weights.pop('stim_a_off')

    net_weights = net_weights | {
        'ex2mn_weight' : 50.0,
        'in2mn_weight' : 1.0,
    }

    # Proprioceptive feedback
    ps_min_activation_deg = 5.0
    ps_weight             = 30.0

    # Run simulation
    sim_rhythmic_bending.run(
        duration             = duration,
        timestep             = timestep,
        stim_a_off           = stim_a_off,
        simulation_tag       = simulation_tag,
        tot_angle_deg        = tot_angle_deg,
        frequency            = stimulus_frequency,
        net_weights          = net_weights,
        ps_min_activation_deg= ps_min_activation_deg,
        ps_weight            = ps_weight,
        cpg_connections_range= cpg_connections_range,
        ps_connections_range = ps_connections_range,
        video                = False,
        plot                 = True,
        save                 = True,
        save_all_data        = True,
    )

    # Collect the main figures
    move_files(
        network_frequency  = float(network_frequency),
        stimulus_frequency = float(stimulus_frequency),
        dest_tag           = '_fast',
        delete_existing    = True,
    )

    return

def run_experiment(
    experiments_list: list[tuple[float, float]],
    n_processes     : int = 5,
):
    ''' Run multiple simulations in parallel '''

    if len(experiments_list) == 1:
        run_rhythmic_bending_experiment(experiments_list[0])
        return

    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.map(run_rhythmic_bending_experiment, experiments_list)

    return


def main():
    ''' Run the pacing experiments for all baseline and stimulus frequencies '''

    network_freq_list  = [4.0, 5.0, 6.0]
    stimulus_freq_list = [2.0, 4.0, 6.0, 8.0, 10.0]

    experiments_list = [
        (network_freq, stimulus_freq)
        for network_freq  in network_freq_list
        for stimulus_freq in stimulus_freq_list
    ]

    run_experiment(experiments_list)

    return

if __name__ == '__main__':
    main()
