import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import sim_rhythmic_bending
import multiprocessing

def _get_network_pars_from_frequency(network_frequency):
    ''' Get network parameters from network frequency '''

    # Weights
    slow_weights = {
        'ex2ex_weight': 3.83037,
        'ex2in_weight': 49.47920,
        'in2ex_weight': 0.84541,
        'in2in_weight': 0.10330,
        'rs2ex_weight': 8.74440,
        'rs2in_weight': 3.28338,
    }

    fast_weights = {
        'ex2ex_weight': 4.36998,
        'ex2in_weight': 44.12017,
        'in2ex_weight': 0.31253,
        'in2in_weight': 0.21465,
        'rs2ex_weight': 9.96475,
        'rs2in_weight': 3.08688,
    }

    # Weights and stimulation offset
    network_freq_to_pars = {
        '2.50' : slow_weights | {'stim_a_off': -1.00},
        '2.75' : slow_weights | {'stim_a_off': -0.75},
        '3.00' : slow_weights | {'stim_a_off': -0.50},
        '3.25' : slow_weights | {'stim_a_off': -0.25},
        '3.50' : slow_weights | {'stim_a_off':  0.00},
        '3.75' : fast_weights | {'stim_a_off': -0.25},
        '4.00' : fast_weights | {'stim_a_off':  0.00},
        '4.25' : fast_weights | {'stim_a_off': +0.50},
    }

    return network_freq_to_pars[network_frequency]

def run_rhythmic_bending_experiment( frequency_args : tuple[float, float] ):

    network_frequency, stimulus_frequency = frequency_args

    # Simulation tag
    net_f_str      = str(round(  network_frequency * 100)).zfill(3)  # E.g. 2.50 -> 250
    stm_f_str      = str(round( stimulus_frequency * 100)).zfill(3)  # E.g. 3.75 -> 375
    simulation_tag = f'rhythmic_bending_from_{net_f_str}_to_{stm_f_str}'

    print(f'Rhythmic bending simulation: {simulation_tag}')

    # Simulation
    duration          = 15.0
    timestep          = 0.001
    network_frequency = f'{network_frequency:.2f}'

    # Kinematics
    tot_angle_deg = 30

    # Connections
    cpg_connections_range = 0.65
    ps_connections_range  = 0.50 # 0.50 - 1.00

    # Weights
    net_weights = _get_network_pars_from_frequency(network_frequency)
    stim_a_off  = net_weights.pop('stim_a_off')

    # Weights
    net_weights = net_weights | {
        'ex2mn_weight' : 25.0,
        'in2mn_weight' : 1.0,
    }

    # Feedback
    ps_min_activation_deg = 10.0
    ps_weight             = 20.0 # 10.0 - 20.0

    # Plotting
    video         = False
    plot          = True
    save          = True
    save_all_data = True

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
        video                = video,
        plot                 = plot,
        save                 = save,
        save_all_data        = save_all_data,
    )

def main():

    # Example
    # run_rhythmic_bending_experiment((2.75, 2.00))

    network_frequencies  = [2.75, 3.00, 3.25, 3.50, 3.75, 4.00]
    stimulus_frequencies = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25]

    # freq_inputs = [
    #     (nf, sf)
    #     for nf in network_frequencies
    #     for sf in stimulus_frequencies
    # ]

    missing = [
        [3.75, 4.25],
        [4.00, 2.00],
        [4.00, 4.25],
    ]

    # 6 * 10 = 60 simulations

    pool = multiprocessing.Pool(processes=5)
    pool.map(
        func     = run_rhythmic_bending_experiment,
        iterable = missing,
    )
    pool.close()
    pool.join()

    return

if __name__ == '__main__':
    main()

