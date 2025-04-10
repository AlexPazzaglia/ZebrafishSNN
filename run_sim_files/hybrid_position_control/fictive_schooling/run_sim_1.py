''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import sim_fictive_schooling

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    duration       = 10.0
    timestep       = 0.001
    simulation_tag = 'fictive_schooling'

    # Simulation parameters
    # -4.00 --> 0.00 Hz
    #  0.00 --> 3.50 Hz
    # -0.25 --> 3.25 Hz
    # -0.50 --> 3.00 Hz
    # -1.00 --> 2.50 Hz

    stim_a_off = 0.50

    # Kinematics
    frequency_scaling = 0.20
    time_offset       = 0.0
    signal_repeats    = 1
    make_sinusoidal   = False

    # Connections
    cpg_connections_range = 0.65
    ps_connections_range  = 1.00

    # Weights
    net_weights = {

        # Original
        # -4.00 --> 0.00 Hz
        #  0.00 --> 3.50 Hz
        # -0.25 --> 3.25 Hz
        # -0.50 --> 3.00 Hz
        # -0.75 --> 2.75 Hz
        # -1.00 --> 2.50 Hz

        # 'ex2ex_weight': 3.83037,
        # 'ex2in_weight': 49.47920,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 3.28338,

        # Faster
        # -4.00 --> 0.00 Hz
        #  0.00 --> 4.00 Hz
        # -0.25 --> 3.75 Hz
        # +0.50 --> 4.25 Hz

        'ex2ex_weight': 4.36998,
        'ex2in_weight': 44.12017,
        'in2ex_weight': 0.31253,
        'in2in_weight': 0.21465,
        'rs2ex_weight': 9.96475,
        'rs2in_weight': 3.08688,


        'ex2mn_weight' : 50.0,
        'in2mn_weight' : 1.0,
    }

    # Feedback
    open_loop = False

    if open_loop:
        ps_min_activation_deg = 720.0
        ps_weight             = 0.0

    else:
        ps_min_activation_deg = 5.0
        ps_weight             = 10.0

    # Plotting
    video         = False
    plot          = True
    save          = False
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
    )


if __name__ == '__main__':
    main()