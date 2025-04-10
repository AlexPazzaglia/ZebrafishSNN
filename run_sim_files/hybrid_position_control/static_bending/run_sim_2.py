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
        'tau'         : 0.1,
    }

    # Connections
    cpg_connections_range = 1.00
    ps_connections_range  = 0.50 # 0.50 - 1.00

    # Weights
    net_weights = {
        # 'ex2mn_weight' : 40.0,
        # 'in2mn_weight' : 10.0,

        # # Original
        # 'ex2ex_weight': 1.0,
        # 'ex2in_weight': 1.0,
        # 'in2ex_weight': 1.25,
        # 'in2in_weight': 1.25,
        # 'rs2ex_weight': 10.0,
        # 'rs2in_weight': 10.0,

        # # Official
        # 'ex2ex_weight': 3.83037,
        # 'ex2in_weight': 49.47920,
        # 'in2ex_weight': 0.84541,
        # 'in2in_weight': 0.10330,
        # 'rs2ex_weight': 8.74440,
        # 'rs2in_weight': 3.28338,

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
        # 'rs2ex_weight': 8.25,
        # 'rs2in_weight': 3.30,

        # # Quite good with PS 10.0, 18.0 at range 1.0, 0.5
        # 'ex2ex_weight': 3.13859,
        # 'ex2in_weight': 29.21290,
        # 'in2ex_weight': 0.53887,
        # 'in2in_weight': 0.11511,
        # 'rs2ex_weight': 8.96591,
        # 'rs2in_weight': 1.22565,

        # # Test
        # # Not bad with PS 10.0, 20.0 at range 1.0, 0.5
        # 'ex2ex_weight': 3.50,
        # 'ex2in_weight': 50.0,
        # 'in2ex_weight': 1.00,
        # 'in2in_weight': 0.10,
        # 'rs2ex_weight': 9.00,
        # 'rs2in_weight': 2.00,

        # Test
        # 'ex2ex_weight': 3.50,
        # 'ex2in_weight': 50.0,
        # 'in2ex_weight': 1.00,
        # 'in2in_weight': 0.10,
        # 'rs2ex_weight': 8.00,
        # 'rs2in_weight': 2.00,

        # Test
        # Very high re2ex + High in2ex?
        'ex2ex_weight': 3.50,
        'ex2in_weight': 50.0,
        'in2ex_weight': 1.00,
        'in2in_weight': 0.10,
        'rs2ex_weight': 10.00,
        'rs2in_weight': 2.00,


        'ex2mn_weight' : 40.0,
        'in2mn_weight' : 10.0,


    }

    # Feedback
    ps_min_activation_deg = 10.0
    ps_weight             = [10, 30]      # [10, 20]  # 7.0  # 10.0 - 20.0 (8.5)

    # New parameters
    new_pars_prc = {
        'ps_connection_amp'      : 0.50,
        'pars_synapses_filename' : 'pars_synapses_zebrafish_exp_E_glyc',

        # 'monitor_states' : {
        #     'active'   : True,
        #     'save'     : False,
        #     'to_csv'   : False,
        #     'indices'  : True,
        #     'rate'     : 1,
        #     'variables': [
        #         'v', 'w1', 'I_tot',
        #         'I_glyc', 'g_glyc1_tot',
        #         'I_nmda', 'g_nmda1_tot',
        #     ],
        #     'plotpars' : {
        #         'showit' : True,
        #         'figure' : False,
        #         'animate': False,

        #         'voltage_traces' : {
        #             'modules' : ['cpg.axial.ex'],   # 'mn.axial.mn'
        #             'showit'  : True,
        #             'save'    : True,
        #             'close'   : True,
        #         }
        #     }
        # }

    }
    new_pars_run = None

    # Plotting
    video         = False
    plot          = True
    save          = True
    save_all_data = True

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