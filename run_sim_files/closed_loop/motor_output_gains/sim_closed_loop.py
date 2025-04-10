''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

import network_experiments.default_parameters.zebrafish.closed_loop.default as default
import motor_gains_optimization

###############################################################################
# SAVED PARAMETERS ############################################################
###############################################################################

def get_muscle_tag_and_drive_amp_from_muscle_freq(muscle_resonant_freq: float):
    ''' Get the muscle tag and drive amplitude from the muscle resonant frequency '''

    stored_solutions = {

        '5000' : {
            'min_error'  : 'NDA',

            'muscle_tag' : 'new_FN_5000_ZC_1000_G0_419_gen_100',
            'muscle_note': '1 head and 3 tail joints capped',

            'mc_gain_axial' : [
                0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
                0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
                0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
            ],
        },

        '7000' : {
            'note'      : 'BAD MATCH',
            'min_error' : '33.18%',
            'muscle_tag': 'FN_7000_ZC_1000_G0_419_gen_199',
            'mo_gains_axial' : [
                0.17296, 0.05268, 0.07437, 0.08956, 0.11122,
                0.12335, 0.13738, 0.13922, 0.15144, 0.20753,
                0.22263, 0.16034, 0.03595, 0.00000, 0.00000,
            ],
        },

        '8000' : {
            'note'      : 'Good match but low IPL_AX_A',
            'min_error' : '1.22%',
            'muscle_tag': 'FN_8000_ZC_1000_G0_419_gen_199',
            'mo_gains_axial' : [

                # GAIN = 0, OFF = 1
                0.36884, 0.10991, 0.09771, 0.10309, 0.11669,
                0.12986, 0.15152, 0.16642, 0.20119, 0.27481,
                0.30989, 0.32010, 0.28163, 0.00000, 0.00000,

                # GAIN = 1, OFF = 0
                # 0.20861, 0.06785, 0.05948, 0.06097, 0.06729,
                # 0.07324, 0.08466, 0.09167, 0.10859, 0.14977,
                # 0.16699, 0.17065, 0.10521, 0.00000, 0.00000,
            ],
        },

    }

    muscle_freq_str = str(int(muscle_resonant_freq * 1000))

    if muscle_freq_str not in stored_solutions:
        raise ValueError(f'No stored solution for muscle frequency {muscle_resonant_freq}')

    solution   = stored_solutions[muscle_freq_str]
    muscle_tag = solution['muscle_tag']
    drive_amp  = np.array( solution['mo_gains_axial'] )

    return muscle_tag, drive_amp

###############################################################################
# MAIN ########################################################################
###############################################################################

def run(**kwargs):
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path   = '/data/pazzagli/simulation_results'
    simulation_tag = 'matching_kinematics_closed_loop'
    modname        = f'{CURRENTDIR}/net_farms_zebrafish.py'

    # Unpack parameters
    muscle_tag            = kwargs.get('muscle_tag')
    mo_gains_axial        = kwargs.get('mo_gains_axial')
    muscle_resonant_freq  = kwargs.get('muscle_resonant_freq')

    duration              = kwargs['duration']
    stim_a_off            = kwargs['stim_a_off']

    mo_cocontraction_gain = kwargs['mo_cocontraction_gain']
    mo_cocontraction_off  = kwargs['mo_cocontraction_off']

    ps_connections_range  = kwargs['ps_connections_range']
    cpg_connections_range = kwargs['cpg_connections_range']

    ex2ex_weight          = kwargs['ex2ex_weight']
    ex2in_weight          = kwargs['ex2in_weight']
    in2ex_weight          = kwargs['in2ex_weight']
    in2in_weight          = kwargs['in2in_weight']
    rs2ex_weight          = kwargs['rs2ex_weight']
    rs2in_weight          = kwargs['rs2in_weight']

    ps_min_activation_deg = kwargs['ps_min_activation_deg']
    ps_weight             = kwargs['ps_weight']

    n_trials              = kwargs['n_trials']
    tolerance             = kwargs['tolerance']
    learn_rate            = kwargs['learn_rate']

    video                 = kwargs['video']
    plot                  = kwargs['plot']
    save                  = kwargs['save']

    # Muscle parameters
    if muscle_tag is None or mo_gains_axial is None:
        muscle_tag, mo_gains_axial = get_muscle_tag_and_drive_amp_from_muscle_freq(
            muscle_resonant_freq = muscle_resonant_freq,
        )

    # PARAMETERS
    simulation_params = default.get_default_parameters(
        muscle_parameters_tag = muscle_tag,
    )

    simulation_params['results_path']   = results_path
    simulation_params['simulation_tag'] = simulation_tag
    simulation_params['modname']        = modname

    # Simulation parameters
    simulation_params['duration']   = duration
    simulation_params['stim_a_off'] = stim_a_off

    # Cocontraction terms
    simulation_params['mo_cocontraction_gain'] = mo_cocontraction_gain
    simulation_params['mo_cocontraction_off']  = mo_cocontraction_off

    # Connectivity parameters
    simulation_params['ps_connections_range']  = ps_connections_range
    simulation_params['cpg_connections_range'] = cpg_connections_range

    # Connection weights
    simulation_params['ex2ex_weight'] = ex2ex_weight
    simulation_params['ex2in_weight'] = ex2in_weight
    simulation_params['in2ex_weight'] = in2ex_weight
    simulation_params['in2in_weight'] = in2in_weight
    simulation_params['rs2ex_weight'] = rs2ex_weight
    simulation_params['rs2in_weight'] = rs2in_weight

    # Feedback parameters
    simulation_params['ps_min_activation_deg'] = ps_min_activation_deg
    simulation_params['ps_weight']             = ps_weight

    # Kinematics data
    kinematics_data  = simulation_params['kinematics_data']
    ref_joint_angles = np.array( kinematics_data['joints_displ_amp'] )

    ########################################################
    # OPTIMIZE AND RUN RESULT ##############################
    ########################################################
    run_iteration_kwargs = {
        'sim_pars': simulation_params,
        'video'   : False,
        'plot'    : False,
        'save'    : False,
    }

    run_result_kwargs = {
        'sim_pars': simulation_params,
        'video'   : video,
        'plot'    : plot,
        'save'    : save,
    }

    (
        mo_gains_axial,
        results_run,
    ) = default.optimize_motor_output_gains(
        mo_gains_axial       = mo_gains_axial,
        ref_joint_angles     = ref_joint_angles,
        run_iteration_func   = motor_gains_optimization.run_iteration,
        run_iteration_kwargs = run_iteration_kwargs,
        n_trials             = n_trials,
        tolerance            = tolerance,
        learn_rate           = learn_rate,
        run_result           = True,
        run_result_kwargs    = run_result_kwargs,
    )

    # Check the results
    default.study_sim_results(
        metrics_runs   = results_run,
        reference_data = simulation_params,
        plot           = True
    )

    return
