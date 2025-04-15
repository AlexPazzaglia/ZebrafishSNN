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
            'min_error'  : '1.90%',

            'muscle_tag' : 'new_FN_5000_ZC_1000_G0_419_gen_100',
            'muscle_note': '1 head and 3 tail joints capped',

            'mc_gain_axial' : [
                0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
                0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
                0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
            ],
        },

        '7000' : {
            'note'      : '',
            'min_error' : '8.91%',
            'muscle_tag': 'FN_7000_ZC_1000_G0_419_gen_199',
            'mc_gain_axial' : [
                0.32395, 0.13874, 0.09938, 0.12272, 0.10301,
                0.10099, 0.11082, 0.10114, 0.05633, 0.12465,
                0.14184, 0.00270, 0.00108, 0.00000, 0.00000,
            ],
        },

        '8000' : {
            'note'      : 'Good match but very low IPL_AX_A = 0.01095',
            'min_error' : '-0.60%',
            'muscle_tag': 'FN_8000_ZC_1000_G0_419_gen_199',
            'mc_gain_axial' : [
                0.35513, 0.09846, 0.08382, 0.08764, 0.09799,
                0.10896, 0.12629, 0.13680, 0.16420, 0.22136,
                0.24820, 0.26054, 0.21957, 0.00000, 0.00000,
            ],
        },

    }

    muscle_freq_str = str(int(muscle_resonant_freq * 1000))

    if muscle_freq_str not in stored_solutions:
        raise ValueError(f'No stored solution for muscle frequency {muscle_resonant_freq}')

    solution   = stored_solutions[muscle_freq_str]
    muscle_tag = solution['muscle_tag']
    drive_amp  = np.array( solution['mc_gain_axial'] )

    return muscle_tag, drive_amp

###############################################################################
# MAIN ########################################################################
###############################################################################

def run(**kwargs):
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path   = 'simulation_results'
    simulation_tag = 'matching_kinematics_open_loop'
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

    n_trials              = kwargs['n_trials']
    tolerance             = kwargs['tolerance']
    learn_rate            = kwargs['learn_rate']

    video                 = kwargs['video']
    plot                  = kwargs['plot']
    save                  = kwargs['save']

    ps_min_activation_deg = kwargs.get('ps_min_activation_deg')
    ps_weight             = kwargs.get('ps_weight')

    if ps_min_activation_deg is not None or ps_weight is not None:
        print('Warning: PS parameters are not used in this simulation, will be ignored')

    ps_min_activation_deg = 360.0
    ps_weight             = 0.0

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

    # Open loop
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

