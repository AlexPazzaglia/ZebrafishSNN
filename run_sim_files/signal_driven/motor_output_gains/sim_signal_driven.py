''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import network_experiments.default_parameters.zebrafish.signal_driven.default as default
import motor_gains_optimization

###############################################################################
# SAVED PARAMETERS ############################################################
###############################################################################

def get_muscle_tag_and_drive_amp_from_muscle_freq(muscle_resonant_freq: float):
    ''' Get the muscle tag and drive amplitude from the muscle resonant frequency '''

    stored_solutions = {
        '7000' : {
            'note'      : 'The tail is unable to follow the commanded wave',
            'min_error' : '14.61%',
            'muscle_tag': 'FN_7000_ZC_1000_G0_419_gen_199',
            'drive_amp' : [
                # 0.2690949, 0.0016843, 0.1607933, 0.1854641, 0.1701128,
                # 0.1588735, 0.1652406, 0.1599254, 0.0000178, 0.2286149,
                # 0.2584264, 0.0000100, 0.0000100, 0.0000100, 0.0000100,

                # BSL = 10.0
                0.94408, 0.49145, 0.42450, 0.42433, 0.46729,
                0.50857, 0.57207, 0.57723, 0.62129, 0.83237,
                0.89036, 0.45668, 0.87675, 0.00000, 0.00000,            ],
        },

        '8000' : {
            'note'      : 'Good match',
            'min_error' : '0.20%',
            'muscle_tag': 'FN_8000_ZC_1000_G0_419_gen_199',
            'drive_amp' : [
                0.18517, 0.06995, 0.05880, 0.05807, 0.06465,
                0.07052, 0.08064, 0.08626, 0.10184, 0.13916,
                0.15195, 0.15326, 0.03031, 0.00000, 0.00000,

                # BSL = 10.0
                # 0.84143, 0.35652, 0.34896, 0.37986, 0.44980,
                # 0.51046, 0.60034, 0.66353, 0.79728, 1.09374,
                # 1.24533, 1.40108, 1.61011, 0.00000, 0.00000,
            ],
        },

        '9000' : {
            'note'      : 'Good match',
            'min_error' : '0.14%',
            'muscle_tag': 'FN_9000_ZC_1000_G0_419_gen_199',
            'drive_amp' : [
                0.16104, 0.06055, 0.05227, 0.05298, 0.06008,
                0.06654, 0.07676, 0.08368, 0.09928, 0.13709,
                0.15275, 0.16098, 0.07325, 0.00000, 0.00000,
            ],
        }
    }

    muscle_freq_str = str(int(muscle_resonant_freq * 1000))

    if muscle_freq_str not in stored_solutions:
        raise ValueError(f'No stored solution for muscle frequency {muscle_resonant_freq}')

    solution   = stored_solutions[muscle_freq_str]
    muscle_tag = solution['muscle_tag']
    drive_amp  = solution['drive_amp']
    drive_amp  = np.array(drive_amp)

    return muscle_tag, drive_amp

###############################################################################
# MAIN ########################################################################
###############################################################################

def run(**kwargs):
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path   = '/data/pazzagli/simulation_results'
    simulation_tag = 'matching_kinematics_signal_driven'

    # Muscle parameters
    muscle_resonant_freq = kwargs['muscle_resonant_freq']

    # Simulation parameters
    duration              = kwargs['duration']

    # Activation signal
    frequency             = kwargs['frequency']
    muscle_bsl            = kwargs['muscle_bsl']

    # Optimize
    n_trials              = kwargs['n_trials']
    tolerance             = kwargs['tolerance']
    learn_rate            = kwargs['learn_rate']

    # Result
    video                 = kwargs['video']
    plot                  = kwargs['plot']
    save                  = kwargs['save']

    # Muscle parameters
    muscle_tag, drive_amp_axial = get_muscle_tag_and_drive_amp_from_muscle_freq(
        muscle_resonant_freq = muscle_resonant_freq,
    )

    # PARAMETERS
    simulation_params   = default.get_default_parameters(
        muscle_parameters_tag = muscle_tag,
    )

    simulation_params['results_path']   = results_path
    simulation_params['simulation_tag'] = simulation_tag
    simulation_params['duration']       = duration
    simulation_params['frequency']      = frequency
    simulation_params['muscle_bsl']     = muscle_bsl

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
        drive_amp_axial,
        results_run,
    ) = default.optimize_motor_output_gains(
        mo_gains_axial       = drive_amp_axial,
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

