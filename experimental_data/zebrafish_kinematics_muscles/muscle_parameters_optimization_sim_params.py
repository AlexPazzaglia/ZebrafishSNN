''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from typing import Callable

###############################################################################
# GLOBALS #####################################################################
###############################################################################

OPT_RESULTS_DIR = 'experimental_data/zebrafish_kinematics_muscles/results'

N_JOINTS_AXIS   = 15
DEFAULT_ALPHA   = 8.4e-9
DEFAULT_BETA    = 1.0e-8
DEFAULT_DELTA   = 1.0e-8
DEFAULT_GAMMA   = 1.0
DEFAULT_EPSILON = 0

WATERLESS_OPTIONS = {
    'gravity'                         : (0.0, 0.0, 0.0),
    'spawn_parameters_options'        : { 'pose' : ( 0.0, 0.0, 1.01, 0.0, 0.0, np.pi )},
    'linear_drag_coefficients_options': [
        [
            [f'link_{i}' for i in range(N_JOINTS_AXIS+1)],
            [-0.0, -0.0, -0.0]
        ]
    ],
}


###############################################################################
# UTILS #######################################################################
###############################################################################
def get_optimization_name(
    target_wn   : float,
    target_zc   : float,
    target_g0   : float,
    name_prefix : str = 'muscle_parameters_optimization',
):
    ''' Get optimization name '''
    optimization_name = (
        f'{name_prefix}_'
        f'FN_{round(target_wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_zc*1e3)}_'
        f'G0_{round(target_g0*1e3)}'
    )
    return optimization_name

def _get_scaled_muscle_parameters_options(
    n_joints_axis       : int,
    gains_scalings_alpha: np.ndarray,
    gains_scalings_beta : np.ndarray,
    gains_scalings_delta: np.ndarray,
):
    ''' Get scaled muscle parameters options '''
    muscle_parameters_options = [
        [
            [f'joint_{muscle}'],
            {
                'alpha'  : DEFAULT_ALPHA * gains_scalings_alpha[muscle],
                'beta'   : DEFAULT_BETA  * gains_scalings_beta[muscle],
                'delta'  : DEFAULT_DELTA * gains_scalings_delta[muscle],
                'gamma'  : DEFAULT_GAMMA,
                'epsilon': DEFAULT_EPSILON,
            }
        ]
        for muscle in range(n_joints_axis)
    ]

    return muscle_parameters_options

def _get_scaled_muscle_parameters_options_with_stiff_joints(
    n_joints_axis       : int,
    gains_scalings_alpha: np.ndarray,
    gains_scalings_beta : np.ndarray,
    gains_scalings_delta: np.ndarray,
    joint_indices       : list[int],
    stiff_factor        : float = 10.0,
):
    ''' Get scaled muscle parameters options '''

    if isinstance(joint_indices, int):
        joint_indices = [joint_indices]
    if isinstance(joint_indices, list):
        joint_indices = np.array(joint_indices)

    # STIFF JOINTS
    gains_alpha = np.ones_like(gains_scalings_alpha) * stiff_factor
    gains_beta  = np.ones_like(gains_scalings_beta)  * stiff_factor
    gains_delta = np.ones_like(gains_scalings_delta) * stiff_factor**0.5

    # SOFT JOINTS (ACTIVE ONLY)
    gains_alpha[joint_indices] = gains_scalings_alpha[joint_indices]
    gains_beta [joint_indices] = gains_scalings_beta [joint_indices]
    gains_delta[joint_indices] = gains_scalings_delta[joint_indices]

    # Get muscle parameters options
    muscle_parameters_options = [
        [
            [f'joint_{muscle}'],
            {
                'alpha'  : DEFAULT_ALPHA * gains_alpha[muscle],
                'beta'   : DEFAULT_BETA  * gains_beta[muscle],
                'delta'  : DEFAULT_DELTA * gains_delta[muscle],
                'gamma'  : DEFAULT_GAMMA,
                'epsilon': DEFAULT_EPSILON,
            }
        ]
        for muscle in range(n_joints_axis)
    ]

    return muscle_parameters_options

def _get_available_iterations(
    optimization_path: str,
    optimization_name: str,
):
    ''' Get available iterations '''

    folder_name = f'{optimization_path}/{optimization_name}'
    iteration_files = [
        file for file in os.listdir(folder_name)
        if file.startswith('performance_iteration_')
    ]
    iteration_indices = [
        int(file.split('_')[-1].split('.')[0])
        for file in iteration_files
    ]
    return sorted(iteration_indices)

def _get_first_iteration(
    optimization_path: str,
    optimization_name: str,
):
    ''' Get the first iteration '''

    iteration_indices = _get_available_iterations(
        optimization_path = optimization_path,
        optimization_name = optimization_name,
    )

    return iteration_indices[0]

def _get_last_iteration(
    optimization_path: str,
    optimization_name: str,
):
    ''' Get the last iteration '''

    iteration_indices = _get_available_iterations(
        optimization_path = optimization_path,
        optimization_name = optimization_name,
    )

    return iteration_indices[-1]

def _get_results_from_iteration(
    optimization_path: str,
    optimization_name: str,
    iteration_ind    : int,
):
    ''' Get performance from iteration '''

    folder_name = f'{optimization_path}/{optimization_name}'

    if iteration_ind == -1:
        iteration_ind = _get_last_iteration(
            optimization_path = optimization_path,
            optimization_name = optimization_name,
        )

    file_name   = f'{folder_name}/performance_iteration_{iteration_ind}.dill'
    with open(file_name, 'rb') as file:
        performance = dill.load(file)

    return performance

###############################################################################
# MOTOR OUTPUT FUNCTION #######################################################
###############################################################################
def get_sweep_function(
    f_min   : float,
    f_max   : float,
    duration: float,
):
    ''' Get sweep function '''
    freq_fun  = lambda time : f_min + (f_max - f_min) / duration * time
    sweep_fun = lambda time : np.sin( 2*np.pi * freq_fun(time) * time )
    return sweep_fun

def get_motor_output_function_params_multi_joint(
    n_joints_axis : int,
    joints_indices: np.ndarray[int],
    sig_function  : Callable,
    sig_amplitude : float,
    sig_baseline  : float,
    sig_frequency : float,
    ipl_arr       : np.ndarray[float] = None,
    amp_arr       : np.ndarray[float] = None,
    off_arr       : np.ndarray[float] = None,
    bsl_arr       : np.ndarray[float] = None,
) -> dict:
    ''' Get motor output function params for multiple joints '''

    # Joints phase lags
    if ipl_arr is None:
        ipl_arr = np.zeros(n_joints_axis)

    # Joints offsets
    if off_arr is None:
        off_arr = np.zeros(n_joints_axis)

    # Joints amplitude
    if amp_arr is None:
        amp_arr                 = np.zeros(n_joints_axis)
        amp_arr[joints_indices] = sig_amplitude

    # Joints baseline
    if bsl_arr is None:
        bsl_arr                 = np.zeros(n_joints_axis)
        bsl_arr[joints_indices] = sig_baseline

    # Parameters
    motor_output_function_params = {
        'sig_function' : sig_function,

        'axis': {
            'frequency': sig_frequency,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['AXIS'],

            'n_joints': n_joints_axis,
            'ipl_arr' : ipl_arr,
            'amp_arr' : amp_arr,
            'off_arr' : off_arr,
            'bsl_arr' : bsl_arr,
        },
    }

    return motor_output_function_params

def get_motor_output_function_params_single_joint(
    n_joints_axis       : int,
    joint_index         : int,
    sig_function        : Callable,
    sig_amplitude       : float,
    sig_baseline        : float,
    sig_frequency       : float,
) -> dict:
    ''' Get motor output function params for a single joint '''

    return get_motor_output_function_params_multi_joint(
        n_joints_axis  = n_joints_axis,
        joints_indices = np.array([joint_index]),
        sig_function   = sig_function,
        sig_amplitude  = sig_amplitude,
        sig_baseline   = sig_baseline,
        sig_frequency  = sig_frequency,
    )

###############################################################################
# RUN PARAMS ##################################################################
###############################################################################
def get_run_params_for_multi_joint_activations(
    waterless                   : bool,
    n_joints_axis               : int,
    tested_joints               : list[int],
    sig_function                : Callable,
    sig_amplitude               : float,
    sig_baseline                : float,
    sig_frequency               : float,
    gains_scalings_alpha        : np.ndarray,
    gains_scalings_beta         : np.ndarray,
    gains_scalings_delta        : np.ndarray,
    ipl_arr                     : np.ndarray[float] =  None,
    amp_arr                     : np.ndarray[float] =  None,
    off_arr                     : np.ndarray[float] =  None,
    bsl_arr                     : np.ndarray[float] =  None,
    inactive_joints_stiff       : bool  = False,
    inactive_joints_stiff_factor: float = 10.0,
):
    ''' Get run params for multiple joints '''

    # Motor output signals
    sig_pars = get_motor_output_function_params_multi_joint(
        n_joints_axis  = n_joints_axis,
        joints_indices = np.array(tested_joints),
        sig_function   = sig_function,
        sig_amplitude  = sig_amplitude,
        sig_baseline   = sig_baseline,
        sig_frequency  = sig_frequency,
        ipl_arr        = ipl_arr,
        amp_arr        = amp_arr,
        off_arr        = off_arr,
        bsl_arr        = bsl_arr,
    )

    # Muscle parameters options
    if inactive_joints_stiff:
        muscle_pars = _get_scaled_muscle_parameters_options_with_stiff_joints(
            n_joints_axis        = n_joints_axis,
            gains_scalings_alpha = gains_scalings_alpha,
            gains_scalings_beta  = gains_scalings_beta,
            gains_scalings_delta = gains_scalings_delta,
            joint_indices        = tested_joints,
            stiff_factor         = inactive_joints_stiff_factor,
        )
    else:
        muscle_pars = _get_scaled_muscle_parameters_options(
            n_joints_axis        = n_joints_axis,
            gains_scalings_alpha = gains_scalings_alpha,
            gains_scalings_beta  = gains_scalings_beta,
            gains_scalings_delta = gains_scalings_delta,
        )

    # Water options
    water_options = {} if not waterless else WATERLESS_OPTIONS

    # Params runs
    params_runs = [
        {
            'tag_run'                 : 0,
            'motor_output_signal_pars': sig_pars,

            'mech_sim_options' : {
                'muscle_parameters_options': muscle_pars,
                'save_all_metrics_data'    : True,
            } | water_options
        }
    ]

    return params_runs

def get_run_params_for_single_joint_activations(
    waterless                   : bool,
    n_joints_axis               : int,
    tested_joints               : list[int],
    sig_function                : Callable,
    sig_amplitude               : float,
    sig_baseline                : float,
    sig_frequency               : float,
    gains_scalings_alpha        : np.ndarray,
    gains_scalings_beta         : np.ndarray,
    gains_scalings_delta        : np.ndarray,
    inactive_joints_stiff       : bool = False,
    inactive_joints_stiff_factor: float = 10.0,
):
    ''' Get run params for a single joint '''


    # Motor output signals
    sig_pars = [
        get_motor_output_function_params_single_joint(
            n_joints_axis  = n_joints_axis,
            joint_index    = joint_index,
            sig_function   = sig_function,
            sig_amplitude  = sig_amplitude,
            sig_baseline   = sig_baseline,
            sig_frequency  = sig_frequency,
        )
        for joint_index in tested_joints
    ]

    # Muscle parameters options
    if inactive_joints_stiff:
        muscle_pars = [
            _get_scaled_muscle_parameters_options_with_stiff_joints(
                n_joints_axis        = n_joints_axis,
                gains_scalings_alpha = gains_scalings_alpha,
                gains_scalings_beta  = gains_scalings_beta,
                gains_scalings_delta = gains_scalings_delta,
                joint_indices        = joint_index,
                stiff_factor         = inactive_joints_stiff_factor,
            )
            for joint_index in tested_joints
        ]
    else:
        muscle_pars = [
            _get_scaled_muscle_parameters_options(
                n_joints_axis        = n_joints_axis,
                gains_scalings_alpha = gains_scalings_alpha,
                gains_scalings_beta  = gains_scalings_beta,
                gains_scalings_delta = gains_scalings_delta,
            )
            for joint_index in tested_joints
        ]

    # Water options
    water_options = {} if not waterless else WATERLESS_OPTIONS

    # Params runs
    params_runs = [
        {
            'tag_run'                 : run_index,
            'motor_output_signal_pars': sig_pars[run_index],

            'mech_sim_options' : {
                'muscle_parameters_options': muscle_pars[run_index],
                'save_all_metrics_data'    : True,
            } | water_options
        }
        for run_index, joint_index in enumerate(tested_joints)
    ]

    return params_runs

###############################################################################
# MUSCLE SCALING ##############################################################
###############################################################################
def _get_muscle_scalings_from_iteration(
    optimization_path: str,
    optimization_name: str,
    iteration_ind    : int,
):
    ''' Get muscle scaling factors from iteration'''

    iteration_results = _get_results_from_iteration(
        optimization_path = optimization_path,
        optimization_name = optimization_name,
        iteration_ind     = iteration_ind,
    )

    return (
        iteration_results['gains_scalings_alpha'],
        iteration_results['gains_scalings_beta'],
        iteration_results['gains_scalings_delta']
    )

def get_muscle_scalings(
    n_joints_axis            : int,
    manual_scaling_alpha     : np.ndarray,
    manual_scaling_beta      : np.ndarray,
    manual_scaling_delta     : np.ndarray,
    use_original_scalings    : bool  = None,
    use_optimization_scalings: bool  = None,
    optimization_path        : str   = None,
    optimization_name        : str   = None,
    optimization_iteration   : int   = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Get muscle scaling factors '''

    if np.count_nonzero((use_optimization_scalings, use_original_scalings)) != 1:
        raise ValueError('Only one of the three options can be True')

    # Scalings of the original model
    if use_original_scalings:
        scalings = (
            np.ones(n_joints_axis),
            np.ones(n_joints_axis),
            np.ones(n_joints_axis),
        )

    # Scalings from optimization results
    if use_optimization_scalings:
        scalings =  _get_muscle_scalings_from_iteration(
            optimization_path = optimization_path,
            optimization_name = optimization_name,
            iteration_ind     = optimization_iteration,
        )

    # Apply manual scalings
    gains_scalings_alpha, gains_scalings_beta, gains_scalings_delta = scalings

    gains_scalings_alpha *= manual_scaling_alpha
    gains_scalings_beta  *= manual_scaling_beta
    gains_scalings_delta *= manual_scaling_delta

    return gains_scalings_alpha, gains_scalings_beta, gains_scalings_delta

def get_starting_muscle_scalings(
    start_iteration: int,
    opt_name       : str,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_joints_axis   = kwargs['n_joints_axis']
    opt_path        = kwargs['opt_results_dir']

    # Get muscle scalings
    scalings = get_muscle_scalings(
        n_joints_axis             = n_joints_axis ,
        manual_scaling_alpha      = np.ones(n_joints_axis) ,
        manual_scaling_beta       = np.ones(n_joints_axis) ,
        manual_scaling_delta      = np.ones(n_joints_axis) ,
        use_original_scalings     = ( start_iteration == 0 ),
        use_optimization_scalings = ( start_iteration != 0 ),
        optimization_path         = opt_path ,
        optimization_name         = opt_name ,
        optimization_iteration    = start_iteration ,
    )

    return scalings

###############################################################################
# MUSCLE PARAMETERS OPTIONS ###################################################
###############################################################################

def get_muscle_parameters_options_from_solution(
    n_joints_axis            : int,
    optimization_path        : str,
    optimization_name        : str,
    optimization_iteration   : int,
):
    ''' Get muscle parameters options from an optimization solution '''

    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta,
    ) = get_muscle_scalings(
        n_joints_axis            = n_joints_axis,
        manual_scaling_alpha     = np.ones(n_joints_axis),
        manual_scaling_beta      = np.ones(n_joints_axis),
        manual_scaling_delta     = np.ones(n_joints_axis),
        use_original_scalings    = False,
        use_optimization_scalings= True,
        optimization_path        = optimization_path,
        optimization_name        = optimization_name,
        optimization_iteration   = optimization_iteration,
    )

    muscle_parameters_options = _get_scaled_muscle_parameters_options(
        n_joints_axis        = n_joints_axis,
        gains_scalings_alpha = gains_scalings_alpha,
        gains_scalings_beta  = gains_scalings_beta,
        gains_scalings_delta = gains_scalings_delta,
    )

    return muscle_parameters_options
