import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Callable, Any
from brian2.units.allunits import second, msecond

from network_experiments.snn_utils import MODELS_FARMS, MODELS_OPENLOOP

from experimental_data.zebrafish_kinematics_muscles. \
    muscle_parameters_optimization_load_results import (
    load_muscle_parameters_options_from_optimization
)

from network_experiments.snn_signals_kinematics import get_kinematics_output_signal

###############################################################################
## CONSTANTS ##################################################################
###############################################################################

### MODEL DATA
MODEL_NAME     = 'zebrafish_experimental'
N_JOINTS_AXIS  = 15
N_LINKS_AXIS   = N_JOINTS_AXIS + 1

POINTS_POSITIONS_MODEL = np.array(
    [
        0.000,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.010,
        0.011,
        0.012,
        0.013,
        0.014,
        0.015,
        0.016,
        0.017,
        0.018,
    ]
)

LINKS_COM_POSITIONS_MODEL = np.array(
    [
        ( POINTS_POSITIONS_MODEL[i] + POINTS_POSITIONS_MODEL[i+1] ) / 2
        for i in range(N_LINKS_AXIS)
    ]
)

LINKS_MASSES = np.array(
    [
        3.2734375504999984e-06,
        4.0521463491666665e-06,
        4.854872365333333e-06,
        5.157439432833333e-06,
        5.498348945e-06,
        5.479874165666666e-06,
        5.159457690333333e-06,
        4.249855453833333e-06,
        3.4060886698333332e-06,
        2.650774316333333e-06,
        1.7336472188333334e-06,
        1.1791477706666665e-06,
        8.259051296666667e-07,
        9.194377328333334e-07,
        9.366918884999999e-07,
        8.46120548e-07,
    ]
)

LENGTH_AXIS_MODEL      = POINTS_POSITIONS_MODEL[-1]
JOINTS_POSITIONS_MODEL = POINTS_POSITIONS_MODEL[1:-1]
LENGTHS_LINKS_MODEL    = np.diff(POINTS_POSITIONS_MODEL)
COM_POS_MODEL          = np.sum(LINKS_COM_POSITIONS_MODEL * LINKS_MASSES) / np.sum(LINKS_MASSES)

POINTS_POSITIONS_R_MODEL = POINTS_POSITIONS_MODEL / LENGTH_AXIS_MODEL
JOINTS_POSITIONS_R_MODEL = JOINTS_POSITIONS_MODEL / LENGTH_AXIS_MODEL
LENGTHS_LINKS_R_MODEL    = LENGTHS_LINKS_MODEL / LENGTH_AXIS_MODEL
COM_POS_R_MODEL          = COM_POS_MODEL / LENGTH_AXIS_MODEL

N_SEGMENTS_AXIS          = 32
FIRST_ACTIVE_SEGMENT_POS = POINTS_POSITIONS_MODEL[1]
LAST_ACTIVE_SEGMENT_POS  = POINTS_POSITIONS_MODEL[-4]
LENGTH_ACTIVE_BODY       = LAST_ACTIVE_SEGMENT_POS - FIRST_ACTIVE_SEGMENT_POS
ACTIVE_BODY_RATIO        = LENGTH_ACTIVE_BODY / LENGTH_AXIS_MODEL

TARGET_WAVE_NUMBER = 0.95
TARGET_NEUR_TWL    = TARGET_WAVE_NUMBER * ACTIVE_BODY_RATIO
TARGET_NEUR_IPL    = TARGET_NEUR_TWL / (N_SEGMENTS_AXIS - 1)

### SIMULATION PARAMETERS
PARAMS_SIMULATION = {
    'animal_model'             : 'zebrafish_v1',
    'timestep'                 : 1 * msecond,
    'duration'                 : 10 * second,
    'verboserun'               : False,
    'load_connectivity_indices': False,
    'set_seed'                 : True,
    'seed_value'               : 3678586,
    'gaitflag'                 : 0,
    'ps_gain_axial'            : 0.0,
    'stim_a_off'               : 0.0,
}

### FEEDBACK
RHEOBASE_CURRENT_PS = 100.0

### MUSCLE PARAMETERS
# MUSCLE_PARAMETERS_TAG = 'FN_30000_ZC_1000_G0_419_gen_99'
# MUSCLE_PARAMETERS_TAG = 'FN_15000_ZC_1000_G0_419_gen_99'
# MUSCLE_PARAMETERS_TAG = 'FN_8000_ZC_1000_G0_419_gen_199'
MUSCLE_PARAMETERS_TAG = 'all_FN_10000_ZC_1000_G0_419_gen_100'


### SIGNAL-DRIVEN AMPS
SIGNAL_DRIVEN_AMPS = {
    'FN_8000_ZC_1000_G0_419_gen_199' : np.array([0.18517, 0.06995, 0.05880, 0.05807, 0.06465, 0.07052, 0.08064, 0.08626, 0.10184, 0.13916, 0.15195, 0.15326, 0.03031, 0.00000, 0.00000]),
    'FN_30000_ZC_1000_G0_419_gen_99' : np.array([0.18195, 0.07197, 0.06235, 0.06190, 0.06905, 0.07548, 0.08525, 0.09064, 0.10876, 0.14818, 0.16362, 0.17330, 0.10699, 0.00000, 0.00000]),

    # Not optimized
    'all_FN_10000_ZC_1000_G0_419_gen_100' : np.array([0.11807, 0.06749, 0.05767, 0.05748, 0.06420, 0.07038, 0.08006, 0.08562, 0.09979, 0.13205, 0.14232, 0.14135, 0.20913, 0.00000, 0.00000]),
}

### MC GAINS (CLOSED LOOP)
MC_GAINS = {
    'FN_8000_ZC_1000_G0_419_gen_199' : np.array([0.35961, 0.11618, 0.10161, 0.10581, 0.11585, 0.12822, 0.14842, 0.15795, 0.18554, 0.25109, 0.27617, 0.28025, 0.13421, 0.00000, 0.00000]),

    'FN_15000_ZC_1000_G0_419_gen_99' : np.array([0.24461, 0.09457, 0.09017, 0.09715, 0.11201, 0.12863, 0.15503, 0.17278, 0.21305, 0.30060, 0.34317, 0.39064, 0.43644, 0.00000, 0.00000]),
    'FN_30000_ZC_1000_G0_419_gen_99' : np.array([0.23178, 0.08624, 0.08315, 0.09155, 0.10738, 0.12095, 0.14442, 0.16446, 0.20367, 0.28890, 0.34091, 0.39566, 0.50805, 0.00000, 0.00000]),

    # Not optimized
    'all_FN_10000_ZC_1000_G0_419_gen_100' : np.array([0.35961, 0.11618, 0.10161, 0.10581, 0.11585, 0.12822, 0.14842, 0.15795, 0.18554, 0.25109, 0.27617, 0.28025, 0.13421, 0.00000, 0.00000]),
}

### KINEMATICS DATA
KINEMATICS_DATA = {

    # From Jensen 2023
    'frequency'     : 15.0,
    'wave_number_ax': 1.00,
    'speed_fwd_bl'  : 3.91,
    'tail_beat_bl'  : 0.027,

    'ipl_array' : np.linspace(0, TARGET_NEUR_TWL, N_JOINTS_AXIS),

    # Joints and points positions in the tracking data
    'joints_pos' : np.array(
        [
            0.00177778,
            0.00355556,
            0.00533333,
            0.00711111,
            0.00888889,
            0.01066667,
            0.01244445,
            0.01422222,
            0.016,
        ]
    ),
    'points_pos' : np.array(
        [
            0.0,
            0.00177778,
            0.00355556,
            0.00533333,
            0.00711111,
            0.00888889,
            0.01066667,
            0.01244445,
            0.01422222,
            0.016,
            0.018,
        ]
    ),

    # Joints and links amplitudes (mapped to the model)
    # In degrees:
    # array(
    #   [
    #       3.77006229, 1.6100114 , 1.59339563,
    #       1.7458024 , 2.07582609, 2.36459682,
    #       2.78686672, 3.09282618, 3.72880933,
    #       5.12510748, 5.88484951, 6.75459945,
    #       8.55368692, 0.        , 0.
    #   ]
    # )
    'joints_displ_amp': np.array(
        [
            0.06580,
            0.02810,
            0.02781,
            0.03047,
            0.03623,
            0.04127,
            0.04864,
            0.05398,
            0.06508,
            0.08945,
            0.10271,
            0.11789,
            0.14929,
            0.0,      # Note: Tail moves passively,
            0.0,      # Note: Tail moves passively,
        ]
    ),

    # TODO: To get from the tracking data
    'links_displ_amp': np.interp(
        x  = POINTS_POSITIONS_R_MODEL,
        xp = [1.00],
        fp = [0.027],
    ),

    # Signal-driven control
    'joints_signals_amps' : SIGNAL_DRIVEN_AMPS[MUSCLE_PARAMETERS_TAG],

    # Closed-loop control
    'mc_gains' : MC_GAINS[MUSCLE_PARAMETERS_TAG],
}

###############################################################################
## SIMULATION OPTIONS #########################################################
###############################################################################

def get_save_all_parameters_options(
    save_cycle_freq : bool = True,
    save_voltages   : bool = True,
    save_emg_traces : bool = True,
    save_to_csv     : bool = True,
    save_synapses   : bool = True,
    save_currents   : bool = True,
):
    ''' Save all parameters option '''

    mon_pars = {'active': True, 'save': True, 'to_csv': save_to_csv, 'indices': True, 'rate': 1}
    plt_pars = {'showit': True, 'animate': False, 'gridon': False, 'densegrid': False}

    # Raster plots
    monitor_spikes = {
        'plotpars': {
            'side_ids'      : [0],
            'excluded_mods' : [],
            'order_mods'    : ['rs', 'cpg', 'mn', 'ps'],
            'insert_limbs'  : False,
            'zoom_plot'     : False,
            'mirror_plot'   : True,
            'sampling_ratio': 1.00,
            'isi_plot'      : {'showit': False, 'modules': ['ex', 'in'] },
            'emg_traces'    : {
                'showit': save_emg_traces,
                'save'  : save_emg_traces,
                'close' : save_emg_traces
            },
        } | plt_pars
    } | mon_pars

    # Smooth pool activations
    monitor_pools_activation = {
        'target_modules': [
            { 'mod_name': 'cpg', 'ner_name': 'ex', 'smooth_factor': 0.999 },
            { 'mod_name': 'cpg', 'ner_name': 'in', 'smooth_factor': 0.999 },
            { 'mod_name':  'mn', 'ner_name': 'mn', 'smooth_factor': 0.999 },
        ],
        'plotpars': {
            'sampling_ratio': 0.50,
            'cycle_freq'    : {
                'ner_name': 'mn',
                'showit'  : save_cycle_freq,
                'save'    : save_cycle_freq,
                'close'   : save_cycle_freq,
            }
        } | plt_pars
    }

    # Neuronal variables
    current_states = []
    if save_currents:
        current_states = ['I_tot', 'I_ext']

    synaptic_states = []
    if save_synapses:
        synaptic_states = [
            'I_glyc', 'g_glyc1_tot',
            'I_nmda', 'g_nmda1_tot',
            'I_ampa', 'g_ampa1_tot',
        ]

    monitor_states = {
        'variables': [ 'v' ] + current_states + synaptic_states, # 'w1'
        'plotpars' : {
            'showit' : True,
            'figure' : False,
            'animate': False,

            'voltage_traces' : {
                'modules' : ['cpg.axial.ex', 'mn.axial.mn'],   # 'mn.axial.mn'
                'showit'  : save_voltages,
                'save'    : save_voltages,
                'close'   : save_voltages,
            }
        }
    } | mon_pars

    # Muscle cells
    current_states = []
    if save_currents:
        current_states = ['I_tot']

    synaptic_states = []
    if save_synapses:
        synaptic_states = [
            'I_glyc', 'g_glyc_mc_tot',
            'I_nmda', 'g_nmda_mc_tot',
            'I_ampa', 'g_ampa_mc_tot',
        ]

    monitor_musclecells = {
        'variables': [ 'v' ] + current_states + synaptic_states,
        'plotpars' : {
            'showit'        : True,
            'filtering'     : False,
            'sampling_ratio': 0.50,
            'duty_cycle_ax' : {'showit': True, 'filter': True, 'target_seg': [5, 6, 7, 8, 9]}
        }
    } | mon_pars

    # Options
    save_everything_pars = {
        'monitor_spikes'          : monitor_spikes,
        'monitor_pools_activation': monitor_pools_activation,
        'monitor_states'          : monitor_states,
        'monitor_musclecells'     : monitor_musclecells,
    }

    return save_everything_pars

def get_signal_driven_gait_params(
    frequency   : float,
    amp_arr     : np.ndarray,
    ipl_arr     : np.ndarray,
    off_arr     : np.ndarray = None,
    bsl_arr     : np.ndarray = None,
    sig_function: Callable = None,
):
    ''' Get the gait parameters '''

    # Ex: Square signal
    # sig_function = lambda phase : np.tanh( 20 * np.cos( 2*np.pi * phase ) )

    if off_arr is None:
        off_arr = np.zeros(N_JOINTS_AXIS)

    if bsl_arr is None:
        bsl_arr = np.zeros(N_JOINTS_AXIS)

    if sig_function is None:
        sig_function = lambda phase : np.sin( 2*np.pi * phase )

    params_gait = {
        'sig_function' : sig_function,

        'axis': {
            'frequency': frequency,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['AXIS'],

            'n_joints': N_JOINTS_AXIS,
            'ipl_arr' : ipl_arr,

            'amp_arr' : amp_arr,
            'off_arr' : off_arr,
            'bsl_arr' : bsl_arr,
        },
    }

    return params_gait

def get_linear_drag_coefficients_options(
    drag_coeffs        : np.ndarray,
    scalings_x         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_y         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_z         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
) -> list:
    ''' Get the linear drag coefficients '''

    drag_coeffs[:, 0] *= scalings_x
    drag_coeffs[:, 1] *= scalings_y
    drag_coeffs[:, 2] *= scalings_z

    linear_drag_coefficients_options = [
        [
            [f'link_{link}'],
            - drag_coeffs[link],
        ]
        for link in range(N_JOINTS_AXIS+1)
    ]

    return linear_drag_coefficients_options

def get_rotational_drag_coefficients_options(
    drag_coeffs        : np.ndarray,
    scalings_x         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_y         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
    scalings_z         : np.ndarray = np.ones(N_JOINTS_AXIS+1),
) -> list:
    ''' Get the angular drag coefficients '''

    drag_coeffs[:, 0] *= scalings_x
    drag_coeffs[:, 1] *= scalings_y
    drag_coeffs[:, 2] *= scalings_z

    angular_drag_coefficients_options = [
        [
            [f'link_{link}'],
            - drag_coeffs[link],
        ]
        for link in range(N_JOINTS_AXIS+1)
    ]

    return angular_drag_coefficients_options

def get_position_control_options(
    gains_p : Union[float, list, np.ndarray],
    gains_d : Union[float, list, np.ndarray],
) -> list:
    ''' Get the drag coefficients '''

    if isinstance(gains_p, (float, int)):
        gains_p = np.ones(N_JOINTS_AXIS) * gains_p
    if isinstance(gains_d, (float, int)):
        gains_d = np.ones(N_JOINTS_AXIS) * gains_d

    position_control_gains = [
        [
            [f'joint_{link}'],
            [
                gains_p[link],
                gains_d[link],
            ]
        ]
        for link in range(N_JOINTS_AXIS)
    ]

    return position_control_gains

def get_scaled_ps_gains(
    alpha_fraction_th       : float,
    joint_displacements_amps: np.ndarray = KINEMATICS_DATA['joints_displ_amp'],
    tail_joints             : int         = 0,
):
    ''' Get the scaled proprioceptive feedback gains '''
    final_joint        = N_JOINTS_AXIS - tail_joints
    joint_amp_fraction = alpha_fraction_th * joint_displacements_amps
    scaled_ps_gains    = np.zeros(N_JOINTS_AXIS)

    scaled_ps_gains[:final_joint] = RHEOBASE_CURRENT_PS / joint_amp_fraction[:final_joint]

    return scaled_ps_gains

def get_uniform_ps_gains(
    min_activation_deg: float = 10,
    n_joints_tail     : int   = 2,
):
    ''' Get uniform PS gains '''

    n_joints_bend     = N_JOINTS_AXIS - n_joints_tail
    reference_angles  = np.ones(N_JOINTS_AXIS)
    reference_angles *= np.deg2rad(min_activation_deg) / n_joints_bend

    ps_gains = get_scaled_ps_gains(
        alpha_fraction_th        = 1.00,
        joint_displacements_amps = reference_angles,
        tail_joints              = n_joints_tail,
    )

    return ps_gains

###########################################################################
# MUSCLE PARAMETERS #######################################################
###########################################################################
def _plot_muscle_params(
    muscle_pars_options: dict,
    original_alphas    : np.ndarray,
    original_betas     : np.ndarray,
    original_deltas    : np.ndarray,
):
    ''' Plot the muscle parameters '''
    joints = np.arange(N_JOINTS_AXIS)
    alphas_new = np.array([muscle_pars_options[joint][1]['alpha'] for joint in joints])
    betas_new  = np.array([muscle_pars_options[joint][1]['beta']  for joint in joints])
    deltas_new = np.array([muscle_pars_options[joint][1]['delta'] for joint in joints])

    def _subplot(value_new, value_old, var_name, sub_plot_ind):
        plt.subplot(3, 1, sub_plot_ind)
        plt.plot(joints, value_new, 'o-', label=f'{var_name} New')
        plt.plot(joints, value_old, 'o-', label=f'{var_name} Original')
        plt.ylabel(var_name)
        plt.yscale('log')
        plt.legend()

    plt.figure(figsize=(12, 8))
    _subplot(alphas_new, original_alphas, 'Alpha', 1)
    _subplot( betas_new,  original_betas,  'Beta', 2)
    _subplot(deltas_new, original_deltas, 'Delta', 3)

    plt.xlabel('Joint Index')
    plt.tight_layout()
    plt.show()

def get_scaled_muscle_parameters_options(
    muscle_parameters_tag: str,
    muscle_factors       : np.ndarray = np.ones(N_JOINTS_AXIS),
    head_joints          : int = 1,
    tail_joints          : int = 3,
    plot_muscles         : bool = False,
):
    ''' Get the muscle parameters options '''

    # Muscle parameters
    muscle_parameters_options = load_muscle_parameters_options_from_optimization(
        optimization_name = f'muscle_parameters_optimization_{muscle_parameters_tag}.csv',
    )

    # Target joints
    head_inds = np.arange(head_joints)
    tail_inds = np.arange(N_JOINTS_AXIS - tail_joints, N_JOINTS_AXIS)

    alphas = np.array([muscle_parameters_options[joint][1]['alpha'] for joint in range(N_JOINTS_AXIS)])
    betas  = np.array([muscle_parameters_options[joint][1]['beta']  for joint in range(N_JOINTS_AXIS)])
    deltas = np.array([muscle_parameters_options[joint][1]['delta'] for joint in range(N_JOINTS_AXIS)])

    # Cap head joints
    first_non_head_joint = head_joints
    for head_i in head_inds:
        muscle_parameters_options[head_i][1]['alpha'] = alphas[first_non_head_joint]
        muscle_parameters_options[head_i][1]['beta']  =  betas[first_non_head_joint]
        muscle_parameters_options[head_i][1]['delta'] = deltas[first_non_head_joint]

    # Cap tail joints
    last_non_tail_joint = N_JOINTS_AXIS - tail_joints - 1
    for tail_i in tail_inds:
        muscle_parameters_options[tail_i][1]['alpha'] = alphas[last_non_tail_joint]
        muscle_parameters_options[tail_i][1]['beta']  =  betas[last_non_tail_joint]
        muscle_parameters_options[tail_i][1]['delta'] = deltas[last_non_tail_joint]

    # Joint stiffness scaling
    for joint_i, joint_factor in enumerate(muscle_factors):
        muscle_parameters_options[joint_i][1]['alpha'] *= joint_factor
        muscle_parameters_options[joint_i][1]['beta']  *= joint_factor
        muscle_parameters_options[joint_i][1]['delta'] *= joint_factor**0.5

    # Plot muscle parameters
    if plot_muscles:
        _plot_muscle_params(
            muscle_pars_options = muscle_parameters_options,
            original_alphas     = alphas,
            original_betas      = betas,
            original_deltas     = deltas,
        )

    return muscle_parameters_options

###############################################################################
## SIM RESULTS ################################################################
###############################################################################

def study_neural_sim_results(
    metrics_runs  : dict,
    reference_data: dict,
    run_index     : int = 0,
):
    ''' Study the results of the simulation '''

    if reference_data is None:
        reference_data = get_default_parameters()

    # Check the results
    if isinstance( metrics_runs['neur_ptcc_ax'], float ):
        assert run_index == 0, 'Only one run available'
        ptcc           = metrics_runs['neur_ptcc_ax']
        freq           = metrics_runs['neur_freq_ax']
        ipls           = metrics_runs['neur_ipl_ax_a']
        total_wave_lag = metrics_runs['neur_twl']
    else:
        ptcc           = metrics_runs['neur_ptcc_ax'][run_index]
        freq           = metrics_runs['neur_freq_ax'][run_index]
        ipls           = metrics_runs['neur_ipl_ax_a'][run_index]
        total_wave_lag = metrics_runs['neur_twl'][run_index]

    wave_number     = total_wave_lag / ACTIVE_BODY_RATIO
    wave_number_err = (wave_number - TARGET_WAVE_NUMBER) / TARGET_WAVE_NUMBER * 100

    print(f'PTCC           : {ptcc:.4f}')
    print(f'Frequency      : {freq:.4f}')
    print(f'IPL            : {ipls:.4f}')
    print(f'Total wave lag : {total_wave_lag:.4f}')
    print(f'Wave number    : {wave_number:.4f} (target: {TARGET_WAVE_NUMBER:.4f}, error: {wave_number_err:.2f}%)')

    return

def study_mechanical_sim_results(
    metrics_runs  : dict,
    reference_data: dict,
    run_index     : int = 0,
    plot          : bool = True,
):
    ''' Study the results of the simulation '''

    # Check the results
    if isinstance( metrics_runs['mech_freq_ax'], float ):
        assert run_index == 0, 'Only one run available'
        actual_joints_displ = metrics_runs['mech_joints_disp_amp']

    else:
        actual_joints_displ = metrics_runs['mech_joints_disp_amp'][run_index]

    # Compare actual and commanded positions
    kinematics_data      = reference_data['kinematics_data']
    desired_joints_displ = np.array( kinematics_data['joints_displ_amp'] )

    # Compute norm
    tail_joints      = 2
    joint_angles     = actual_joints_displ[:-tail_joints]
    ref_joint_angles = desired_joints_displ[:-tail_joints]

    joints_diff_s = ( joint_angles - ref_joint_angles ) / ref_joint_angles
    joint__diff_u = np.abs( joints_diff_s )

    joints_error_s = np.mean( joints_diff_s )
    joints_error_u = np.mean( joint__diff_u )

    print(f'Mean signed error   = {joints_error_s * 100 :.2f}%')
    print(f'Mean unsigned error = {joints_error_u * 100 :.2f}%')

    # Plot
    if not plot:
        return

    plt.figure('Joints displacement')
    plt.plot( actual_joints_displ, label='Actual' )
    plt.plot( desired_joints_displ, label='Desired' )
    plt.xlabel('Joints')
    plt.ylabel('Displacement')
    plt.xlim(0, N_JOINTS_AXIS-1)
    plt.legend()

    plt.show()
    return

###############################################################################
## MOTOR OUTPUT TUNING ########################################################
###############################################################################

def update_motor_output_gains(
    mo_gains_axial  : np.ndarray,
    results_run     : dict,
    ref_joint_angles: np.ndarray,
    trial           : int = 0,
    learn_rate      : float = 0.5,
    tail_joints     : int = 2,
):
    ''' Update the axial motor command gains '''

    # Compare actual and commanded positions
    joint_angles     = results_run['mech_joints_disp_amp'][0]
    joint_angles     = joint_angles[:-tail_joints]
    ref_joint_angles = ref_joint_angles[:-tail_joints]

    # Compute norm
    joints_diff_s = ( joint_angles - ref_joint_angles ) / ref_joint_angles
    joint__diff_u = np.abs( joints_diff_s )

    joints_error_s = np.mean( joints_diff_s )
    joints_error_u = np.mean( joint__diff_u )

    # Logging
    print(f'Trial {trial}')
    print('Motor output gains: [')
    for i in range(0, len(mo_gains_axial), 5):
        print(
            '   ' + ', '.join([f'{mc:.5f}' for mc in mo_gains_axial[i:i+5]]) + ','
        )
    print(']')
    print(f'Mean signed error   = {joints_error_s * 100 :.2f}%')
    print(f'Mean unsigned error = {joints_error_u * 100 :.2f}%')
    print('')

    # Update joints_amps_scalings
    mo_gains                       = mo_gains_axial[:-tail_joints]
    mo_gains                      *= (1 + learn_rate * (-1 + ref_joint_angles / joint_angles))
    mo_gains[mo_gains < 1e-5]      = 1e-5
    mo_gains_axial[:-tail_joints]  = mo_gains

    return mo_gains_axial, joints_error_u

def optimize_motor_output_gains(
    mo_gains_axial      : np.ndarray,
    ref_joint_angles    : np.ndarray,
    run_iteration_func  : Callable,
    run_iteration_kwargs: dict[str, Any],
    n_trials            : int,
    tolerance           : float,
    learn_rate          : float = 0.5,
    run_result          : bool = True,
    run_result_kwargs   : dict[str, Any] = None,
):
    ''' Optimize the axial motor command gains '''

    # OPTIMIZE
    trial         = 0
    best_error    = np.inf
    best_mo_gains = np.copy(mo_gains_axial)

    mo_gains_axial_old = np.copy(mo_gains_axial)
    mo_gains_axial_new = np.copy(mo_gains_axial)

    for trial in range(n_trials):

        mo_gains_axial_old = np.copy(mo_gains_axial_new)

        results_run = run_iteration_func(
            mo_gains_axial = mo_gains_axial_new,
            trial          = trial,
            **run_iteration_kwargs,
        )

        # Compare actual and commanded positions
        mo_gains_axial_new, joints_error_u = update_motor_output_gains(
            mo_gains_axial   = mo_gains_axial_new,
            results_run      = results_run,
            ref_joint_angles = ref_joint_angles,
            trial            = trial,
            learn_rate       = learn_rate,
        )

        if joints_error_u < best_error:
            best_error    = joints_error_u
            best_mo_gains = mo_gains_axial_old

        if abs(joints_error_u) < tolerance:
            break

    mo_gains_axial = best_mo_gains

    # SIMULATE
    if not run_result:
        return mo_gains_axial, results_run

    # Run the simulation
    results_run = run_iteration_func(
        mo_gains_axial = mo_gains_axial,
        trial          = trial + 1,
        **run_result_kwargs,
    )

    return mo_gains_axial, results_run

###############################################################################
## BENDING FUNCTIONS ##########################################################
###############################################################################

def _get_interval_points(times, points_list):
    ''' Get the interval points '''
    tmin, tmax = np.amin(times), np.amax(times)
    return tmin + (tmax - tmin) * np.array(points_list)

## STEPS

def _exp_fun(
    times: np.ndarray,
    time0: float,
    time1: float,
    tau  : float,
    f_inf: float,
    sign : float,
) -> np.ndarray:
    ''' Exponential function '''
    f_exp             = np.zeros_like(times)
    valid_time        = (time0 <= times) & (times < time1)
    f_exp[valid_time] = f_inf + sign * np.exp(-(times[valid_time] - time0) / tau)
    return f_exp

def sig_funcion_step(
    times      : np.ndarray,
    points_list: list[float] = [0.5, 1.0],
    tau        : float = 0.05,
) -> np.ndarray:
    ''' Single step function '''
    t0, t1 = _get_interval_points(times, points_list)

    return _exp_fun(times, t0, t1, tau, 1, -1)

def sig_funcion_single_square(
    times      : np.ndarray,
    points_list: list[float] = [0.25, 0.75, 1.00],
    tau        : float = 0.05,
) -> np.ndarray:
    ''' Single square function '''
    t0, t1, t2  = _get_interval_points(times, points_list)
    fun_output  = np.zeros_like(times)
    fun_output += _exp_fun(times, t0, t1, tau, +1, -1) # Positive Step
    fun_output += _exp_fun(times, t1, t2, tau,  0, +1)

    return fun_output

def sig_funcion_double_square(
    times      : np.ndarray,
    points_list: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    tau        : float = 0.05,
) -> np.ndarray:
    ''' Double square function with positive and negative steps '''
    t0, t1, t2, t3, t4 = _get_interval_points(times, points_list)

    fun_output  = np.zeros_like(times)
    fun_output += _exp_fun(times, t0, t1, tau, +1, -1) # Positive Step
    fun_output += _exp_fun(times, t1, t2, tau,  0, +1)
    fun_output += _exp_fun(times, t2, t3, tau, -1, +1) # Negative Step
    fun_output += _exp_fun(times, t3, t4, tau,  0, -1)

    return fun_output

## SINE WAVES

def _sin_fun(
    phases       : np.ndarray,
    phase0       : float,
    phase1       : float,
    remove_offset: bool =False,
) -> np.ndarray:
    ''' Sine function '''
    f_sin             = np.zeros_like(phases)
    valid_time        = (phase0 <= phases) & (phases < phase1)
    offset            = 0 if not remove_offset else phase0
    f_sin[valid_time] = np.sin( 2*np.pi * (phases[valid_time] - offset) )
    return f_sin

def sig_function_sine_square(
    phases       : np.ndarray,
    points_list  : list[float] = [ 0.333, 0.666],
    remove_offset: bool = False,
) -> float:
    ''' Sine square function '''
    p0, p1 = _get_interval_points(phases, points_list)

    fun_output  = np.zeros_like(phases)
    fun_output += _sin_fun(phases, p0, p1, remove_offset)

    return fun_output

###############################################################################
## KINEMATICS FILES ###########################################################
###############################################################################

def create_kinematics_file(
    filename          : str,
    duration          : float,
    timestep          : float,
    frequency         : float,
    amp_arr           : np.ndarray,
    ipl_arr           : np.ndarray,
    off_arr           : np.ndarray = None,
    sig_funcion       : Callable = None,
    sig_funcion_kwargs: dict[str, Any] = None,
):
    ''' Create kinematics file '''

    if off_arr is None:
        off_arr = np.zeros(N_JOINTS_AXIS)

    motor_output_signal_pars = {
        'axis': {
            'frequency'  : frequency,
            'n_copies'   : 1,
            'ipl_off'    : [0.0],
            'names'      : ['AXIS'],

            'n_joints': N_JOINTS_AXIS,
            'amp_arr' : amp_arr,
            'ipl_arr' : ipl_arr,
            'off_arr' : off_arr,
        },
    }

    # Create kinematics file
    # E.g. sig_function = lambda phase : np.sin( 2*np.pi * phase ) [Default]
    # E.g. sig_funcion  = lambda time: 1 - np.exp( - time / 0.2 ),

    get_kinematics_output_signal(
        times               = np.arange(0, duration, timestep),
        chains_params       = motor_output_signal_pars,
        sig_funcion         = sig_funcion,
        sig_function_kwargs = sig_funcion_kwargs,
        save_file           = filename,
    )

def create_static_bending_kinematics_file(
    tot_angle_deg     : float,
    duration          : float,
    timestep          : float,
    filename          : str = 'data_static_bending',
    sig_funcion       : Callable = sig_funcion_single_square,
    sig_funcion_kwargs: dict[str, Any] = None,
):
    ''' Create static bending kinematics file '''

    kinematics_file = (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        f'kinematics/{filename}.csv'
    )

    # Active joints
    no_bend           = [0, N_JOINTS_AXIS-2, N_JOINTS_AXIS-1]
    yes_bend          = np.ones(N_JOINTS_AXIS, dtype=bool)
    yes_bend[no_bend] = False

    joint_angles           = np.zeros(N_JOINTS_AXIS)
    joint_angles[yes_bend] = np.deg2rad(tot_angle_deg) / np.sum(yes_bend)

    # Create kinematics file
    create_kinematics_file(
        filename           = kinematics_file,
        duration           = duration,
        timestep           = timestep,
        frequency          = 1.0,
        amp_arr            = joint_angles,
        ipl_arr            = np.zeros(N_JOINTS_AXIS),
        sig_funcion        = sig_funcion,
        sig_funcion_kwargs = sig_funcion_kwargs,
    )

    return kinematics_file

def create_rhythmic_bending_kinematics_file(
    tot_angle_deg: float,
    frequency    : float,
    duration     : float,
    timestep     : float,
    square       : bool = False,
    filename     : str = 'data_rhythmic_bending',
):
    ''' Create rhythmic bending kinematics file '''

    kinematics_file = (
        'farms_experiments/experiments/zebrafish_v1_position_control_swimming/'
        f'kinematics/{filename}.csv'
    )

    # Active joints
    no_bend           = [0, N_JOINTS_AXIS-2, N_JOINTS_AXIS-1]
    yes_bend          = np.ones(N_JOINTS_AXIS, dtype=bool)
    yes_bend[no_bend] = False

    joint_angles           = np.zeros(N_JOINTS_AXIS)
    joint_angles[yes_bend] = np.deg2rad(tot_angle_deg) / np.sum(yes_bend)

    # Check for square sine signal
    sig_function        = None if not square else sig_function_sine_square
    sig_function_kwargs = {'remove_offset': True}

    # Create kinematics file
    create_kinematics_file(
        filename           = kinematics_file,
        duration           = duration,
        timestep           = timestep,
        frequency          = frequency,
        amp_arr            = joint_angles,
        ipl_arr            = np.zeros(N_JOINTS_AXIS),
        sig_funcion        = sig_function,
        sig_funcion_kwargs = sig_function_kwargs,
    )

    return kinematics_file

###############################################################################
## WEIGHTS MODULATION #########################################################
###############################################################################

def _get_syn_weight_dict(name, weight):
    ''' Get synaptic weight '''

    tag_to_name_dict = {
        'ex' : 'cpg.axial.ex',
        'in' : 'cpg.axial.in',
        'cpg': 'cpg.axial',
        'mn' : 'mn.axial',
        'rs' : 'rs.axial',
        'ps' : 'ps.axial',
        'mc' : 'mc.axial',
    }

    if '2' in name:

        # E.g. 'ex2in_weight' -> 'ex2in' -> ['ex', 'in']
        source, target = name.replace('_weight', '').split('2')
        source_name    = tag_to_name_dict[source]
        target_name    = tag_to_name_dict[target]

        # Log
        message   = f'Scale {source} to {target} synaptic strength by factor {weight}'
        logging.info(message)

        weight_dict = {
            'source_name' : source_name,
            'target_name' : target_name,
            'weight_ampa' : [weight, ''],
            'weight_nmda' : [weight, ''],
            'weight_glyc' : [weight, ''],
        }

    elif name == 'ps_weight':

        # Log
        message = f'Scale PS synaptic strength by factor {weight}'
        logging.info(message)

        weight_dict = {
            'source_name' : 'ps.axial',
            'target_name' : 'cpg.axial',
            'weight_ampa' : [weight, ''],
            'weight_nmda' : [weight, ''],
            'weight_glyc' : [weight, ''],
        }

    else:
        raise ValueError(f'Invalid connection name: {name}')

    return weight_dict

def get_syn_weights_list(new_pars: dict[str, float]) -> None:
        ''' Get list of syn weights to update the network '''
        syn_weights_list = []
        for par_name in list( new_pars.keys() ):
            if par_name.endswith('_weight'):
                syn_weights_list.append(
                    _get_syn_weight_dict(par_name, new_pars.pop(par_name))
                )
        return syn_weights_list

def get_original_syn_weights() -> list:
    ''' Get the original synaptic weights '''
    return {
        'ex2ex_weight' : 1.00,
        'ex2in_weight' : 1.00,
        'in2ex_weight' : 1.25,
        'in2in_weight' : 1.25,
        'rs2ex_weight' : 10.00,
        'rs2in_weight' : 10.00,
        'ex2mn_weight' : 40.00,
        'in2mn_weight' : 10.00,
        'mn2mc_weight' : 10.00,
        'ps_weight'    : 50.00,
    }

###############################################################################
## CONNECTIVITY MODULATION ####################################################
###############################################################################

def get_scaled_ex_to_cpg_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.70,
    cond_str        : str   = '',
):
    ''' Modulate EX to CPG connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_V2a -> AX_All Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_neur',
            'amp'     : amp,
            'sigma_up': 0.50 * range_scaling_0,
            'sigma_dw': 2.50 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'same', 'ax', 'ex.V2a', 'ax', ['ex.V2a', 'in.V0d']]
        ],
        'cond_str'  : cond_str
    }

def get_scaled_in_to_cpg_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.70,
    cond_str        : str   = '',
):
    ''' Modulate IN to CPG connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_V0d -> AX_All Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_neur',
            'amp'     : amp,
            'sigma_up': 1.00 * range_scaling_0,
            'sigma_dw': 3.00 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'opposite', 'ax', 'in.V0d', 'ax', ['ex.V2a', 'in.V0d']]
        ],
        'cond_str'  : cond_str
    }

def get_scaled_ex_to_mn_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.75,
    cond_str        : str   = '',
):
    ''' Modulate EX to MN connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_V2a -> AX_mn Ipsi',
        'synapse'   : 'syn_ex',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_neur',
            'amp'     : amp,
            'sigma_up': 0.50 * range_scaling_0,
            'sigma_dw': 2.00 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'same', 'ax', 'ex.V2a', 'ax', 'mn']
        ],
        'cond_str'  : cond_str
    }

def get_scaled_in_to_mn_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.35,
    cond_str        : str   = '',
):
    ''' Modulate IN to MN connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_V0d -> AX_mn Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_neur',
            'amp'     : amp,
            'sigma_up': 1.00 * range_scaling_0,
            'sigma_dw': 2.00 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'opposite', 'ax', 'in.V0d', 'ax', 'mn']
        ],
        'cond_str'  : cond_str
    }

def get_scaled_ps_to_cpg_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.50,
    cond_str        : str   = '',
):
    ''' Modulate PS to CPG connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_ps -> AX_V2a Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': 6.00 * range_scaling_0,
            'sigma_dw': 0.00 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'ex.V2a']
        ],
        'cond_str'  : cond_str
    }

def get_scaled_ps_to_ps_connections(
    range_scaling_0 : float,
    range_scaling_1 : float = None,
    amp             : float = 0.50,
    cond_str        : str   = '',
):
    ''' Modulate PS to CPG connections '''

    if range_scaling_1 is None:
        range_scaling_1 = range_scaling_0

    return {
        'name'      : 'AX_ps -> AX_ps Contra',
        'synapse'   : 'syn_in',
        'type'      : 'gaussian_identity',
        'parameters': {
            'y_type'  : 'y_mech',
            'amp'     : amp,
            'sigma_up': 6.00 * range_scaling_0,
            'sigma_dw': 0.00 * range_scaling_1,
        },
        'cond_list' : [
            ['', 'contra', 'ax', 'ps', 'ax', 'ps']
        ],
        'cond_str'  : cond_str
    }