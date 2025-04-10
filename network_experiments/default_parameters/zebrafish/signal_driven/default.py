import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import matplotlib.pyplot as plt

from network_experiments.snn_signals_neural import get_motor_output_signal
from network_experiments.snn_simulation_setup import get_mech_sim_options

from network_experiments.default_parameters.zebrafish.swimming_default import *


###############################################################################
## CONSTANTS ##################################################################
###############################################################################
MODNAME  = MODELS_FARMS[MODEL_NAME][0]
PARSNAME = MODELS_FARMS[MODEL_NAME][1]

###############################################################################
## SIM PARAMETERS #############################################################
###############################################################################

def get_default_parameters(
    muscle_parameters_tag: str         = MUSCLE_PARAMETERS_TAG,
    muscle_factors        : np.ndarray = np.ones(N_JOINTS_AXIS),
):
    ''' Get the default parameters for the analysis '''

    # Muscle parameters
    muscle_parameters_options = get_scaled_muscle_parameters_options(
        muscle_parameters_tag = muscle_parameters_tag,
        muscle_factors        = muscle_factors,
        head_joints           = 1,
        tail_joints           = 3,
        plot_muscles          = False
    )

    # Mech sim options
    mech_sim_options = get_mech_sim_options(
        muscle_parameters_options = muscle_parameters_options,
    )

    # Process parameters
    params_process = PARAMS_SIMULATION | {
        'mech_sim_options'        : mech_sim_options,
        'motor_output_signal_func': get_motor_output_signal,
    }

    return {
        'kinematics_data'      : KINEMATICS_DATA,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
        'muscle_parameters_tag': muscle_parameters_tag,
    }

###############################################################################
## SIM RESULTS ################################################################
###############################################################################

def study_sim_results(
    metrics_runs  : dict,
    reference_data: dict = None,
    run_index     : int = 0,
    plot          : bool = True,
):
    ''' Study the results of the simulation '''

    if reference_data is None:
        reference_data = get_default_parameters()

    # Check the mechanical results
    study_mechanical_sim_results(
        metrics_runs   = metrics_runs,
        reference_data = reference_data,
        run_index      = run_index,
        plot           = plot,
    )

    return