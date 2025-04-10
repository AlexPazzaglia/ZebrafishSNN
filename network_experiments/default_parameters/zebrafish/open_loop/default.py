import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_utils import MODELS_OPENLOOP
from network_experiments.default_parameters.zebrafish.swimming_default import *

###############################################################################
## CONSTANTS ##################################################################
###############################################################################
MODNAME  = MODELS_OPENLOOP[MODEL_NAME][0]
PARSNAME = MODELS_OPENLOOP[MODEL_NAME][1]

###############################################################################
## SIM PARAMETERS #############################################################
###############################################################################

def get_default_parameters():
    ''' Get the default parameters for the analysis '''

    # Process parameters
    params_process = PARAMS_SIMULATION

    return {
        'kinematics_data'      : KINEMATICS_DATA,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
    }

###############################################################################
## SIM RESULTS ################################################################
###############################################################################

def study_sim_results(
    metrics_runs  : dict,
    reference_data: dict = None,
    run_index     : int = 0,
):
    ''' Study the results of the simulation '''

    if reference_data is None:
        reference_data = get_default_parameters()

    # Check the neural results
    study_neural_sim_results(
        metrics_runs   = metrics_runs,
        reference_data = reference_data,
        run_index      = run_index,
    )

    return