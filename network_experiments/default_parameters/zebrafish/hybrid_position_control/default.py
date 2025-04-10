import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

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

def get_default_parameters():
    ''' Get the default parameters for the analysis '''

    # Process parameters
    params_process = PARAMS_SIMULATION | {
        'mech_sim_options': get_mech_sim_options(),
    }

    return {
        'kinematics_data'      : KINEMATICS_DATA,
        'modname'              : MODNAME,
        'parsname'             : PARSNAME,
        'params_process'       : params_process,
    }
