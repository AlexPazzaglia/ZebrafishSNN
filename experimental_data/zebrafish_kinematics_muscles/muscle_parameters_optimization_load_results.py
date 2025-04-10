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
import pandas as pd

N_JOINTS_AXIS = 15

def load_muscle_parameters_options_from_optimization(
    optimization_name: str,
    folder_name      : str = (
        'experimental_data/zebrafish_kinematics_muscles/optimized_parameters'
    ),
):
    ''' Load the muscle parameters '''

    # Load the parameter
    muscle_params_df = pd.read_csv(f'{folder_name}/{optimization_name}')

    # Organize for simulation
    # muscle_parameters_options = [
    #     [['joint_0'],  {'alpha': 6.99905e-08,  'beta': 8.33e-08,  'delta': 2.0729e-09  }],
    #     [['joint_1'],  {'alpha': 3.867022e-07, 'beta': 4.60e-07,  'delta': 9.9484e-09  }],
    #     [['joint_2'],  {'alpha': 9.09911e-07,  'beta': 1.083e-06, 'delta': 2.2656e-08  }],
    #     [['joint_3'],  {'alpha': 1.895313e-06, 'beta': 2.25e-06,  'delta': 4.5077e-08  }],
    #     [['joint_4'],  {'alpha': 3.020483e-06, 'beta': 3.595e-06, 'delta': 6.9371e-08  }],
    #     [['joint_5'],  {'alpha': 3.023565e-06, 'beta': 3.59e-06,  'delta': 7.0830e-08  }],
    #     [['joint_6'],  {'alpha': 2.7499e-06,   'beta': 3.273e-06, 'delta': 6.5320e-08  }],
    #     [['joint_7'],  {'alpha': 3.249942e-06, 'beta': 3.86e-06,  'delta': 7.4085e-08  }],
    #     [['joint_8'],  {'alpha': 2.016e-06,    'beta': 2.40e-06,  'delta': 4.65283e-08 }],
    #     [['joint_9'],  {'alpha': 1.198723e-06, 'beta': 1.427e-06, 'delta': 2.82726e-08 }],
    #     [['joint_10'], {'alpha': 7.24792e-07,  'beta': 8.62e-07,  'delta': 1.72191e-08 }],
    #     [['joint_11'], {'alpha': 2.82111e-07,  'beta': 3.3e-07,   'delta': 7.0236e-09  }],
    #     [['joint_12'], {'alpha': 8.94351e-08,  'beta': 1.064e-07, 'delta': 2.4076e-09  }],
    #     [['joint_13'], {'alpha': 2.585919e-08, 'beta': 3.07e-08,  'delta': 7.5689e-10  }],
    #     [['joint_14'], {'alpha': 7.12220e-09,  'beta': 8.47e-09,  'delta': 2.26387e-10 }]
    # ]

    muscle_parameters_options = [
        [
            [f'joint_{i}'],
            {
                'alpha'  : muscle_params_df.loc[i, 'alpha'],
                'beta'   : muscle_params_df.loc[i, 'beta'],
                'delta'  : muscle_params_df.loc[i, 'delta'],
                'gamma'  : 1.0,
                'epsilon': 0,
            }
        ]
        for i in range(N_JOINTS_AXIS)
    ]

    return muscle_parameters_options

if __name__ == '__main__':

    optimization_name = 'muscle_parameters_optimization'
    target_G0         = 2 * np.pi / N_JOINTS_AXIS
    target_Wn         = 30.00 * 2 * np.pi
    target_Zc         = 1.00
    index_iteration   = 99

    optimization_name = (
        f'{optimization_name}_'
        f'FN_{round(target_Wn/2/np.pi*1e3)}_'
        f'ZC_{round(target_Zc*1e3)}_'
        f'G0_{round(target_G0*1e3)}_'
        f'gen_{index_iteration}.csv'
    )

    load_muscle_parameters_options_from_optimization(
        optimization_name = optimization_name,
    )