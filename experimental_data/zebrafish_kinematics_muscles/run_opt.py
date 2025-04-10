''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import muscle_parameters_optimization_sim_params as opt_params
import muscle_parameters_optimization_runner as opt_runner

# SIMULATION
RESULTS_PATH = '/data/pazzagli/simulation_results'

N_JOINTS_AXIS = 15
DURATION      = 10.0
TIMESTEP      = 0.001
PLOTTING      = False
WATERLESS     = True

# OPTIMIZATION
SAVE_ITERATIONS = True
TESTED_JOINTS   = range(N_JOINTS_AXIS)

# MUSCLE ACTIVATION
MUSCLE_BASELINE  = 0.0
MUSCLE_AMPLITUDE = 0.1

# SWEEP FUNCTION
F_MIN     = 0.0
F_MAX     = 10.0
FREQ_FUN  = lambda time : F_MIN + (F_MAX - F_MIN) / DURATION * time
SWEEP_FUN = lambda time : np.sin( 2*np.pi * FREQ_FUN(time) * time )

def main():

    # TARGET PARAMETERS
    target_G0 = 2 * np.pi/ N_JOINTS_AXIS
    target_Zc = np.array( [1.00] )
    # target_Fn = np.arange(3.0, 5.0, 0.25,)
    target_Fn = np.arange(10.5, 20.1, 0.5)

    target_G0_list = np.array( [target_G0] )
    target_Zc_list = target_Zc
    target_Wn_list = target_Fn * 2 * np.pi

    # OPTIMIZATION
    start_iteration = 0
    n_iterations    = 51 - start_iteration
    rate_beta_arr   = np.linspace(0.3, 0.1, n_iterations)
    rate_delta_arr  = np.linspace(0.3, 0.1, n_iterations)

    optimization_settings = {

        # Simulation
        'results_path'   : RESULTS_PATH,
        'n_joints_axis'  : N_JOINTS_AXIS,
        'duration'       : DURATION,
        'timestep'       : TIMESTEP,
        'plotting'       : PLOTTING,
        'waterless'      : WATERLESS,

        # Optimization
        'opt_name'       : 'muscle_parameters_optimization_all',
        'opt_results_dir': opt_params.OPT_RESULTS_DIR,
        'tested_joints'  : TESTED_JOINTS,

        'muscle_baseline' : MUSCLE_BASELINE,
        'muscle_amplitude': MUSCLE_AMPLITUDE,
        'muscle_function' : SWEEP_FUN,

        'save_iterations': SAVE_ITERATIONS,
        'start_iteration': start_iteration,
        'n_iterations'   : n_iterations,

        'target_G0_list' : target_G0_list,
        'target_Wn_list' : target_Wn_list,
        'target_Zc_list' : target_Zc_list,

        'rate_beta_arr' : rate_beta_arr,
        'rate_delta_arr': rate_delta_arr,

        'n_batch'        : 10,

        # Muscles
        'default_alpha'  : opt_params.DEFAULT_ALPHA,
        'default_beta'   : opt_params.DEFAULT_BETA,
        'default_delta'  : opt_params.DEFAULT_DELTA,
        'default_gamma'  : opt_params.DEFAULT_GAMMA,
        'default_epsilon': opt_params.DEFAULT_EPSILON,

        'inactive_joints_stiff'       : True,
        'inactive_joints_stiff_factor': 10.0,
    }

    opt_runner.run_multiple_optimization(**optimization_settings)

if __name__ == '__main__':
    main()