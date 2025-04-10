''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import multiprocessing

import numpy as np

from network_experiments.snn_utils import start_deletion_schedule
from farms_core.model.options import SpawnMode
from sim_runner import run_simulation

RESULTS_PATH   = '/data/pazzagli/simulation_results'
MODULE_NAME    = 'net_farms_zebrafish'
SIMULATION_TAG = 'dynamic_water_vortices_fixed_head'

def get_sim_settings(**kwargs):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation
    analysis_mode            = True
    results_path             = RESULTS_PATH
    simulation_data_file_tag = SIMULATION_TAG

    delay_start = 5.0
    duration    = 30.0 + delay_start

    # Spawn
    spawn_x = 1.00
    spawn_y = 0.00

    # Vortex field
    leader_freq = None
    leader_name = None

    speed_mult_x = 1.0
    speed_mult_y = 1.0

    speed_offset_x = 0.0
    speed_offset_y = 0.0

    # Connections
    ps_connections_range  = 0.50
    cpg_connections_range = 0.65

    # Muscles
    muscle_resonant_freq = 5.0
    muscle_tag           = None

    mc_gain_axial = np.array(
        [
            # 5Hz
            0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
            0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
            0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
        ]
    )

    # Weights
    net_weights = {
        'ex2ex_weight': 3.83037,
        'ex2in_weight': 49.47920,
        'in2ex_weight': 0.84541,
        'in2in_weight': 0.10330,
        'rs2ex_weight': 8.74440,
        'rs2in_weight': 3.28338,
    }

    ###########################################################################
    # Simulation settings #####################################################
    ###########################################################################

    sim_settings = {
        'analysis_mode'       : analysis_mode,
        'duration'            : duration,
        'results_path'        : results_path,
        'sim_file_tag'        : simulation_data_file_tag,

        # Constraints
        # 'sim_spawn_mode'      : SpawnMode.FIXED,
        'sim_spawn_mode'      : SpawnMode.ROTZ,

        # Connections
        'load_connectivity'    : False,
        'ps_connections_range' : ps_connections_range,
        'cpg_connections_range': cpg_connections_range,

        # Muscles
        'muscle_tag'          : muscle_tag,
        'muscle_resonant_freq': muscle_resonant_freq,
        'mc_gain_axial'       : mc_gain_axial,

        # Weights
        'net_weights'         : net_weights,

        # Feedback
        'ps_weight'           : None,
        'ps_act_deg'          : None,

        # Functionalities
        'video'               : False,
        'plot'                : False,
        'animate'             : False,
        'save'                : True,

        # Water dynamics
        'spawn_x'             : spawn_x,
        'spawn_y'             : spawn_y,
        'leader_freq'         : leader_freq,
        'leader_name'         : leader_name,
        'speed_mult_x'        : speed_mult_x,
        'speed_mult_y'        : speed_mult_y,
        'speed_offset_x'      : speed_offset_x,
        'speed_offset_y'      : speed_offset_y,
        'delay_start'         : delay_start,
    }

    # Update the simulation settings with the keyword arguments
    sim_settings.update(kwargs)

    return sim_settings

def get_files_deletion_process(
    np_random_seed_vals: list[int],
) -> multiprocessing.Process:
    ''' Get the process to delete the files '''

    target_folders = [
        f'{RESULTS_PATH}/data/{MODULE_NAME}_{SIMULATION_TAG}_{seed}_SIM'
        for seed in np_random_seed_vals
    ]

    target_files = [
        'network_state.brian',
        'snn_connectivity_indices.dill',
        'run_0/statemon.dill',
        'run_0/spikemon.dill',
        'run_0/musclemon.dill',
        'run_0/farms/simulation.hdf5',
    ]

    return multiprocessing.Process(
        target = start_deletion_schedule,
        args   = (target_folders, target_files)
    )

def run_analysis(
    sim_settings_list: list[dict],
    n_batch          : int = 10,
):
    ''' Run the analysis '''

    # Estimate the time needed to run the simulations
    n_batch        = min( n_batch, multiprocessing.cpu_count() )
    n_simulations  = len(sim_settings_list)
    n_batches      = n_simulations // n_batch + 1
    time_per_batch = 90.0
    total_time     = time_per_batch * n_batches / 3600

    print(f'Number of simulations: {n_simulations}')
    print(f'Needed batches: {n_batches}')
    print(f'Projected time: {total_time :.2f} hours')

    # Get random seed values from the simulation settings
    np_random_seed_vals = list(
        set( [ sim['np_random_seed'] for sim in sim_settings_list ] )
    )

    # Start the deletion process
    deletion_process = get_files_deletion_process(np_random_seed_vals)
    deletion_process.start()

    # Run simulations in batches
    with multiprocessing.Pool(processes = n_batch) as pool:
        pool.map(run_simulation, sim_settings_list)

    # Terminate the deletion process
    deletion_process.terminate()

    return