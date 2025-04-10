''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_hybrid_position_control_build
from interpolate_experimental_angles import create_fictive_schooling_file

import network_experiments.default_parameters.zebrafish.hybrid_position_control.default as default

from run_sim_files.zebrafish.hybrid_position_control import simulation_runner


def run(
    duration             : float,
    timestep             : float,
    stim_a_off           : float,
    simulation_tag       : str,
    frequency_scaling    : float,
    make_sinusoidal      : bool,
    time_offset          : float,
    signal_repeats       : int,
    net_weights          : dict[str, float],
    ps_min_activation_deg: float,
    ps_weight            : float,
    cpg_connections_range: float,
    ps_connections_range : float,
    video                : bool = False,
    plot                 : bool = True,
    save                 : bool = False,
    save_all_data        : bool = False,
    random_seed          : int  = 100,
    new_pars_prc         : dict = None,
    new_pars_run         : dict = None,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    ########################################################
    # CREATE KINEMATICS FILE ###############################
    ########################################################
    kinematics_file_path = create_fictive_schooling_file(
        timestep        = timestep,
        total_duration  = duration,
        freq_scaling    = frequency_scaling,
        time_offset     = time_offset,
        signal_repeats  = signal_repeats,
        make_sinusoidal = make_sinusoidal,
        plot            = False,
    )

    ########################################################
    # SIM PARAMETERS #######################################
    ########################################################

    # PROCESS
    if new_pars_prc is None:
        new_pars_prc = {}

    # RUN
    if new_pars_run is None:
        new_pars_run = {}

    ########################################################
    # SIMULATION ###########################################
    ########################################################

    results_runs = simulation_runner.run(
        module_path           = f'{CURRENTDIR}/net_farms_zebrafish_fictive_schooling.py',
        duration              = duration,
        timestep              = timestep,
        kinematics_file_path  = kinematics_file_path,
        stim_a_off            = stim_a_off,
        simulation_tag        = simulation_tag,
        net_weights           = net_weights,
        ps_min_activation_deg = ps_min_activation_deg,
        ps_weight             = ps_weight,
        cpg_connections_range = cpg_connections_range,
        ps_connections_range  = ps_connections_range,
        new_pars_prc          = new_pars_prc,
        new_pars_run          = new_pars_run,
        video                 = video,
        plot                  = plot,
        save                  = save,
        save_all_data         = save_all_data,
        random_seed           = random_seed,
    )

    return results_runs