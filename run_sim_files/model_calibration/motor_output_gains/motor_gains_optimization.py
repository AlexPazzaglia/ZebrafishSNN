import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
import network_experiments.default_parameters.zebrafish.closed_loop.default as default

###############################################################################
# SIMULATION ##################################################################
###############################################################################

def _get_sim_params(
    mc_gains_axial   : np.ndarray,
    sim_pars         : dict,
    load_connectivity: bool = False,
    video            : bool = False,
):
    '''
    Get the parameters for the simulation
    '''

    # Simulation
    simulation_tag = sim_pars['simulation_tag']
    duration       = sim_pars['duration']
    stim_a_off     = sim_pars['stim_a_off']

    # Connection params
    ps_connections_range  = sim_pars['ps_connections_range']
    cpg_connections_range = sim_pars['cpg_connections_range']

    connectivity_axial_newpars = {
        'ax2ax': [
            default.get_scaled_ex_to_cpg_connections(cpg_connections_range),
            default.get_scaled_in_to_cpg_connections(cpg_connections_range),
        ],
        'ps2ax': [
            default.get_scaled_ps_to_cpg_connections(ps_connections_range),
            default.get_scaled_ps_to_ps_connections(ps_connections_range),
        ],
    }

    # Connection weights
    ex2ex_weight = sim_pars['ex2ex_weight']
    ex2in_weight = sim_pars['ex2in_weight']
    in2ex_weight = sim_pars['in2ex_weight']
    in2in_weight = sim_pars['in2in_weight']
    rs2ex_weight = sim_pars['rs2ex_weight']
    rs2in_weight = sim_pars['rs2in_weight']

    # Cocontraction terms
    mo_cocontraction_gain = sim_pars['mo_cocontraction_gain']
    mo_cocontraction_off  = sim_pars['mo_cocontraction_off']

    # PS gains
    ps_min_activation_deg = sim_pars['ps_min_activation_deg']
    ps_weight             = sim_pars['ps_weight']

    ps_gains = default.get_uniform_ps_gains(
        min_activation_deg = ps_min_activation_deg,
        n_joints_tail      = 2,
    )

    # Process parameters
    params_process = sim_pars['params_process'] | {
        'simulation_data_file_tag'  : simulation_tag,
        'load_connectivity_indices' : load_connectivity,
        'connectivity_axial_newpars': connectivity_axial_newpars,

        'mo_cocontraction_gain': mo_cocontraction_gain,
        'mo_cocontraction_off' : mo_cocontraction_off,
        'duration'             : duration,

        'stim_a_off'    : stim_a_off,
        'mc_gain_axial' : mc_gains_axial,
    }

    # Params runs
    params_runs = [
        {

            'ex2ex_weight' : ex2ex_weight,
            'ex2in_weight' : ex2in_weight,
            'in2ex_weight' : in2ex_weight,
            'in2in_weight' : in2in_weight,
            'rs2ex_weight' : rs2ex_weight,
            'rs2in_weight' : rs2in_weight,

            'ps_weight'    : ps_weight,
            'ps_gain_axial': ps_gains,

            'mech_sim_options' : {
                'video'      : video,
                'video_fps'  : 30,
                'video_speed': 1.0,
            }
        }
    ]

    return params_process, params_runs

def run_iteration(
    mo_gains_axial: np.ndarray,
    trial         : int,
    sim_pars      : dict,
    video         : bool = False,
    plot          : bool = False,
    save          : bool = False,
):
    ''' Run a single iteration of the simulation '''

    # Get params
    params_process, params_runs = _get_sim_params(
        mc_gains_axial    = mo_gains_axial,
        sim_pars          = sim_pars,
        load_connectivity = (trial != 0),
        video             = video
    )

    # Simulate
    results_runs = simulate_single_net_multi_run_closed_loop_build(
        modname             = sim_pars['modname'],
        parsname            = sim_pars['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = plot,
        results_path        = sim_pars['results_path'],
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
    )

    return results_runs
