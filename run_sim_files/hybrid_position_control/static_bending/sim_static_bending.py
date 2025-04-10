''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import network_experiments.default_parameters.zebrafish.hybrid_position_control.default as default
from run_sim_files.zebrafish.hybrid_position_control import simulation_runner

def run(
    duration             : float,
    timestep             : float,
    stim_a_off           : float,
    simulation_tag       : str,
    tot_angle_deg        : float,
    net_weights          : dict[str, float],
    ps_min_activation_deg: float,
    ps_weight            : float,
    cpg_connections_range: float,
    ps_connections_range : float,
    video                : bool = False,
    plot                 : bool = True,
    save                 : bool = False,
    save_all_data        : bool = False,
    new_pars_prc         : dict = None,
    new_pars_run         : dict = None,
    random_seed          : int  = 100,
    sig_funcion          : callable = None,
    sig_funcion_kwargs   : dict = None,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    ########################################################
    # CREATE KINEMATICS FILE ###############################
    ########################################################

    sig_funcion        = sig_funcion if sig_funcion is not None else default.sig_funcion_single_square
    sig_funcion_kwargs = sig_funcion_kwargs if sig_funcion_kwargs is not None else {}

    kinematics_file_path = default.create_static_bending_kinematics_file(
        tot_angle_deg      = tot_angle_deg,
        duration           = duration,
        timestep           = timestep,
        filename           = 'data_static_bending',
        sig_funcion        = sig_funcion,
        sig_funcion_kwargs = sig_funcion_kwargs,
    )

    ########################################################
    # SIM PARAMETERS #######################################
    ########################################################

    # PROCESS
    if new_pars_prc is None:
        new_pars_prc = {

            'monitor_musclecells' : {
                'active'   : True,
                'save'     : True,
                'to_csv'   : False,
                'indices'  : True,
                'variables': ['v', 'I_tot'],
                'rate'     : 1,
                'plotpars' : {
                    'showit'        : True,
                    'filtering'     : False,
                    'sampling_ratio': 0.50,
                    'duty_cycle_ax' : {
                            'showit'    : True,
                            'filter'    : True,
                            'target_seg': [5, 6, 7, 8, 9], # [8, 9, 10, 11, 12]
                    }
                }
            },

        }

    # RUN
    if new_pars_run is None:
        new_pars_run = {}

    ########################################################
    # SIMULATION ###########################################
    ########################################################

    results_runs = simulation_runner.run(
        module_path           = f'{CURRENTDIR}/net_farms_zebrafish_static_bending.py',
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
