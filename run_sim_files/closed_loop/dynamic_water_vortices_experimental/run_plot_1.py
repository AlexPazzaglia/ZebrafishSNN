''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from farms_core.model.options import SpawnMode
from sim_runner import run_simulation

from network_experiments.snn_simulation_replay import replicate_network_post_processing
import network_experiments.default_parameters.zebrafish.closed_loop.default as default


def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path = '/data/pazzagli/simulation_results'
    module_name  = 'net_farms_zebrafish'
    folder_tag   = 'dynamic_water_vortices_closed_loop_fixed_head_040'
    folder_name  = f'{module_name}_{folder_tag}_100_SIM'

    default_params = default.get_default_parameters()

    replicate_network_post_processing(
        control_type = 'closed_loop',
        modname      = f'{CURRENTDIR}/{module_name}.py',
        parsname     = default_params['parsname'],
        folder_name  = folder_name,
        tag_folder   = 'SIM',
        tag_process  = '0',
        results_path = results_path,
        run_id       = 0,
        plot_figures = True,
        save_prompt  = False,
    )




if __name__ == '__main__':
    main()