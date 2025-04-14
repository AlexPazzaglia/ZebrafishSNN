"""Prompt"""

import os
from distutils.util import strtobool
from farms_core import pylog
from farms_core.simulation.options import Simulator


def prompt(query, default):
    """Prompt"""
    val = input(f'{query} [{"Y/n" if default else "y/N"}]: ')
    try:
        ret = strtobool(val) if val != '' else default
    except ValueError:
        pylog.error('Did not recognise \'%s\', please reply with a y/n', val)
        return prompt(query, default)
    return ret


def prompt_postprocessing(sim, animat_options, query=True, **kwargs):
    """Prompt postprocessing"""
    # Arguments
    log_path = kwargs.pop('log_path', '')
    verify = kwargs.pop('verify', False)
    extension = kwargs.pop('extension', 'pdf')
    simulator = kwargs.pop('simulator', Simulator.MUJOCO)
    assert not kwargs, kwargs

    # Post-processing
    pylog.info('Simulation post-processing')
    save_data = (
        (query and prompt('Save data', False))
        or log_path and not query
    )
    if log_path:
        os.makedirs(log_path, exist_ok=True)
    show_plots = prompt('Show plots', False) if query else False
    iteration = (
        sim.iteration
        if simulator == Simulator.PYBULLET
        else sim.task.iteration  # Simulator.MUJOCO
    )
    sim.postprocess(
        iteration=iteration,
        log_path=log_path if save_data else '',
        plot=show_plots,
        video=(
            os.path.join(log_path, 'simulation.mp4')
            if sim.options.record
            else ''
        ),
    )

    # Save MuJoCo MJCF
    if simulator == Simulator.MUJOCO:
        sim.save_mjcf_xml(os.path.join(log_path, 'sim_mjcf.xml'))

