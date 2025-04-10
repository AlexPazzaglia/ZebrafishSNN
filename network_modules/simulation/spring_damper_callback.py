"""Callbacks"""

import numpy as np

from typing import Union
from farms_mujoco.simulation.task import TaskCallback
from farms_amphibious.model.options import AmphibiousOptions, AmphibiousArenaOptions

class SpringDamper():

    def __init__(
        self,
        spring_constant : Union[float, np.ndarray],
        damping_constant: Union[float, np.ndarray],
        fixed_point     : np.ndarray,
        target_link     : int,
        rest_length     : Union[float, np.ndarray] = 0.0,
        inactive_band   : Union[float, np.ndarray] = 0.0,
    ):
        self.spring_constant  = spring_constant
        self.damping_constant = damping_constant
        self.rest_length      = rest_length
        self.inactive_band    = inactive_band
        self.fixed_point      = fixed_point
        self.target_link      = target_link

        # Ignore coordinates that are NaN
        self.ignore_p = np.isnan(self.fixed_point)
        assert sum(self.ignore_p) <= 2, \
            "At most 2 coordinates can be ignored"

    def compute_force(
        self,
        link_position,
        link_velocity,
    ):
        ''' Compute spring-damper force '''

        # Distance and velocity arrays
        distance_arr = np.array( link_position - self.fixed_point )
        velocity_arr = np.array( link_velocity[:] )

        # Ignore coordinates
        distance_arr[self.ignore_p] = 0
        velocity_arr[self.ignore_p] = 0

        # Compute spring force
        spring_elongation = distance_arr - self.rest_length

        dl_amp = np.abs(spring_elongation) - self.inactive_band
        dl_sgn = np.sign(spring_elongation)
        dl_ok  = (dl_amp > 0)

        spring_force_arr = - self.spring_constant * dl_amp * dl_sgn * dl_ok

        # Compute damping force
        damping_force_arr = - self.damping_constant * velocity_arr

        return spring_force_arr + damping_force_arr


class SpringDamperCallback(TaskCallback):
    """Swimming callback"""

    def __init__(
            self,
            animat_options: AmphibiousOptions,
            arena_options: AmphibiousArenaOptions,
            spring_damper_properties: dict,
    ):
        super().__init__()
        self.animat_options    = animat_options
        self.arena_options     = arena_options
        self.spring_properties = spring_damper_properties
        self.spring_damper     = SpringDamper(**spring_damper_properties)
        self.target_link       = self.spring_damper.target_link

    def before_step(self, task, action, physics):
        """Step spring-damper """

        # Get link position and velocity
        link_position = task.data.sensors.links.com_position(
            iteration = task.iteration,
            link_i    = self.target_link,
        )

        link_velocity = task.data.sensors.links.com_lin_velocity(
            iteration = task.iteration,
            link_i    = self.target_link,
        )

        # Compute spring-damper force
        spring_damper_force = self.spring_damper.compute_force(
            link_position = link_position,
            link_velocity = link_velocity,
        )

        # Set spring-damper forces in physics engine
        target_ind = task.maps['sensors']['data2xfrc'][self.target_link]
        physics.data.xfrc_applied[target_ind, :3] += spring_damper_force * task.units.newtons

        return