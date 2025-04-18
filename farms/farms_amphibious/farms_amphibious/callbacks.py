"""Callbacks"""

import numpy as np
from imageio import imread

from farms_core.sensors.sensor_convention import sc
from farms_mujoco.simulation.task import TaskCallback
from farms_mujoco.swimming.drag import SwimmingHandler

from .model.options import (
    SpikingOptions,
    AmphibiousArenaOptions
)


def setup_callbacks(
        animat_options,
        arena_options,
        camera=None,
        water_properties=None,
        other_callbacks : list = None,
):
    """Callbacks for amphibious simulation"""
    callbacks = []
    if arena_options.water.sdf:
        callbacks += [
            SwimmingCallback(
                animat_options,
                arena_options,
                water_properties=water_properties,
            ),
        ]
    if camera is not None:
        callbacks += [camera]
    if other_callbacks is not None:
        callbacks += other_callbacks
    return callbacks


def water_velocity_from_maps(position, water_maps):
    """Water velocity from maps"""
    vel = np.zeros(3)
    if all(
            water_maps['pos_min'][i] < position[i] < water_maps['pos_max'][i]
            for i in range(2)
    ):
        vel[:2] = [
            water_maps[png][tuple(
                (
                    max(0, min(
                        water_maps[png].shape[index]-1,
                        round(water_maps[png].shape[index]*(
                            (
                                position[index]
                                - water_maps['pos_min'][index]
                            ) / (
                                water_maps['pos_max'][index]
                                - water_maps['pos_min'][index]
                            )
                        ))
                    ))
                )
                for index in range(2)
            )]
            for png_i, png in enumerate(['vel_x', 'vel_y'])
        ]
    # vel[1] *= -1
    return vel


class SwimmingCallback(TaskCallback):
    """Swimming callback"""

    def __init__(
            self,
            animat_options: SpikingOptions,
            arena_options: AmphibiousArenaOptions,
            substep=True,
            water_properties=None,
    ):
        super().__init__(substep=substep)
        self.animat_options = animat_options
        self.arena_options = arena_options
        self._handler: SwimmingHandler = None
        self._water_properties = water_properties

        self.constant_velocity: bool = (
            len(arena_options.water.velocity) == 3
        )
        if not self.constant_velocity:
            water_velocity = arena_options.water.velocity
            water_maps = arena_options.water.maps
            pngs = [np.flipud(imread(water_maps[i])).T for i in range(2)]
            pngs_info = [np.iinfo(png.dtype) for png in pngs]
            vels = [
                (
                    png - info.min  # .astype(np.double)
                ) * (
                    water_velocity[png_i+3] - water_velocity[png_i+0]
                ) / (
                    info.max - info.min
                ) + water_velocity[png_i+0]
                for png_i, (png, info) in enumerate(zip(pngs, pngs_info))
            ]
            self.water_maps = {
                'pos_min': np.array(water_velocity[6:8]),
                'pos_max': np.array(water_velocity[8:10]),
                'vel_x': vels[0],
                'vel_y': vels[1],
            }

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self._handler = SwimmingHandler(
            data=task.data,
            animat_options=self.animat_options,
            arena_options=self.arena_options,
            units=task.units,
            physics=physics,
            water=self._water_properties,
        )

    def before_step(self, task, action, physics):
        """Step hydrodynamics"""

        # Water maps
        if not self.constant_velocity:
            # NOTE: This only samples the water velocity of the head
            # NOTE: Use custom SwimmingHandler for advanced water velocity options
            water_velocity = water_velocity_from_maps(
                position=task.data.sensors.links.urdf_position(
                    iteration=task.iteration,
                    link_i=0,
                ),
                water_maps=self.water_maps,
            )
            self._handler.set_water_velocity(water_velocity)


        # Compute fluid forces
        self._handler.step(physics.time(), task.iteration)

        # Set fluid forces in physics engine
        indices = task.maps['sensors']['data2xfrc']
        physics.data.xfrc_applied[indices, :] = (
            task.data.sensors.xfrc.array[
                task.iteration, :,
                sc.xfrc_force_x:sc.xfrc_torque_z+1,
            ]
        )
        for force_i, (rotation_mat, force_local) in enumerate(zip(
                physics.data.xmat[indices],
                physics.data.xfrc_applied[indices],
        )):
            physics.data.xfrc_applied[indices[force_i]] = (
                rotation_mat.reshape([3, 3])  # Local to global frame
                @ force_local.reshape([3, 2], order='F')
            ).flatten(order='F')
        physics.data.xfrc_applied[indices, :3] *= task.units.newtons
        physics.data.xfrc_applied[indices, 3:] *= task.units.torques
