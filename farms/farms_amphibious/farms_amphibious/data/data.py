"""Amphibious data"""

from typing import Dict

import numpy as np

from farms_core import pylog
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.array.array import to_array
from farms_core.array.array_cy import IntegerArray2D
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions, ControlOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.sensors.data import SensorsData

from ..model.options import (
    SpikingControlOptions,
    KinematicsControlOptions,
)

from .data_cy import SpikingDataCy

from .network import (
    SpikingNetworkState,
    SpikingNetworkParameters,
    SpikingOscillators,
)


def get_amphibious_data(animat_options, simulation_options):
    """Get amphibious_data"""
    return (
        AmphibiousKinematicsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
        if isinstance(animat_options.control, KinematicsControlOptions)
        else SpikingData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
        if isinstance(animat_options.control, SpikingControlOptions)
        else AnimatData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )
    )

class AmphibiousKinematicsData(AnimatData):
    """Amphibious data"""

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""
        return cls(
            timestep=simulation_options.timestep,
            sensors=SensorsData.from_options(
                animat_options=animat_options,
                simulation_options=simulation_options,
            ),
        )

# --------------------- [ SPIKING ] ---------------------
class SpikingData(SpikingDataCy, AnimatData):
    """Spiking data"""

    def __init__(
            self,
            state: SpikingNetworkState,
            network: SpikingNetworkParameters,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.network = network

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From animat and simulation options"""

        # Sensors
        sensors = SensorsData.from_options(
            animat_options=animat_options,
            simulation_options=simulation_options,
        )

        # Oscillators
        oscillators = SpikingOscillators.from_options(
            network=animat_options.control.network,
        ) if animat_options.control.network is not None else None


        # State
        state = (
            SpikingNetworkState.from_initial_state(
                initial_state=animat_options.state_init(),
                n_iterations=simulation_options.n_iterations,
                n_oscillators=animat_options.control.network.n_oscillators(),
            )
            if animat_options.control.network is not None
            else None
        )

        # Network
        network = (
            SpikingNetworkParameters(oscillators=oscillators)
            if animat_options.control.network is not None
            else None
        )

        return cls(
            timestep=simulation_options.timestep,
            sensors=sensors,
            state=state,
            network=network,
        )

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        data['n_oscillators'] = len(data['network']['oscillators']['names'])
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        n_oscillators = dictionary.pop('n_oscillators')
        return cls(
            timestep=dictionary['timestep'],
            state=SpikingNetworkState(dictionary['state'], n_oscillators),
            network=SpikingNetworkParameters.from_dict(dictionary['network']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
        })
        return data_dict

# \-------------------- [ SPIKING ] ---------------------