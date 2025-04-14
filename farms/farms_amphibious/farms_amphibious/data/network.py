"""Network"""

from typing import List, Dict

import numpy as np

from farms_core.array.types import NDARRAY_V1


from .data_cy import (
    SpikingNetworkParametersCy,
    SpikingNetworkStateCy,
    SpikingOscillatorsCy,
)

# pylint: disable=no-member


NPDTYPE = np.float64
NPUITYPE = np.uintc

# --------------------- [ SPIKING ] ---------------------
class SpikingNetworkState(SpikingNetworkStateCy):
    """Network state"""

    @classmethod
    def from_initial_state(
            cls,
            initial_state: NDARRAY_V1,
            n_iterations: int,
            n_oscillators: int,
    ):
        """From initial state"""
        state_size = len(initial_state)
        state_array = np.full(
            shape=[n_iterations, state_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        state_array[0, :] = initial_state
        return cls(array=state_array, n_oscillators=n_oscillators)

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
class SpikingOscillators(SpikingOscillatorsCy):
    """Oscillator array"""

    def __init__(
            self,
            names: List[str],
    ):
        super().__init__(n_oscillators=len(names))
        self.names = names

    @classmethod
    def from_options(cls, network):
        """Default"""
        return cls(network.osc_names())

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(names=dictionary['names'])

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {'names': self.names}

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
class SpikingNetworkParameters(SpikingNetworkParametersCy):
    """Network parameter"""

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            oscillators=SpikingOscillators.from_dict(
                dictionary['oscillators']
            ),
        ) if dictionary else None

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {'oscillators': self.oscillators.to_dict()}

# \-------------------- [ SPIKING ] ---------------------