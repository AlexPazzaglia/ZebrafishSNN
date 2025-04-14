"""Amphibious data"""

from typing import Any
import numpy as np
cimport numpy as np
from nptyping import NDArray, Shape


cdef class AmphibiousDataCy(AnimatDataCy):
    """Amphibious data"""
    pass

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingDataCy(AnimatDataCy):
    """Spiking data"""
    pass

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingNetworkParametersCy:
    """Spiking network parameters"""

    def __init__(
            self,
            oscillators: SpikingOscillatorsCy,
    ):
        super().__init__()
        self.oscillators = oscillators

# --------------------- [ SPIKING ] ---------------------

cdef class NetworkStateCy(DoubleArray2D):
    """Network state"""


# --------------------- [ SPIKING ] ---------------------
cdef class SpikingNetworkStateCy(NetworkStateCy):
    """Spiking network state"""

    def __init__(
            self,
            array: NDArray[Shape['*, *'], np.double],
            n_oscillators: int,
    ):
        assert np.ndim(array) == 2, 'Ndim {np.ndim(array)} != 2'
        assert n_oscillators > 1, f'n_oscillators={n_oscillators} must be > 1'
        super().__init__(array=array)
        self.n_oscillators = n_oscillators

    cpdef DTYPEv1 offsets(self, unsigned int iteration):
        """Offset"""
        return self.array[iteration, self.n_oscillators:]

    cpdef DTYPEv2 offsets_all(self):
        """Offset"""
        return self.array[:, self.n_oscillators:]

    cpdef np.ndarray outputs(self, unsigned int iteration):
        """Outputs"""
        return self.array[iteration, :self.n_oscillators] * np.ones(self.n_oscillators)

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingOscillatorsCy:
    """Spiking oscillators"""

    def __init__(
            self,
            n_oscillators: int,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators

# \-------------------- [ SPIKING ] ---------------------



