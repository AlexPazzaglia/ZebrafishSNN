"""Amphibious data"""

include 'types.pxd'
import numpy as np
cimport numpy as np
from farms_core.sensors.data_cy cimport SensorsDataCy
from farms_core.model.data_cy cimport AnimatDataCy
from farms_core.array.array_cy cimport (
    DoubleArray1D,
    DoubleArray2D,
    IntegerArray1D,
    IntegerArray2D,
)

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingDataCy(AnimatDataCy):
    """Spiking data"""
    cdef public SpikingNetworkStateCy state
    cdef public SpikingNetworkParametersCy network

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingNetworkParametersCy:
    """SpikingNetwork parameters"""
    cdef public SpikingOscillatorsCy oscillators

# \-------------------- [ SPIKING ] ---------------------

cdef class NetworkStateCy(DoubleArray2D):
    """Network state"""

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingNetworkStateCy(NetworkStateCy):
    """Spiking network state"""
    cdef public unsigned int n_oscillators

    cpdef public DTYPEv1 offsets(self, unsigned int iteration)
    cpdef public DTYPEv2 offsets_all(self)
    cpdef public np.ndarray outputs(self, unsigned int iteration)

# \-------------------- [ SPIKING ] ---------------------

# --------------------- [ SPIKING ] ---------------------
cdef class SpikingOscillatorsCy:
    """Spiking oscillator array"""
    cdef public unsigned int n_oscillators

# \-------------------- [ SPIKING ] ---------------------
