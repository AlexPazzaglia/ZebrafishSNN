"""Network"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from queue import Queue

from farms_core.model.data import AnimatData

class AnimatNetwork(ABC):
    """Animat network"""

    def __init__(self, data, n_iterations):
        super().__init__()
        self.data: AnimatData = data
        self.n_iterations = n_iterations

    @abstractmethod
    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            **kwargs,
    ):
        """Step function called at each simulation iteration"""


# --------------------- [ SPIKING ] ---------------------
class NetworkSNN(AnimatNetwork):
    """NetworkSNN"""

    def __init__(
        self,
        data,
        snn_network,
        **kwargs
    ):
        state_array = data.state.array
        super().__init__(
            data=data,
            n_iterations=np.shape(state_array)[0]
        )

        # Inter-thread communication
        self.snn_network   = snn_network
        self.q_in  : Queue = snn_network.q_out
        self.q_out : Queue = snn_network.q_in

        self.callback_dt = float(self.snn_network.params.simulation.callback_dt)
        self.callback_st = round( self.callback_dt / self.data.timestep )

        # Motor output signal
        self.motor_output_signal : np.ndarray = kwargs.pop(
            'motor_output_signal',
            None
        )

        self.old_motor_output = np.zeros_like(self.data.state.array[0])
        self.new_motor_output = np.zeros_like(self.data.state.array[0])

        # Step function
        self.step_function : Callable = (
            self.get_queue_communication_step
            if self.motor_output_signal is None
            else
            self.get_motor_output_signal_step
        )

        assert not kwargs, kwargs

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        **kwargs,
    ):
        """Control step"""
        self.data.state.array[iteration] = self.step_function(
            iteration   = iteration,
        )
        return

    def get_queue_communication_step(self, iteration, **kwargs):
        ''' Communicate with the controller via Queue '''

        if iteration % self.callback_st == 0:

            current_time        = iteration * self.data.timestep
            transient_time      = 1.0
            transient_weighting = np.clip(current_time / transient_time, 0, 1)

            # Send joint positions and get handshake
            positions = np.asarray( self.data.sensors.joints.positions(iteration) )

            self.q_out.put(positions)
            self.q_in.get(
                block   = True,
                timeout = 600,
            )

            # Get motor output and offsets
            muscle_activations_step = np.concatenate(
                [
                    self.snn_network.get_motor_output(iteration)* transient_weighting,
                    self.snn_network.get_motor_offsets(iteration),
                ]
            )
            assert (
                not np.any(np.isnan(muscle_activations_step)) and
                not np.any(np.isinf(muscle_activations_step))
                ), 'Invalid values found in the muscle activations'

            # Final handshake
            self.q_out.put(True)
            self.q_in.get(
                block   = True,
                timeout = 600,
            )

            self.old_motor_output = self.new_motor_output
            self.new_motor_output = muscle_activations_step

        # else:
        #     muscle_activations_step = np.array( self.data.state.array[iteration - 1] )

        motor_output = (
            self.old_motor_output +
            (self.new_motor_output - self.old_motor_output) *
            (iteration % self.callback_st) / self.callback_st
        )

        return motor_output

    def get_motor_output_signal_step(self, iteration, **kwargs):
        ''' Test pre-selected oscillators' outputs '''
        return self.motor_output_signal[iteration]

class NetworkHYBRID(AnimatNetwork):
    """NetworkHYBRID"""

    def __init__(
        self,
        data,
        snn_network,
        **kwargs
    ):
        super().__init__(
            data=data,
            n_iterations=int( snn_network.params.simulation.duration / data.timestep )
        )

        # Inter-thread communication
        self.snn_network   = snn_network
        self.q_in  : Queue = self.snn_network.q_out
        self.q_out : Queue = self.snn_network.q_in

        self.callback_dt = float(self.snn_network.params.simulation.callback_dt)
        self.callback_st = round( self.callback_dt / self.data.timestep )

        assert not kwargs, kwargs

    def step(
        self,
        iteration: int,
        time: float,
        timestep: float,
        **kwargs,
    ):
        """Control step"""
        self.get_queue_communication_step(iteration)
        return

    def get_queue_communication_step(self, iteration, **kwargs):
        ''' Communicate with the controller via Queue '''

        if iteration % self.callback_st != 0:
            return

        # Send joint positions and get handshake
        positions = np.asarray( self.data.sensors.joints.positions(iteration) )

        self.q_out.put(positions)
        self.q_in.get(
            block   = True,
            timeout = 600,
        )

        # Final handshake
        self.q_out.put(True)
        self.q_in.get(
            block   = True,
            timeout = 600,
        )

        return

# \-------------------- [ SPIKING ] ---------------------
