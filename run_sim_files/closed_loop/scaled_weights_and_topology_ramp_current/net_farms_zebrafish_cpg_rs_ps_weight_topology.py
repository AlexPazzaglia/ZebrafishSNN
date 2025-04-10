'''
Network simulation using farms and 1DOF limbs
'''

import logging
import numpy as np
from typing import Any

from brian2.units.allunits import pamp
from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment

import network_experiments.default_parameters.zebrafish.swimming_default as default

class SalamandraSimulation(SnnExperiment):
    '''
    Class used to run a simulation in closed loop with multi-dof limbs
    '''

    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        ''' Update network parameters from a dictionary '''

        # Check for missing keys
        expected_keys = [
            'ex2ex_weight',
            'ex2in_weight',
            'in2ex_weight',
            'in2in_weight',
            'rs2ex_weight',
            'rs2in_weight',
            'ps_weight',
        ]
        for key in expected_keys:
            assert key in new_pars, f'Missing key: {key}'

        # Update synaptic weights
        syn_weights_list = default.get_syn_weights_list(new_pars)

        self._assign_synaptic_weights_by_list(
            syn_weights_list = syn_weights_list,
            std_val         = 0.0,
        )

        # Update network parameters
        super().update_network_parameters_from_dict(new_pars)

    # Callback function
    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        stim_max  = 1.00 # + 1.5
        stim_min  = -0.5 # - 1.5

        self.step_ramp_current_axis(
            curtime = curtime,
            curr0   = stim_max,
            curr1   = stim_min,
        )

        # Silence feedback
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return
