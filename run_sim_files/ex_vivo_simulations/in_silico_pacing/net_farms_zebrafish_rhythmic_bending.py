'''
Network module for the in silico pacing experiments.

The network receives proprioceptive feedback from a position-controlled body
(hybrid position control).
'''

from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment

import network_experiments.default_parameters.zebrafish.swimming_default as default

class SalamandraSimulation(SnnExperiment):
    '''
    Spiking network coupled to the position-controlled zebrafish body
    '''

    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        ''' Update network parameters from a dictionary '''

        # Neural parameters gain
        neural_pars_gain = new_pars.pop('neural_pars_gain', 1.0)

        # Update synaptic weights
        syn_weights_list = default.get_syn_weights_list(new_pars)

        self._assign_synaptic_weights_by_list(
            syn_weights_list = syn_weights_list,
            std_val         = 0.0,
        )

        # Update neural params
        gain = neural_pars_gain

        self.pop.delta_w1[:] = self.pop.delta_w1[:] * gain * 1.2
        self.pop.tau1[:]     = self.pop.tau1[:]     / gain

        self.pop.tau_memb[:] = self.pop.tau_memb[:] / gain
        self.pop.t_refr[:]   = self.pop.t_refr[:]   / gain


        # Update network parameters
        super().update_network_parameters_from_dict(new_pars)

    # Callback function
    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return
