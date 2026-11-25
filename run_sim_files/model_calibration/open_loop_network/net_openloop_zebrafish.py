'''
Network simulation using farms and 1DOF limbs
'''

import logging
import numpy as np
from typing import Any

from brian2.units.allunits import pamp
from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment


class SalamandraSimulation(SnnExperiment):
    '''
    Class used to run a simulation in closed loop with multi-dof limbs
    '''


    def update_network_parameters_from_dict(self, new_pars: dict) -> None:
        ''' Update network parameters from a dictionary '''

        ex2ex_weight = new_pars.pop('ex2ex_weight')
        ex2in_weight = new_pars.pop('ex2in_weight')
        in2ex_weight = new_pars.pop('in2ex_weight')
        in2in_weight = new_pars.pop('in2in_weight')
        rs2ex_weight = new_pars.pop('rs2ex_weight')
        rs2in_weight = new_pars.pop('rs2in_weight')

        logging.info('CPG SCALINGS')
        message = f'Scale ex to ex synaptic strength by factor {ex2ex_weight}\n'
        logging.info(message)
        message = f'Scale ex to in synaptic strength by factor {ex2in_weight}\n'
        logging.info(message)
        message = f'Scale in to ex synaptic strength by factor {in2ex_weight}\n'
        logging.info(message)
        message = f'Scale in to in synaptic strength by factor {in2in_weight}\n'
        logging.info(message)

        logging.info('RS SCALINGS')
        message = f'Scale rs to ex synaptic strength by factor {rs2ex_weight}\n'
        logging.info(message)
        message = f'Scale rs to in synaptic strength by factor {rs2in_weight}\n'
        logging.info(message)

        syn_weights_list = [

            ########################################
            # CPG to CPG ###########################
            ########################################

            # Ex2Ex
            {
                'source_name' : 'cpg.axial.ex',
                'target_name' : 'cpg.axial.ex',
                'weight_ampa' : [ex2ex_weight, ''],
                'weight_nmda' : [ex2ex_weight, ''],
            },
            # Ex2In
            {
                'source_name' : 'cpg.axial.ex',
                'target_name' : 'cpg.axial.in',
                'weight_ampa' : [ex2in_weight, ''],
                'weight_nmda' : [ex2in_weight, ''],
            },
            # In2Ex
            {
                'source_name' : 'cpg.axial.in',
                'target_name' : 'cpg.axial.ex',
                'weight_glyc' : [in2ex_weight, ''],
            },
            # In2In
            {
                'source_name' : 'cpg.axial.in',
                'target_name' : 'cpg.axial.in',
                'weight_glyc' : [in2in_weight, ''],
            },

            ########################################
            # RS to CPG ############################
            ########################################

            # Rs2Ex
            {
                'source_name' : 'rs',
                'target_name' : 'cpg.axial.ex',
                'weight_ampa' : [rs2ex_weight, ''],
                'weight_nmda' : [rs2ex_weight, ''],
            },
            # Rs2In
            {
                'source_name' : 'rs',
                'target_name' : 'cpg.axial.in',
                'weight_ampa' : [rs2in_weight, ''],
                'weight_nmda' : [rs2in_weight, ''],
            },

        ]

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

        # Silence feedback
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return
