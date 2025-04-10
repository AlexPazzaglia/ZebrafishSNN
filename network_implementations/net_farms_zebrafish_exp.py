'''
Network simulation using farms and 1DOF limbs
'''

import numpy as np
from typing import Any

from brian2.units.allunits import pamp
from brian2.core.variables import VariableView
from network_modules.experiment.network_experiment import SnnExperiment


class SalamandraSimulation(SnnExperiment):
    '''
    Class used to run a simulation in closed loop with multi-dof limbs
    '''

    # Callback function

    def step_function(self, curtime: VariableView) -> None:
        '''
        When used (include_callback = True), called at every integration step.\n
        Exchanges data with external thread via input and output queues.
        '''

        # # Turn right, Turn left
        # self.step_update_turning(curtime, self.params.simulation.duration*1/3, +0.1)
        # self.step_update_turning(curtime, self.params.simulation.duration*2/3, -0.1)

        # # Silence feedback
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 0/2, active= False )
        # self.step_feedback_toggle( curtime, self.params.simulation.duration * 1/2, active= True )

        # Silence drive
        # self.step_drive_toggle( curtime, self.params.simulation.duration * 1/2, active= False )

        # # Desired time
        # if self.step_desired_time(curtime, 0.5*self.params.simulation.duration):
        #     print(curtime)

        # Send handshake
        self.q_out.put(True)

        # Proprioceptive feedback
        self.get_mechanics_input()

        # Final handshake
        self.queue_handshake()

        return
