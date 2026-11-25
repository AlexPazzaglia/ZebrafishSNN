'''
Calibration of the motor output gains G_MC (Supplementary Material, Eq. 11).

The isolated neural network (no proprioceptive feedback) drives the
mechanical model through the Ekeberg muscle model. The per-joint gains G_MC
are iteratively adjusted until the joint angle amplitudes match the
sub-carangiform kinematics of zebrafish swimming.

The starting gains below are the result of the calibration for muscles with a
natural frequency of 5.5 Hz, and are the values used in all the closed-loop
(in vivo) simulations.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import sim_open_loop

def main():
    ''' Optimize the motor output gains in open loop '''

    sim_pars = {}

    # Muscle parameters (natural frequency 5.5 Hz)
    sim_pars['muscle_resonant_freq'] = None
    sim_pars['muscle_tag']           = 'new_FN_5500_ZC_1000_G0_419_gen_100'

    # Initial motor output gains
    sim_pars['mo_gains_axial'] = np.array(
        [
            0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
            0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
            0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
        ]
    )

    # Simulation parameters
    sim_pars['duration']   = 20.0
    sim_pars['stim_a_off'] = 0.0

    # Cocontraction terms
    sim_pars['mo_cocontraction_gain'] = 1.0
    sim_pars['mo_cocontraction_off']  = 0.0

    # Connectivity parameters
    sim_pars['ps_connections_range']  = 0.50
    sim_pars['cpg_connections_range'] = 0.65

    # Connection weights
    sim_pars.update(
        {
            'ex2ex_weight': 3.83037,
            'ex2in_weight': 49.47920,
            'in2ex_weight': 0.84541,
            'in2in_weight': 0.10330,
            'rs2ex_weight': 8.74440,
            'rs2in_weight': 3.28338,
        }
    )

    # Optimization
    sim_pars['n_trials']   = 10
    sim_pars['tolerance']  = 0.02
    sim_pars['learn_rate'] = 0.2

    # Result
    sim_pars['video'] = False
    sim_pars['plot']  = True
    sim_pars['save']  = False

    sim_open_loop.run(**sim_pars)

if __name__ == '__main__':
    main()
