''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import sim_signal_driven

def main():
    ''' Run the analysis-specific post-processing script '''


    sim_pars = {}

    # Muscle parameters
    sim_pars['muscle_resonant_freq'] = 7.0

    # Simulation parameters
    sim_pars['duration']   = 20.0

    # Activation signal
    sim_pars['frequency']  = 3.5
    sim_pars['muscle_bsl'] = 10.0

    # Optimize
    sim_pars['n_trials']   = 1
    sim_pars['tolerance']  = 0.02
    sim_pars['learn_rate'] = 0.2

    # Result
    sim_pars['video'] = True
    sim_pars['plot']  = True
    sim_pars['save']  = False

    # Run simulation
    sim_signal_driven.run(**sim_pars)

    return

if __name__ == '__main__':
    main()
