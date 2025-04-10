import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import sim_open_loop
import sim_closed_loop

def main():
    ''' Run the analysis-specific post-processing script '''

    OPEN_LOOP   = True
    CLOSED_LOOP = False

    sim_pars = {}

    # Muscle parameters
    sim_pars['muscle_resonant_freq'] = None

    sim_pars['muscle_tag']     = 'new_FN_5500_ZC_1000_G0_419_gen_100'
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

        # 'ex2ex_weight': 11.03748,
        # 'ex2in_weight': 8.37228,
        # 'in2ex_weight': 4.36458,
        # 'in2in_weight': 0.23006,
        # 'rs2ex_weight': 9.97614,
        # 'rs2in_weight': 3.09240,
        }
    )

    # Feedback parameters (closed loop)
    sim_pars['ps_min_activation_deg'] = 10.0
    sim_pars['ps_weight']             = 10.0

    # Optimize
    sim_pars['n_trials']   = 10
    sim_pars['tolerance']  = 0.02
    sim_pars['learn_rate'] = 0.2

    # Result
    sim_pars['video'] = False
    sim_pars['plot']  = True
    sim_pars['save']  = False

    # Run simulation
    if OPEN_LOOP:
        sim_open_loop.run(**sim_pars)
    elif CLOSED_LOOP:
        sim_closed_loop.run(**sim_pars)

    return

if __name__ == '__main__':
    main()