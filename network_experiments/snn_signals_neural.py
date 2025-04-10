
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm

from typing import Any, Callable

from network_modules.experiment.network_experiment import SnnExperiment

def _plot_signals(
    times         : np.ndarray,
    signals_chains: list[np.ndarray],
    names_chains  : list[str],
    )             :
    ''' Plot signals '''

    n_chains = len(signals_chains)

    fig, axs = plt.subplots(
        n_chains,
        1,
        sharex  = True,
        figsize = (8.0, 10.5),
        dpi     = 300
    )
    axs[0].set_title('Motor outputs')
    axs[0].set_xlim(times[0], times[-1])
    axs[-1].set_xlabel('Time')
    fig.subplots_adjust(hspace=0)

    for chain, signals in enumerate(signals_chains):

        max_val   = np.amax(signals)
        min_val   = np.amin(signals)
        range_val = max_val - min_val

        n_signals   = signals.shape[1]
        n_joints    = n_signals // 2
        signal_jump = range_val * 1.0

        colors = matplotlib.cm.winter(np.linspace(0.10, 0.90, n_joints))

        for joint in range(n_joints):
            axs[chain].plot(
                times,
                + signals[:, 2*joint + 0] + joint * signal_jump,
                color     = colors[joint],
                linewidth = 2,
            )
            axs[chain].plot(
                times,
                + signals[:, 2*joint + 1] + joint * signal_jump,
                color = colors[joint],
                linestyle = 'dashed',
            )

        y_min, y_max = axs[chain].get_ylim()
        axs[chain].set_ylim(y_max, y_min)
        axs[chain].set_ylabel(names_chains[chain])
        axs[chain].set_yticks([])
        axs[chain].grid()

    return

def _neural_output_single_chain(
    frequency   : float,
    times       : np.ndarray[float],
    amp_joints  : np.ndarray[float],
    off_joints  : np.ndarray[float],
    bsl_joints  : np.ndarray[float],
    ipl_joints  : np.ndarray[float],
    sig_function: Callable = None,
):
    ''' Neural output for the DOFs of a single chain '''

    assert len(amp_joints) == len(ipl_joints), \
        'amp_joints and ipl_joints should have the same length'
    assert len(amp_joints) == len(off_joints), \
        'amp_joints and off_joints should have the same length'
    assert len(amp_joints) == len(bsl_joints), \
        'amp_joints and bsl_joints should have the same length'

    n_joints         = len(amp_joints)
    times_expanded   = np.vstack([times] * n_joints).T

    # Signal function
    if sig_function is None:
        sig_function = lambda phase : np.sin( 2*np.pi * phase )

    chain_signal_aux = sig_function( frequency*times_expanded - ipl_joints )

    # Two motor outputs for each joint
    chain_signal = np.array(
        [
            # j_sign * j_off/2 + j_amp * (1 + j_sign * chain_signal_aux[:, j_ind])/2

            (
                0.5 * j_bsl +                                           # Baseline (M_L + M_R)
                0.5 * j_sign * j_off +                                  # Offset (M_L - M_R)
                0.5 * j_sign * j_amp * chain_signal_aux[:, j_ind]       # Signal (M_L - M_R)
            )

            for j_ind, (j_off, j_amp, j_bsl) in enumerate(zip(off_joints, amp_joints, bsl_joints))
            for j_sign in [+1, -1]
        ]
    ).T

    return chain_signal

def _neural_output_multi_chain(
    times            : np.ndarray[float],
    frq_joints_chains: list[np.ndarray[float]],
    amp_joints_chains: list[np.ndarray[float]],
    off_joints_chains: list[np.ndarray[float]],
    bsl_joints_chains: list[np.ndarray[float]],
    ipl_joints_chains: list[np.ndarray[float]],
    ipl_chains       : list[float],
    sig_funcion      : Callable = None,
):
    ''' Neural output for the DOFs of multiple chains '''

    n_chains = len(ipl_chains)

    assert len(frq_joints_chains) == n_chains, \
        'frq_joints_chains should have length n_chains'
    assert len(amp_joints_chains) == n_chains, \
        'amp_joints_chains should have length n_chains'
    assert len(off_joints_chains) == n_chains, \
        'off_joints_chains should have length n_chains'
    assert len(bsl_joints_chains) == n_chains, \
        'bsl_joints_chains should have length n_chains'
    assert len(ipl_joints_chains) == n_chains, \
        'ipl_joints_chains should have length n_chains'

    multi_chains_signals_separated = [
        _neural_output_single_chain(
            times      = times,
            frequency  = frq_joints_chains[chain],
            amp_joints = amp_joints_chains[chain],
            off_joints = off_joints_chains[chain],
            bsl_joints = bsl_joints_chains[chain],
            ipl_joints = ipl_joints_chains[chain] + ipl_chains[chain],
            sig_function    = sig_funcion
        )
        for chain in range(n_chains)
    ]
    multi_chains_signals = np.concatenate(multi_chains_signals_separated, axis=1)

    return multi_chains_signals, multi_chains_signals_separated

def _get_neural_output_signal(
    times        : np.ndarray,
    chains_params: dict[str, dict[str, Any]],
    sig_funcion  : Callable = None,
):
    ''' Get neural output '''

    ipl_chains        = []
    names_chains      = []
    frq_joints_chains = []
    amp_joints_chains = []
    off_joints_chains = []
    bsl_joints_chains = []
    ipl_joints_chains = []

    # CHAINS
    for chain_name, chain_pars in chains_params.items():

        if chain_pars['n_joints'] == 0:
            continue

        frequency_chain = chain_pars.get('frequency')

        # AMPLITUDE
        amp_joints_min = chain_pars.get('amp_min')
        amp_joints_max = chain_pars.get('amp_max')
        amp_joints_arr = chain_pars.get('amp_arr')

        if amp_joints_min is not None and amp_joints_max is not None:
            assert amp_joints_arr is None, \
                'Either amp_joints_arr or amp_joints_min and amp_joints_max must be None'
            amp_joints_arr = np.linspace(amp_joints_min, amp_joints_max, chain_pars['n_joints'])

        # OFFSET
        off_joints_min = chain_pars.get('off_min')
        off_joints_max = chain_pars.get('off_max')
        off_joints_arr = chain_pars.get('off_arr')

        if off_joints_min is not None and off_joints_max is not None:
            assert off_joints_arr is None, \
                'Either off_joints_arr or off_joints_min and off_joints_max must be None'
            off_joints_arr = np.linspace(off_joints_min, off_joints_max, chain_pars['n_joints'])

        # BASELINE
        bsl_joints_min = chain_pars.get('bsl_min')
        bsl_joints_max = chain_pars.get('bsl_max')
        bsl_joints_arr = chain_pars.get('bsl_arr')

        if bsl_joints_min is not None and bsl_joints_max is not None:
            assert bsl_joints_arr is None, \
                'Either bsl_joints_arr or bsl_joints_min and bsl_joints_max must be None'
            bsl_joints_arr = np.linspace(bsl_joints_min, bsl_joints_max, chain_pars['n_joints'])

        # PHASE
        ipl_joints_min = chain_pars.get('ipl_min')
        ipl_joints_max = chain_pars.get('ipl_max')
        ipl_joints_arr = chain_pars.get('ipl_arr')

        if ipl_joints_min is not None and ipl_joints_max is not None:
            assert ipl_joints_arr is None, \
                'Either ipl_joints_arr or ipl_joints_min and ipl_joints_max must be None'
            ipl_joints_arr = np.linspace(ipl_joints_min, ipl_joints_max, chain_pars['n_joints'])

        # COPIES
        n_copies     = chain_pars.get('n_copies', 1)
        ipl_off      = chain_pars.get('ipl_off', [0.0])
        names_copies = chain_pars.get('names', [chain_name.upper()])

        assert len(ipl_off) == n_copies, \
            'ipl_off must have the same length as n_copies'
        assert len(names_copies) == n_copies, \
            'names_copies must have the same length as n_copies'

        # APPEND
        ipl_chains        = ipl_chains   + ipl_off
        names_chains      = names_chains + names_copies
        frq_joints_chains = frq_joints_chains + [frequency_chain] * n_copies
        amp_joints_chains = amp_joints_chains + [amp_joints_arr]  * n_copies
        off_joints_chains = off_joints_chains + [off_joints_arr]  * n_copies
        bsl_joints_chains = bsl_joints_chains + [bsl_joints_arr]  * n_copies
        ipl_joints_chains = ipl_joints_chains + [ipl_joints_arr]  * n_copies

    # MOTOR OUTPUT SIGNALS
    motor_output_signals, multi_chains_signals_separated = _neural_output_multi_chain(
        times             = times,
        frq_joints_chains = frq_joints_chains,
        amp_joints_chains = amp_joints_chains,
        off_joints_chains = off_joints_chains,
        bsl_joints_chains = bsl_joints_chains,
        ipl_joints_chains = ipl_joints_chains,
        ipl_chains        = ipl_chains,
        sig_funcion       = sig_funcion,
    )

    return motor_output_signals, multi_chains_signals_separated, names_chains

def get_motor_output_signal(
    motor_output_params: dict[str, dict[str, Any]],
    snn_network        : SnnExperiment,
    **kwargs
):
    ''' Get pre-selected oscillators' motor outputs '''

    duration = float( snn_network.params.simulation.duration )
    timestep = float( snn_network.params.mechanics.mech_timestep )
    times = np.arange(
        0,
        duration + timestep,
        timestep,
    )

    # Neural output
    sig_function = motor_output_params.pop('sig_function', None)
    (
        motor_output_signals,
        multi_chains_signals_separated,
        names_chains,
    ) = _get_neural_output_signal(
        times         = times,
        chains_params = motor_output_params,
        sig_funcion   = sig_function,
    )

    # Motor offsets
    motor_offsets = np.vstack(
        [
            snn_network.get_motor_offsets(0)
            for _ in range(len(times))
        ]
    )

    # Motor output
    motor_output = np.hstack(
        [
            motor_output_signals,
            motor_offsets,
        ]
    )

    return motor_output

def run_example():

    times = np.arange(0, 10, 0.001)
    n_joints_trunk = 4
    n_joints_tail  = 4
    n_joints_limb  = 4

    motor_output_signal_pars = {
        'trunk': {
            'frequency': 1.0,
            'n_copies' : 1,
            'ipl_off'  : [0.0],
            'names'    : ['TRUNK'],

            'n_joints': n_joints_trunk,
            'ipl_min' : 0.0,
            'ipl_max' : 0.5,
            'amp_min' : 1.0,
            'amp_max' : 1.0,
            'off_arr' : np.zeros(n_joints_trunk),
            'bsl_arr' : np.zeros(n_joints_trunk),
        },
        'tail': {
            'frequency': 1.0,
            'n_copies' : 1,
            'ipl_off'  : [0.5],
            'names'    : ['TAIL'],

            'n_joints': n_joints_tail,
            'ipl_min' : 0.0,
            'ipl_max' : 0.5,
            'amp_min' : 1.0,
            'amp_max' : 1.0,
            'off_arr' : np.zeros(n_joints_tail),
            'bsl_arr' : np.zeros(n_joints_tail),
        },
        'forelimbs': {
            'frequency': 1.0,
            'n_copies' : 2,
            'ipl_off'  : [0.0, 0.5],
            'names'    : ['LF', 'RF'],

            'n_joints': n_joints_limb,
            'ipl_arr' : np.array([0.25, 0.00, 0.00, 0.00]),
            'amp_arr' : np.zeros(n_joints_limb),
            'off_arr' : np.zeros(n_joints_limb),
            'bsl_arr' : np.zeros(n_joints_limb),
        },
        'hindlimbs': {
            'frequency': 1.0,
            'n_copies' : 2,
            'ipl_off'  : [0.0, 0.5],
            'names'    : ['LH', 'RH'],

            'n_joints': n_joints_limb,
            'ipl_arr' : np.array([0.25, 0.00, 0.00, 0.00]),
            'amp_arr' : np.zeros(n_joints_limb),
            'off_arr' : np.zeros(n_joints_limb),
            'bsl_arr' : np.zeros(n_joints_limb),
        },
    }

    (
        motor_output_signals,
        multi_chains_signals_separated,
        names_chains,
    ) = _get_neural_output_signal(
        times         = times,
        chains_params = motor_output_signal_pars,
        sig_funcion   = lambda phase : np.tanh( 3 * np.sin( 2*np.pi * phase ) ),
    )

    _plot_signals(
        times,
        multi_chains_signals_separated,
        names_chains = names_chains,
    )

    plt.show()



if __name__ == '__main__':
    run_example()