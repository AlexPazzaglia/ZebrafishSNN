'''
Module to run experiments with the defined neuronal models
'''

import os
import sys
import inspect
import warnings

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from neuron_experiments.inputs import input_factory
from neuron_experiments.experiments import neuronal_model

import logging

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# COMPUTE PARAMETERS OF THE NEURONAL MODEL

def compute_spike_times(
    statemon,
    neuron_index = 0,
    upswing      = True
):
    '''
    Returns the spike times, computed from threshold crossing
    '''
    th_mv = -30
    vm_mv = np.array( statemon[neuron_index].vm / b2.mV )
    times_ms = np.array( statemon.t / b2.ms )

    # Find spike onsets
    threshold_crossings = np.diff(vm_mv > th_mv, prepend=False)

    if upswing:
        spike_onsets_inds = np.argwhere(threshold_crossings)[::2,0]
        spike_times_ms = times_ms[spike_onsets_inds]
    else:
        spike_offsets_inds = np.argwhere(threshold_crossings)[1::2,0]
        spike_times_ms = times_ms[spike_offsets_inds]

    return spike_times_ms

def compute_inter_spike_intervals(
    spikemon : b2.SpikeMonitor,
):
    ''' Computes the interspike intervals from the spike times '''
    spike_trains    = spikemon.spike_trains()
    n_neurons       = len(spike_trains)
    spike_intervals = [
        np.diff( spike_trains[ner_id] )
        for ner_id in range(n_neurons)
    ]
    return spike_intervals

def compute_equilibrium_values(
    monitor_vars   : list[str],
    neuron_group   : b2.NeuronGroup,
    simulation_time: float = 10000 * b2.ms,
):
    '''
    Run simulation with zero input to get equilibrium values
    '''

    statemon, _spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = input_factory.get_zero_current(),
        simulation_time = simulation_time,
        monitor_vars    = monitor_vars,
    )

    eq_vals = {
        var_name : getattr(statemon, var_name)[0,-1]
        for var_name in monitor_vars
    }

    logging.info('Equilibrium values:')
    for key, value in eq_vals.items():
        logging.info(f'{key} = {value:.5f}')

    return eq_vals

def compute_adaptation_time(
    neuron_group: b2.NeuronGroup,
    duration_ms : float,
    amplitude_pa: float,
    spike_times : np.ndarray = None,
    neuron_index: int = 0,
):
    '''
    Computes the time constant for the adaptation (with selected current)
    '''

    if spike_times is None:
        # Stimulation not provided, run simulation

        i_stim = input_factory.get_step_current(
            t_start   = 0,
            t_end     = duration_ms,
            unit_time = b2.ms,
            amplitude = amplitude_pa * b2.pA,
        )

        _statemon, spikemon = neuronal_model.simulate_neuron_population(
            neuron_group    = neuron_group,
            i_stim          = i_stim,
            simulation_time = duration_ms * b2.ms,
        )
        spike_times = np.array(spikemon.t)[spikemon.i == neuron_index]

    isi    = np.diff(spike_times)
    isi_th = isi[0] + 0.632 * (isi[-1] - isi[0])

    if isi[-1] >= isi[0]:
        # Find rise time
        th_crossing = np.diff(isi > isi_th, prepend=False)
    else:
        # Find decay time
        th_crossing = np.diff(isi < isi_th, prepend=False)

    crossing_index = np.argwhere(th_crossing)[0,0]
    time_constant  = spike_times[0] + np.sum( isi[:crossing_index] )

    logging.info(f'Adaptation time constant: {time_constant:.2f} s')
    return time_constant

def compute_adaptation_level(
    neuron_group: b2.NeuronGroup,
    duration_ms : float,
    amplitude_pa: float,
    spike_times : np.ndarray = None,
    neuron_index: int = 0,
):
    '''
    Compute the level of adaptation with respect to the initial firing rate
    '''

    if spike_times is None:
        # Stimulation not provided, run simulation

        i_stim = input_factory.get_step_current(
            t_start   = 0,
            t_end     = duration_ms,
            unit_time = b2.ms,
            amplitude = amplitude_pa * b2.pA,
        )

        _statemon, spikemon = neuronal_model.simulate_neuron_population(
            neuron_group    = neuron_group,
            i_stim          = i_stim,
            simulation_time = duration_ms * b2.ms,
        )
        spike_times = np.array(spikemon.t)[spikemon.i == neuron_index]

    isi = np.diff(spike_times)
    adaptation_level = (isi[-1] - isi[0]) / isi[0]

    logging.info(f'Adaptation level: {adaptation_level:.2f}')
    return adaptation_level

def compute_membrane_time_constant(
    neuron_group: b2.NeuronGroup,
    t_start     : float,
    duration_ms : float,
    amplitude_pa: float,
    statemon    : b2.StateMonitor = None,
    neuron_index: int = 0,
):
    '''
    Computes the time constant for the adaptation (with selected current)
    '''

    if statemon is None:
        # Stimulation not provided, run simulation
        i_stim = input_factory.get_step_current(
            t_start   = 0,
            t_end     = duration_ms,
            unit_time = b2.ms,
            amplitude = amplitude_pa * b2.pA,
        )

        statemon, _spikemon = neuronal_model.simulate_neuron_population(
            neuron_group    = neuron_group,
            i_stim          = i_stim,
            simulation_time = duration_ms * b2.ms,
        )

    times = np.array(statemon.t)
    t_start = float(t_start)
    v_memb = np.array(statemon[neuron_index].v)
    v_start = v_memb[0]
    v_end = v_memb[-1]

    # Get an estimate of the rise time
    v_th = v_start + 0.632 * (v_end - v_start)
    th_crossing = np.diff(v_memb > v_th, prepend=False)
    crossing_index = np.argwhere(th_crossing)[0,0]
    time_constant = times[crossing_index] - t_start

    # NOTE: Refine the estimate with linear fitting of the voltage
    # V(t) = V_0 + (V_inf - V_0) * ( 1 - exp(-t/tau) )
    # V(t) ~ V_0 + (V_inf - V_0) * t / tau
    t_rise = times[times > t_start] - t_start
    v_rise = v_memb[times > t_start] - v_start

    # Consider only the linear part (time_constant / 4)
    t_rise = ( t_rise[t_rise < time_constant / 4] ).reshape((-1,1))
    v_rise = v_rise[:len(t_rise)]

    # Linear regression
    model = LinearRegression(fit_intercept= False ).fit(t_rise, v_rise)
    time_constant = (v_end - v_start) / model.coef_

    logging.info(f'Membrane time constant: {time_constant[0]:.2f} s')
    return time_constant[0]

def compute_rheobase_current(
    neuron_group   : b2.NeuronGroup,
    current_max_pa : float,
    simulation_time: float,
    spike_times    : np.ndarray = None,
    neuron_index   : int = 0,
):


    '''
    Compute the rheobase current

    This calculation is most accurate when assessed over extended periods.
    As it allows the applied current to reach its full potential to induce a spike.
    If the evaluation time is too short, the current measured at the spike's occurrence might not represent the true rheobase.
    As the neuron may require more time to reach the threshold for firing.

    '''

    if spike_times is None:
        ramp_current = input_factory.get_ramp_current(
            t_start         = 0,
            t_end           = simulation_time,
            unit_time       = b2.ms,
            amplitude_start = 0.0 * b2.pA,
            amplitude_end   = current_max_pa * b2.pA,
            append_zero     = True
        )
        # Simulate
        _statemon, spikemon = neuronal_model.simulate_neuron_population(
            neuron_group    = neuron_group,
            i_stim          = ramp_current,
            simulation_time = simulation_time * b2.ms
        )

        spike_times = np.array(spikemon.t)[spikemon.i == neuron_index]

    # Computation
    rheobase_current = ramp_current(spike_times[0] * b2.ms, 0)

    logging.info(f'Rheobase current: {rheobase_current:.2f} A')
    return rheobase_current

def compute_membrane_capacitance(
    neuron_group   : b2.NeuronGroup,
    min_current_pa : float = 0.5,
    max_current_pa : float = 2.5,
    pulse_duration : float = 0.001,
    simulation_time: float = 50,
):
    ''' Compute the membrane capacitance '''

    n_currents        = neuron_group.N
    pulse_duration_ms = int( pulse_duration * 1000 )
    curvals           = np.linspace(
        min_current_pa,
        max_current_pa,
        n_currents
    )

    # Create the input current
    currents = [
        input_factory.get_step_current(
            20,
            20 + pulse_duration_ms - 1,
            b2.ms,
            curval * b2.pA
        )
        for curval in curvals
    ]
    currents = input_factory.stack_currents(currents, b2.ms)

    # Simulate all neurons at once
    statemon, _spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = currents,
        simulation_time = simulation_time,
        monitor_vars    = ['v']
    )

    # Computation
    vmemb       = np.array(statemon.v)
    vmemb_rest  = vmemb[:,0]
    vmemb_maxs  = np.amax(vmemb, axis=1)
    c_estimates = (curvals * 1e-6) * pulse_duration / ( vmemb_maxs - vmemb_rest )

    c_memb = np.mean(c_estimates) * b2.farad

    logging.info(f'Membrane capacitance: {c_memb:.2f} F')
    return c_memb

# COMPUTE RESPONSE OF THE NEURONAL MODEL

def simulate_response_to_step_currents(
    neuron_group   : b2.NeuronGroup,
    t_start_ms     : float,
    t_end_ms       : float,
    current_min_pa : float,
    current_max_pa : float,
    simulation_time: float,
    unit_time      : float = 0.001,
    current_vals_pa: np.ndarray = None,
):
    ''' Simulate the response of the neuronal model to a step current '''

    n_neurons = neuron_group.N

    if current_vals_pa is None:
        current_vals_pa = np.linspace(current_min_pa, current_max_pa, n_neurons)

    currents = [
        input_factory.get_step_current(
            t_start   = t_start_ms,
            t_end     = t_end_ms,
            unit_time = unit_time * b2.second,
            amplitude = curval * b2.pA
        )
        for curval in current_vals_pa
    ]
    currents = input_factory.stack_currents(currents, unit_time * b2.second)

    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = currents,
        simulation_time = simulation_time * unit_time * b2.second
    )

    return statemon, spikemon, currents

def simulate_response_to_ramp_currents(
    neuron_group      : b2.NeuronGroup,
    t_start_ms        : float,
    t_end_ms          : float,
    current_start_pa  : float,
    current_end_min_pa: float,
    current_end_max_pa: float,
    simulation_time   : float,
):
    ''' Simulate the response of the neuronal model to a step current '''

    n_neurons = neuron_group.N

    currents = [
        input_factory.get_ramp_current(
            t_start         = t_start_ms,
            t_end           = t_end_ms,
            unit_time       = b2.ms,
            amplitude_start = current_start_pa * b2.pA,
            amplitude_end   = curval * b2.pA
        )
        for curval in np.linspace(current_end_min_pa, current_end_max_pa, n_neurons)
    ]
    currents = input_factory.stack_currents(currents, b2.ms)

    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = currents,
        simulation_time = simulation_time * b2.ms
    )

    return statemon, spikemon, currents

# COMPUTE CURVES

def compute_gain_function(
    neuron_group   : b2.NeuronGroup,
    t_start_ms     : float,
    t_end_ms       : float,
    current_min_pa : float,
    current_max_pa : float,
    simulation_time: float,
):
    ''' Compute the gain function of the neuronal model '''

    # Simulate the response to a step current
    statemon, spikemon, currents = simulate_response_to_step_currents(
        neuron_group    = neuron_group,
        t_start_ms      = t_start_ms,
        t_end_ms        = t_end_ms,
        current_min_pa  = current_min_pa,
        current_max_pa  = current_max_pa,
        simulation_time = simulation_time
    )

    # Get inter-spike intervals
    inter_spike_intervals = compute_inter_spike_intervals(
        spikemon = spikemon,
    )

    return statemon, spikemon, currents, inter_spike_intervals

def get_freq_over_stim(
        neuron_group   : b2.NeuronGroup,
        t_end_ms       : float,
        current_max_pa : float,
    ):
    ##############################################################################
    # The number of neurons in your group is the number of stim you want to test #
    ##############################################################################

    # Simulate the response to a step current
    statemon, spikemon, currents = simulate_response_to_step_currents(
        neuron_group    = neuron_group,
        t_start_ms      = 0,
        t_end_ms        = t_end_ms,
        current_min_pa  = 0,
        current_max_pa  = current_max_pa,
        simulation_time = t_end_ms,
    )

    ISIs = compute_inter_spike_intervals(spikemon)

    freqs = [1/np.mean(arr) for arr in ISIs]
    return freqs

def get_rheobase_current_new(
    neuron_group   : b2.NeuronGroup,
    current_max_pa : float,
    simulation_time: float,
    t_start_ms     : float = 0,
    t_end_ms       : float = 1000,
    current_min_pa : float = 0,
):
    ####################################################################################################
    # The number of neurons in your group is the number of stim gives you the precision of the measure #
    ####################################################################################################

    if neuron_group.N == 1:
        raise ValueError('Only one neuron in the group, cannot compute rheobase current')
    if neuron_group.N < 10:
         warnings.warn('Less than 10 neurons in the group, the precision of the measure might be low')


    # Simulate the N response to a step current
    statemon, spikemon, currents = simulate_response_to_step_currents(
        neuron_group    = neuron_group,
        t_start_ms      = t_start_ms,
        t_end_ms        = t_end_ms,
        current_min_pa  = current_min_pa,
        current_max_pa  = current_max_pa,
        simulation_time = simulation_time,
    )

        # Get spike count for each idx
    spike_count = [len(arr) for arr in spikemon.spike_trains().values()]
        # Get last neuron index with 0 spike
    last_idx = len(spike_count) - 1 - spike_count[::-1].index(0)
        # Check if the next neuron has a spike
    if spike_count[last_idx + 1] != 0:
        # If yes, the rheobase current is the current of the next neuron after 0 spike
        rheobase_current = currents( simulation_time * b2.ms, last_idx)
        return rheobase_current
    else:
        # error no rheobase current found
        raise ValueError('No rheobase current found')
