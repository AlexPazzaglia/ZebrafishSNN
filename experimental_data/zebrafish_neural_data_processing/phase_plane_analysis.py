import numpy as np
import matplotlib.pyplot as plt

import brian2 as b2

def get_v_nullcline_crossings(
    neuron_group: b2.NeuronGroup,
    statemon    : b2.StateMonitor,
    i_stim      : b2.TimedArray,
    start_time  = 2 * b2.second,
    isIzhikevich  : bool = False,
):
    """
    Returns the index of the first crossing of the v-nullcline.
    """

    if isIzhikevich:
        # Parameter values
        start_ind = int(start_time / neuron_group.dt)

        V_rest   = neuron_group.V_rest
        k_gain   = neuron_group.k_gain
        V_crit   = neuron_group.V_crit
        I_ext    = neuron_group.I_ext[:]
        # Get nullcline
        def __v_nullcline(v):
            ''' Returns the V-nullcline for the neuron model'''
            return (k_gain * (v - V_rest) * (v - V_crit)+ I_ext)

        # Get first crossing of the v-nullcline
        w1_vals_nullcline = __v_nullcline(statemon.v[:, start_ind:].T)
        w1_vals           = statemon.w1[:, start_ind:].T

        # Find crossings
        (
            v_nullcline_crossings_step,
            v_nullcline_crossings_neur,
        ) = np.where(
            np.diff( (w1_vals > w1_vals_nullcline) == 1, axis=0 )
        )

        v_nullcline_crossings_inds = [
            v_nullcline_crossings_step[v_nullcline_crossings_neur == neuron_index] + 1
            for neuron_index in range(neuron_group.N)
        ]

    else:
    
        # Parameter values
        start_ind = int(start_time / neuron_group.dt)

        V_rest   = neuron_group.V_rest
        delta_t  = neuron_group.delta_t
        exp_term = neuron_group.exp_term
        V_rh     = neuron_group.V_rh
        R_memb   = neuron_group.R_memb
        I_ext    = neuron_group.I_ext[:]

        # Get nullcline
        def __v_nullcline(v):
            ''' Returns the V-nullcline for the neuron model'''
            return ( (V_rest - v) + exp_term * delta_t * np.exp( (v - V_rh) / delta_t ) + R_memb * I_ext ) /  R_memb


        # Get first crossing of the v-nullcline
        w1_vals_nullcline = __v_nullcline(statemon.v[:, start_ind:].T)
        w1_vals           = statemon.w1[:, start_ind:].T

        # Find crossings
        (
            v_nullcline_crossings_step,
            v_nullcline_crossings_neur,
        ) = np.where(
            np.diff( (w1_vals > w1_vals_nullcline) == 1, axis=0 )
        )

        v_nullcline_crossings_inds = [
            v_nullcline_crossings_step[v_nullcline_crossings_neur == neuron_index] + 1
            for neuron_index in range(neuron_group.N)
        ]

    return v_nullcline_crossings_inds

def get_rheobase_currents(
    neuron_group: b2.NeuronGroup,
    use_v_thres : bool = False,
):
    ''' Returns the rheobase current for each neuron in the group'''
    V_rest   = neuron_group.V_rest
    exp_term = neuron_group.exp_term
    delta_t  = neuron_group.delta_t
    a_gain1  = neuron_group.a_gain1
    V_rh     = neuron_group.V_rh
    V_th     = neuron_group.V_thres
    R_memb   = neuron_group.R_memb

    thres = V_th if use_v_thres else V_rh

    return ( a_gain1 + 1 / R_memb ) * ( thres - V_rest ) - exp_term * delta_t / R_memb

def phase_plane_analysis(
    neuron_group: b2.NeuronGroup,
    statemon    : b2.StateMonitor,
    i_stim      : b2.TimedArray,
    plot        : bool = True,
    start_time  : float = 2 * b2.second,
    num_points  : int = 50,
    arrow_scale : int = 100,
    isIzhikevich  : bool = False,
):
    """
    Plots the phase plane distribution of derivatives as vectors.
    """
    if isIzhikevich:
        # Parameter values
        start_ind = int(start_time / neuron_group.dt)

        C_memb   = neuron_group.C_memb
        k_gain   = neuron_group.k_gain
        b_gain   = neuron_group.b_gain
        tau1     = neuron_group.tau1
        delta_w1  = neuron_group.delta_w1
        V_rest   = neuron_group.V_rest
        V_crit   = neuron_group.V_crit
        V_thres  = neuron_group.V_thres
        V_reset  = neuron_group.V_reset
        I_ext    = neuron_group.I_ext[:]
       

        # Nullclines and state equation
        def __state_equation(v, w):
            ''' Returns the state equation for the neuron model '''
            dv_dt = (k_gain * (v - V_rest) * (v - V_crit)- w + I_ext)/C_memb
            dw_dt = ( b_gain * (v - V_rest) - w ) / tau1
            return np.array(dv_dt), np.array(dw_dt)

        def __v_nullcline(v):
            ''' Returns the V-nullcline for the neuron model'''
            return (k_gain * (v - V_rest) * (v - V_crit)+ I_ext)

        def __w_nullcline(v):
            ''' Returns the w-nullcline for the neuron model'''
            return b_gain * (v - V_rest)
        
        # Get v range
        v_min  = V_rest - 15*b2.mV
        v_max  = V_crit   + 15*b2.mV
        v_arr = np.linspace(v_min, v_max, 1000)

        v_nullcline_w1 = __v_nullcline(v_arr)
        w_nullcline_w1 = __w_nullcline(v_arr)

        # Get w range
        v_min_all = min(V_crit) - 15*b2.mV
        v_max_all = max(V_thres)

        w_min_all = 0
        w_max_all = np.amax(statemon.w1[:, start_ind:])
 
    else:
        # Parameter values
        start_ind = int(start_time / neuron_group.dt)

        tau_memb = neuron_group.tau_memb
        V_rest   = neuron_group.V_rest
        delta_t  = neuron_group.delta_t
        exp_term = neuron_group.exp_term
        a_gain1  = neuron_group.a_gain1
        tau1     = neuron_group.tau1
        V_rh     = neuron_group.V_rh
        V_thres  = neuron_group.V_thres
        R_memb   = neuron_group.R_memb
        I_ext    = neuron_group.I_ext[:]

        # Nullclines and state equation
        def __state_equation(v, w):
            ''' Returns the state equation for the neuron model '''
            dv_dt = ( (V_rest - v) + exp_term * delta_t * np.exp( (v - V_rh) / delta_t ) + R_memb * I_ext - R_memb * w ) /  tau_memb
            dw_dt = ( a_gain1 * (v - V_rest) - w ) / tau1
            return np.array(dv_dt), np.array(dw_dt)

        def __v_nullcline(v):
            ''' Returns the V-nullcline for the neuron model'''
            return ( (V_rest - v) + exp_term * delta_t * np.exp( (v - V_rh) / delta_t ) + R_memb * I_ext ) /  R_memb

        def __w_nullcline(v):
            ''' Returns the w-nullcline for the neuron model'''
            return a_gain1 * (v - V_rest)

        # Get v range
        v_min  = V_rest - 15*b2.mV
        v_max  = V_rh   + 15*b2.mV
        v_arr = np.linspace(v_min, v_max, 1000)

        v_nullcline_w1 = __v_nullcline(v_arr)
        w_nullcline_w1 = __w_nullcline(v_arr)

        # Get w range
        v_min_all = min(V_rh) - 15*b2.mV
        v_max_all = max(V_thres)

        w_min_all = 0
        w_max_all = np.amax(statemon.w1[:, start_ind:])

    # Get phase plane vectors
    v_values = np.linspace(v_min_all, v_max_all, num_points)
    w_values = np.linspace(w_min_all, w_max_all, num_points)

    V_grid, W_grid = np.meshgrid(v_values, w_values)

    V_vect = np.zeros((num_points, num_points, neuron_group.N))
    W_vect = np.zeros((num_points, num_points, neuron_group.N))

    for i in range(num_points):
        for j in range(num_points):
            dv_dt, dw_dt = __state_equation(V_grid[i, j], W_grid[i, j])

            dv_length     = np.sqrt(dv_dt**2 + dw_dt**2)
            dv_lenght_ind = (dv_length != 0)

            V_vect[i, j, dv_lenght_ind] = dv_dt[dv_lenght_ind] / dv_length[dv_lenght_ind]
            W_vect[i, j, dv_lenght_ind] = dw_dt[dv_lenght_ind] / dv_length[dv_lenght_ind]

    # Get first crossing of the v-nullcline
    neurons_v_nullcline_crossings_inds = get_v_nullcline_crossings(
        neuron_group = neuron_group,
        statemon     = statemon,
        i_stim       = i_stim,
        start_time   = start_time,
        isIzhikevich   = isIzhikevich,
    )

    # PLOT
    if not plot:
        return

    for neuron_index in range(neuron_group.N):

        plt.figure(f'Phase plane - Neuron {neuron_index}')
        plt.plot(v_arr[:, neuron_index], v_nullcline_w1[:, neuron_index], 'b', label='V-nullcline')
        plt.plot(v_arr[:, neuron_index], w_nullcline_w1[:, neuron_index], 'g', label='W-nullcline')

        # Plot phase plane vectors
        plt.quiver(
            V_grid,
            W_grid,
            V_vect[:, :, neuron_index],
            W_vect[:, :, neuron_index],
            color          = 'grey',
            scale          = arrow_scale,
            width          = 0.001,
            headwidth      = 3,
            headlength     = 3,
            headaxislength = 3,
        )

        # Plot trajectory after first crossing
        first_crossing_ind = (
            neurons_v_nullcline_crossings_inds[neuron_index][0]
            if len(neurons_v_nullcline_crossings_inds[neuron_index]) > 0
            else 0
        )

        plt.plot(
            statemon.v[neuron_index][start_ind + first_crossing_ind:],
            statemon.w1[neuron_index][start_ind + first_crossing_ind:],
            'k'
        )

        # Decorate plot
        v_neur_min = np.amin(statemon.v[neuron_index][start_ind + first_crossing_ind:] )
        v_neur_max = V_thres[neuron_index]
        v_neur_0   = v_neur_min - 0.1 * (v_neur_max - v_neur_min)
        v_neur_1   = v_neur_max + 0.01 * (v_neur_max - v_neur_min)
        plt.xlim(v_neur_0, v_neur_1)


        w_neur_min = np.amin(statemon.w1[neuron_index][start_ind + first_crossing_ind:] )
        w_neur_max = np.amax(statemon.w1[neuron_index][start_ind + first_crossing_ind:] )
        w_neur_0   = w_neur_min - 0.1 * (w_neur_max - w_neur_min)
        w_neur_1   = w_neur_max + 0.1 * (w_neur_max - w_neur_min)
        plt.ylim(w_neur_0, w_neur_1)

        plt.vlines(
            V_crit[neuron_index],
            w_neur_0,
            w_neur_1,
            'r',
            label='V-threshold'
        )

        plt.xlabel('V (V)')
        plt.ylabel('w (pA)')
        plt.title(f'Phase plane - Neuron {neuron_index}')
        plt.legend()
        plt.grid()

    return
