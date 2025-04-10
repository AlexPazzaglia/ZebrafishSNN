'''
Network simulation including metrics to evaluate the network
Included network redefine method to initialize monitors
Included method to assign new parameters to the network
'''
import os
import dill
import json
import logging
import numpy as np
import pandas as pd
import brian2 as b2

from typing import Any
from queue import Queue

import network_modules.performance.signal_processor_snn as SigProc
import network_modules.performance.signal_processor_mech as MechProc

from network_modules.simulation.network_simulation import SnnSimulation

SNN_METRICS = [

    # Periodicity
    'neur_ptcc_ax',
    'neur_ptcc_lb',

    # Frequency
    'neur_freq_ax',
    'neur_freq_lb',
    'neur_freq_diff',

    # Phase lags
    'neur_ipl_ax_c',
    'neur_ipl_ax_a',
    'neur_ipl_ax_t',
    'neur_twl',
    'neur_wave_number_a',
    'neur_wave_number_t',
    'neur_ipl_lb_h',
    'neur_ipl_lb_c',
    'neur_ipl_lb_d',

    # Motor effort
    'neur_eff_mn_all',
    'neur_eff_mn_ax',
    'neur_eff_mn_lb',
    'neur_eff_mc_all',
    'neur_eff_mc_ax',
    'neur_eff_mc_lb',

    # Muscle cells duty cycle
    'neur_duty_cycle_r',
    'neur_duty_cycle_l',

    # Oscillation means
    'neur_mean_ex_ax',
    'neur_mean_ex_lb',
    'neur_mean_in_ax',
    'neur_mean_in_lb',
    'neur_mean_mn_ax',
    'neur_mean_mn_lb',
    'neur_mean_mc_ax',
    'neur_mean_mc_lb',
    'neur_mean_mo_ax',
    'neur_mean_mo_lb',

    # Oscillation amplitudes
    'neur_amp_ex_ax',
    'neur_amp_ex_lb',
    'neur_amp_in_ax',
    'neur_amp_in_lb',
    'neur_amp_mn_ax',
    'neur_amp_mn_lb',
    'neur_amp_mc_ax',
    'neur_amp_mc_lb',
    'neur_amp_mo_ax',
    'neur_amp_mo_lb',
]

class SnnPerformance(SnnSimulation):
    '''
    Class used to evaluate the performance of a simulation
    '''

    ## Initialization
    def __init__(
        self,
        network_name: str,
        params_name : str,
        results_path: str,
        q_in        : Queue = None,
        q_out       : Queue = None,
        new_pars    : dict  = None,
        **kwargs,
    ) -> None:
        ''' Parameter initialization, network setup, defines metrics '''
        super().__init__(
            network_name = network_name,
            params_name  = params_name,
            results_path = results_path,
            q_in         = q_in,
            q_out        = q_out,
            new_pars     = new_pars,
            **kwargs,
        )
        self._define_metrics()

    ## RUN SIMULATIONS
    def simulation_run(self, param_scalings: np.ndarray= None) -> None:
        '''
        Run the simulation either with nominal or scaled parameter values.
        Save the simulation data for the following post-processing.
        '''
        self._define_metrics()
        super().simulation_run(param_scalings)
        self.save_simulation_data()

    ## INITIALIZE METRICS
    def _define_online_metrics(self) -> None:
        ''' Initializes the values of the metrics computed online '''

        pars_sim = self.params.simulation

        if not pars_sim.include_online_act:
            return

        pools_lb = 2 * self.params.topology.segments_limbs
        window_n = round( float( pars_sim.online_act_window // pars_sim.timestep ) )

        self.online_onsets_lb  = [ [] for _ in range(pools_lb) ]
        self.online_offsets_lb = [ [] for _ in range(pools_lb) ]

        self.online_count_lb     = np.zeros((pools_lb, window_n))

        self.all_online_activities_lb = np.zeros((pools_lb, pars_sim.callback_steps))
        self.all_online_periods_lb    = np.zeros((pools_lb, pars_sim.callback_steps))
        self.all_online_duties_lb     = np.zeros((pools_lb, pars_sim.callback_steps))

        self.online_activities_lb = np.zeros(pools_lb)
        self.online_periods_lb    = np.zeros(pools_lb)
        self.online_duties_lb     = np.zeros(pools_lb)

        self.online_crossings = [
            {
                'on1'  : False,
                'on2'  : False,
                'off1' : False,
                'off2' : False,
                'onset_candidate' : 0,
                'offset_candidate' : 0,
            }
            for _ in range(pools_lb)
        ]

    def _define_metrics(self) -> None:
        ''' Initialize the value of the metrics '''

        if hasattr(self, 'initialized_metrics') and self.initialized_metrics:
            return

        # SIGNAL PROCESSING
        self.spikemon_dict    : dict[str, np.ndarray] = {}
        self.statemon_dict    : dict[str, np.ndarray] = {}
        self.musclemon_dict   : dict[str, np.ndarray] = {}

        # Filtered spike counts
        self.smooth_activations : dict[str, dict[str, Any]] = {}

        self.spike_count_fft  = np.array([], dtype= float)   # FFT of the filtered spike count
        self.com_x            : list[list[float]]     = []
        self.com_y            : list[list[float]]     = []   # COM of oscillations
        self.start_indices    : list[list[int]]       = []
        self.stop_indices     : list[list[int]]       = []   # Onsets and offsets

        # DEFINE METRICS
        self._define_online_metrics()
        self.ptccs           = np.array([], dtype= float)  # Peak to through correlation coefficients
        self.mean_freqs           = np.array([], dtype= float)  # Mean frequenct of every segment
        self.ipls_evolutions_dict = {}                          # Evolution of intersegmental phase lags amd their mean
        self.ipls_parts      = {}                          # Phase lag between different parts of the network
        self.hilb_freqs      = np.array([], dtype= float)  # Instantaneous frequency of every segment
        self.hilb_phases     = np.array([], dtype= float)  # Instantaneous phase of every segment
        self.freq_evolution_dict  = np.array([], dtype= float)  # Evolution of the mean and std of the instantaneous frequency
        self.effs_mn         = np.array([], dtype= float)  # Efforts for the motor neurons
        self.effs_mc         = np.array([], dtype= float)  # Efforts for the muscle cells

        self.metrics : dict[str, float ]= {}

        self.initialized_metrics = True

    ## POST-PROCESSING
    def simulation_post_processing(self, load_from_file = False) -> dict[str, float]:
        ''' Computes the performance metrics of the simulation '''
        self.load_simulation_data(load_from_file= load_from_file)
        return self.network_performance_metrics()

    # MONITOR DATA
    def _monitor_data_to_dict(self, monitor_name) -> dict[str, np.ndarray]:
        ''' Converts monitor data to a dictionary of array '''
        return getattr(self, monitor_name).get_states(units= False)

    def _convert_monitor_data(self, monitor_pars, monitor_name) -> dict[str, np.ndarray]:
        ''' Converts monitor data as a dictionary of array '''
        if not monitor_pars['active']:
            return {}
        return self._monitor_data_to_dict(monitor_name)

    def _save_monitor_data(self, monitor_pars, monitor_name) -> None:
        ''' Saves monitor data as a dictionary of array '''

        if not monitor_pars['active'] or not monitor_pars['save']:
            return

        monitor_dict = self._monitor_data_to_dict(monitor_name)
        data_path    = self.params.simulation.results_data_folder_run
        file_name    = f'{data_path}/{monitor_name}'
        os.makedirs(data_path, exist_ok=True)

        # Single DILL FILE
        dill_file = f'{file_name}.dill'
        logging.info('Saving self.%s data in %s', monitor_name, dill_file)

        with open(dill_file, 'wb') as outfile:
            dill.dump( monitor_dict, outfile)

        ## Separate CSV FILE
        if not monitor_pars.get('to_csv'):
            return

        for key, value in monitor_dict.items():

            if not isinstance(value, np.ndarray) or len(value.shape) > 2:
                continue
            if len(value.shape) == 1:
                df = pd.Series(value)
            if len(value.shape) == 2:
                df = pd.DataFrame(value)

            csv_file = f'{file_name}_{key}.csv'
            logging.info('Saving self.%s data in %s', monitor_name, csv_file)

            df.to_csv(f'{file_name}_{key}.csv', index= False)

    def _load_monitor_data(self, monitor_pars, monitor_name) -> dict[str, np.ndarray]:
        ''' Loads monitor data as a dictionary of array '''
        data_path = self.params.simulation.results_data_folder_run
        data_file = f'{data_path}/' + '{mon_name}.dill'

        if monitor_pars['active'] and monitor_pars['save']:
            filename = data_file.format(mon_name= monitor_name)
            logging.info('Loading self.%s data from %s', monitor_name, filename)
            with open(filename, 'rb') as infile:
                return dill.load(infile)
        return {}

    # SIMULATION DATA SAVING
    def save_simulation_data(self) -> None:
        ''' Saves the data from the last simulation '''
        self._save_monitor_data(self.params.monitor.spikes, 'spikemon')
        self._save_monitor_data(self.params.monitor.states, 'statemon')
        self._save_monitor_data(self.params.monitor.muscle_cells, 'musclemon')
        return

    # SIMULATION DATA RETRIEVAL
    def load_simulation_data(self, load_from_file = False) -> None:
        ''' Loads the data from the last simulation '''
        loader = self._load_monitor_data if load_from_file else self._convert_monitor_data
        self.spikemon_dict  = loader(self.params.monitor.spikes, 'spikemon')
        self.statemon_dict  = loader(self.params.monitor.states, 'statemon')
        self.musclemon_dict = loader(self.params.monitor.muscle_cells, 'musclemon')
        return

    # METRICS COMPUTATION
    def _get_smooth_activations(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate smooth oscillations of the CPG segments if not already computed'''

        # self.times_ex_f, self.spike_count_ex_f
        # self.times_in_f, self.spike_count_in_f
        # self.times_mn_f, self.spike_count_mn_f

        monitor     = self.params.monitor.pools_activation
        net_modules = self.params.topology.network_modules

        if recompute:
            self.smooth_activations = {}

        for target_pars in monitor['target_modules']:
            mod_name      = target_pars['mod_name']
            ner_name      = target_pars['ner_name']
            smooth_factor = target_pars['smooth_factor']
            net_module    = net_modules.get_sub_module_from_full_name(mod_name)

            if not net_module.include:
                continue

            if not recompute and self.smooth_activations.get(ner_name) is not None:
                continue

            (
                times_f,
                spike_count_f
            ) = SigProc.compute_smooth_activation_module(
                spikemon_t    = self.spikemon_dict['t'],
                spikemon_i    = self.spikemon_dict['i'],
                dt_sig        = b2.defaultclock.dt,
                duration      = self.params.simulation.duration,
                net_module    = net_module,
                ner_type      = ner_name,
                smooth_factor = smooth_factor,
            )

            self.smooth_activations[ner_name] = {
                'times'        : times_f,
                'time_step'    : times_f[1] - times_f[0],
                'spike_count'  : spike_count_f,
                'mod_name'     : mod_name,
                'ner_name'     : ner_name,
                'smooth_factor': smooth_factor,
            }

        return

    def _get_oscillations_com(self, recompute: bool = False) -> tuple[list]:
        ''' Calculate COM of oscillations if not already computed'''

        if recompute or [] in [ self.com_x, self.com_y, self.start_indices, self.stop_indices ]:
            (
                self.com_x,
                self.com_y,
                self.start_indices,
                self.stop_indices
            ) = SigProc.process_network_smooth_signals_com(
                times         = self.smooth_activations['ex']['times'],
                signals       = self.smooth_activations['ex']['spike_count'],
                signals_ptccs = self.ptccs,
                discard_time  = 1 * b2.second
            )

        return self.com_x, self.com_y, self.start_indices, self.stop_indices

    def _get_hilbert_transform(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate phases and frequencies of oscillations if not already computed'''

        if not self.params.monitor.hilbert['active']:
            return self.hilb_phases, self.hilb_freqs

        if not np.any(self.mean_freqs):
            raise ValueError('Frequency values must be computed before Hilbert Transform')

        mean_freq = np.nanmean(self.mean_freqs)
        ner_name  = self.params.monitor.hilbert['ner_name']
        times     = self.smooth_activations[ner_name]['times']
        signals   = self.smooth_activations[ner_name]['spike_count']

        if recompute or not np.any(self.hilb_freqs) or not np.any(self.hilb_phases) :
            self.hilb_phases, self.hilb_freqs = SigProc.compute_hilb_transform(
                times        = times,
                signals      = signals,
                filtering    = self.params.simulation.metrics_filtering,
                mean_freq    = mean_freq,
            )

        return self.hilb_phases, self.hilb_freqs

    def _get_hilbert_metrics(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate phases and frequencies of oscillations if not already computed'''

        hilb_monitor = self.params.monitor.hilbert

        if not hilb_monitor['active']:
            return self.freq_evolution_dict, self.ipls_evolutions_dict

        if not np.any(self.mean_freqs):
            raise ValueError('Frequency values must be computed before Hilbert Transform')

        mean_freq = np.nanmean(self.mean_freqs)
        top_pars  = self.params.topology

        # FREQUENCY EVOLUTIONS
        hilb_freq_active  = self.params.monitor.hilbert_freq['active']
        hilb_freq_compute = recompute or not np.any(self.freq_evolution_dict)

        if hilb_freq_active and hilb_freq_compute:
            self.freq_evolution_dict = SigProc.compute_frequency_hilb(
                freqs     = self.hilb_freqs,
                mean_freq = mean_freq,
            )
            self.freq_evolution_dict['ner_name'] = hilb_monitor['ner_name']

        # IPLS EVOLUTIONS
        hilb_ipl_active  = self.params.monitor.hilbert_ipl['active']
        hilb_ipl_compute = recompute or not np.any(self.ipls_evolutions_dict)

        if hilb_ipl_active and hilb_ipl_compute:
            self.ipls_evolutions_dict = SigProc.compute_ipls_hilb(
                phases                  = self.hilb_phases,
                segments                = top_pars.segments_axial,
                limbs_pairs_i_positions = top_pars.limbs_pairs_i_positions,
                jump_at_girdles         = self.params.simulation.gait not in ['swim'],
                trunk_only              = self.params.simulation.metrics_trunk_only
            )
            self.ipls_evolutions_dict['ner_name'] = hilb_monitor['ner_name']

        return self.freq_evolution_dict, self.ipls_evolutions_dict

    def _get_fourier_transform(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate phases and frequencies of oscillations if not already computed'''

        if recompute or not np.any(self.spike_count_fft):
            self.spike_count_fft = SigProc.compute_fft_transform(
                times   = self.smooth_activations['ex']['times'],
                signals = self.smooth_activations['ex']['spike_count'],
            )

        return self.spike_count_fft

    def _get_network_ptcc(self, recompute: bool = False) -> np.ndarray:
        ''' Computes the PTCC of oscillations if not already computed '''

        if recompute or not np.any(self.ptccs):
            self.ptccs = SigProc.compute_ptcc(
                times   = self.smooth_activations['ex']['times'],
                signals = self.smooth_activations['ex']['spike_count'],
            )

        n_osc_ax = 2 * self.params.topology.segments_axial
        incl_ax  = self.params.topology.include_cpg_axial
        incl_lb  = self.params.topology.include_cpg_limbs
        ptcc_ax  = float( np.mean(self.ptccs[:n_osc_ax]) ) if incl_ax else np.nan
        ptcc_lb  = float( np.mean(self.ptccs[n_osc_ax:]) ) if incl_lb else np.nan

        return ptcc_ax, ptcc_lb

    def _get_network_freq(self, recompute: bool = False) -> float:
        ''' Computes the frequencies of oscillations if not already computed '''

        n_osc_ax = 2 * self.params.topology.segments_axial
        incl_ax  = self.params.topology.include_cpg_axial
        incl_lb  = self.params.topology.include_cpg_limbs
        times    = self.smooth_activations['ex']['times']

        # FFT (Mean Frequency Values)
        if recompute or not np.any(self.mean_freqs):
            self.mean_freqs, _ = SigProc.compute_frequency_fft(
                times       = times,
                signals_fft = self.spike_count_fft
            )

        # Mean values
        freq_ax   = float( np.mean(self.mean_freqs[:n_osc_ax]) ) if incl_ax else np.nan
        freq_lb   = float( np.mean(self.mean_freqs[n_osc_ax:]) ) if incl_lb else np.nan
        freq_diff = np.abs(freq_ax - freq_lb)

        return freq_ax, freq_lb, freq_diff

    def _get_network_ipls(self, recompute: bool = False) -> float:
        '''
        Computes the IPLS of oscillations if not already computed
        For swiming, consider the entire network.
        For walking, consider the phase jump at girdles.
        '''
        top_pars = self.params.topology
        signals  = self.smooth_activations['ex']['spike_count']
        times    = self.smooth_activations['ex']['times']

        incl_ax  = top_pars.include_cpg_axial
        incl_lb  = top_pars.include_cpg_limbs

        segments_axial = top_pars.segments_axial
        limb_pairs     = top_pars.limb_pairs
        limb_pair_pos  = top_pars.limbs_pairs_i_positions
        osc_ax         = 2* top_pars.segments_axial
        osc_lb         = 2* top_pars.segments_per_limb
        osc_lb_pair    = 4* top_pars.segments_per_limb

        # Axial indices
        commissural_ax_lr = lambda inds : [ [2*s,     2*( s )+1] for s in inds ]
        consecutive_ax_l  = lambda inds : [ [2*s,     2*(s+1)  ] for s in inds ]
        consecutive_ax_r  = lambda inds : [ [2*s + 1, 2*(s+1)+1] for s in inds ]
        make_arr_ax       = lambda fun, inds : np.array( fun(inds) if incl_ax else [] )

        # Commisural (left/right)
        inds_ax_com = np.arange(segments_axial)
        axial_inds_c0 = make_arr_ax(commissural_ax_lr, inds_ax_com)

        # All Axial (left/right)
        inds_ax_all   = np.arange(segments_axial - 1)
        axial_inds_a0 = make_arr_ax(consecutive_ax_l, inds_ax_all)
        axial_inds_a1 = make_arr_ax(consecutive_ax_r, inds_ax_all)

        # Trunk Axial (left/right)
        trunk_ind_0 = limb_pair_pos[0] if limb_pair_pos else 0
        trunk_ind_1 = limb_pair_pos[1] if limb_pair_pos else segments_axial + 1

        inds_ax_trunk = np.arange(trunk_ind_0, trunk_ind_1 - 2)
        axial_inds_t0 = make_arr_ax(consecutive_ax_l, inds_ax_trunk)
        axial_inds_t1 = make_arr_ax(consecutive_ax_r, inds_ax_trunk)

        # Limbs indices
        lb_inds     = np.arange(limb_pairs - 1)
        flex_l      = lambda lb_pair : osc_ax + osc_lb_pair * lb_pair
        flex_r      = lambda lb_pair : osc_ax + osc_lb_pair * lb_pair + osc_lb
        make_arr_lb = lambda fun : np.array( [ fun(lb) for lb in lb_inds ] if incl_lb else [] )

        # Homolateral
        homo_ll = lambda pair : [flex_l(pair), flex_l(pair+1)]
        homo_rr = lambda pair : [flex_r(pair), flex_r(pair+1)]

        limbs_inds_h0, limbs_inds_h1 = make_arr_lb(homo_ll), make_arr_lb(homo_rr)

        # Contralateral
        contra_lr = lambda pair : [flex_l(pair), flex_r(pair)]
        contra_rl = lambda pair : [flex_r(pair), flex_l(pair)]

        limbs_inds_c0, limbs_inds_c1 = make_arr_lb(contra_lr), make_arr_lb(contra_rl)

        # Diagonal
        diag_lr = lambda pair : [flex_l(pair), flex_r(pair+1)]
        diag_rl = lambda pair : [flex_r(pair), flex_l(pair+1)]

        limbs_inds_d0, limbs_inds_d1 = make_arr_lb(diag_lr), make_arr_lb(diag_rl)

        # CROSS-CORRELATION (Mean IPLs values)
        if recompute or self.ipls_parts == {}:
            get_ipls = lambda inds : SigProc.compute_ipls_corr(times, signals, self.mean_freqs, inds)

            self.ipls_parts['ipls_ax_c0'] = get_ipls(axial_inds_c0)
            self.ipls_parts['ipls_ax_a0'] = get_ipls(axial_inds_a0)
            self.ipls_parts['ipls_ax_a1'] = get_ipls(axial_inds_a1)
            self.ipls_parts['ipls_ax_t0'] = get_ipls(axial_inds_t0)
            self.ipls_parts['ipls_ax_t1'] = get_ipls(axial_inds_t1)
            self.ipls_parts['ipls_lb_h0'] = get_ipls(limbs_inds_h0)
            self.ipls_parts['ipls_lb_h1'] = get_ipls(limbs_inds_h1)
            self.ipls_parts['ipls_lb_c0'] = get_ipls(limbs_inds_c0)
            self.ipls_parts['ipls_lb_c1'] = get_ipls(limbs_inds_c1)
            self.ipls_parts['ipls_lb_d0'] = get_ipls(limbs_inds_d0)
            self.ipls_parts['ipls_lb_d1'] = get_ipls(limbs_inds_d1)

        # COMPUTE MEAN IPLS
        ipls = self.ipls_parts

        ipl_ax_c = np.mean( [ ipls['ipls_ax_c0'] ] )
        ipl_ax_a = np.mean( [ ipls['ipls_ax_a0'], ipls['ipls_ax_a1'] ] )
        ipl_ax_t = np.mean( [ ipls['ipls_ax_t0'], ipls['ipls_ax_t1'] ] )
        ipl_lb_h = np.mean( [ ipls['ipls_lb_h0'], ipls['ipls_lb_h1'] ] )
        ipl_lb_c = np.mean( [ ipls['ipls_lb_c0'], ipls['ipls_lb_c1'] ] )
        ipl_lb_d = np.mean( [ ipls['ipls_lb_d0'], ipls['ipls_lb_d1'] ] )

        # # PLOT OF IPLS
        # import matplotlib.pyplot as plt
        # ipls_ax_bilat = np.array( [ self.ipls_parts['ipls_ax_a0'], self.ipls_parts['ipls_ax_a1'] ] )
        # ipls_ax_mean  = np.mean(ipls_ax_bilat, axis= 0)
        # ipls_ax_cum   = np.cumsum(ipls_ax_mean) - np.cumsum(ipls_ax_mean)[7]
        # plt.plot( ipls_ax_cum / self.freqs[0] )
        # plt.grid()

        # COMPUTE WAVE NUMBER
        total_wave_lag = ipl_ax_a * (segments_axial - 1)

        ax_length = self.params.mechanics.mech_axial_length
        inds_cpg  = top_pars.network_modules.cpg.axial.indices
        y_ax_0    = np.amax( top_pars.neurons_y_mech[0][inds_cpg] )
        y_ax_1    = np.amin( top_pars.neurons_y_mech[0][inds_cpg] )
        y_ax_fr   = (y_ax_0 - y_ax_1) * (1 - 1/segments_axial) / ax_length

        wave_number_a = ipl_ax_a * (segments_axial - 1) / y_ax_fr

        segments_trunk = trunk_ind_1 - trunk_ind_0 - 1
        lb_pos = top_pars.limbs_pairs_y_positions
        y_tr_0 = float( top_pars.limbs_pairs_y_positions[0] ) if lb_pos else 0
        y_tr_1 = float( top_pars.limbs_pairs_y_positions[1] ) if lb_pos else ax_length
        y_tr_fr = (y_tr_1 - y_tr_0) * (1 - 1/segments_trunk) / ax_length

        wave_number_t = ipl_ax_t * (segments_trunk - 1) / y_tr_fr

        return (
            ipl_ax_c,
            ipl_ax_a,
            ipl_ax_t,
            total_wave_lag,
            wave_number_a,
            wave_number_t,
            ipl_lb_h,
            ipl_lb_c,
            ipl_lb_d,
        )

    def _get_motor_neurons_effort(self, recompute: bool = False) -> float:
        '''
        Compute effort of the motor neurons activations
        based on the integral of their activations
        '''

        if self.smooth_activations.get('mn') is None:
            return np.nan, np.nan, np.nan

        if recompute or not np.any(self.effs_mn):
            self.effs_mn = SigProc.compute_effort(
                times   = self.smooth_activations['mn']['times'],
                signals = self.smooth_activations['mn']['spike_count'],
            )

        top = self.params.topology
        eff_mn_all = np.mean(self.effs_mn)
        eff_mn_ax  = np.mean(self.effs_mn[:2*top.segments_axial]) if top.include_motor_neurons_axial else np.nan
        eff_mn_lb  = np.mean(self.effs_mn[2*top.segments_axial:]) if top.include_motor_neurons_limbs else np.nan

        return eff_mn_all, eff_mn_ax, eff_mn_lb

    def _get_muscle_cells_effort(self, recompute: bool = False) -> float:
        '''
        Compute effort of the muscle cells activations
        based on the integral of their activations
        '''

        if not self.params.monitor.muscle_cells['active']:
            return np.nan, np.nan, np.nan

        if recompute or not np.any(self.effs_mc):
            self.effs_mc = SigProc.compute_effort(
                times   = self.musclemon_dict['t'],
                signals = self.musclemon_dict['v'],
            )

        top = self.params.topology
        eff_mc_all = np.mean(self.effs_mc)
        eff_mc_ax  = np.mean(self.effs_mc[:2*top.segments_axial]) if top.include_muscle_cells_axial else np.nan
        eff_mc_lb  = np.mean(self.effs_mc[2*top.segments_axial:]) if top.include_muscle_cells_limbs else np.nan

        return eff_mc_all, eff_mc_ax, eff_mc_lb

    def _get_duty_cycle(self, recompute: bool = False) -> float:
        '''
        Compute duty cycle of the muscle cells activations
        based on their activation period and total period
        '''

        if not self.params.monitor.muscle_cells['active']:
            return np.nan, np.nan

        duty_cycles_l, duty_cycles_r = SigProc.compute_duty_cycle(
            times      = self.musclemon_dict['t'],
            signals    = self.musclemon_dict['v'],
            threshold  = 0.25,
        )

        duty_cycle_l = np.mean(duty_cycles_l)
        duty_cycle_r = np.mean(duty_cycles_r)

        return duty_cycle_l, duty_cycle_r

    def _get_oscillations_means(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate amplitudes of oscillations if not already computed'''

        pars_top  = self.params.topology
        n_seg_ax  = pars_top.segments_axial
        n_seg_lb  = pars_top.segments_limbs
        mc_active = self.params.monitor.muscle_cells['active']
        mc_mon    = self.musclemon_dict

        net_activations = self.smooth_activations

        # AUXILIARY FUNCTIONS
        def _means_mod( mod_signal, chain_multiplier = 2):
            ''' Average of the segmental activity of a signal'''
            n_ax, n_lb  = np.array([n_seg_ax, n_seg_lb]) * chain_multiplier
            means_mod   = np.mean(mod_signal, axis=0)
            mean_mod_ax = np.mean(means_mod[0:n_ax]) if n_ax else np.nan
            mean_mod_lb = np.mean(means_mod[n_ax:])  if n_lb else np.nan
            return mean_mod_ax, mean_mod_lb

        def _get_pop_mean(name):
            ''' Get the mean of the population '''
            if self.smooth_activations.get(name) is None:
                return np.nan, np.nan
            return _means_mod(net_activations[name]['spike_count'].T)

        # EX, IN, MN neurons
        mean_ex_ax, mean_ex_lb = _get_pop_mean('ex')
        mean_in_ax, mean_in_lb = _get_pop_mean('in')
        mean_mn_ax, mean_mn_lb = _get_pop_mean('mn')

        # Muscle cells and Motor output
        if mc_active:
            mc_vals = mc_mon['v']
            mc_diff = mc_mon['v'][0::2] - mc_mon['v'][1::2]
            mean_mc_ax, mean_mc_lb = _means_mod( mc_vals )
            mean_mo_ax, mean_mo_lb = _means_mod( mc_diff, chain_multiplier = 1)
        else:
            mean_mc_ax, mean_mc_lb = np.nan, np.nan
            mean_mo_ax, mean_mo_lb = np.nan, np.nan

        return (
            mean_ex_ax, mean_ex_lb,
            mean_in_ax, mean_in_lb,
            mean_mn_ax, mean_mn_lb,
            mean_mc_ax, mean_mc_lb,
            mean_mo_ax, mean_mo_lb,
        )

    def _get_oscillations_amplitudes(self, recompute: bool = False) -> tuple[np.ndarray]:
        ''' Calculate amplitudes of oscillations if not already computed'''

        pars_top  = self.params.topology
        n_seg_ax  = pars_top.segments_axial
        n_seg_lb  = pars_top.segments_limbs
        mc_active = self.params.monitor.muscle_cells['active']
        mc_mon    = self.musclemon_dict

        # NOTE: FFT can be unstable with multiple frequencies
        # amp_func  = SigProc.compute_amplitudes_fft
        amp_func = SigProc.compute_amplitudes_peaks

        net_activations = self.smooth_activations

        # AUXILIARY FUNCTION
        def _amp_mod( mod_signal, chain_multiplier = 2):
            ''' Average of the segmental activity of a signal'''
            n_ax, n_lb = np.array([n_seg_ax, n_seg_lb]) * chain_multiplier
            means_mod  = amp_func(mod_signal)
            amp_mod_ax = np.mean(means_mod[0:n_ax]) if n_ax else np.nan
            amp_mod_lb = np.mean(means_mod[n_ax:])  if n_lb else np.nan
            return amp_mod_ax, amp_mod_lb

        def _get_pop_amp(name):
            ''' Get the mean of the population '''
            if self.smooth_activations.get(name) is None:
                return np.nan, np.nan
            return _amp_mod(net_activations[name]['spike_count'].T)

        # EX, IN, MN neurons
        amp_ex_ax, amp_ex_lb = _get_pop_amp('ex')
        amp_in_ax, amp_in_lb = _get_pop_amp('in')
        amp_mn_ax, amp_mn_lb = _get_pop_amp('mn')

        # Muscle cells and Motor output
        if mc_active:
            mc_vals = mc_mon['v']
            mc_diff = mc_mon['v'][0::2] - mc_mon['v'][1::2]
            amp_mc_ax, amp_mc_lb = _amp_mod( mc_vals )
            amp_mo_ax, amp_mo_lb = _amp_mod( mc_diff, chain_multiplier = 1)
        else:
            amp_mc_ax, amp_mc_lb = np.nan, np.nan
            amp_mo_ax, amp_mo_lb = np.nan, np.nan

        return (
            amp_ex_ax, amp_ex_lb,
            amp_in_ax, amp_in_lb,
            amp_mn_ax, amp_mn_lb,
            amp_mc_ax, amp_mc_lb,
            amp_mo_ax, amp_mo_lb,
        )

    def network_performance_metrics(self, recompute: bool = False) -> tuple[float]:
        ''' Compute metrics to evaluate network's performance '''

        recompute = recompute or self.initialized_metrics
        metrics   = self.metrics

        self._get_smooth_activations(recompute)
        self._get_fourier_transform(recompute)

        (
            metrics['neur_ptcc_ax'],
            metrics['neur_ptcc_lb'],
        ) = self._get_network_ptcc(recompute)

        (
            metrics['neur_freq_ax'],
            metrics['neur_freq_lb'],
            metrics['neur_freq_diff'],
        ) = self._get_network_freq(recompute)


        self._get_hilbert_transform(recompute)
        self._get_hilbert_metrics(recompute)

        (
            metrics['neur_ipl_ax_c'],
            metrics['neur_ipl_ax_a'],
            metrics['neur_ipl_ax_t'],
            metrics['neur_twl'],
            metrics['neur_wave_number_a'],
            metrics['neur_wave_number_t'],
            metrics['neur_ipl_lb_h'],
            metrics['neur_ipl_lb_c'],
            metrics['neur_ipl_lb_d'],
        ) = self._get_network_ipls(recompute)
        (
            metrics['neur_eff_mn_all'],
            metrics['neur_eff_mn_ax'],
            metrics['neur_eff_mn_lb'],
        ) = self._get_motor_neurons_effort(recompute)
        (
            metrics['neur_eff_mc_all'],
            metrics['neur_eff_mc_ax'],
            metrics['neur_eff_mc_lb'],
        ) = self._get_muscle_cells_effort(recompute)
        (
            metrics['neur_duty_cycle_l'],
            metrics['neur_duty_cycle_r'],
        ) = self._get_duty_cycle(recompute)
        (
            metrics['neur_mean_ex_ax'],  metrics['neur_mean_ex_lb'],
            metrics['neur_mean_in_ax'],  metrics['neur_mean_in_lb'],
            metrics['neur_mean_mn_ax'],  metrics['neur_mean_mn_lb'],
            metrics['neur_mean_mc_ax'],  metrics['neur_mean_mc_lb'],
            metrics['neur_mean_mo_ax'],  metrics['neur_mean_mo_lb'],
        ) = self._get_oscillations_means(recompute)
        (
            metrics['neur_amp_ex_ax'],  metrics['neur_amp_ex_lb'],
            metrics['neur_amp_in_ax'],  metrics['neur_amp_in_lb'],
            metrics['neur_amp_mn_ax'],  metrics['neur_amp_mn_lb'],
            metrics['neur_amp_mc_ax'],  metrics['neur_amp_mc_lb'],
            metrics['neur_amp_mo_ax'],  metrics['neur_amp_mo_lb'],
        ) = self._get_oscillations_amplitudes(recompute)

        assert all( [key in SNN_METRICS  for key in metrics.keys()] ), 'Not all metrics are listed'
        assert all( [key in metrics for key in SNN_METRICS] ),         'Not all metrics are computed'

        self.initialized_metrics = False

        logging.info(f'NEURAL METRICS: {json.dumps(metrics, indent=4)}')
        return self.metrics

# TEST
def main():
    ''' Test case '''

    from queue import Queue

    logging.info('TEST: SNN Performance ')

    performance = SnnPerformance(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    performance.define_network_topology()
    performance.simulation_run()
    performance.simulation_post_processing()

    return performance

if __name__ == '__main__':
    main()
