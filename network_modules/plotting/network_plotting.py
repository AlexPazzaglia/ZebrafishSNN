'''
Network simulation including metrics to evaluate the network
Included network redefine method to initialize monitors
Included method to assign new parameters to the network
'''

import os
import time
import shutil
import logging
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from typing import Union
from queue import Queue
from scipy.signal import find_peaks
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import network_modules.plotting.plots_utils as plots_utils
import network_modules.plotting.plots_snn as snn_plotting
import network_modules.plotting.animations_snn as net_anim

from network_modules.performance.network_performance import SnnPerformance

# FIGURE PARAMETERS
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font',      size = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend',  fontsize = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure',   figsize = (10.0, 5.0)) # size of the figure
plt.rc('lines',  linewidth = 2.0         ) # linewidth of the figure

# PLOTTING
class SnnPlotting(SnnPerformance):
    '''
    Class used plot the results of neuronal simulations
    '''

    def __init__(
        self,
        network_name: str,
        params_name : str,
        results_path: str,
        control_type: str,
        q_in        : Queue = None,
        q_out       : Queue = None,
        new_pars    : dict  = None,
        **kwargs,
    ) -> None:
        ''' Parameter initialization, network setup '''
        super().__init__(
            network_name = network_name,
            params_name  = params_name,
            results_path = results_path,
            control_type = control_type,
            q_in         = q_in,
            q_out        = q_out,
            new_pars     = new_pars,
            **kwargs
        )

        # Change matplotlib logging level to avoid undesired messages
        plt.set_loglevel("warning")

        # Initialize figures
        self.figures : dict[str, Union[Figure, list[Figure, FuncAnimation]]] = {}

        return

    # ALL RESULTS
    def simulation_plots(self) -> None:
        ''' Plots showing the network's behavior '''

        # Initialize figures
        self.figures : dict[str, Union[Figure, list[Figure, FuncAnimation]]] = {}

        # inds     = self.params.topology.network_modules.cpg.axial.ex.indices_pools_sides

        # G_glyc   = self._get_synaptic_conductance_evolution('glyc', save=True)
        # I_neuron = self._get_total_current_evolution()

        # var_name = 'I_tot - I_adapt'
        # var_vals = I_neuron

        # for seg in range(5, 15):
        #     self._plot_variable_evolution(
        #         f'{var_name} - Segment {seg}',
        #         var_vals,
        #         inds[0, seg, :],
        #         inds[1, seg, :],
        #     )

        # Plot

        self._plot_network_states_evolution()
        self._plot_voltage_traces_evolution()
        self._plot_connectivity_matrix()
        self._plot_raster_plot()
        self._plot_simulated_emg_signals()
        self._plot_isi_distribution()
        self._plot_processed_pool_activation()
        self._plot_hilbert_freq_evolution()
        self._plot_hilbert_ipl_evolution()
        self._plot_musclecells_evolution()
        self._plot_online_limb_activations_evolution()
        self._plot_spike_count_cycle_frequencies_evolution()

        # Animations
        self._plot_raster_plot_animation()
        self._plot_network_states_animation()
        self._plot_processed_pool_animation()


    # AUXILIARY
    def _check_monitor_conditions(
            self,
            monitor: dict,
            mon_condlist: list[str],
            plot_condlist: list[str]
        ) -> bool:
        ''' Checks conditions from monitor '''

        condition = True
        for cond in mon_condlist:
            condition = condition and monitor[cond]
        for cond in plot_condlist:
            condition = condition and monitor['plotpars'][cond]

        return condition

    def _get_total_current_evolution(
        self,
        save: bool = False,
    ):
        ''' Plot the evolution of the total current '''

        R_memb  = getattr(self.pop, 'R_memb')
        I_adapt = getattr(self.statemon, 'w1')
        I_tot   = getattr(self.statemon, 'I_tot')
        I_tot   = (I_tot.T / R_memb).T

        I_neuron = I_tot - I_adapt

        if not save:
            return I_neuron

        # Save data
        import pandas as pd

        data_path = self.params.simulation.results_data_folder_run
        file_name = f'{data_path}/statemon_I_neuron.csv'

        logging.info(f'Saving self.statemon.I_neuron data in {file_name}')

        I_neuron_df = pd.DataFrame(I_neuron.T)
        I_neuron_df.to_csv(file_name, index= False)

        return I_neuron

    def _get_synaptic_conductance_evolution(
        self,
        syn_name: str,
        save    : bool = False
    ) -> None:
        ''' Plot the evolution of the current '''
        v      = getattr(self.statemon, 'v')
        R_memb = getattr(self.pop, 'R_memb')
        E_syn  = getattr(self.pop, f'E_{syn_name}')
        I_syn  = getattr(self.statemon, f'I_{syn_name}')

        I_syn = (I_syn.T / R_memb).T
        G_syn = (I_syn.T / (E_syn - v.T)).T

        if not save:
            return G_syn

        # Save data
        import pandas as pd

        data_path = self.params.simulation.results_data_folder_run
        file_name = f'{data_path}/statemon_G_{syn_name}.csv'

        logging.info(f'Saving self.statemon.G_{syn_name} data in {file_name}')

        G_syn_df = pd.DataFrame(G_syn.T)
        G_syn_df.to_csv(file_name, index= False)

        return G_syn

    def _get_inds_to_inds_connections(
        self,
        src_inds : np.ndarray,
        trg_inds : np.ndarray,
        plot_data: bool = False
    ):
        ''' Get indices of connections between different populations '''
        wmat         = self.get_wmat()
        wmat         = ( wmat != 0.0 ).astype(int)
        wmat_src_trg = wmat[src_inds][:, trg_inds]

        if not plot_data:
            return wmat_src_trg

        n_conn_per_trg = np.sum( wmat_src_trg, axis=0 )
        nmin, nmax     = np.min(n_conn_per_trg), np.max(n_conn_per_trg)
        plt.hist(
            n_conn_per_trg,
            bins      = nmax - nmin + 1,
            color     = 'royalblue',
            edgecolor = 'black',
            alpha     = 0.75,
            linewidth = 1.2,
            range     = ( nmin - 0.5, nmax + 0.5 ),
        )

        return wmat_src_trg

    def _plot_variable_evolution(
        self,
        var_name : str,
        var_vals : np.ndarray,
        inds_l   : np.ndarray,
        inds_r   : np.ndarray,
    ):
        ''' Plot the evolution of the variable '''

        var_l   = var_vals[ inds_l ]
        var_r   = var_vals[ inds_r ]
        n_l     = var_l.shape[0]
        n_r     = var_r.shape[0]
        var_l_m = np.mean(var_l, axis=0)
        var_r_m = np.mean(var_r, axis=0)

        plt.figure(f'{var_name}')
        plt.subplot(2, 1, 1)
        plt.title(f'{var_name} - Evolution- Left')
        # plt.plot( [ self.statemon.t ] * n_l, var_l, 'orange', lw=0.25 )
        plt.plot( self.statemon.t, var_l_m, 'black', lw=2 )
        plt.subplot(2, 1, 2)
        plt.title(f'{var_name} - Evolution- Right')
        # plt.plot( [ self.statemon.t ] * n_r, var_r,  'black', lw=0.25 )
        plt.plot( self.statemon.t, var_r_m, 'black', lw=2 )

        return

    # INDIVIDUAL PLOTS
    # ------ [ Variables evolution ] ------
    def _plot_network_states_evolution(self) -> None:
        '''
        Evolution of the neuronal parameters
        '''
        monitor = self.params.monitor.states
        if not self._check_monitor_conditions(monitor, ['active'], ['showit', 'figure'] ):
            return

        inds   = range(len(self.pop)) if monitor['indices'] is True else monitor['indices']
        subpop = self.pop[inds]

        target_times = self.statemon_dict['t'] >= float(self.initial_time)
        times        = self.statemon_dict['t'][target_times]

        R_memb = np.array(subpop.R_memb)

        def _get_var(var_name: str):
            ''' Return the desired recorded quantity in the target time interval '''
            return self.statemon_dict.get(var_name)[target_times]

        variables_values = {}

        if 'I_tot' in self.statemon_dict.keys():
            i_tot = _get_var('I_tot')
            i_tot_unit = self.pop.I_tot[0].get_best_unit()
            i_tot_unit = ( i_tot_unit / i_tot_unit.base ).get_best_unit().name

            variables_values[f'Total input [{i_tot_unit}]'] = i_tot.T

        if 'I_ext' in self.statemon_dict.keys():
            i_drv = _get_var('I_ext')
            i_drv_unit = self.pop.I_ext[0].get_best_unit()
            i_drv_unit = ( i_drv_unit / i_drv_unit.base ).get_best_unit().name

            variables_values[f'External drive [{i_drv_unit}]'] = i_drv.T

        if 'v' in self.statemon_dict.keys():
            v_memb = _get_var('v')
            v_rest = np.array(subpop.V_rest[inds])

            variables_values['Membrane potential'] = v_memb.T
            variables_values['Leak current']       = ( (v_memb - v_rest) / R_memb ).T

        if 'w1' in self.statemon_dict.keys():
            w1 = _get_var('w1')
            variables_values['Fast adaptation'] = w1.T

        if 'w2' in self.statemon_dict.keys():
            w2 = _get_var('w2')
            variables_values['Slow adaptation'] = w2.T


        for key, value in variables_values.items():
            fig_te = plt.figure(f'{key} - Evolution')
            snn_plotting.plot_temporal_evolutions(
                times            = times,
                variables_values = [ value ],
                variables_names  = [ key ],
                inds             = inds,
                three_dim        = False
            )
            key = key.replace(' ', '_').replace('[','').replace(']','')
            self.figures[f'fig_te_{key}'] = fig_te

        return

    def _plot_voltage_traces_evolution(self) -> None:
        ''' Plot peak frequencies '''

        monitor = self.params.monitor.states
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        monitor_voltage_traces = monitor['plotpars']['voltage_traces']
        if not monitor_voltage_traces['showit']:
            return

        network_mods      = self.params.topology.network_modules
        target_times      = self.statemon_dict['t'] >= float(self.initial_time)
        times             = self.statemon_dict['t'][target_times]
        v_memb            = self.statemon_dict['v'][target_times]

        fig_list = snn_plotting.plot_voltage_traces_evolution(
            times           = times,
            v_memb          = v_memb,
            network_modules = network_mods,
            module_names    = monitor_voltage_traces['modules'],
            close           = monitor_voltage_traces['close'],
            save            = monitor_voltage_traces['save'],
            save_path       = self.params.simulation.figures_data_folder_run,
            ref_freq        = self.metrics['neur_freq_ax'],
        )

        return

    def _plot_network_states_animation(self) -> None:
        '''
        Animation of the evolution of the neuronal parameters
        '''
        monitor = self.params.monitor.states
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit', 'animate'] ):
            return

        fig_na = plt.figure('Neuronal activity animation')
        anim_na = net_anim.animation_neuronal_activity(
            fig            = fig_na,
            pop            = self.pop,
            statemon_times = self.statemon_dict['t'],
            statemon_dict  = self.statemon_dict,
            net_cpg_module = self.params.topology.network_modules['cpg'],
            height         = self.params.topology.height_segment_row,
            limb_positions = self.params.topology.limbs_i_positions
        )
        self.figures['fig_na'] = [fig_na, anim_na]

    def _plot_musclecells_evolution(self) -> None:
        '''
        Plot evolution of the muscle cells' variables
        '''
        monitor = self.params.monitor.muscle_cells
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        mc_module = self.params.topology.network_modules['mc']
        plot_pars = monitor['plotpars']

        if mc_module['axial'].include:
            fig_mc_a = plt.figure('Muscle cells evolution - AXIS')
            snn_plotting.plot_musclecells_evolutions_axial(
                musclemon_times = self.musclemon_dict['t'],
                musclemon_dict  = self.musclemon_dict,
                module_mc       = mc_module,
                plotpars        = plot_pars,
                starting_time   = self.initial_time,
            )
            self.figures['fig_mc_a'] = fig_mc_a

        if mc_module['axial'].include and plot_pars['duty_cycle_ax']['showit']:
            duty_pars = plot_pars['duty_cycle_ax']

            fig_mc_duty_a = plt.figure('Muscle cells duty factor - AXIS')
            snn_plotting.plot_musclecells_duty_cycle_evolutions_axial(
                t_muscle      = self.musclemon_dict['t'],
                v_muscle      = self.musclemon_dict['v'],
                module_mc     = mc_module,
                neur_freq_ax  = self.metrics['neur_freq_ax'],
                starting_time = self.initial_time,
                target_seg    = duty_pars['target_seg'],
                filtering     = duty_pars['filter'],
            )
            self.figures['fig_mc_duty_a'] = fig_mc_duty_a

        if mc_module['limbs'].include:
            fig_mc_l = plt.figure('Muscle cells evolution - LIMBS')
            snn_plotting.plot_musclecells_evolutions_limbs(
                musclemon_times = self.musclemon_dict['t'],
                musclemon_dict  = self.musclemon_dict,
                module_mc       = mc_module,
                plotpars        = plot_pars,
                starting_time   = self.initial_time,
            )
            self.figures['fig_mc_l'] = fig_mc_l

    def _plot_simulated_emg_signals(self) -> None:
        '''
        Evolution of the simulated sEMG signals
        '''

        monitor = self.params.monitor.spikes
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        monitor_emg_traces = monitor['plotpars'].get('emg_traces')
        if not monitor_emg_traces or not monitor_emg_traces['showit']:
            return

        # Parameters
        duration  = float( self.params.simulation.duration )
        timestep  = float( self.params.simulation.timestep )
        mn_module = self.params.topology.network_modules['mn']

        if not mn_module.include:
            return

        fig_list = snn_plotting.plot_simulated_emg_evolution(
            duration  = duration,
            timestep  = timestep,
            mn_module = mn_module.axial.mn,
            spikemon  = self.spikemon,
            close     = monitor_emg_traces['close'],
            save      = monitor_emg_traces['save'],
            save_path = self.params.simulation.figures_data_folder_run,
        )

        return

    # \----- [ Variables evolution ] ------

    # ------ [ Connectivity matrices ] ------
    def _plot_connectivity_matrix(self) -> None:
        '''
        Show connections within the network
        '''
        monitor = self.params.monitor.connectivity
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        fig_cm = plt.figure('Connectivity matrix')
        self.get_wmat()

        # TODO: Automatize depth to select for the modules
        leaf_modules = self.params.topology.network_leaf_modules[0]

        snn_plotting.plot_connectivity_matrix(
            pop_i                  = self.pop,
            pop_j                  = self.pop,
            w_syn                  = self.wmat,
            network_modules_list_i = leaf_modules,
            network_modules_list_j = leaf_modules,
        )
        self.figures['fig_cm'] = fig_cm

    def _plot_limbs_connectivity_matrix(self, label: str = '') -> None:
        '''
        Show connections within the limbs of the network
        '''
        fig_cm_lb = plt.figure(f'{label.upper()}_Limbs connectivity matrix')
        self.get_wmat()
        snn_plotting.plot_limb_connectivity(
            wsyn             = self.wmat,
            cpg_limbs_module = self.params.topology.network_modules['cpg']['limbs'],
            plot_label       = self.params.simulation.gait
        )
        self.figures[f'fig_cm_lb_{label}'] = fig_cm_lb

    # \----- [ Connectivity matrices ] ------

    # ------ [ Raster plots ] ------
    def _plot_raster_plot(self) -> None:
        '''
        Raster plot of spiking activity
        '''
        monitor = self.params.monitor.spikes
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        # TODO: Automatize depth to select for the modules
        plot_pars    : dict = monitor['plotpars']
        leaf_modules = self.params.topology.network_leaf_modules[0]
        neurons_y    = self.params.topology.neurons_y_mech[0]

        # RASTER PLOT
        fig_rp = plt.figure('Raster plot - Y scaling')
        snn_plotting.plot_raster_plot(
            spikemon_t           = self.spikemon_dict['t'],
            spikemon_i           = self.spikemon_dict['i'],
            duration             = self.params.simulation.duration,
            network_modules_list = leaf_modules,
            neurons_h            = neurons_y,
            mirrored             = False,
            side_ids             = plot_pars.get('side_ids', [0, 1]),
            excluded_mods        = plot_pars.get('excluded_mods', []),
            modules_order        = plot_pars.get('order_mods', []),
            sampling_ratio       = plot_pars.get('sampling_ratio', 1),
            starting_time        = self.initial_time,
            duration_ratio       = 1.0,
        )
        self.figures['fig_rp'] = fig_rp

        if plot_pars['zoom_plot']:
            fig_rp_zoom = plt.figure('Raster plot - Y scaling - Zoom')
            snn_plotting.plot_raster_plot(
                spikemon_t           = self.spikemon_dict['t'],
                spikemon_i           = self.spikemon_dict['i'],
                duration             = self.params.simulation.duration,
                network_modules_list = leaf_modules,
                neurons_h            = neurons_y,
                mirrored             = False,
                side_ids             = plot_pars.get('side_ids', [0, 1]),
                excluded_mods        = plot_pars.get('excluded_mods', []),
                modules_order        = plot_pars.get('order_mods', []),
                sampling_ratio       = plot_pars.get('sampling_ratio', 1),
                starting_time        = self.initial_time,
                duration_ratio       = 0.2,
            )
            self.figures['fig_rp_zoom'] = fig_rp_zoom

        if plot_pars['mirror_plot']:
            fig_rp = plt.figure('Raster plot - Y scaling -Mirrored')
            snn_plotting.plot_raster_plot(
                spikemon_t           = self.spikemon_dict['t'],
                spikemon_i           = self.spikemon_dict['i'],
                duration             = self.params.simulation.duration,
                network_modules_list = leaf_modules,
                neurons_h            = neurons_y,
                mirrored             = True,
                side_ids             = [0, 1],
                excluded_mods        = plot_pars.get('excluded_mods', []),
                modules_order        = plot_pars.get('order_mods', []),
                sampling_ratio       = plot_pars.get('sampling_ratio', 1),
                duration_ratio       = 1.0,
            )
            self.figures['fig_rp_mirr'] = fig_rp

        return

    def _plot_isi_distribution(self) -> None:
        ''' Plot ISI distribution for each module '''

        monitor       = self.params.monitor.spikes
        isi_plot_pars = monitor['plotpars']['isi_plot']

        # Checks
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit'] ):
            return

        if not isi_plot_pars['showit']:
            return

        # Target modules
        isi_neurons_modules = [
            module
            for target_module in isi_plot_pars['modules']
            for module in self.params.topology.network_leaf_modules[0]
            if target_module in module.name.split('.')
        ]

        isi_figures = snn_plotting.plot_isi_distribution(
            neuron_modules = isi_neurons_modules,
            spikemon_dict  = self.spikemon_dict,
            timestep       = self.params.simulation.timestep,
        )

        self.figures.update(isi_figures)

    def _plot_raster_plot_animation(self) -> None:
        '''
        Raster plot of spiking activity
        '''
        monitor = self.params.monitor.spikes
        if not self._check_monitor_conditions(monitor, ['active','save'], ['showit', 'animate'] ):
            return

        fig_rpa = plt.figure('Raster plot animation')
        anim_rpa = net_anim.animation_raster_plot(
            fig                  = fig_rpa,
            pop                  = self.pop,
            spikemon_t           = self.spikemon_dict['t'],
            spikemon_i           = self.spikemon_dict['i'],
            duration             = self.params.simulation.duration,
            timestep             = self.params.simulation.timestep,
            network_modules_list = self.params.topology.network_leaf_modules[0],
            plotpars             = self.params.monitor.spikes['plotpars']
        )
        self.figures['fig_rpa'] = [fig_rpa, anim_rpa]

    # \----- [ Raster plots ] ------

    # ------ [ Processed activations ] ------

    def _plot_processed_pool_activation(self) -> None:
        '''
        Evolution of the processed pools' activations
        '''
        monitor = self.params.monitor.pools_activation
        if not self._check_monitor_conditions(monitor, [], ['showit'] ):
            return

        self._get_oscillations_com()

        # CPG ACTIVATIONS
        # if self.params.topology.include_cpg:
        if self.smooth_activations.get('ex') is not None:
            fig_pa_cpg = plt.figure('CPG_EX - Processed pools activations')

            signals_cpg = {
                'times_f'      : self.smooth_activations['ex']['times'],
                'spike_count_f': self.smooth_activations['ex']['spike_count'],
            }
            points_cpg  = {
                # 'com_x'   : self.com_x,
                # 'com_y'   : self.com_y,
                # 'strt_ind': self.start_indices,
                # 'stop_ind': self.stop_indices,
            }
            snn_plotting.plot_processed_pools_activations(
                signals        = signals_cpg,
                points         = points_cpg,
                seg_axial      = self.params.topology.segments_axial,
                seg_limbs      = self.params.topology.segments_limbs,
                duration       = self.params.simulation.duration,
                plotpars       = monitor['plotpars'],
            )
            self.figures['fig_pa_cpg'] = fig_pa_cpg

        # MN ACTIVATIONS
        if self.smooth_activations.get('mn') is not None:
            fig_pa_mn = plt.figure('MN - Processed pools activations')

            signals_mn = {
                'times_f'      : self.smooth_activations['mn']['times'],
                'spike_count_f': self.smooth_activations['mn']['spike_count'],
            }
            points_mn  = {}

            snn_plotting.plot_processed_pools_activations(
                signals        = signals_mn,
                points         = points_mn,
                seg_axial      = self.params.topology.segments_axial,
                seg_limbs      = self.params.topology.segments_limbs,
                duration       = self.params.simulation.duration,
                plotpars       = monitor['plotpars'],
            )
            self.figures['fig_pa_mn'] = fig_pa_mn

    def _plot_processed_pool_animation(self) -> None:
        '''
        Animation of the evolution of the processed pools' activations
        '''
        monitor = self.params.monitor.pools_activation
        if not self._check_monitor_conditions(monitor, [], ['showit','animate'] ):
            return

        if self.smooth_activations.get('ex') is None:
            return

        signals  = self.smooth_activations['ex']['spike_count']
        timestep = self.smooth_activations['ex']['time_step']

        fig_spa = plt.figure('Smooth neuronal activity animation')
        anim_spa = net_anim.animation_smooth_neural_activity(
            fig            = fig_spa,
            signals        = signals,
            timestep       = timestep,
            limb_positions = self.params.topology.limbs_i_positions
        )
        self.figures['fig_spa'] = [fig_spa, anim_spa]

    # \----- [ Processed activations ] ------

    # ------ [ Metrics evolution ] ------
    def _plot_hilbert_freq_evolution(self) -> None:
        '''
        Plot evolution of the measured oscillations' frequencies
        '''
        monitor = self.params.monitor.hilbert_freq
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        fig_fe = plt.figure('Frequency evolution')
        snn_plotting.plot_hilb_freq_evolution(self.freq_evolution_dict)
        self.figures['fig_fe'] = fig_fe

    def _plot_hilbert_ipl_evolution(self) -> None:
        '''
        Plot evolution of the measured oscillations' IPL
        '''
        monitor = self.params.monitor.hilbert_ipl
        if not self._check_monitor_conditions(monitor, ['active'], ['showit'] ):
            return

        fig_le = plt.figure('IPL evolution')
        snn_plotting.plot_hilbert_ipl_evolution(
            ipls_dict           = self.ipls_evolutions_dict,
            plotpars            = monitor['plotpars'],
            limb_pair_positions = self.params.topology.limbs_pairs_i_positions
        )
        self.figures['fig_le'] = fig_le

    def _plot_spike_count_cycle_frequencies_evolution(self) -> None:
        ''' Plot peak frequencies '''

        cycle_freq_plot_pars = self.params.monitor.pools_activation['plotpars']['cycle_freq']

        if not cycle_freq_plot_pars['showit']:
            return

        ner_name     = cycle_freq_plot_pars['ner_name']
        ner_activity = self.smooth_activations.get(ner_name)

        if ner_activity is None:
            return

        times   = ner_activity['times']
        signals = ner_activity['spike_count']
        mod_tag = ner_name.upper()

        fig_list = snn_plotting.plot_spike_count_cycle_frequencies_evolution(
            times       = times,
            signals     = signals,
            module_name = mod_tag,
            close       = cycle_freq_plot_pars['close'],
            save        = cycle_freq_plot_pars['save'],
            save_path   = self.params.simulation.figures_data_folder_run,
        )

        return

    # \----- [ Metrics evolution ] ------

    # ------ [ Online metrics evolution ] ------
    def _plot_online_limb_activations_evolution(self) -> None:
        '''
        Plot evolution of the measured limb activations
        '''
        monitor = self.params.monitor.online_metrics
        if not monitor['active'] or not self.online_activities_lb.any():
            return

        callback_dt = self.params.simulation.callback_dt

        if monitor['plotpars']['activity']:
            fig_oa = plt.figure('Online activity evolution')
            snn_plotting.plot_online_activities_lb(self.all_online_activities_lb, callback_dt)
            self.figures['fig_oa'] = fig_oa

        if monitor['plotpars']['period']:
            fig_op = plt.figure('Online period evolution')
            snn_plotting.plot_online_periods_lb(self.all_online_periods_lb, callback_dt)
            self.figures['fig_op'] = fig_op

        if monitor['plotpars']['period']:
            fig_od = plt.figure('Online duty evolution')
            snn_plotting.plot_online_duties_lb(self.all_online_duties_lb, callback_dt)
            self.figures['fig_od'] = fig_od

    # \----- [ Online metrics evolution ] ------

    # ------ [ Data saving ] ------
    def save_prompt(
        self,
        figures_dict : dict[str, Figure] = None
    ) -> None:
        ''' Prompt user to choose whether to save the figures '''

        figures_snn  = self.figures if hasattr(self, 'figures') else {}
        figures_dict = figures_dict if figures_dict is not None else {}
        results_path = self.params.simulation.results_data_folder_run
        figures_base = self.params.simulation.figures_data_folder_run

        saved_figures, figures_path = plots_utils.save_prompt(
            figures_dict = figures_snn | figures_dict,
            folder_path  = self.params.simulation.figures_data_folder_run,
            default_save = self.params.simulation.save_by_default,
        )

        # Clear figures
        self.clear_figures()

        if not saved_figures:
            return

        def _figures_mover( plot_pars: dict, folder_name: str):
            if not plot_pars or not ( plot_pars['showit'] and plot_pars['save'] ):
                return
            plots_utils.move_files_if_recent(
                source_folder = f'{figures_base}/{folder_name}',
                target_folder = f'{figures_path}/{folder_name}',
            )

        # Move cycle frequency plots
        cycle_freq_plot_pars = self.params.monitor.pools_activation['plotpars'].get('cycle_freq')
        _figures_mover(cycle_freq_plot_pars, 'cycle_frequencies')

        # Move voltage traces plots
        voltage_traces_plot_pars = self.params.monitor.states['plotpars'].get('voltage_traces')
        _figures_mover(voltage_traces_plot_pars, 'voltage_traces')

        # Move emg traces plots
        emg_traces_plot_pars = self.params.monitor.spikes['plotpars'].get('emg_traces')
        _figures_mover(emg_traces_plot_pars, 'emg_traces')

        # Move saved .csv files (raw data)
        plots_utils.move_files_if_recent(
            source_folder = results_path,
            target_folder = f'{figures_path}/raw_data',
            file_type     = '.csv',
        )

        # Move farms video files
        plots_utils.move_files_if_recent(
            source_folder = f'{results_path}/farms',
            target_folder = figures_path,
            file_type     = '.mp4',
            delete_src    = False,
        )

        video_path_src = f'{results_path}/farms/animation.mp4'
        video_path_dst = f'{figures_path}/animation.mp4'
        if os.path.isfile(video_path_src):
            logging.info('Copying %s file to %s', video_path_src,  video_path_dst)
            shutil.copyfile(video_path_src, video_path_dst, )

        return

    # \----- [ Data saving ] ------

    # ------ [ Data cleaning ] ------
    def clear_figures(self) -> None:
        ''' Clear figures '''
        self.figures = {}
        plt.close('all')

    # \----- [ Data cleaning ] ------

# TEST
def main():
    ''' Test case '''

    from queue import Queue

    logging.info('TEST: SNN Plotting ')

    plotting = SnnPlotting(
        network_name = 'snn_core_test',
        params_name  = 'pars_simulation_test',
        q_in         = Queue(),
        q_out        = Queue(),
    )

    plotting.define_network_topology()
    plotting.simulation_run()
    plotting.simulation_post_processing()
    plotting.simulation_plots()
    plt.show()
    plotting.save_prompt()

    return plotting

if __name__ == '__main__':
    main()