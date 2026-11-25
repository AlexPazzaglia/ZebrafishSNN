'''
Base plotting class for the vortex matching simulations.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, CURRENTDIR)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import data_loading
import histogram_fitting

BODY_MASS    = 5.022e-05
COM_POS_REL  = 0.4321805865303308
PHASE_OFFSET = 0.0 * np.pi

class BaseSchoolingPlotting:
    def __init__(
        self,
        results_root_folder: str,
        feedback_mode      : int,
        leader_signal_str  : str,
        spawn_x_vals       : np.ndarray,
        np_random_seed_vals: np.ndarray,
        stim_a_off         : float = 0.0,
        leader_freq        : float = 3.5,
        use_actual_com_pos : bool  = True,
        use_actual_phases  : bool  = True,
        module_name        : str  = data_loading.MODULE_NAME,
        simulation_tag     : str  = data_loading.SIMULATION_TAG,
    ):
        """ Initialize an incremental 2D histogram."""

        # Parameters
        self.module_name         = module_name
        self.simulation_tag      = simulation_tag
        self.results_root_folder = results_root_folder
        self.stim_a_off          = stim_a_off
        self.feedback_mode       = feedback_mode
        self.leader_signal_str   = leader_signal_str
        self.leader_freq         = leader_freq
        self.spawn_x_vals        = spawn_x_vals
        self.np_random_seed_vals = np_random_seed_vals
        self.use_actual_com_pos  = use_actual_com_pos
        self.use_actual_phases   = use_actual_phases

        self.n_spawn_x = len(spawn_x_vals)
        self.n_seeds   = len(np_random_seed_vals)
        self.ps_tag    = self._get_ps_tag()

        # Define parameters
        self._define_parameters()

        # Folders names
        self._get_folder_names()

        # Initialize the histogram counts
        self._initialize_histogram()

        return

    ###########################################################################
    # PARAMETERS ##############################################################
    ###########################################################################

    def _define_parameters(self):
        ''' Define the parameters for the simulation '''
        self.process_parameters = {}
        self.all_parameters     = {}
        return

    def _get_folder_names(self):
        ''' Get the folder names for the simulations '''

        self.folder_names = [
            data_loading.get_folder_name_from_seed(
                np_random_seed = seed,
                mod_name       = self.module_name,
                sim_tag        = self.simulation_tag,
            )
            for seed in self.np_random_seed_vals
        ]

        self.save_file_root = None
        self.save_folder    = None

        return

    def _get_ps_tag(self) -> str:
        ''' Get the ps tag '''
        return data_loading.get_ps_tag(
            stim_a_off    = self.stim_a_off,
            feedback_mode = self.feedback_mode,
        )

    ###########################################################################
    # DATA COLLECTION ##########################################################
    ###########################################################################

    def _load_saved_parameters(self):
        '''
        Load the collected data if it exists
        Return False if the loaded parameters match the expected ones.
        '''

        # Load the parameters
        save_folder        = self.save_folder
        save_file_root     = self.save_file_root
        process_parameters = self.process_parameters

        parameters_file = f'{save_folder}/{save_file_root}_parameters.csv'

        if not os.path.exists(parameters_file):
            return False

        with open(parameters_file, 'r') as f:
            # Read every line and compare the values
            for line in f:
                parts = line.strip().split(',')
                key   = parts[0]
                value = parts[1:]

                # Get the correct value
                if key in process_parameters:
                    par_value = process_parameters[key]
                elif hasattr(self, key):
                    par_value = getattr(self, key)
                else:
                    raise ValueError(f'Unknown key {key}')

                # Get the type of the value
                par_is_array = isinstance(par_value, (list, tuple, np.ndarray))
                par_type     = (
                    type(par_value)
                    if not par_is_array
                    else type(par_value[0])
                )

                # Handle parentheses
                if par_is_array:
                    value = [ val.replace('[', '').replace(']', '') for val in value ]

                # Convert the values to the correct type
                if not par_is_array:
                    value = par_type(value[0])
                else:
                    value = np.array([par_type(val) for val in value])

                # Compare the values
                if not par_is_array and ( par_value != value ):
                    return False

                if par_is_array and not np.allclose(par_value, value):
                    return False

        return True

    def _load_saved_histogram(self):
        ''' Load the histogram data if it exists '''

        save_folder    = self.save_folder
        save_file_root = self.save_file_root

        #####################
        # LOAD THE HISTOGRAM
        #####################

        histogram_file = f'{save_folder}/{save_file_root}_histogram.csv'

        if not os.path.exists(histogram_file):
            return False

        histogram = pd.read_csv(histogram_file, index_col=0).astype(int)
        vals_1    = np.array(histogram.columns, dtype=float)
        vals_2    = np.array(histogram.index, dtype=float)

        if len(vals_1) == self.x_bins and len(vals_2) == self.phi_bins:
            x_vals    = vals_1
            phi_vals  = vals_2
            hist_vals = histogram.values.T
        elif len(vals_1) == self.phi_bins and len(vals_2) == self.x_bins:
            x_vals    = vals_2
            phi_vals  = vals_1
            hist_vals = histogram.values
        else:
            print(f'ERROR: Histogram dimensions do not match expected values.')
            return False

        # Check bins values
        if not (
            np.allclose(  x_vals,   self.x_vals) and
            np.allclose(phi_vals, self.phi_vals)
        ):
            print(f'ERROR: Histogram values do not match expected values.')
            return False

        # Build histogram analysis object
        self.hist_analysis = histogram_fitting.HistogramPlotFitting(
            histogram = hist_vals,
            x_vals    = x_vals,
            phi_vals  = phi_vals,
            verbose   = True,
        )
        self.fit_results  = self.hist_analysis.fit_results
        self.corr_results = self.hist_analysis.corr_results

        # Assign histogram data
        self.histogram   = hist_vals
        self.loaded_hist = True

        return True

    def _load_saved_performance_data(self):
        ''' Load the performance data if it exists '''

        save_folder    = self.save_folder
        save_file_root = self.save_file_root

        # Load the performance data
        performance_file = f'{save_folder}/{save_file_root}_performance.csv'

        if not os.path.exists(performance_file):
            return False

        performance_df = pd.read_csv(performance_file)

        self.performance_data = {
            key : performance_df[key].values
            for key in performance_df.columns
        }

        return True

    def load_saved_data(self):
        ''' Load the collected data if it exists '''

        print(f'Searching existing save for {self.save_file_root} ... ')
        fail_str = (
            'FAILED \n'
            'Loading the data from scratch ...'
        )

        # Load the parameters
        if not self._load_saved_parameters():
            print(f'Matching parameters not found ... {fail_str}')
            return False

        # Load the histogram
        if not self._load_saved_histogram():
            print(f'Matching histogram not found ... {fail_str}')
            return False

        # Load the performance data
        if not self._load_saved_performance_data():
            print(f'Matching performance data not found ... {fail_str}')
            return False

        # Print message
        print(f'Found existing save for {self.save_file_root} ... LOADED')

        return True

    def _collect_time_and_metrics_data(
        self,
        folder_name   : str,
        process_folder: str,
    ):
        ''' Collect the time data from the first simulation '''

        time_data, performance_data = data_loading.load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
            target_quantities   = ['times'],
        )

        # Metrics data
        self.performance_data = {
            key : []
            for key, value in performance_data.items()
            if (
                not isinstance(value[0], np.ndarray) and
                not value[0] == np.nan
            )
        }

        # Time data
        self.times        = np.array( time_data['times'] )
        self.time_step    = self.times[1] - self.times[0]
        self.duration     = self.times[-1] - self.times[0]
        self.n_iterations = len(self.times)

        self.discard_time  = 5.0
        self.discard_ratio = self.discard_time / self.times[-1]
        self.n_iter_cons   = round(self.n_iterations * (1 - self.discard_ratio))

        return

    def _update_histogram_counts(
        self,
        x_values  : np.ndarray,
        phi_values: np.ndarray,
    ):
        """ Update the histogram with new x_values and phi_values. """

        # Compute the histogram for the new data
        hist, x_edges, phi_edges = np.histogram2d(
            x     = x_values,
            y     = phi_values,
            bins  = [ self.x_bins,  self.phi_bins],
            range = [self.x_range, self.phi_range],
        )

        # Add the new histogram counts to the existing one
        self.histogram += hist

        # Checks
        assert np.allclose(self.x_edges, x_edges), \
            f'Error: x_edges {self.x_edges} != {x_edges}'
        assert np.allclose(self.phi_edges, phi_edges), \
            f'Error: phi_edges {self.phi_edges} != {phi_edges}'

        return

    def _collect_single_simulation_com_vs_angle_data(
        self,
        folder_name   : str,
        process_folder: str,
    ) -> pd.DataFrame:
        ''' Collect com vs angle data for a single simulation '''

        data_df = data_loading.load_com_vs_angle_data_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
        )

        return data_df

    def _get_stacked_com_pos(
        self,
        data_df: pd.DataFrame,
        spawn_x: float,
        n_iter : int,
    ) -> np.ndarray:
        ''' Get stacked com positions for histogram '''

        # Positions evolution
        if self.use_actual_com_pos:
            com_pos_diff = data_df.com_positions_diff[-n_iter:] - COM_POS_REL
        else:
            com_pos_diff = spawn_x * np.ones(n_iter)

        # Stack
        com_pos_stack = np.concatenate( [ com_pos_diff ] * 5 )

        return com_pos_stack

    def _get_stacked_phases(
        self,
        data_df: pd.DataFrame,
        n_iter : int,
    ):
        ''' Get stacked phases for histogram '''

        # Phases evolution
        if self.use_actual_phases:
            phases_diff = data_df.joint_phases_diff[-n_iter:]
        else:
            phases_diff = data_df.joint_phases_diff[-n_iter:] + PHASE_OFFSET

        phases_diff = histogram_fitting.wrap_phase_to_pi_pi(phases_diff)

        # Stack
        phases_stack = np.concatenate(
            [
                phases_diff - 2*np.pi,   # 1. Mostly < range
                phases_diff,             # 2.
                phases_diff + 2*np.pi,   # 3.
                phases_diff + 4*np.pi,   # 4.
                phases_diff + 6*np.pi,   # 5. Mostly > range

            ]
        )
        return phases_stack

    def _collect_single_simulation_data(
        self,
        folder_name   : str,
        process_folder: str,
        spawn_x       : float = None,
        excluded      : bool = False,
    ):
        ''' Collect data for a single simulation '''

        n_iter = self.n_iter_cons

        # Load data
        data_df, performance_dict = data_loading.load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
        )

        ########################
        ### PERFORMANCE DATA ###
        ########################

        # Store performance data
        for key in self.performance_data.keys():
            self.performance_data[key].append(performance_dict[key][0])

        ########################
        ### HISTOGRAM DATA   ###
        ########################

        if excluded:
            return

        # STACK POSITIONS
        com_positions_stack = self._get_stacked_com_pos(
            data_df = data_df,
            spawn_x = spawn_x,
            n_iter  = n_iter,
        )

        # STACK PHASES
        joints_phases_stack = self._get_stacked_phases(
            data_df = data_df,
            n_iter  = n_iter,
        )

        # Update histogram with new data
        self._update_histogram_counts(
            x_values   = com_positions_stack,
            phi_values = joints_phases_stack,
        )

        return

    def _collect_all_simulation_com_vs_angle_data(
        self,
        folder_names_list   : list[str],
        process_folders_list: list[list[str]],
        vals_xys_list       : list[list[float, float, float]],
    ):
        """ Collect com vs angle data from all simulations """

        # Verify number of simulations
        n_simulations = len(folder_names_list)
        assert n_simulations == len(process_folders_list) == len(vals_xys_list), \
            f'Error: {n_simulations} simulations found'

        # Verify existing save
        csv_path = (
            f'{self.results_root_folder}/{self.save_file_root}_'
            f'com_vs_angle_all.csv'
        )
        if os.path.exists(csv_path):
            print(f'Found existing {csv_path}')
            data_com_vs_angle_all = pd.read_csv(csv_path)
            print('LOADED')
            return data_com_vs_angle_all

        # Collect data
        n_print = n_simulations // 10
        print(f'Collecting data for {self.save_file_root}', end=' ... ', flush=True)

        data_com_vs_angle_all_list = []

        for sim_ind in range(n_simulations):

            if sim_ind % n_print == 0:
                print(f'{ 100 * sim_ind / n_simulations :.1f} %', end=' ... ', flush=True)

            data_df = self._collect_single_simulation_com_vs_angle_data(
                folder_name    = folder_names_list[sim_ind],
                process_folder = process_folders_list[sim_ind],
            )

            # ADD METADATA
            n_points           = len(data_df)
            data_df['step']    = np.arange(n_points)
            data_df['spawn_x'] = vals_xys_list[sim_ind][0]
            data_df['spawn_y'] = vals_xys_list[sim_ind][1]
            data_df['seed']    = vals_xys_list[sim_ind][2]

            # Append
            data_com_vs_angle_all_list.append(data_df)

        # Concatenate all data
        data_com_vs_angle_all = pd.concat( data_com_vs_angle_all_list, ignore_index=True)

        # Log info
        mem_bytes = data_com_vs_angle_all.memory_usage(deep=True).sum()
        mem_gb    = mem_bytes / (1024 ** 3)
        print(f"DataFrame size: {mem_gb:.2f} GB")

        # Save as csv
        print(f'Saving collected data to {csv_path}')
        data_com_vs_angle_all.to_csv(csv_path, index=False)

        print('DONE')

        return data_com_vs_angle_all

    def _collect_all_simulation_data(
        self,
        folder_names_list   : list[str],
        process_folders_list: list[list[str]],
        spawn_x_vals_list   : np.ndarray,
        excluded_map        : np.ndarray = None,
    ):
        """ Collect data from all simulations and update the histogram. """

        n_sim = len(folder_names_list)
        if excluded_map is None:
            excluded_map = np.zeros(n_sim, dtype=bool)

        # Collect the time and metrics data
        self._collect_time_and_metrics_data(
            folder_name    = folder_names_list[0],
            process_folder = process_folders_list[0],
        )

        # Load the saved data if it exists
        if self.load_saved_data():
            return

        # Verify number of simulations
        assert n_sim == len(process_folders_list) == len(spawn_x_vals_list), \
            f'Error: {n_sim} simulations found'

        # Collect data
        n_print = n_sim // 10
        print(f'Collecting data for {self.save_file_root}', end=' ... ', flush=True)

        for sim_ind in range(n_sim):

            if sim_ind % n_print == 0:
                print(f'{ 100 * sim_ind / n_sim :.1f} %', end=' ... ', flush=True)

            self._collect_single_simulation_data(
                folder_name    = folder_names_list[sim_ind],
                process_folder = process_folders_list[sim_ind],
                spawn_x        = spawn_x_vals_list[sim_ind],
                excluded       = excluded_map[sim_ind],
            )

        # Prune non-informative performance data
        prune_keys = []
        for key in self.performance_data.keys():
            if isinstance(self.performance_data[key][0], (str)):
                continue
            if isinstance(self.performance_data[key][0], (np.ndarray)):
                continue
            if (
                np.all(np.isnan(self.performance_data[key]))
            ):
                prune_keys.append(key)

        for key in prune_keys:
            del self.performance_data[key]

        # Build histogram analysis object
        self.hist_analysis = histogram_fitting.HistogramPlotFitting(
            histogram = self.histogram,
            x_vals    = self.x_vals,
            phi_vals  = self.phi_vals,
            verbose   = True,
        )
        self.fit_results  = self.hist_analysis.fit_results
        self.corr_results = self.hist_analysis.corr_results

        print('DONE')
        return

    ###########################################################################
    # PLOTS ####################################################################
    ###########################################################################

    def plots(
        self,
        performance : bool = True,
        histogram   : bool = True,
    ):
        ''' Plot the performance data and the histogram '''

        self.figures_dict : dict[str, plt.Figure] = {}

        if performance:
            self.plot_performance()

        if histogram:
            self.plot_histogram()

        return

    def plot_performance(self):
        ''' Plot the performance data '''
        print('WARNING: plot_performance not implemented yet')
        return

    def _initialize_histogram(self):

        # Initialize the histogram counts
        self.x_bins  = len(self.spawn_x_vals)
        if self.x_bins >= 2:
            self.x_step  = self.spawn_x_vals[1] - self.spawn_x_vals[0]
        else:
            self.x_step  = 0.10
        self.x_range = (
            self.spawn_x_vals[0]  - self.x_step / 2,
            self.spawn_x_vals[-1] + self.x_step / 2
        )
        self.x_edges = np.arange(self.x_bins + 1) * self.x_step + self.x_range[0]
        self.x_vals  = self.x_edges[:-1] + self.x_step / 2

        # Phi bins
        phi_min = -np.pi
        phi_max = 5 * np.pi

        self.phi_bins  = 40
        self.phi_step  = (phi_max - phi_min) / ( self.phi_bins - 1 )
        self.phi_range = (
            phi_min - self.phi_step / 2,
            phi_max + self.phi_step / 2,
        )
        self.phi_edges = np.arange(self.phi_bins + 1) * self.phi_step + self.phi_range[0]
        self.phi_vals  = self.phi_edges[:-1] + self.phi_step / 2

        # Histogram
        self.histogram   = np.zeros((self.x_bins, self.phi_bins))
        self.loaded_hist = False

        return

    def plot_histogram(
        self,
        max_density: float,
        **kwargs
    ):
        """Plot the current histogram as a heatmap."""

        # Plot standard histogram
        fig, _ax = self.hist_analysis.plot_histogram(
            max_density = max_density,
        )
        self.figures_dict['histogram'] = fig

        # Plot non-uniform histogram
        fig, _ax = self.hist_analysis.plot_histogram_non_uniform_image(
            max_density = max_density,
        )
        self.figures_dict['histogram_non_uniform'] = fig

        # Plot 3D histogram
        fig, _ax = self.hist_analysis.plot_histogram_3d(
            max_density = max_density,
        )
        self.figures_dict['histogram_3d'] = fig

        # Plot polar histogram
        fig, _ax = self.hist_analysis.plot_histogram_polar(
            max_density = max_density,
        )
        self.figures_dict['histogram_polar'] = fig

        return

    def _plot_line_fitting_histogram_theory(
        self,
        empirical_fit    : tuple[float, float],
        theoretical_speed: bool = False,
    ):
        ''' Get speed of fluid from frequency '''

        frequency = self.leader_freq

        if theoretical_speed:
            # Brainbridge 1958
            speed = (3 * frequency - 4) / 4

        else:
            # Empirical data
            f0, v0 = [ 15.0, 3.90 ] # Jensen et al. 2023 (BL / s)
            f1, v1 = [  8.0, 2.10 ] # Mwaffo et al. 2017 (BL / s)

            slope     = (v1 - v0) / (f1 - f0)
            intercept = v0 - slope * f0
            speed     = slope * frequency + intercept


        # Get the phase values
        x_values   = self.x_edges[:-1]
        phi_values = ( 2 * np.pi * frequency / speed ) * x_values

        # Shift the line to have the same origin as the empirical data
        x0         = x_values[0]
        phi0       = empirical_fit[0] * x0 + empirical_fit[1]
        phi_values = phi_values - phi_values[0] + phi0

        # Plot the line
        for offset in [-2, -1, 0, +1, +2]:
            plt.plot(
                x_values,
                phi_values + offset * 2 * np.pi,
                'w--',
                linewidth = 2,
                color     = 'cyan',
            )

        return speed

    ###########################################################################
    # SAVE #####################################################################
    ###########################################################################

    def _load_fit_results(self):
        ''' Load histogram fitting results '''

        save_folder      = self.save_folder
        save_file_root   = self.save_file_root
        fit_results_file = f'{save_folder}/{save_file_root}_histogram_fit.json'

        if not os.path.exists(fit_results_file):
            return False

        with open(fit_results_file, 'r') as f:
            histogram_fit = json.load(f)

        # Assign results
        self.fit_results  = {
            'params' : [
                histogram_fit['fit_results']['slope'],
                histogram_fit['fit_results']['intercept'],
            ],
            'loss'   : histogram_fit['fit_results']['loss'],
            'r2'     : histogram_fit['fit_results']['r2'],
        }
        self.corr_results = histogram_fit['corr_results']

        return True

    def _save_fit_results(self):
        ''' Save histogram fitting results 00'''

        save_folder      = self.save_folder
        save_file_root   = self.save_file_root
        fit_results_file = f'{save_folder}/{save_file_root}_histogram_fit.json'

        fit_results = {
            'fit_results' : {
                'slope'    : self.fit_results['params'][0],
                'intercept': self.fit_results['params'][1],
                'loss'     : self.fit_results['loss'],
                'r2'       : self.fit_results['r2'],
            },
            'corr_results': {
                'rho_mean' : self.corr_results['rho_mean'],
                'pval_mean': self.corr_results['pval_mean'],
                'rho_dist' : self.corr_results['rho_dist'],
                'pval_dist': self.corr_results['pval_dist'],
            },
        }

        with open(fit_results_file, 'w') as f:
            json.dump(fit_results, f, indent=4)

        print(f'Saved histogram fit results to {fit_results_file}')
        return

    def save_all_simulation_data(self):
        ''' Save the collected data '''

        # Save the parameters as csv (without pandas)
        pars_file  = f'{self.save_folder}/{self.save_file_root}_parameters.csv'
        with open(pars_file, 'w') as f:
            for key, value in self.all_parameters.items():
                if isinstance(value, np.ndarray):
                    value = ','.join([str(val) for val in value])
                f.write(f'{key},{value}\n')
        print(f'Saved parameters to {pars_file}')

        # Save the histogram counts (convert to DataFrame and save as csv)
        x_edges   = ( self.x_edges + (self.x_edges[1] - self.x_edges[0]) / 2 )[:-1]
        phi_edges = ( self.phi_edges + (self.phi_edges[1] - self.phi_edges[0]) / 2 )[:-1]

        histogram_df = pd.DataFrame(
            self.histogram.T,
            columns = x_edges,
            index   = phi_edges,
        )

        histogram_file = f'{self.save_folder}/{self.save_file_root}_histogram.csv'
        histogram_df.to_csv(histogram_file)
        print(f'Saved histogram to {histogram_file}')

        # Save histogram fitting results
        self._save_fit_results()

        # Save the performance data (convert to DataFrame and save as csv)
        performance_df = pd.DataFrame(self.performance_data)
        performance_df.to_csv(f'{self.save_folder}/{self.save_file_root}_performance.csv', index=False)

        return

    def save_plots(self):
        """Save the current histogram as a heatmap."""

        # Create the save folder if it does not exist
        os.makedirs(self.save_folder, exist_ok=True)

        # Save the plots
        for name, figure in self.figures_dict.items():
            fig_path = f'{self.save_folder}/{self.save_file_root}_{name}'
            print(f'Saving {name} to {fig_path}.pdf')
            try:
                figure.savefig(f'{fig_path}.pdf', format='pdf')
            except Exception as e:
                print(f'Error saving {name} to {fig_path}.pdf: {e}')

        return

