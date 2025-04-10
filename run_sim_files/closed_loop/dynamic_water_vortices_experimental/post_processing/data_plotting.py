import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.signal import find_peaks

import data_loading

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

    def _load_saved_simulation_data(self):
        ''' Load the collected data if it exists '''

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

        # Load the histogram
        histogram_file = f'{save_folder}/{save_file_root}_histogram.csv'

        if not os.path.exists(histogram_file):
            return False

        histogram = pd.read_csv(histogram_file, index_col=0).astype(int)
        x_edges   = np.array( histogram.columns, dtype=float )
        phi_edges = np.array( histogram.index, dtype=float )

        dx = (   x_edges[1] -   x_edges[0] )
        dy = ( phi_edges[1] - phi_edges[0] )

        x_edges   =   x_edges - dx / 2
        phi_edges = phi_edges - dy / 2

        x_edges   = np.append(  x_edges,   x_edges[-1] + dx)
        phi_edges = np.append(phi_edges, phi_edges[-1] + dy)

        self.histogram   = histogram.values.T
        self.x_edges     = x_edges
        self.phi_edges   = phi_edges
        self.loaded_hist = True

        if (
            np.isclose(self.phi_edges[0], self.x_range[0]) and
            np.isclose(self.phi_edges[-1], self.x_range[-1]) and
            np.isclose(self.x_edges[0], self.phi_range[0]) and
            np.isclose(self.x_edges[-1], self.phi_range[-1])
        ):
            aux            = self.x_edges
            self.x_edges   = self.phi_edges
            self.phi_edges = aux
            self.histogram = self.histogram.T

        # Load the performance data
        performance_file = f'{save_folder}/{save_file_root}_performance.csv'

        if not os.path.exists(performance_file):
            return False

        performance_df = pd.read_csv(performance_file)

        self.performance_data = {
            key : performance_df[key].values
            for key in performance_df.columns
        }

        # Print message
        print(f'Found existing save for {save_file_root} ... LOADED')

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
            if not isinstance(value[0], np.ndarray) and not np.isnan(value[0])
        }

        # Time data
        self.times            = np.array( time_data['times'] )
        self.n_iterations     = len(self.times)

        self.discard_time  = 5.0
        self.discard_ratio = self.discard_time / self.times[-1]
        self.n_iter_cons   = round(self.n_iterations * (1 - self.discard_ratio))

        return

    def _update_histogram_counts(self, x_values, phi_values):
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

        # Store bin edges (only needs to be set once)
        if self.x_edges is None or self.phi_edges is None:
            self.x_edges   = x_edges
            self.phi_edges = phi_edges

        return

    def _collect_single_simulation_data(
        self,
        folder_name   : str,
        process_folder: str,
        spawn_x_vals  : np.ndarray = None,
    ):
        ''' Collect data for a single simulation '''

        n_iter = self.n_iter_cons

        # Load data
        data_df, performance_dict = data_loading.load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
        )

        # Store performance data
        for key in self.performance_data.keys():
            self.performance_data[key].append(performance_dict[key][0])

        # STACK POSITIONS
        if self.use_actual_com_pos:
            com_positions_stack = np.concatenate(
                [
                    data_df.com_positions_diff[-n_iter:] - COM_POS_REL,
                    data_df.com_positions_diff[-n_iter:] - COM_POS_REL,
                    data_df.com_positions_diff[-n_iter:] - COM_POS_REL,
                ]
            )
        else:
            com_positions_stack = spawn_x_vals * np.ones(n_iter * 3)

        # STACK PHASES
        if self.use_actual_phases:
            joints_phases_diff = data_df.joint_phases_diff[-n_iter:]
        else:
            joints_phases_diff = np.mod(
                data_df.joint_phases_diff[-n_iter:] + PHASE_OFFSET,
                2 * np.pi,
            )
            joints_phases_diff[joints_phases_diff > np.pi] -= 2*np.pi

        joints_phases_stack = np.concatenate(
            [
                joints_phases_diff,
                joints_phases_diff + 2*np.pi,
                joints_phases_diff + 4*np.pi,
            ]
        )

        # Update histogram with new data
        self._update_histogram_counts(
            x_values   = com_positions_stack,
            phi_values = joints_phases_stack,
        )

    def _collect_all_simulation_data(
        self,
        folder_names_list   : list[str],
        process_folders_list: list[list[str]],
        spawn_x_vals_list   : np.ndarray,
    ):
        """ Collect data from all simulations and update the histogram. """

        # Collect the time and metrics data
        self._collect_time_and_metrics_data(
            folder_name    = folder_names_list[0],
            process_folder = process_folders_list[0],
        )

        # Load the saved data if it exists
        if self._load_saved_simulation_data():
            return

        # Verify number of simulations
        n_simulations = len(folder_names_list)
        assert n_simulations == len(process_folders_list) == len(spawn_x_vals_list), \
            f'Error: {n_simulations} simulations found'

        # Collect data
        n_print = n_simulations // 10
        print(f'Collecting data for {self.save_file_root}', end=' ... ', flush=True)

        for sim_ind in range(n_simulations):

            if sim_ind % n_print == 0:
                print(f'{ 100 * sim_ind / n_simulations :.1f} %', end=' ... ', flush=True)

            self._collect_single_simulation_data(
                folder_name    = folder_names_list[sim_ind],
                process_folder = process_folders_list[sim_ind],
                spawn_x_vals   = spawn_x_vals_list[sim_ind],
            )

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

        x_step                  = self.spawn_x_vals[1] - self.spawn_x_vals[0]
        # x_edges               = np.zeros(self.x_bins + 1)
        # x_edges[:self.x_bins] = self.spawn_x_vals - x_step / 2
        # x_edges[-1]           = self.spawn_x_vals[-1] + x_step / 2

        self.x_range = (
            self.spawn_x_vals[0]  - x_step / 2,
            self.spawn_x_vals[-1] + x_step / 2
        )

        # Phi bins
        phi_min = -np.pi
        phi_max =  5 * np.pi
        self.phi_bins  = 40

        phi_step                    = (phi_max - phi_min) / ( self.phi_bins - 1 )
        # phi_edges                 = np.zeros(self.phi_bins + 1)
        # phi_vals                  = np.linspace(-np.pi, 5 * np.pi, self.phi_bins)
        # phi_edges[:self.phi_bins] = phi_vals - phi_step / 2
        # phi_edges[-1]             = phi_vals[-1] + phi_step / 2

        self.phi_range = (
            phi_min - phi_step / 2,
            phi_max + phi_step / 2,
        )

        # Histogram
        self.histogram = np.zeros((self.x_bins, self.phi_bins))

        # Retrieve bin edges
        self.x_edges     = None
        self.phi_edges   = None
        self.loaded_hist = False

        return

    def _decorate_histogram_plot(self):
        ''' Decorate the histogram plot '''

        # Add dashed lines at multiples of pi
        for i in range(-1, 6):
            plt.axhline(i * np.pi, color='white', linestyle='--', linewidth=0.8)

        # Put ylabels and ticks at multiples of pi
        plt.yticks(
            ticks  = np.arange(-np.pi, 5*np.pi+1, np.pi),
            labels = [ r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$']
        )

        # Increas thickness of ticks
        plt.tick_params(axis='both', which='major', width=1.5)

        # Label axes
        plt.xlabel("FB distance (bl)", fontsize=14)
        plt.ylabel("Phase difference $\\Delta \\Phi$", fontsize=14)
        plt.title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$", fontsize=16)

        # Set limits
        plt.xlim(self.x_range)
        plt.ylim(self.phi_range)

        return

    def _plot_histogram_non_uniform_image(
        self,
        histogram: np.ndarray,
    ):
        ''' Plot the histogram using NonUniformImage '''
        from matplotlib.image import NonUniformImage
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=self.x_range, ylim=self.phi_range)
        im = NonUniformImage(ax, interpolation='bilinear')
        xcenters = (  self.x_edges[:-1] +   self.x_edges[1:]) / 2
        ycenters = (self.phi_edges[:-1] + self.phi_edges[1:]) / 2
        im.set_data(xcenters, ycenters, histogram.T)
        ax.add_image(im)

        # Decorate the plot
        self._decorate_histogram_plot()

        return

    def _plot_histogram_3d(
        self,
        histogram: np.ndarray,
    ):
        ''' Plot the histogram in 3D '''
        ## 3D figure
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        x_vals =  self.x_edges[:-1] + (self.x_edges[1] - self.x_edges[0]) / 2
        y_vals = self.phi_edges[:-1] + (self.phi_edges[1] - self.phi_edges[0]) / 2

        mesh_x, mesh_Y = np.meshgrid(x_vals, y_vals)

        ax.plot_surface( mesh_x, mesh_Y, histogram.T, cmap='hot')

        # Labels
        ax.set_xlabel("FB distance (bl)", fontsize=14)
        ax.set_xticks(np.arange(self.x_range[0], self.x_range[1], step=1))
        ax.set_yticks(np.arange(self.phi_range[0], self.phi_range[1], step=np.pi))
        ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'])
        ax.set_ylabel("Phase difference $\\Delta \\Phi$", fontsize=14)
        ax.set_zlabel("Phase matching ratio", fontsize=14)
        ax.set_title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$", fontsize=16)

        return

    def plot_histogram(
        self,
        max_density: float,
    ):
        """Plot the current histogram as a heatmap."""

        # Normalize the histogram bt the total number of iterations
        total_data_points    = np.sum(self.histogram)
        normalized_histogram = self.histogram / total_data_points
        normalized_histogram = normalized_histogram / max_density

        normalized_histogram[:,  0]  *= 2.0
        normalized_histogram[:, -1] *= 2.0

        # Plot the histogram
        fig   , ax     = plt.subplots()
        mesh_x, mesh_Y = np.meshgrid(self.x_edges, self.phi_edges)
        pcm = ax.pcolormesh(
            mesh_x,
            mesh_Y,
            normalized_histogram.T,
            cmap    = 'hot',
            shading = 'auto',
            vmin    = 0.0,
            vmax    = 1.0,
        )
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Phase matching ratio", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Decorate the plot
        self._decorate_histogram_plot()

        ## NonUniformImage
        # self._plot_histogram_non_uniform_image(normalized_histogram)

        ## 3D figure
        # self._plot_histogram_3d(normalized_histogram)

        # Plot the line fitting the histogram
        empirical_fit = self._plot_line_fitting_histogram(normalized_histogram)
        # self._plot_line_fitting_histogram_theory(
        #     empirical_fit     = empirical_fit,
        #     theoretical_speed = True,
        # )

        self.figures_dict['histogram'] = fig

        return

    def _plot_line_fitting_histogram(
        self,
        normalized_histogram: np
    ):
        ''' Plot the line fitting the histogram '''

        peak_points = []
        dx          = (self.x_edges[1] - self.x_edges[0])
        dphi        = (self.phi_edges[1] - self.phi_edges[0])
        phi_values  = self.phi_edges[:-1] + dphi / 2

        for i in range(len(self.x_edges) - 1):
            x_value  = self.x_edges[i] + dx / 2
            counts   = normalized_histogram[i, :]

            # Find local maxima
            peaks, _ = find_peaks(
                counts,
                distance   = self.phi_bins // 10,
                prominence = 0.1,
            )

            if not peaks.size:
                continue

            # Normalize the phase values
            phi_max_values                          = phi_values[peaks]
            phi_max_values                          = np.mod(phi_max_values, 2*np.pi)
            phi_max_values[phi_max_values > np.pi] -= 2 * np.pi

            # Extend previous phase values
            if peak_points:
                prev_max_value = peak_points[-1][1]
            else:
                prev_max_value = min( phi_values[peaks] )

            phi_max_values = [
                phi + 2 * np.pi * round((prev_max_value - phi) / (2 * np.pi))
                for phi in phi_max_values
            ]

            # Add the peak points
            peak_points.extend([ [x_value, phi] for phi in phi_max_values ])

        if not peak_points:
            return

        x_vals, phi_vals = zip(*peak_points)
        fit_params       = np.polyfit(x_vals, phi_vals, 1)
        fit_line         = np.poly1d(fit_params)

        # Evaluate quality of fit (pearson correlation)
        r_squared = r2_score(phi_vals, fit_line(x_vals))

        if r_squared < 0.8:
            print(f'Fitting line has R^2 = {r_squared} -- skipping plot')
            return

        for offset in [-2, -1, 0, +1, +2]:
            plt.plot(
                self.x_edges,
                fit_line(self.x_edges) + offset * 2 * np.pi,
                'w--',
                linewidth = 2
            )

        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.plot(phi_vals, x_vals)
        # ax.grid(True)

        return fit_params

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

            # Extrapolate (y = mx + c)
            slope     = (v1 - v0) / (f1 - f0)
            intercept = v0 - slope * f0
            speed     = slope * frequency + intercept

        # Li et al. 2020
        # PHI = ( 2 * pi * f / V ) * D + PHI_0

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

    def save_all_simulation_data(self):
        ''' Save the collected data '''

        # Save the parameters as csv (without pandas)
        with open(f'{self.save_folder}/{self.save_file_root}_parameters.csv', 'w') as f:
            for key, value in self.all_parameters.items():
                if isinstance(value, np.ndarray):
                    value = ','.join([str(val) for val in value])
                f.write(f'{key},{value}\n')

        # Save the histogram counts (convert to DataFrame and save as csv)
        x_edges   = ( self.x_edges + (self.x_edges[1] - self.x_edges[0]) / 2 )[:-1]
        phi_edges = ( self.phi_edges + (self.phi_edges[1] - self.phi_edges[0]) / 2 )[:-1]

        histogram_df = pd.DataFrame(
            self.histogram.T,
            columns = x_edges,
            index   = phi_edges,
        )

        histogram_df.to_csv(f'{self.save_folder}/{self.save_file_root}_histogram.csv')

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
            figure.savefig(f'{fig_path}.pdf', format='pdf')

        return

