import os
import numpy as np
import matplotlib.pyplot as plt

import data_loading_1

from data_plotting import BaseSchoolingPlotting

COM_POS_REL = 0.4321805865303308

###############################################################################
# HISTOGRAM ###################################################################
###############################################################################

class SchoolingPlotting(BaseSchoolingPlotting):
    def __init__(
        self,
        results_root_folder: str,
        feedback_mode      : int,
        leader_freq        : float,
        np_random_seed_vals: np.ndarray,
        use_actual_com_pos : bool,
        spawn_x_vals       : np.ndarray,
        speed_mult_y       : float,
    ):
        """ Initialize an incremental 2D histogram."""

        # Parameters
        self.speed_mult_y = speed_mult_y

        super().__init__(
            results_root_folder = results_root_folder,
            feedback_mode       = feedback_mode,
            leader_freq         = leader_freq,
            spawn_x_vals        = spawn_x_vals,
            np_random_seed_vals = np_random_seed_vals,
            use_actual_com_pos  = use_actual_com_pos,
        )

        # Processes names
        self._get_processes_names()

        return

    ###########################################################################
    # PARAMETERS ##############################################################
    ###########################################################################

    def _define_parameters(self):
        ''' Define the parameters for the simulation '''

        self.process_parameters = {
            'leader_freq'  : self.leader_freq,
            'feedback_mode': self.feedback_mode,
            'ps_tag'       : self.ps_tag,
            'speed_mult_y' : self.speed_mult_y,
        }

        self.all_parameters = self.process_parameters |{
            'spawn_x_vals'       : self.spawn_x_vals,
            'np_random_seed_vals': self.np_random_seed_vals,
        }

        return

    def _get_processes_names(self):
        ''' Get the folder names for the simulations '''

        # Processes
        self.process_folders = [
            [
                data_loading_1.get_process_folder_from_parameters(
                    self.process_parameters | {
                        'spawn_x'        : spawn_x,
                        'np_random_seed' : seed,
                    }
                )
                for seed in self.np_random_seed_vals
            ]
            for spawn_x in self.spawn_x_vals
        ]

        # Save folder and file
        self.save_file_root = data_loading_1.get_process_folder_nickname_from_parameters(
            parameters = self.process_parameters,
        )
        self.save_folder = f'{self.results_root_folder}/figures/{self.save_file_root}'

        os.makedirs(self.save_folder, exist_ok=True)

        return

    ###########################################################################
    # DATA COLLECTION ##########################################################
    ###########################################################################

    def collect_all_simulation_data(self):
        """ Collect data from all simulations and update the histogram. """

        # List of all files
        folder_names_list = [
            self.folder_names[seed_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for seed_ind in range(self.n_seeds)
        ]

        process_folders_list = [
            self.process_folders[spawn_x_ind][seed_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for seed_ind in range(self.n_seeds)
        ]

        spawn_x_vals_list = [
            self.spawn_x_vals[spawn_x_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for seed_ind in range(self.n_seeds)
        ]

        # Load all simulation data
        self._collect_all_simulation_data(
            folder_names_list    = folder_names_list,
            process_folders_list = process_folders_list,
            spawn_x_vals_list    = spawn_x_vals_list,
        )

        return

    ###########################################################################
    # PLOTS ####################################################################
    ###########################################################################

    def plot_performance(self):
        ''' Plot the performance data '''

        target_quantities = [
            'mech_energy',
            'mech_torque',
        ]

        for target_quantity in target_quantities:
            fig = plt.figure(figsize=(8, 6))

            quantity_values = np.array(
                [
                    performance_dict[target_quantity][0]
                    for performance_dict in self.performance_dict_list
                ]
            ).reshape(self.n_spawn_x, self.n_seeds)

            # Mean and std across seeds
            mean_values = np.mean(quantity_values, axis=1)
            std_values  = np.std(quantity_values, axis=1)

            plt.plot(
                self.spawn_x_vals,
                mean_values,
                color     = 'blue',
                linewidth = 2,
                label     = 'Mean'
            )
            plt.fill_between(
                self.spawn_x_vals,
                mean_values - std_values,
                mean_values + std_values,
                color = 'blue',
                alpha = 0.2,
                label = 'Std'
            )

            plt.xlim(self.x_range)
            plt.title(f'{target_quantity} vs FB distance')
            plt.xlabel('FB distance (bl)')
            plt.ylabel(target_quantity)

            plt.legend()

            self.figures_dict[target_quantity] = fig

        return

    def plot_histogram(self):
        """Plot the current histogram as a heatmap."""

        # Normalization factor
        total_data_points = np.sum(self.histogram)
        max_density       = ( 0.25 *  self.n_seeds * self.n_iter_cons ) / total_data_points

        super().plot_histogram(max_density = max_density)

        return



###############################################################################
# PLOT PARAMETERS COMBINATIONS ################################################
###############################################################################

def plot_parameters_combination(
    results_root_folder: str,
    feedback_mode      : int,
    leader_freq        : float,
    speed_mult_y       : float,
    spawn_x_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    use_actual_com_pos : bool = True,
    save               : bool = True,
    show               : bool = True,
):
    ''' Plot the histogram for a given set of parameters '''

    # Initialize the incremental 2D histogram
    plotting = SchoolingPlotting(
        results_root_folder = results_root_folder,
        feedback_mode       = feedback_mode,
        leader_freq         = leader_freq,
        speed_mult_y        = speed_mult_y,
        spawn_x_vals        = spawn_x_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
    )

    # Collect all simulation data
    plotting.collect_all_simulation_data()

    # Save collected data
    plotting.save_all_simulation_data()

    # Plot
    plotting.plots()

    # Save
    if save:
        plotting.save_plots()

    # Show the plot
    if show:
        plt.show()

    return