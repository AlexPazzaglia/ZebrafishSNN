import os
import numpy as np
import matplotlib.pyplot as plt

import data_loading_3

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
        leader_pars        : list[float, str],
        np_random_seed_vals: np.ndarray,
        use_actual_com_pos : bool,
        use_actual_phases  : bool,
        spawn_x_vals       : np.ndarray,
        spawn_y_vals       : np.ndarray,
    ):
        """ Initialize an incremental 2D histogram."""

        # Parameters
        self.leader_name  = leader_pars[1]
        self.spawn_y_vals = spawn_y_vals
        self.n_spawn_y    = len(spawn_y_vals)

        super().__init__(
            results_root_folder = results_root_folder,
            feedback_mode       = feedback_mode,
            leader_freq         = leader_pars[0],
            spawn_x_vals        = spawn_x_vals,
            np_random_seed_vals = np_random_seed_vals,
            use_actual_com_pos  = use_actual_com_pos,
            use_actual_phases   = use_actual_phases,
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
            'leader_name'  : self.leader_name,
            'feedback_mode': self.feedback_mode,
            'ps_tag'       : self.ps_tag,
        }

        self.all_parameters = self.process_parameters |{
            'spawn_x_vals'       : self.spawn_x_vals,
            'spawn_y_vals'       : self.spawn_y_vals,
            'np_random_seed_vals': self.np_random_seed_vals,
        }

        return

    def _get_processes_names(self):
        ''' Get the folder names for the simulations '''

        process_pars = self.process_parameters

        # Processes
        self.process_folders = [
            [
                data_loading_3.get_process_folder_from_parameters(
                    process_pars | {
                        'spawn_x'        : spawn_x,
                        'spawn_y'        : spawn_y,
                        'np_random_seed' : seed,
                    }
                )
                for seed in self.np_random_seed_vals
            ]
            for spawn_x in self.spawn_x_vals
            for spawn_y in self.spawn_y_vals
        ]

        # Save folder and file
        self.save_file_root = data_loading_3.get_process_folder_nickname_from_parameters(process_pars)
        self.save_folder    = f'{self.results_root_folder}/figures/{self.save_file_root}'

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
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)
        ]

        process_folders_list = [
            self.process_folders[spawn_x_ind * self.n_spawn_y + spawn_y_ind][seed_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)
        ]

        spawn_x_vals_list = [
            self.spawn_x_vals[spawn_x_ind]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
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

        y_range = (
            np.amin( self.spawn_y_vals ),
            np.amax( self.spawn_y_vals ),
        )

        target_quantities = [
            [ 'mech_energy', [2.5*1e-6, 4.0*1e-6] ],
            [ 'mech_torque', [3.8*1e-3, 4.0*1e-3] ],
        ]

        for target_pars in target_quantities:

            target_quantity, target_range = target_pars

            fig = plt.figure(figsize=(8, 6))

            quantity_values = np.array(self.performance_data[target_quantity])
            quantity_values = quantity_values.reshape(
                (self.n_spawn_x, self.n_spawn_y, self.n_seeds)
            )

            # Mean and std across seeds
            mean_values = np.mean(quantity_values, axis=2)
            # std_values  = np.std(quantity_values, axis=2)

            plt.contourf(
                self.spawn_x_vals,
                self.spawn_y_vals,
                mean_values.T,
                levels = 20,
                cmap   = 'viridis',
                vmin   = target_range[0],
                vmax   = target_range[1],
            )
            plt.colorbar(label=target_quantity)

            plt.xlim(self.x_range)
            plt.ylim(y_range)

            plt.title(f'{target_quantity} vs FB position')
            plt.xlabel('FB distance (bl)')
            plt.ylabel('FB height (bl)')
            plt.ylabel(target_quantity)

            self.figures_dict[target_quantity] = fig

        return

    def plot_histogram(self):
        """Plot the current histogram as a heatmap."""

        # Normalization factor
        total_data_points = np.sum(self.histogram)
        max_density       = ( 0.10 *  self.n_seeds * self.n_iter_cons * self.n_spawn_y ) / total_data_points

        super().plot_histogram(max_density = max_density)

        return


###############################################################################
# PLOT PARAMETERS COMBINATIONS ################################################
###############################################################################

def plot_parameters_combination(
    results_root_folder: str,
    feedback_mode      : int,
    leader_pars        : list[float, str],
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    use_actual_com_pos : bool,
    use_actual_phases  : bool,
    save               : bool = True,
    show               : bool = True,
):

    # Initialize the incremental 2D histogram
    plotting = SchoolingPlotting(
        results_root_folder = results_root_folder,
        feedback_mode       = feedback_mode,
        leader_pars         = leader_pars,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
        use_actual_phases   = use_actual_phases,
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