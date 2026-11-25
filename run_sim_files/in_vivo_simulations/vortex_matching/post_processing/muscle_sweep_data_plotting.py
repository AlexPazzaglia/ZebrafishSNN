'''
Plotting of the results of the muscle stiffness sweep.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, CURRENTDIR)

import numpy as np
import matplotlib.pyplot as plt

import data_loading
import muscle_sweep_data_loading as data_loading_8

from data_plotting import BaseSchoolingPlotting

COM_POS_REL  = 0.4321805865303308
PHASE_OFFSET = 0.0 * np.pi

###############################################################################
# HISTOGRAM ###################################################################
###############################################################################

class SchoolingPlotting(BaseSchoolingPlotting):
    def __init__(
        self,
        results_root_folder: str,
        muscle_tag          : str,
        stim_a_off         : float,
        feedback_mode      : int,
        leader_signal_str  : str,
        spawn_x_vals       : np.ndarray,
        spawn_y_vals       : np.ndarray,
        np_random_seed_vals: np.ndarray,
        use_actual_com_pos : bool,
        simulation_tag     : str = None,
    ):
        """ Initialize an incremental 2D histogram."""

        # Parameters
        self.spawn_y_vals = spawn_y_vals
        self.n_spawn_y    = len(spawn_y_vals)
        self.muscle_tag   = muscle_tag

        if simulation_tag is None:
            simulation_tag = f'dynamic_water_vortices_free_head_muscle_FN_{muscle_tag}'

        super().__init__(
            results_root_folder = results_root_folder,
            stim_a_off          = stim_a_off,
            feedback_mode       = feedback_mode,
            leader_signal_str   = leader_signal_str,
            spawn_x_vals        = spawn_x_vals,
            np_random_seed_vals = np_random_seed_vals,
            use_actual_com_pos  = use_actual_com_pos,
            simulation_tag      = simulation_tag,
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
            'leader_signal_str': self.leader_signal_str,
            'feedback_mode'    : self.feedback_mode,
            'ps_tag'           : self.ps_tag,
        }

        self.all_parameters = self.process_parameters |{
            'spawn_x_vals'       : self.spawn_x_vals,
            'spawn_y_vals'       : self.spawn_y_vals,
            'np_random_seed_vals': self.np_random_seed_vals,
        }

        return

    def _get_processes_names(self):
        ''' Get the folder names for the simulations '''

        # Processes
        self.process_folders = [
            [
                data_loading_8.get_process_folder_from_parameters(
                    self.process_parameters | {
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
        self.save_file_root = data_loading_8.get_process_folder_nickname_from_parameters(
            parameters = self.process_parameters,
        )
        self.save_folder = f'{self.results_root_folder}/figures/{self.save_file_root}'

        os.makedirs(self.save_folder, exist_ok=True)

        return

    ###########################################################################
    # DATA COLLECTION ##########################################################
    ###########################################################################

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
        target_quantities = [
            'com_positions_diff',
            'joint_phases_diff',
            'coherence_pos_vel',
            'coherence_vel_vel',
            'coherence_acc_vel',
            'propulsion_fx',
            'propulsion_fy',
        ]

        data_df, performance_dict = data_loading.load_results_single_simulation(
            results_root_folder = self.results_root_folder,
            folder_name         = folder_name,
            process_folder      = process_folder,
            target_quantities   = target_quantities,
        )

        ########################
        ### PERFORMANCE DATA ###
        ########################

        # Add performance data
        n_start = len( self.times[self.times < 5.0] )

        coherence_pos_vel = data_df['coherence_pos_vel'][n_start:].mean()
        coherence_vel_vel = data_df['coherence_vel_vel'][n_start:].mean()
        coherence_acc_vel = data_df['coherence_acc_vel'][n_start:].mean()
        propulsion_fx     = data_df['propulsion_fx'][n_start:].mean()
        propulsion_fy     = data_df['propulsion_fy'][n_start:].mean()

        performance_dict : dict = performance_dict | {
            'coherence_pos_vel' : np.array([coherence_pos_vel]),
            'coherence_vel_vel' : np.array([coherence_vel_vel]),
            'coherence_acc_vel' : np.array([coherence_acc_vel]),
            'propulsion_fx'     : np.array([propulsion_fx]),
            'propulsion_fy'     : np.array([propulsion_fy]),
        }

        # Store performance data
        for key in performance_dict.keys():
            if key not in self.performance_data:
                self.performance_data[key] = []
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

    def collect_all_simulation_com_vs_angle_data(self):
        """ Collect com vs angle data from all simulations """

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

        vals_xys_list = [
            [
                self.spawn_x_vals[spawn_x_ind],
                self.spawn_y_vals[spawn_y_ind],
                self.np_random_seed_vals[seed_ind]
            ]
            for spawn_x_ind in range(self.n_spawn_x)
            for spawn_y_ind in range(self.n_spawn_y)
            for seed_ind in range(self.n_seeds)
        ]

        # Load all simulation data
        data_com_vs_angle_all = self._collect_all_simulation_com_vs_angle_data(
            folder_names_list    = folder_names_list,
            process_folders_list = process_folders_list,
            vals_xys_list        = vals_xys_list,
        )

        return data_com_vs_angle_all

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
            # 'mech_torque_total',
            # 'mech_torque_active',
            'mech_energy_total',
            # 'mech_energy_active',
            'mech_power_total',
            # 'mech_power_active',
            # 'mech_tail_beat_amp_bl',
            # 'coherence_pos_vel',
            'coherence_vel_vel',
            # 'coherence_acc_vel',
            # 'propulsion_fx',
        ]

        for target_quantity in target_quantities:

            quantity_values_flat = np.array(self.performance_data[target_quantity])
            quantity_values = quantity_values_flat.reshape(
                (self.n_spawn_x, self.n_spawn_y, self.n_seeds)
            )

            # Mean and std across seeds
            mean_values = np.mean(quantity_values, axis=2)
            std_values  = np.std(quantity_values, axis=2)

            # Boxplot
            fig = plt.figure(figsize=(8, 6))
            plt.boxplot(quantity_values_flat)
            plt.title(f'{target_quantity} Distribution')
            plt.ylabel(f'{target_quantity}')

            self.figures_dict[f'{target_quantity} Boxplot'] = fig

            # Contour plot
            if len(self.spawn_x_vals) < 2 or len(self.spawn_y_vals) < 2:
                continue

            fig = plt.figure(figsize=(8, 6))
            plt.contourf(
                self.spawn_x_vals,
                self.spawn_y_vals,
                mean_values.T,
                levels = 20,
                cmap   = 'viridis',
            )
            plt.colorbar(label=target_quantity)

            plt.xlim(self.x_range)
            plt.ylim(y_range)

            plt.title(f'{target_quantity} vs FB position')
            plt.xlabel('FB distance (bl)')
            plt.ylabel('FB height (bl)')

            self.figures_dict[f'{target_quantity} Contour'] = fig

        return

    def plot_histogram(self):
        """Plot the current histogram as a heatmap."""

        # Normalization factor
        max_density = 0.10

        super().plot_histogram(max_density = max_density)

        return


###############################################################################
# DATA COLLECTION AND SAVING ÙÙ################################################
###############################################################################

def collect_all_simulation_com_vs_angle_data(
    results_root_folder: str,
    muscle_tag         : str,
    stim_a_off         : float,
    feedback_mode      : int,
    leader_signal_str  : str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    use_actual_com_pos : bool = False,
):
    ''' Plot the histogram for a given set of parameters '''

    # Initialize the incremental 2D histogram
    plotting = SchoolingPlotting(
        results_root_folder = results_root_folder,
        muscle_tag          = muscle_tag,
        stim_a_off          = stim_a_off,
        feedback_mode       = feedback_mode,
        leader_signal_str   = leader_signal_str,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
    )

    # Collect all simulation data
    plotting.collect_all_simulation_com_vs_angle_data()

    return

###############################################################################
# PLOT PARAMETERS COMBINATIONS ################################################
###############################################################################

def plot_parameters_combination(
    results_root_folder: str,
    muscle_tag         : str,
    stim_a_off         : float,
    feedback_mode      : int,
    leader_signal_str  : str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    use_actual_com_pos : bool = False,
    save               : bool = True,
    show               : bool = False,
    simulation_tag     : str = None,
):
    ''' Plot the histogram for a given set of parameters '''

    # Initialize the incremental 2D histogram
    plotting = SchoolingPlotting(
        results_root_folder = results_root_folder,
        muscle_tag          = muscle_tag,
        stim_a_off          = stim_a_off,
        feedback_mode       = feedback_mode,
        leader_signal_str   = leader_signal_str,
        spawn_x_vals        = spawn_x_vals,
        spawn_y_vals        = spawn_y_vals,
        np_random_seed_vals = np_random_seed_vals,
        use_actual_com_pos  = use_actual_com_pos,
        simulation_tag      = simulation_tag,
    )

    # Collect all simulation data
    plotting.collect_all_simulation_data()

    # Save collected data
    plotting.save_all_simulation_data()

    # Plot
    plotting.plots(performance=True)

    # Save
    if save:
        plotting.save_plots()

    # Show the plot
    if show:
        plt.show()

    return