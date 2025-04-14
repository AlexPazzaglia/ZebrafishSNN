import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from typing import Tuple, Callable
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from farms_mujoco.swimming.drag import WaterProperties

from network_modules.vortices.yaml_operations import yaml2pyobject

###############################################################################
# WATER DYNAMICS ##############################################################
###############################################################################

class WaterDynamicsCallback(WaterProperties):

    def __init__(
        self,
        results_path            : str,
        invert_x                : bool = False,
        invert_y                : bool = False,
        translation             : np.ndarray = np.array([0.0, 0.0]),
        speed_mult              : np.ndarray = np.array([1.0, 1.0, 1.0]),
        speed_offset            : np.ndarray = np.array([0.0, 0.0, 0.0]),
        pos_offset_function     : Callable = None,
        pos_offset_function_args: Tuple = None,
        delay_start             : float = 0.0,
    ):
        super().__init__()
        self.results_path = results_path

        # Applied transformations
        # TODO: Add possibility to scale, rotate and translate the velocity field
        self.data_translation = translation
        self.data_invert_x    = invert_x
        self.data_invert_y    = invert_y
        self.speed_mult       = speed_mult
        self.speed_offset     = speed_offset

        if pos_offset_function is None:
            pos_offset_function = lambda time : np.zeros(3)

        if pos_offset_function_args is None:
            pos_offset_function_args = []

        self.pos_offset_function      = pos_offset_function
        self.pos_offset_function_args = pos_offset_function_args

        # Sim parameters
        sim_pars      = yaml2pyobject(f"{results_path}/parameters.yaml")
        self.sim_pars = sim_pars

        # Integration parameters
        self.time_step   = sim_pars["solver"]["dt"]
        self.iterations  = sim_pars["solver"]["nt"]
        self.grid_n      = sim_pars["solver"]["N"]
        self.delay_start = delay_start

        # Coordinates
        self.xmin   = sim_pars["solver"]["xmin"]
        self.xmax   = sim_pars["solver"]["xmax"]
        self.ymin   = sim_pars["solver"]["ymin"]
        self.ymax   = sim_pars["solver"]["ymax"]

        self._apply_limits_transformation()

        self.grid_dx = (self.xmax - self.xmin) / (self.grid_n - 1)
        self.grid_dy = (self.ymax - self.ymin) / (self.grid_n - 1)

        # Boundary conditions
        self.constant_vx = sim_pars['boundary_conditions']['BC_values_u'][0]
        self.constant_vy = sim_pars['boundary_conditions']['BC_values_v'][0]

        self._apply_boundary_conditions_transformation()

        self.x_vals = np.linspace(self.xmin, self.xmax, self.grid_n + 1)
        self.y_vals = np.linspace(self.ymin, self.ymax, self.grid_n + 1)

        self.grid_x = np.meshgrid(self.x_vals, self.y_vals)[0]
        self.grid_y = np.meshgrid(self.x_vals, self.y_vals)[1]

        # Saved fields
        self.iterations_saved = sim_pars["output"]["save_every"]

        # Last uploaded field
        self.last_loaded_iteration = None
        self.last_loaded_vx_field  = None
        self.last_loaded_vy_field  = None

        # Steady state
        self.steady_loop = None
        self.find_steady_state_loop()

        return


    ###########################################################################
    # CALLBACKS ###############################################################
    ###########################################################################

    def surface(self, t, x, y):
        """Surface"""
        return 0.0

    def density(self, t, x, y, z):
        """Density"""
        return 1000.0

    def viscosity(self, t, x, y, z):
        """Viscosity"""
        return 1.0

    def velocity(self, t, x, y, z):
        """Velocity in global frame"""
        return self._get_single_velocity_iteration(
            time  = t,
            x_pos = x,
            y_pos = y,
            z_pos = z,
        )

    ###########################################################################
    # TRANSFORMATIONS ########################################################
    ###########################################################################
    def _apply_limits_transformation(self):
        ''' Apply transformations to limits '''

        if self.data_invert_x:
            self.xmin, self.xmax = -self.xmax, -self.xmin
        if self.data_invert_y:
            self.ymin, self.ymax = -self.ymax, -self.ymin

        self.xmin += self.data_translation[0]
        self.xmax += self.data_translation[0]
        self.ymin += self.data_translation[1]
        self.ymax += self.data_translation[1]

        return

    def _apply_boundary_conditions_transformation(self):
        ''' Apply transformations to boundary conditions '''
        if self.data_invert_x:
            self.constant_vx = - self.constant_vx
        if self.data_invert_y:
            self.constant_vy = - self.constant_vy
        return

    def _apply_field_transformation(
        self,
        vx_field: np.ndarray,
        vy_field: np.ndarray,
    ):
        ''' Apply transformations to data '''
        if self.data_invert_x:
            vx_field = - np.flip(vx_field, axis=0)
            vy_field = + np.flip(vy_field, axis=0)
        if self.data_invert_y:
            vx_field = + np.flip(vx_field, axis=1)
            vy_field = - np.flip(vy_field, axis=1)
        return vx_field, vy_field

    ###########################################################################
    # DISTRIBUTIONS ###########################################################
    ###########################################################################
    def get_available_iterations(
        self,
        folder_name   = 'uv_field',
        quantity_name = 'u',
    ):
        ''' Get available iterations '''

        # Get available files
        file_list = os.listdir(f"{self.results_path}/{folder_name}")
        file_list = [ f for f in file_list if f.startswith(f'{quantity_name}_') ]

        # Get iterations
        save_frames = [ round( float( f.split("_")[-1].split(".")[0] )) for f in file_list ]
        save_frames = np.sort(save_frames)

        # Check
        save_skip   = self.sim_pars["output"]["save_every"]
        assert np.all( np.diff(save_frames) == save_skip )

        return save_frames

    def _get_grid_coordinates(
        self,
        x_pos       : float,
        y_pos       : float,
        x_pos_offset: float = 0.0,
        y_pos_offset: float = 0.0,
    ):
        ''' Get quantized coordinates '''

        # Get quantized coordinates
        xq = round( (x_pos - self.xmin - x_pos_offset) / self.grid_dx)
        yq = round( (y_pos - self.ymin - y_pos_offset) / self.grid_dy)

        # Handle out of bounds
        if xq < 0 or xq >= self.grid_n:
            xq = np.nan
        if yq < 0 or yq >= self.grid_n:
            yq = np.nan

        return xq, yq

    def _load_all_velocities_iteration(
        self,
        time            : float,
        use_steady_state: bool = True,
    ):
        ''' Load velocity field '''

        # Check if steady state is defined and reached
        if use_steady_state and self.steady_loop is not None:
            time0         = self.steady_loop['time0']
            loop_duration = self.steady_loop['duration']
            cycle_ratio   = (time - time0) / loop_duration
            cycle_frac    = cycle_ratio - np.floor(cycle_ratio)
            time_loop     = cycle_frac * loop_duration
            time          = time0 + time_loop

        # Get iteration
        iteration  = round(time / self.time_step)
        iteration  = iteration - (iteration % self.iterations_saved)
        field_path = f"{self.results_path}/uv_field"

        if iteration != self.last_loaded_iteration:

            # Load velocity field
            vx_field = np.load(f"{field_path}/u_{iteration}.npy")
            vy_field = np.load(f"{field_path}/v_{iteration}.npy")

            # Apply transformations
            vx_field, vy_field = self._apply_field_transformation(
                vx_field = vx_field,
                vy_field = vy_field,
            )

            # Update last uploaded field
            self.last_loaded_iteration = iteration
            self.last_loaded_vx_field  = vx_field
            self.last_loaded_vy_field  = vy_field

        return self.last_loaded_vx_field, self.last_loaded_vy_field

    def _get_single_velocity_iteration(
        self,
        time,
        x_pos,
        y_pos,
        z_pos,
        vx_field        = None,
        vy_field        = None,
        remove_constant = False,
    ):
        ''' Get water velocity at a given position '''

        speed_mult         = self.speed_mult
        time               = time - self.delay_start
        constant_field     = np.zeros(3)
        constant_field[0]  = self.constant_vx * (1 - float(remove_constant))
        constant_field[1]  = self.constant_vy * (1 - float(remove_constant))
        constant_field    *= speed_mult
        constant_field    += self.speed_offset

        # Delayed start
        if time < 0.0:
            return constant_field

        # If vx and vy fields are not provided, load them
        if vx_field is None or vy_field is None:
            vx_field, vy_field = self._load_all_velocities_iteration(time)

        # Get position offset for the current time
        pos_offset = self.pos_offset_function(time, *self.pos_offset_function_args)

        # Get quantized coordinates
        xq, yq = self._get_grid_coordinates(
            x_pos        = x_pos,
            y_pos        = y_pos,
            x_pos_offset = pos_offset[0],
            y_pos_offset = pos_offset[1],
        )

        # Out of bounds
        if np.isnan(xq) or np.isnan(yq):
            return constant_field

        water_v     = np.zeros(3)
        water_v[0] += vx_field[xq, yq] - self.constant_vx * float(remove_constant)
        water_v[1] += vy_field[xq, yq] - self.constant_vy * float(remove_constant)
        water_v    *= speed_mult
        water_v    += self.speed_offset

        return water_v

    def _get_multiple_velocities_iteration(
        self,
        time  : float,
        xy_pos: np.ndarray,
    ):
        ''' Get velocity field values '''

        # Load velocity field for current time
        vx_field, vy_field = self._load_all_velocities_iteration(time)

        # Compute water speed field
        n_pos    = xy_pos.shape[0]
        water_vx = np.zeros(n_pos)
        water_vy = np.zeros(n_pos)

        for i, (x, y) in enumerate(xy_pos):
            water_vx[i], water_vy[i], _ = self._get_single_velocity_iteration(
                time            = time,
                x_pos           = x,
                y_pos           = y,
                z_pos           = 0,
                vx_field        = vx_field,
                vy_field        = vy_field,
                remove_constant = self.sim_pars['plot']['remove_constant'],
            )

        return water_vx, water_vy

    def _get_grid_velocities_iteration(
        self,
        time : float,
        xvals: np.ndarray,
        yvals: np.ndarray,
    ):
        ''' Get velocity field values '''

        # All xy positions
        n_speeds_x = len(xvals)
        n_speeds_y = len(yvals)
        xy_pos     = np.array([ (x, y) for x in xvals for y in yvals])

        water_vx, water_vy = self._get_multiple_velocities_iteration(
            time  = time,
            xy_pos= xy_pos,
        )

        # Reshape
        water_vx = water_vx.reshape((n_speeds_x, n_speeds_y))
        water_vy = water_vy.reshape((n_speeds_x, n_speeds_y))

        return water_vx, water_vy

    ###########################################################################
    # FIND STEADY STATE #######################################################
    ###########################################################################

    def find_steady_state_loop(
        self,
        dist_tolerance  : float = 1.0,
        search_interval : float = None,
        plot            : bool  = False,
        verbose         : bool  = False,
    ) -> FuncAnimation:
        ''' Animation of the trajectory '''

        # Upload parameters
        save_skip   = self.sim_pars["output"]["save_every"]
        save_step   = self.time_step * save_skip

        save_frames = self.get_available_iterations()
        saved_times = save_frames * self.time_step

        # Get reference velocity field
        reference_time = saved_times[-1]
        self._load_all_velocities_iteration(reference_time)

        reference_vx_field = np.copy(self.last_loaded_vx_field)
        reference_vy_field = np.copy(self.last_loaded_vy_field)

        if search_interval is None:
            search_interval = reference_time

        # Loop over frames
        search_times = saved_times[
            ( saved_times > ( reference_time - search_interval ) ) &
            ( saved_times < ( reference_time - save_step       ) )
        ]
        n_search     = len(search_times)
        v_diff_array = np.zeros(n_search)

        for ind, time in enumerate(search_times):
            self._load_all_velocities_iteration(time)

            # Get distance between current and reference field
            v_field_diff = np.sqrt(
                (self.last_loaded_vx_field - reference_vx_field)**2 +
                (self.last_loaded_vy_field - reference_vy_field)**2
            )

            # Sum of differences
            v_field_diff_sum  = np.sum(v_field_diff)
            v_diff_array[ind] = v_field_diff_sum

        # Get steady state time
        dist_min_ind = np.argmin(v_diff_array)
        dist_min_val = v_diff_array[dist_min_ind]

        # Plot sum of differences
        if plot:
            self.plot_sum_of_differences_periodicity(
                save_step    = save_step,
                search_times = search_times,
                v_diff_array = v_diff_array,
                n_search     = n_search,
                dist_min_ind = dist_min_ind,
            )

        # Check convergence
        def _log_message(message):
            if not verbose:
                return
            print(message)

        if dist_min_val > dist_tolerance:
            message = (
                "STEADY STATE NOT REACHED \n"
                f'MSE distance: {dist_min_val :.2e}'
            )
            _log_message(message)
            return None

        time0    = search_times[dist_min_ind]
        time1    = reference_time
        duration = time1 - time0

        self.steady_loop = {
            'time0'   : time0,
            'time1'   : time1,
            'duration': (time1 - time0),
            'ind0'    : round( time0 / self.time_step ),
            'ind1'    : round( time1 / self.time_step ),
            'length'  : round( duration / self.time_step ),
        }

        message = (
            "STEADY STATE REACHED \n"
            f"Start time  : {time0 :.2f} s\n"
            f"End time    : {time1 :.2f} s\n"
            f'MSE distance: {dist_min_val :.2e}'
        )
        _log_message(message)

        return self.steady_loop

    def fill_missing_saved_states(self):

        # Upload parameters
        save_skip   = self.sim_pars["output"]["save_every"]
        save_step   = self.time_step * save_skip
        save_frames = self.iterations // save_skip
        save_iters  = save_skip * np.arange(save_frames, dtype=int)

        # Check saved states function
        def _check_saved_states(
            dir_name  : str,
            files_root: str,
        ):
            ''' Names in the format "{files_root}_{iteration}.{extension} '''
            file_list = os.listdir(f"{self.results_path}/{dir_name}")
            file_list = [ f for f in file_list if f.startswith(files_root) ]
            iter_list = [ round( float( f.split("_")[-1].split(".")[0] )) for f in file_list ]

            return {
                "quantity" : files_root.rstrip('_'),
                "dir_path" : f"{self.results_path}/{dir_name}",
                "file_list": file_list,
                "iter_list": iter_list,
                "last_iter": max(iter_list),
                "n_states" : len(iter_list),
            }

        # Check saved states
        saved_states = {}

        for dir_name in os.listdir(self.results_path):

            if not os.path.isdir(f"{self.results_path}/{dir_name}"):
                continue

            # Velocity fields
            if dir_name == "uv_field":
                saved_states['u_field'] = _check_saved_states(dir_name, "u_")
                saved_states['v_field'] = _check_saved_states(dir_name, "v_")
                continue

            # Other directories
            saved_states[dir_name] = _check_saved_states(dir_name, dir_name)

        # Last iteration fully saved
        last_iter      = min( [ value['last_iter'] for value in saved_states.values() ] )
        missing_states = save_iters[ save_iters > last_iter ]
        steady_ind0    = self.steady_loop['ind0']
        steady_len     = self.steady_loop['length']

        # Copy states exploiting periodicity
        for field_name in saved_states.keys():

            quantity_name = saved_states[field_name]['quantity']
            dir_path      = saved_states[field_name]['dir_path']
            file_list     = saved_states[field_name]['file_list']

            # Add readme file specifying copied states
            with open(f"{dir_path}/README.txt", "w") as readme_file:
                readme_file.write(
                    "This directory contains the saved states copied from the steady state.\n"
                    "The states were copied to fill the missing iterations in the simulation.\n"
                    f"Steady state: {steady_ind0} - {steady_ind0 + steady_len}\n"
                    f"Missing states: {missing_states[0]} - {missing_states[-1]}\n"
                )

            for missing_iter in missing_states:
                closest_iter = steady_ind0 + (missing_iter - steady_ind0) % steady_len

                src_file_root = f"{quantity_name}_{closest_iter}"
                dst_file_root = f"{quantity_name}_{missing_iter}"

                src_file = [ f for f in file_list if f.startswith(src_file_root) ][0]
                file_ext = src_file.split(".")[-1]
                dst_file = f"{dst_file_root}.{file_ext}"

                # Copy file
                shutil.copyfile(f"{dir_path}/{src_file}", f"{dir_path}/{dst_file}")

        return

    ###########################################################################
    # PLOTTING FUNCTIONS ######################################################
    ###########################################################################

    def plot_velocity_amplitude(
        self,
        body_length: float = 1.0,
    ) -> FuncAnimation:
        ''' Plot the amplitude of the velocity field '''

        # Upload parameters
        save_frames    = self.get_available_iterations()
        saved_times    = save_frames * self.time_step

        # Loop over frames
        v_field_amp_sum = None
        n_times         = len(saved_times)

        for ind, time in enumerate(saved_times):
            self._load_all_velocities_iteration(time)

            if v_field_amp_sum is None:
                v_field_amp_sum = np.zeros_like(self.last_loaded_vx_field)

            v_field_amp_sum += np.sqrt(
                ( self.last_loaded_vx_field - self.constant_vx ) **2 +
                ( self.last_loaded_vy_field - self.constant_vy ) **2
            )

        # Get average amplitude
        v_field_amp_avg = v_field_amp_sum / n_times

        plt.figure(figsize=(8, 6))
        plt.contourf(
            self.grid_x / body_length,
            self.grid_y / body_length,
            v_field_amp_avg.T,
            cmap     = 'viridis',
            levels   = 100
        )
        plt.colorbar(label="Velocity Amplitude")
        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.title("Average Velocity Field Amplitude", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Define x-values for vertical cuts
        n_cuts     = 6
        y_vals     = self.y_vals / body_length
        x_vals     = self.x_vals / body_length
        x_cut_vals = np.linspace(0.0, 2.0, n_cuts)

        plt.figure(figsize=(8, 6))
        colors = cm.viridis(np.linspace(0, 1, n_cuts))
        for cut_ind, x_val in enumerate(x_cut_vals):

            x_ind     = np.argmin( np.abs(x_vals - x_val) )
            v_profile = v_field_amp_avg[x_ind]
            plt.plot(
            y_vals,
            v_profile,
            color = colors[cut_ind],
            label = f"x = { x_vals[x_ind] :.2f}"
            )

        plt.xlabel("Y Coordinate", fontsize=14)
        plt.ylabel("Velocity Amplitude", fontsize=14)
        plt.title("Vertical Cuts of Velocity Field Amplitude", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        return

    def plot_sum_of_differences_periodicity(
        self,
        save_step    : float,
        search_times : np.ndarray,
        v_diff_array : np.ndarray,
        n_search     : int,
        dist_min_ind : int,
    ):
        ''' Plot sum of differences between periodic states '''

        # Fourier transform of v_diff_array
        v_diff_fft  = np.fft.fft( v_diff_array - np.mean(v_diff_array) )
        v_diff_fft  = np.abs(v_diff_fft)
        v_diff_fft  = v_diff_fft[:n_search//2]
        v_diff_freq = np.fft.fftfreq(n_search, d=save_step)
        v_diff_freq = v_diff_freq[:n_search//2]

        max_freq_ind = np.argmax(v_diff_fft)
        max_fft      = v_diff_fft[max_freq_ind]
        max_freq     = v_diff_freq[max_freq_ind]

        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(search_times, v_diff_array)
        ax[0].plot(search_times[:dist_min_ind],v_diff_array[:dist_min_ind],'r')
        ax[0].set_xlim([search_times[0], search_times[-1]])
        ax[0].set_ylim([0, 1.1 * np.max(v_diff_array)])
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Sum of differences')
        ax[0].set_title('Sum of differences between velocity fields')

        ax[1].plot(v_diff_freq, v_diff_fft)
        ax[1].plot(max_freq, max_fft, 'ro')
        ax[1].plot([max_freq, max_freq], [0, max_fft], 'r--')
        ax[1].text(
            max_freq + 5 * (v_diff_freq[1] - v_diff_freq[0]),
            max_fft,
            f"{max_freq:.2f} Hz", fontsize=12
        )
        ax[1].set_xlim(v_diff_freq[0], v_diff_freq[-1])
        ax[1].set_ylim(0, 1.1 * max_fft)
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_title('Fourier transform of sum of differences')

        return

    def plot_velocity_component_field(
        self,
        xvals_dyn    : np.ndarray,
        yvals_dyn    : np.ndarray,
        v_field      : np.ndarray,
    ):
        ''' Plot velocity field '''
        dx = ( xvals_dyn[1]-xvals_dyn[0] ) / 2.0
        dy = ( yvals_dyn[1]-yvals_dyn[0] ) / 2.0
        extent = [
            xvals_dyn[ 0] - dx,
            xvals_dyn[-1] + dx,
            yvals_dyn[ 0] - dy,
            yvals_dyn[-1] + dy,
        ]
        plt.imshow(v_field.T, extent=extent, origin='lower')
        return

    def _plot_velocity_vector_field(
        self,
        axis         : plt.Axes,
        time         : float,
        n_speeds_dyn : int = 100,
    ):
        ''' Plot water dynamics '''

        # Compute limits
        xvals_dyn  = np.linspace(self.xmin, self.xmax, n_speeds_dyn)
        yvals_dyn  = np.linspace(self.ymin, self.ymax, n_speeds_dyn)

        # Get velocity field values
        (
            water_vx,
            water_vy,
        ) = self._get_grid_velocities_iteration(
            time  = time,
            xvals = xvals_dyn,
            yvals = yvals_dyn,
        )

        # Plot water speed field as a quiver plot
        quiver_plot = axis.quiver(
            xvals_dyn,
            yvals_dyn,
            water_vx.T,
            water_vy.T,
            alpha = 0.5,
            pivot = 'middle',
            units = 'xy',
            scale = 20.0,
        )
        return quiver_plot

    def _update_plot_velocity_vector_field(
        self,
        quiver_plot: plt.quiver,
        time       : float,
    ):
        ''' Plot water dynamics '''

        # Get x vals from quiver plot
        xvals_dyn  = np.sort(np.unique(quiver_plot.X))
        yvals_dyn  = np.sort(np.unique(quiver_plot.Y))

        # Get velocity field values
        (
            water_vx,
            water_vy,
        ) = self._get_grid_velocities_iteration(
            time     = time,
            xvals    = xvals_dyn,
            yvals    = yvals_dyn,
        )

        # Update water speed field as a quiver plot
        quiver_plot.set_UVC(
            water_vx.T,
            water_vy.T,
        )

        return quiver_plot

    def animate_velocity_vector_field(
        self,
        fig             : Figure,
        anim_step_jump  : int = 1,
        remove_constant : bool = False,
    ) -> FuncAnimation:
        ''' Animation of the trajectory '''

        # Upload parameters
        self.sim_pars['plot'] = { "remove_constant": remove_constant }

        save_skip   = self.sim_pars["output"]["save_every"]
        save_step   = self.time_step * save_skip
        save_frames = self.iterations // save_skip
        save_times  = np.arange(save_frames) * save_step

        anim_steps = np.arange( 0, save_frames, anim_step_jump, dtype= int )

        # Initialize plot
        ax1 = plt.axes()
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.axis('equal')
        ax1.grid(True)
        ax1.set_title('Trajectory')

        time_text = ax1.text(0.02, 0.05, '', transform=ax1.transAxes)

        # Draw water dynamics if available
        quiver_plot = self._plot_velocity_vector_field(
            axis            = ax1,
            time            = save_times[0],
            n_speeds_dyn    = 100,
        )

        # Define animation
        def _animation_step(
            anim_step  : int,
            time_text  : plt.text,
            quiver_plot: plt.quiver,
        ) -> Tuple[plt.Axes, str, plt.quiver]:
            ''' Update animation '''

            # Update time text
            time_text.set_text(f"time: {save_times[anim_step] :.2f} s")

            # Update velocity field
            self._update_plot_velocity_vector_field(
                quiver_plot = quiver_plot,
                time        = save_times[anim_step],
            )
            return (ax1, time_text, quiver_plot)

        anim = FuncAnimation(
            fig,
            _animation_step,
            frames   = anim_steps,
            interval = anim_step_jump,
            blit     = False,
            fargs    = (time_text, quiver_plot,)
        )

        return anim

def main():

    results_path = (
        "/data/pazzagli/simulation_results/fluid_solver/"
        "saved_simulations/"
        # "10s_amp_100_300Hz_abdquickest_fast_speed_sketch_body_2025-03-12T15:18:42.576709"
        "30s_continuous_amp_100_constant_034Hz_abdquickest_exp_12104_12264_scaled_025_bounded_NaN_NaN_Hz_fast_speed_sketch_body_2025-03-11T15:12:31.648789"
    )

    # Water dynamics
    water_dynamics = WaterDynamicsCallback(
        results_path = results_path,
        invert_x     = False,
        invert_y     = False,
        translation  = np.array([0.0, 0.0]),
    )

    # Find steady state
    water_dynamics.find_steady_state_loop(
        search_interval = None,
        plot            = True,
        verbose         = True,
    )

    # Plot velocity amplitude
    water_dynamics.plot_velocity_amplitude(
        body_length = 0.018,
    )

    # Animate velocity field
    # fig  = plt.figure()
    # anim = water_dynamics.animate_velocity_vector_field(
    #     fig             = fig,
    #     anim_step_jump  = 10,
    #     remove_constant = False,
    # )

    plt.show()


if __name__ == "__main__":
    main()
