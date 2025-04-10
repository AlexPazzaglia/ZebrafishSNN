'''
Plotting of the mechanical simulation
'''

import os
import time

import numpy as np
import pandas as pd
import network_modules.plotting.plots_mech as mech_plt

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.animation import FuncAnimation

from scipy.signal import hilbert, find_peaks

from network_modules.performance.mechanics_performance import MechPerformance
from network_modules.plotting import plots_utils
from farms_mujoco.sensors.camera import save_video

from network_modules.performance import signal_processor_mech_plot

# FIGURE PARAMETERS
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.rc(  'font', size      = SMALL_SIZE )  # controls default text sizes
plt.rc(  'axes', titlesize = SMALL_SIZE )  # fontsize of the axes title
plt.rc(  'axes', labelsize = BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc( 'xtick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc( 'ytick', labelsize = MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize  = SMALL_SIZE )  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

class MechPlotting(MechPerformance):
    '''
    Class used plot the results of mechanical simulations
    '''

    # POST PROCESSING
    def simulation_post_processing(self, load_from_file = False) -> dict[str, float]:
        '''
        Post-processing for the mechanical simulation
        Optionally save the video of the simulation
        '''
        mech_metrics = super().simulation_post_processing(load_from_file=load_from_file)

        # Saving video
        if self.mech_sim_options.video:
            os.makedirs(self.results_data_folder, exist_ok=True)
            save_video(
                camera      = self.camera,
                video_path  = f'{self.results_data_folder}/{self.mech_sim_options.video_name}',
                iteration   = self.sim.iteration,
            )

        return mech_metrics

    # ALL RESULTS
    def simulation_plots(
        self,
        figures_dict   = None,
        load_from_file = False,
    ) -> None:
        ''' Plots showing the network's behavior '''

        # Change matplotlib logging level to avoid undesired messages
        plt.set_loglevel("warning")

        # Plot
        self._plot_farms_simulation(figures_dict, load_from_file)

    # ------ [ PLOTTING ] ------
    def _plot_farms_simulation(
        self,
        figures_dict   : dict = None,
        load_from_file : bool = False,
    ) -> None:
        '''
        When included, plot the movemente generated in the farms model
        '''
        if not self.snn_network.params.monitor.farms_simulation['active']:
            return

        self.load_mechanical_simulation_data(load_from_file)

        figures_dict = self.snn_network.figures if figures_dict is None else figures_dict

        # Plot links y position
        axis_length = self.snn_network.params.mechanics.mech_axial_length
        links_pos   = self.all_metrics_data['links_positions']
        links_y_pos = links_pos[:, :, 1] / axis_length
        colors      = plt.cm.jet(np.linspace(0, 1, links_pos.shape[1]))

        fig_links_y = plt.figure('Links y position')
        axis        = fig_links_y.add_subplot(111)

        for i, color in enumerate(colors):

            axis.plot(
                links_y_pos[:, i],
                color = color,
            )

        axis.set_xlabel('Time [s]')
        axis.set_ylabel('Position [BL]')
        axis.set_title('Links y position')

        figures_dict['fig_links_y'] = fig_links_y

        # Plots
        self._plot_joints_angles_amps(figures_dict)
        self._plot_links_disp_amps(figures_dict)
        self._plot_joints_angles(figures_dict)
        self._plot_joints_velocities(figures_dict)
        self._plot_com_trajectory(figures_dict)
        self._plot_leader_vs_follower_schooling_data(figures_dict)
        self._plot_fitted_trajectory(figures_dict)

        # Animations
        self._animate_links_trajectory(figures_dict)

    # \----- [ Mechanical simulation ] ------

    # ------ [ Plots ] ------
    def _plot_joints_angles_amps(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles amplitudes '''

        plotpars : dict = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars.get('joints_angle_amps'):
            return

        metrics     = self.all_metrics_data['mech_metrics']
        mech_pars   = self.snn_network.params.mechanics
        length_body = mech_pars.mech_axial_length

        joints_angle_amps     = metrics['mech_joints_disp_amp']
        joints_angle_amps_deg = np.rad2deg(joints_angle_amps)

        joints_axial_pos    = np.array(mech_pars.mech_axial_joints_position)
        joints_axial_pos_bl = joints_axial_pos / length_body

        fig_joints_angle_amp = plt.figure('Joint angles amplitudes')
        axis = fig_joints_angle_amp.add_subplot(111)

        axis.plot(joints_axial_pos_bl, joints_angle_amps_deg, 'o-')

        axis.set_xlabel('Joint Position [BL]')
        axis.set_ylabel('Joint Angle Amplitude [deg]')
        axis.set_title('Joint Angle Amplitudes')
        axis.set_xlim([0, 1])

        figures_dict['fig_joints_angle_amp'] = fig_joints_angle_amp

        return

    def _plot_links_disp_amps(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot links displacements amplitudes '''

        plotpars : dict = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars.get('links_disp_amps'):
            return

        metrics     = self.all_metrics_data['mech_metrics']
        mech_pars   = self.snn_network.params.mechanics
        length_body = mech_pars.mech_axial_length

        links_disp_amps_bl = metrics['mech_links_disp_amp_bl']

        joints_axial_pos    = np.array(mech_pars.mech_axial_joints_position)
        points_axial_pos    = np.array([0] + joints_axial_pos.tolist() + [length_body])
        points_axial_pos_bl = points_axial_pos / length_body

        fig_links_disp_amp = plt.figure('Links displacements amplitudes')
        axis = fig_links_disp_amp.add_subplot(111)

        axis.plot(points_axial_pos_bl, links_disp_amps_bl, 'o-')

        axis.set_xlabel('Point Position [BL]')
        axis.set_ylabel('Point displacement [BL]')
        axis.set_title('Points displacements')
        axis.set_xlim([0, 1])

        figures_dict['fig_links_disp_amp'] = fig_links_disp_amp

        return

    def _plot_joints_angles(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['joint_angles']:
            return

        joints_positions = np.array( self.sim_res_data.sensors.joints.positions_all() )

        if not joints_positions[-1].any():
            joints_positions  = joints_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = joints_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        fig_ja_dict = mech_plt.plot_joints_signals(
            times         = times,
            joints_angles = np.rad2deg( joints_positions ),
            params        = self.sim_res_pars,
            fig_name      = 'fig_ja',
            signal_name   = 'Joint angles',
        )
        figures_dict.update(fig_ja_dict)

        return

    def _plot_joints_velocities(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['joint_velocities']:
            return

        joints_velocities = np.array( self.sim_res_data.sensors.joints.velocities_all() )

        if not joints_velocities[-1].any():
            joints_velocities  = joints_velocities[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = joints_velocities.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        fig_ja_dict = mech_plt.plot_joints_signals(
            times         = times,
            joints_angles = np.rad2deg( joints_velocities ),
            params        = self.sim_res_pars,
            fig_name      = 'fig_jv',
            signal_name   = 'Joint velocities',
        )
        figures_dict.update(fig_ja_dict)

        return

    def _plot_com_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot COM trajectory '''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        com_pars = plotpars['com_trajectory']
        if not com_pars or not com_pars['showit']:
            return

        links_positions  = np.array( self.sim_res_data.sensors.links.urdf_positions() )

        if not links_positions[-1].any():
            links_positions  = links_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = links_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        axial_joints  = self.sim_res_pars['morphology']['n_joints_body']
        com_positions = np.mean( links_positions[:, :axial_joints], axis= 1 )

        water_parameters = {
            'water_maps'    : self.water_map_parameters,
            'water_dynamics': self.water_dynamics_options,
        }

        fig_ht = mech_plt.plot_com_trajectory(
            times            = times,
            com_positions    = com_positions,
            water_parameters = water_parameters,
            plot_pos_1D      = com_pars['pos_1D'],
            plot_pos_2D      = com_pars['pos_2D'],
            plot_vel_1D      = com_pars['vel_1D'],
        )
        figures_dict.update(fig_ht)

        return

    def _plot_fitted_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot trajectory'''

        plotpars = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars['trajectory_fit']:
            return

        disp_data = self.all_metrics_data['links_displacements_data']
        figures_dict['fig_traj_fit'] = signal_processor_mech_plot.plot_trajectory_fit(
            links_pos_xy               = disp_data['links_positions'],
            direction_fwd              = disp_data['direction_fwd'],
            direction_left             = disp_data['direction_left'],
            quadratic_fit_coefficients = disp_data['quadratic_fit_coefficients'],
        )

    ########################################################
    # COM POSITION VS JOINT PHASE RELATIONSHIP #############
    ########################################################

    def _plot_joint_angle_vs_ref_angle(
        self,
        joint_positions    : np.ndarray,
        joint_positions_ref: np.ndarray,
        n_iter_cons        : int,
        figures_dict       : dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        ## Angles-time plot
        fig_angle_vs_angle_1d = plt.figure('Joint angle vs Reference angle - 1D')
        axis               = fig_angle_vs_angle_1d.add_subplot(111)

        timestep = self.sim_res_data.timestep
        n_iter   = joint_positions.shape[0]
        times   = np.arange(n_iter - n_iter_cons, n_iter) * timestep
        t0, t1  = times[0], times[-1]

        angle_ref = joint_positions_ref[-n_iter_cons:]
        angle_exp = joint_positions[-n_iter_cons:]

        # Plot data
        axis.plot(times, angle_ref, label='Reference')
        axis.plot(times, angle_exp, label='Follower')

        # Decorate
        min_angle = np.amin([angle_ref,angle_exp])
        max_angle = np.amax([angle_ref,angle_exp])

        axis.set_xlim([t0, t1])
        axis.set_ylim([min_angle, max_angle])
        axis.set_xlabel('Time [s]')
        axis.set_ylabel('Joint Angle [rad]')
        axis.set_title('Joint Angle vs Reference Angle')
        axis.legend()

        figures_dict['fig_joint_angle_vs_ref_angle_1D'] = fig_angle_vs_angle_1d

        ## Angle-Angle plot
        fig_angle_vs_angle_2d = plt.figure('Joint angle vs Reference angle - 2D')
        axis                  = fig_angle_vs_angle_2d.add_subplot(111)

        # Plot data
        axis.plot(
            angle_ref,
            angle_exp,
        )

        # Plot diagonal line
        min_angle = np.amin(
            [
                angle_ref,
                angle_exp,
            ]
        )
        max_angle = np.amax(
            [
                angle_ref,
                angle_exp,
            ]
        )
        axis.plot([min_angle, max_angle], [min_angle, max_angle], 'k--')

        # Decorate
        axis.set_xlim([min_angle, max_angle])
        axis.set_ylim([min_angle, max_angle])
        axis.set_xlabel('Reference Angle [rad]')
        axis.set_ylabel('Joint Angle [rad]')
        axis.set_title('Joint Angle vs Reference Angle')

        figures_dict['fig_joint_angle_vs_ref_angle_2D'] = fig_angle_vs_angle_2d

        return

    def _animation_joint_angle_vs_ref_angle(
        self,
        joint_positions    : np.ndarray,
        joint_positions_ref: np.ndarray,
        timestep           : float,
        n_iterations       : int,
        save_video         : bool,
        steps_jump         : int = 20,
        memory_time        : float = 2.5,
        switch_time        : float = 5.0,
    ):
        ''' Plot joint angles evolution '''

        if not save_video:
            return

        # Plot diagonal line
        min_phase = np.amin( [ joint_positions_ref, joint_positions ] )
        max_phase = np.amax( [ joint_positions_ref, joint_positions ] )

        fig_angle_anim, ax_angle_anim = plt.subplots()
        ax_angle_anim.set_xlim([min_phase, max_phase])
        ax_angle_anim.set_ylim([min_phase, max_phase])
        ax_angle_anim.set_xlabel('Reference Angle [rad]')
        ax_angle_anim.set_ylabel('Joint Angle [rad]')
        ax_angle_anim.set_title('Joint Angle vs Reference Angle Animation')

        line_ol,  = ax_angle_anim.plot([], [], lw=2, color='grey')
        line_cl,  = ax_angle_anim.plot([], [], lw=2, color='red')
        fb_text   = ax_angle_anim.text(0.02, 0.95, '', transform=ax_angle_anim.transAxes)
        time_text = ax_angle_anim.text(0.02, 0.05, '', transform=ax_angle_anim.transAxes)

        memory_step = round(memory_time / timestep)
        switch_step = round(switch_time / timestep)

        def init():
            line_ol.set_data([], [])
            line_cl.set_data([], [])
            time_text.set_text(f"time: {0.0 :.1f} s")
            fb_text.set_text(f"Feedback OFF")
            fb_text.set_color('black')
            return line_ol, line_cl, time_text, fb_text

        def update(frame):

            curr_time = frame*timestep

            if curr_time < switch_time:
                line_ol.set_data(
                    joint_positions_ref[: frame],
                    joint_positions[: frame]
                )
            elif curr_time - switch_time < steps_jump * timestep:
                line_ol.set_data(
                    joint_positions_ref[: switch_step + 1],
                    joint_positions[: switch_step + 1]
                )
                line_cl.set_data(
                    joint_positions_ref[switch_step : frame],
                    joint_positions[switch_step : frame]
                )
                fb_text.set_text(f"Feedback ON")
                fb_text.set_color('red')
            else:
                step_min = max(switch_step, frame - memory_step)
                line_cl.set_data(
                    joint_positions_ref[step_min : frame],
                    joint_positions[step_min : frame]
                )

            time_text.set_text(f"time: {frame*timestep :.1f} s")

            return line_ol, line_cl, time_text, fb_text

        ani_angle = FuncAnimation(
            fig_angle_anim,
            update,
            frames    = np.arange(0, n_iterations, steps_jump),
            init_func = init,
            blit      = True,
            interval = 1,
        )

        # Save_video
        plots_utils._save_animation(
            figure      = fig_angle_anim,
            anim        = ani_angle,
            folder_path = self.results_data_folder,
            fig_label   = 'joint_angle_vs_ref_angle',
        )

        # Close the figure
        plt.close(fig_angle_anim)

        return

    def _plot_joint_phase_vs_ref_phase(
        self,
        joint_phases    : np.ndarray,
        joint_phases_ref: np.ndarray,
        times           : np.ndarray,
        n_iter_cons     : int,
        figures_dict    : dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        ### PHASES
        fig_phase_vs_phase = plt.figure('Joint Phase vs Reference Phase')
        axis               = fig_phase_vs_phase.add_subplot(111)

        joint_phases_ref_norm = np.mod(joint_phases_ref, 2*np.pi)
        joint_phases_norm     = np.mod(joint_phases, 2*np.pi)

        axis.plot(
            times[-n_iter_cons:],
            joint_phases_ref_norm[-n_iter_cons:],
            label = 'Reference Phase'
        )
        axis.plot(
            times[-n_iter_cons:],
            joint_phases_norm[-n_iter_cons:],
            label = 'Joint Phase'
        )

        # Decorate
        axis.set_xlabel('Time [s]')
        axis.set_ylabel('Phase [rad]')
        axis.set_title('Joint Phase vs Reference Phase')
        axis.legend()

        figures_dict['fig_joint_phase_vs_ref_phase'] = fig_phase_vs_phase

        ### PHASE DIFFERENCE
        fig_phase_diff = plt.figure('Joint Phase vs Reference Phase - Difference')
        diff_signal = ( joint_phases - joint_phases_ref ) / (2*np.pi)

        plt.plot(times, diff_signal, 'k-')
        plt.xlim(0, times[-1])
        plt.ylim(np.amin(diff_signal), np.amax(diff_signal))
        plt.xlabel('Time [s]')
        plt.ylabel('Phase Difference [cycles]')
        plt.title('Phase Difference Over Time')

        figures_dict['fig_joint_phase_vs_ref_phase_diff'] = fig_phase_diff

        return

    def _animation_joint_phase_vs_ref_phase(
        self,
        joint_phases    : np.ndarray,
        joint_phases_ref: np.ndarray,
        times           : np.ndarray,
        save_video      : bool,
        steps_jump      : int = 20,
        switch_time     : float = 5.0,
    ):
        ''' Plot joint angles evolution '''

        if not save_video:
            return

        fig_phase_anim, (ax_phase_anim, ax_phase_diff) = plt.subplots(1, 2, figsize=(12, 6))

        # Parameters
        timestep     = times[1] - times[0]
        n_iterations = len(times)

        ### CIRCLE PLOT
        ax_phase_anim.set_xlim(-1.5, 1.5)
        ax_phase_anim.set_ylim(-1.5, 1.5)
        ax_phase_anim.set_aspect('equal')
        ax_phase_anim.set_xlabel('cos(Phase)')
        ax_phase_anim.set_ylabel('sin(Phase)')
        ax_phase_anim.set_title('Joint Phase vs Reference Phase Animation')

        fb_text   = ax_phase_anim.text(0.02, 0.95, '', transform=ax_phase_anim.transAxes)
        time_text = ax_phase_anim.text(0.02, 0.05, '', transform=ax_phase_anim.transAxes)

        # Draw unit circle
        unit_circle = plt.Circle((0, 0), 1, color='black', fill=False)
        ax_phase_anim.add_artist(unit_circle)

        arc_diff = Arc((0, 0), 1, 1, theta1=0, theta2=0, color='red', linewidth=2)
        ax_phase_anim.add_patch(arc_diff)

        # Empty lines
        line_ref,        = ax_phase_anim.plot([], [], 'bo-', label='Reference Phase')
        line_joint,      = ax_phase_anim.plot([], [], 'ro-', label='Joint Phase')
        ax_phase_anim.legend()

        ### PHASE DIFFERENCE PLOT
        diff_signal = joint_phases - joint_phases_ref

        ax_phase_diff.set_xlim(0, times[-1])
        ax_phase_diff.set_ylim(np.amin(diff_signal), np.amax(diff_signal))
        ax_phase_diff.set_xlabel('Time [s]')
        ax_phase_diff.set_ylabel('Phase Difference [rad]')
        ax_phase_diff.set_title('Phase Difference Over Time')
        ax_phase_diff.axvline(x=switch_time, color='k', linestyle='--')

        # Empty lines
        line_phase_diff, = ax_phase_diff.plot([], [], 'k-')

        # INITIALIZATION
        def init_phase():
            line_ref.set_data([], [])
            line_joint.set_data([], [])
            line_phase_diff.set_data([], [])
            time_text.set_text(f"time: {0.0 :.1f} s")
            fb_text.set_text(f"Feedback OFF")

            return line_ref, line_joint, line_phase_diff, time_text, fb_text

        # UPDATE
        def update_phase(frame):
            curr_time   = frame*timestep
            ref_phase   = joint_phases_ref[frame]
            joint_phase = joint_phases[frame]

            line_ref.set_data(   [0,   np.cos(ref_phase)], [0,   np.sin(ref_phase)])
            line_joint.set_data( [0, np.cos(joint_phase)], [0, np.sin(joint_phase)])

            line_phase_diff.set_data(
                times[:frame],
                diff_signal[:frame],
            )

            time_text.set_text(f"time: {curr_time :.1f} s")

            if curr_time > switch_time:
                fb_text.set_text(f"Feedback ON")
                fb_text.set_color('red')

            # Draw the red arc
            theta1 = np.degrees(ref_phase % (2 * np.pi))
            theta2 = np.degrees(joint_phase % (2 * np.pi))
            theta2 = theta2 if theta2 > theta1 else theta2 + 360

            arc_diff.theta1 = theta1
            arc_diff.theta2 = theta2

            return line_ref, line_joint, line_phase_diff, time_text, fb_text, arc_diff

        ani_phase = FuncAnimation(
            fig_phase_anim,
            update_phase,
            frames    = np.arange(0, n_iterations, steps_jump),
            init_func = init_phase,
            blit      = True,
            interval  = 1,
        )

        # Save video:
        plots_utils._save_animation(
            figure      = fig_phase_anim,
            anim        = ani_phase,
            folder_path = self.results_data_folder,
            fig_label   = 'joint_phase_vs_ref_phase',
        )

        # Close the figure
        plt.close(fig_phase_anim)

        return

    def _plot_histogram(
        self,
        x_values  : np.ndarray,
        phi_values: np.ndarray,
        fig       : plt.Figure,
        ax        : plt.Axes,
    ):
        ''' Plot histogram of joint angles evolution '''

        # X bins
        x0      = np.amin(x_values)
        x1      = np.amax(x_values)
        x_range = x1 - x0
        x_min   = min( 0.4, x0 - 0.1*x_range )
        x_max   = max( 1.6, x1 + 0.1*x_range )
        x_bins  = 41
        x_step  = ( x_max - x_min ) / ( x_bins - 1 )
        x_range = ( x_min - x_step / 2, x_max + x_step / 2 )

        # Phi bins
        phi_min   = -np.pi
        phi_max   = 5 * np.pi
        phi_bins  = 40
        phi_step  = (phi_max - phi_min) / ( phi_bins - 1 )
        phi_range = ( phi_min - phi_step / 2, phi_max + phi_step / 2)

        # Histogram
        histogram, x_edges, phi_edges = np.histogram2d(
            x     = x_values,
            y     = phi_values,
            bins  = [ x_bins,  phi_bins],
            range = [x_range, phi_range],
        )

        # Normalize the histogram bt the total number of iterations
        n_iterations         = x_values.shape[0] / 3
        max_density          = 0.25 * n_iterations
        normalized_histogram = histogram / max_density

        # Account for periodicity
        hist_aux = normalized_histogram[:, 0] + normalized_histogram[:, -1]
        normalized_histogram[:,  0] = hist_aux
        normalized_histogram[:, -1] = hist_aux

        # Plot the histogram
        mesh_x, mesh_Y = np.meshgrid(x_edges, phi_edges)
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
        plt.xlim(x_range)
        plt.ylim(phi_range)

        return fig, ax

    def _plot_histogram_hexbin(
        self,
        x_values  : np.ndarray,
        phi_values: np.ndarray,
        fig       : plt.Figure,
        ax        : plt.Axes,
    ):
        ''' Plot histogram of joint angles evolution '''

        # Plot hexbin
        x0   = np.amin(x_values)
        x1   = np.amax(x_values)
        x_range = x1 - x0
        x_min      = min( 0.8, x0 - 0.1*x_range )
        x_max      = max( 2.0, x1 + 0.1*x_range )

        hb = ax.hexbin(
            x_values,
            phi_values,
            gridsize = 50,
            cmap     = 'inferno',
            extent   = (x_min, x_max, -np.pi, 5*np.pi),
        )

        # Decorate

        # Put ylabels and ticks at multiples of pi
        ax.set_yticks(np.arange(-np.pi, 5*np.pi+1, np.pi))
        ax.set_yticklabels(
            [
                r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$'
            ]
        )

        ax.set_xlabel('FB distance [Body Length]')
        ax.set_ylabel('Joint Phase Difference [rad]')
        ax.set_title('FB distance vs Joint Phase Difference')

        plt.colorbar(hb, ax = ax)

        return fig, ax

    def _plot_com_position_vs_joint_phase(
        self,
        com_positions_diff: np.ndarray,
        joint_phases_diff : np.ndarray,
        n_iterations      : int,
        figures_dict      : dict[str, plt.Figure] = None,
        pos_offset        : float = 0.0,
    ):
        ''' Plot joint angles evolution '''

        ###############
        ## 1D Histogram
        ###############
        fig_com_vs_angle_1d = plt.figure('COM Position vs Joint Phase Difference - 1D')

        plt.hist(
            joint_phases_diff[-n_iterations:] / np.pi,
            bins      = 80,
            color     = 'royalblue',
            edgecolor = 'black',
            alpha     = 0.75,
            linewidth = 1.2,
        )

        plt.xlabel('Joint Phase Difference [% cycle]', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Histogram of Joint Phase Difference', fontsize=14)
        plt.xlim([-1.0, 1.0])
        plt.tight_layout()

        figures_dict['fig_com_vs_phase_1D'] = fig_com_vs_angle_1d

        ###############
        ## 2D Histogram
        ###############
        fig_com_vs_angle_2d = plt.figure('COM Position vs Joint Phase Difference - 2D')
        axis = fig_com_vs_angle_2d.add_subplot(111)

        # concatenate 3 copies of COM positions
        com_positions_stack = np.concatenate(
            [
                com_positions_diff[-n_iterations:],
                com_positions_diff[-n_iterations:],
                com_positions_diff[-n_iterations:],
            ]
        ) + pos_offset

        joints_phases_stack = np.concatenate(
            [
                joint_phases_diff[-n_iterations:],
                joint_phases_diff[-n_iterations:] + 2*np.pi,
                joint_phases_diff[-n_iterations:] + 4*np.pi,
            ]
        )

        fig_com_vs_angle_2d, axis = self._plot_histogram(
            x_values  = com_positions_stack,
            phi_values= joints_phases_stack,
            fig       = fig_com_vs_angle_2d,
            ax        = axis,
        )

        # fig_com_vs_angle_2d, axis = self._plot_histogram_hexbin(
        #     x_values  = com_positions_stack,
        #     phi_values= joints_phases_stack,
        #     fig       = fig_com_vs_angle_2d,
        #     ax        = axis,
        # )

        # # Brainbridge 1958
        # # V = body_length * (3 * frequency - 4) / 400
        # # Li et al. 2020
        # # PHI = ( 2 * pi * f / V ) * D + PHI_0

        # v_fun       = lambda    f: (3*f - 4) / 400
        # phi_fun     = lambda f, d: np.mod( ( 2*np.pi*f / v_fun(f) ) * d, 2*np.pi )

        # d_vals = np.linspace(0.8, 2.0, 100)
        # phi_vals = phi_fun(3.5, d_vals)

        figures_dict['fig_com_vs_phase_2D'] = fig_com_vs_angle_2d

        return

    def _plot_leader_vs_follower_schooling_data(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Plot joint angles evolution '''

        plotpars          : dict = self.snn_network.params.monitor.farms_simulation['plotpars']
        com_vs_angle_pars : dict = plotpars.get('com_position_joint_phase_relationship')

        if not com_vs_angle_pars or not com_vs_angle_pars['showit']:
            return

        # target_joints      = com_vs_angle_pars['target_joints']
        # target_dim         = com_vs_angle_pars['target_dim']
        # target_freq        = com_vs_angle_pars['target_freq']
        # target_pos         = com_vs_angle_pars['target_pos']
        # target_offset_fun  = com_vs_angle_pars.get('target_pos_offset_fun', None)
        # target_offset_args = com_vs_angle_pars.get('target_pos_offset_args', None)

        discard_ratio = com_vs_angle_pars['discard_ratio']
        save_video    = com_vs_angle_pars.get('save_video', False)

        # Parameters
        timestep     = self.sim_res_data.timestep
        n_iterations =  self.sim_res_data.sensors.joints.size(0)
        times        = np.arange(0, timestep*n_iterations, timestep)

        n_iter_cons = round(n_iterations * (1 - discard_ratio))

        # Data
        joint_positions_ref = self.data_com_pos_vs_joint_phase['joint_positions_ref']
        joint_positions     = self.data_com_pos_vs_joint_phase['joint_positions']
        joint_phases_ref    = self.data_com_pos_vs_joint_phase['joint_phases_ref']
        joint_phases        = self.data_com_pos_vs_joint_phase['joint_phases']
        com_positions       = self.data_com_pos_vs_joint_phase['com_positions']
        com_positions_diff  = self.data_com_pos_vs_joint_phase['com_positions_diff']
        joint_phases_diff   = self.data_com_pos_vs_joint_phase['joint_phases_diff']

        # Joint Angle vs Reference Angle #######################
        self._plot_joint_angle_vs_ref_angle(
            joint_positions    = joint_positions,
            joint_positions_ref= joint_positions_ref,
            n_iter_cons        = n_iter_cons,
            figures_dict       = figures_dict,
        )

        # Animation of Joint Angle vs Reference Angle ##########
        self._animation_joint_angle_vs_ref_angle(
            joint_positions    = joint_positions,
            joint_positions_ref= joint_positions_ref,
            timestep           = timestep,
            n_iterations       = n_iterations,
            save_video         = save_video,
        )

        # Joint Phase vs Reference Phase #######################
        self._plot_joint_phase_vs_ref_phase(
            joint_phases    = joint_phases,
            joint_phases_ref= joint_phases_ref,
            times           = times,
            n_iter_cons     = n_iter_cons,
            figures_dict    = figures_dict,
        )

        # Animation of Joint Phase vs Reference Phase ##########
        self._animation_joint_phase_vs_ref_phase(
            joint_phases    = joint_phases,
            joint_phases_ref= joint_phases_ref,
            times           = times,
            save_video      = save_video,
        )

        # COM vs Joint Phase ###################################
        self._plot_com_position_vs_joint_phase(
            com_positions_diff= com_positions_diff,
            joint_phases_diff = joint_phases_diff,
            n_iterations      = n_iter_cons,
            figures_dict      = figures_dict,
            pos_offset        = com_positions[0],
        )

        return


    # \----- [ Plots ] ------

    # ------ [ Animations ] ------
    def _animate_links_trajectory(
        self,
        figures_dict: dict[str, plt.Figure] = None,
    ):
        ''' Animate links trajectory '''

        plotpars  : dict = self.snn_network.params.monitor.farms_simulation['plotpars']
        if not plotpars:
            return

        anim_pars : dict = plotpars.get('animation')
        if not anim_pars or not anim_pars['active']:
            return

        links_positions = np.copy( self.links_positions )
        if not links_positions[-1].any():
            links_positions  = links_positions[:-1]

        timestep     = self.sim_res_data.timestep
        n_iterations = links_positions.shape[0]
        times        = np.arange(0, timestep*n_iterations, timestep)

        water_parameters = {
            'water_maps'    : self.water_map_parameters,
            'water_dynamics': self.water_dynamics_options,
        }

        fig_lt = plt.figure('Link trajectory animation')
        anim_lt = mech_plt.animate_links_trajectory(
            fig              = fig_lt,
            times            = times,
            links_positions  = links_positions,
            params           = self.sim_res_pars,
            water_parameters = water_parameters,
            show_animation   = anim_pars.get('showit', True),
            save_frames      = anim_pars.get('save_frames', False),
            save_path        = anim_pars.get('save_path', False),
            video_speed      = anim_pars.get('video_speed', False),
        )

        if anim_lt is not None:
            figures_dict['fig_lt'] = [fig_lt, anim_lt]

        return

    # \----- [ Animations ] ------

    # ------ [ Data cleaning ] ------
    def clear_figures(self) -> None:
        ''' Clear figures '''
        plt.close('all')

    # \----- [ Data cleaning ] ------