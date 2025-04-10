import logging
import dill
import json
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import find_peaks, hilbert, butter, filtfilt
from network_modules.simulation.mechanical_simulation import MechSimulation

import network_modules.performance.signal_processor_mech as MechProc

MECH_METRICS = [

    # Frequency
    'mech_freq_ax',
    'mech_freq_lb',
    'mech_freq_diff',

    # Speed
    'mech_speed_fwd',
    'mech_speed_lat',
    'mech_speed_abs',
    'mech_stride_len',

    # Phase lag
    'mech_ipl_ax_a',
    'mech_ipl_ax_t',
    'mech_wave_number_a',
    'mech_wave_number_t',
    'mech_ipl_lb_h',
    'mech_ipl_lb_c',
    'mech_ipl_lb_d',

    # Neuro-muscolar phase lag
    'mech_n2m_lag_ax_all',
    'mech_n2m_lag_ax_trk',
    'mech_n2m_lag_ax_tal',
    'mech_n2m_lag_lb_all',

    # Torque and energy
    'mech_torque',
    'mech_energy',
    'mech_cot',

    # Trajectory
    'mech_traj_mse',
    'mech_traj_curv',

    # Tail beat
    'mech_tail_beat_amp',

    # Dimensionless numbers
    'mech_strouhal_number',
    'mech_swimming_number',
    'mech_reynolds_number',

    # Normalized metrics
    'mech_speed_fwd_bl',
    'mech_speed_lat_bl',
    'mech_speed_abs_bl',
    'mech_stride_len_bl',
    'mech_tail_beat_amp_bl',

    # Displacements
    'mech_joints_disp_mean',
    'mech_joints_disp_amp',
    'mech_links_disp_mean',
    'mech_links_disp_amp',

    # Displacements normalized
    'mech_links_disp_amp_bl',
]

MECH_METRICS_VECT_JOINTS = [
    'mech_joints_disp_mean',
    'mech_joints_disp_amp',
]

MECH_METRICS_VECT_LINKS = [
    'mech_links_disp_mean',
    'mech_links_disp_amp',
    'mech_links_disp_amp_bl',
]

class MechPerformance(MechSimulation):
    '''
    Class used to evaluate the performance of a mechanical simulation
    '''

    ## POST PROCESSING
    def simulation_post_processing(
        self,
        load_from_file: bool = False,
    ) -> dict[str, float]:
        ''' Post-processing for the mechanical simulation '''
        super().simulation_post_processing()
        return self.farms_performance_metrics(load_from_file)

    # DATA LOADING
    def load_mechanical_simulation_data(self, from_file = False):
        ''' Load mechanical simulation data '''

        # NOTE:
        # For the links arrays: positions[iteration, link_id, xyz]
        # For the positions arrays: positions[iteration, xyz]
        # For the joints arrays: positions[iteration, joint]

        if not from_file:
            self.sim_res_data = self.animat_data
            self.sim_res_pars = self.animat_options
            return

        self.sim_res_data, self.sim_res_pars = MechProc.load_mechanical_simulation_data(
            data_folder  = self.results_data_folder,
            control_type = self.snn_network.control_type,
        )

    def load_mechanical_simulation_arrays(self):
        ''' Load mechanical simulation arrays '''

        data       = self.sim_res_data
        n_ax_links = self.snn_network.params.mechanics.mech_axial_joints + 1

        joints_positions      = np.array( data.sensors.joints.positions_all() )
        joints_velocities     = np.array( data.sensors.joints.velocities_all() )
        joints_active_torques = np.array( data.sensors.joints.active_torques() )
        links_positions       = np.array( data.sensors.links.urdf_positions() )
        links_velocities      = np.array( data.sensors.links.com_lin_velocities() )

        joints_commands = (
            None
            if self.snn_network.control_type in ['position_control', 'hybrid_position_control']
            else
            np.array( data.state.array )
        )

        # Take care of last iteration
        if not np.any(joints_positions[-1]):
            joints_positions  = joints_positions[:-1]

        if not np.any(joints_velocities[-1]):
            joints_velocities = joints_velocities[:-1]

        if not np.any(joints_active_torques[-1]):
            joints_active_torques = joints_active_torques[:-1]

        if not np.any(links_positions[-1]):
            links_positions = links_positions[:-1]

        if not np.any(links_velocities[-1]):
            links_velocities = links_velocities[:-1]

        if joints_commands is not None and not np.any(joints_commands[-1]):
            joints_commands = joints_commands[:-1]

        n_steps = min(
            [
                len(state_arr) for state_arr in [
                    joints_positions,
                    joints_velocities,
                    joints_active_torques,
                    links_positions,
                    links_velocities,
                    joints_commands,
                ]
                if state_arr is not None
            ]
        )

        joints_positions      = joints_positions[:n_steps]
        joints_velocities     = joints_velocities[:n_steps]
        joints_active_torques = joints_active_torques[:n_steps]
        links_positions       = links_positions[:n_steps]
        links_velocities      = links_velocities[:n_steps]
        joints_commands       = joints_commands[:n_steps] if joints_commands is not None else None

        #  Compute center of mass position
        com_positions         = np.mean( links_positions[:, :n_ax_links], axis= 1)

        # Extend links positions including the tail
        lenght_tail_link = self.snn_network.params.mechanics.mech_axial_links_length[-1]

        tail_positions = MechProc._compute_tail_positions_from_links_positions(
            links_positions  = links_positions,
            joints_positions = joints_positions,
            length_tail_link = lenght_tail_link,
            n_links_axis     = n_ax_links,
        )
        links_positions = np.insert(
            links_positions,
            n_ax_links,
            tail_positions,
            axis = 1,
        )

        self.joints_positions      = joints_positions
        self.joints_velocities     = joints_velocities
        self.joints_active_torques = joints_active_torques
        self.links_positions       = links_positions
        self.links_velocities      = links_velocities
        self.com_positions         = com_positions
        self.joints_commands       = joints_commands

        return (
            joints_positions,
            joints_velocities,
            joints_active_torques,
            links_positions,
            links_velocities,
            com_positions,
            joints_commands,
        )

    ## METRICS COMPUTATION
    def farms_performance_metrics(
        self,
        load_from_file    : bool  = False,
    ) -> dict[str,float]:
        ''' Campute all the FARMS-related metrics '''

        # Load data
        self.load_mechanical_simulation_data(load_from_file)
        data = self.sim_res_data

        # Sim parameters
        timestep          = data.timestep
        n_steps           = np.shape(data.sensors.links.array)[0]
        duration          = timestep * n_steps

        # Transient time
        transient_time    = min( 2.0, duration / 2.0 )

        sim_fraction      = 1.0 - transient_time / duration
        n_steps_fraction  = round( n_steps * sim_fraction )
        duration_fraction = timestep * n_steps_fraction

        # Mechanics parameters
        n_joints_limb = self.snn_network.params.mechanics.mech_n_lb_joints
        n_lb_joints   = self.snn_network.params.mechanics.mech_limbs_joints
        n_ax_joints   = self.snn_network.params.mechanics.mech_axial_joints
        n_ax_links    = n_ax_joints + 1
        n_ax_points   = n_ax_links + 1

        inds_lb_pairs = self.snn_network.params.mechanics.mech_limbs_pairs_indices

        n_pca_joints = self.snn_network.params.mechanics.mech_pca_joints
        n_pca_links  = n_pca_joints + 1

        # Extract data
        (
            joints_positions,
            joints_velocities,
            joints_active_torques,
            links_positions,
            links_velocities,
            com_positions,
            joints_commands,
        ) = self.load_mechanical_simulation_arrays()

        # Get COM position vs joint phase
        self.get_com_position_vs_joint_phase_data(
            points_positions = links_positions,
        )

        # Compute metrics
        (
            joints_displacements,
            joints_displacements_mean,
            _joints_displacements_std,
            joints_displacements_amp,
        ) = MechProc.compute_joints_displacements_metrics(
            joints_positions = joints_positions,
            n_joints_axis    = n_ax_joints,
            sim_fraction     = sim_fraction,
        )

        (
            links_displacements,
            links_displacements_mean,
            _links_displacements_std,
            links_displacements_amp,
            links_displacements_data,
        ) = MechProc.compute_links_displacements_metrics(
            points_positions = links_positions,
            n_points_axis    = n_ax_points,
            sim_fraction     = sim_fraction,
            n_points_pca     = n_pca_links,
        )

        (
            freq_ax,
            freq_lb,
            freq_diff,
            freq_joints,
        ) = MechProc.compute_frequency(
            joints_positions = joints_positions,
            n_joints_axis    = n_ax_joints,
            n_joints_limbs   = n_lb_joints,
            timestep         = timestep,
            sim_fraction     = sim_fraction,
        )

        plot_joint_phase = False
        if plot_joint_phase:
            self.plot_phase_evolution(
                joints_positions = joints_positions,
                joint_index      = 0,
            )

        (
            ipl_ax_a,
            ipl_ax_t,
            ipl_lb_h,
            ipl_lb_c,
            ipl_lb_d,
            ipls_all,
        ) = MechProc.compute_joints_ipls(
            joints_angles     = joints_positions,
            joints_freqs      = freq_joints,
            n_joints_axis     = n_ax_joints,
            n_joints_per_limb = n_joints_limb,
            n_active_joints   = n_ax_joints + n_lb_joints,
            limb_pairs_inds   = inds_lb_pairs,
            timestep          = timestep,
        )

        (
            wave_number_a,
            wave_number_t
        ) = self.compute_wave_number(
            ipl_ax_a = ipl_ax_a,
            ipl_ax_t = ipl_ax_t,
        )

        # torque_commands = joints_commands[:, 1::2] - joints_commands[:, 0::2]
        # (
        #     ipl_command_ax_a,
        #     ipl_command_ax_t,
        #     ipl_command_lb_h,
        #     ipl_command_lb_c,
        #     ipl_command_lb_d,
        #     ipls_command__all,
        # ) = MechProc.compute_joints_ipls(
        #     joints_angles     = torque_commands,
        #     joints_freqs      = freq_joints,
        #     n_joints_axis     = n_ax_joints,
        #     n_joints_per_limb = n_joints_limb,
        #     n_active_joints   = n_ax_joints + n_lb_joints,
        #     limb_pairs_inds   = inds_lb_pairs,
        #     timestep          = timestep,
        # )

        (
            n2m_lag_ax_all,
            n2m_lag_ax_trk,
            n2m_lag_ax_tal,
            n2m_lag_lb_all,
            n2m_lags_all,
        ) = MechProc.compute_joints_neuro_muscolar_ipls(
            joints_commands = joints_commands,
            joints_angles   = joints_positions,
            joints_freqs    = freq_joints,
            n_joints_axis   = n_ax_joints,
            n_active_joints = n_ax_joints + n_lb_joints,
            limb_pairs_inds = inds_lb_pairs,
            timestep        = timestep,
        )

        speed_fwd, speed_lat, speed_abs = MechProc.compute_speed(
            links_positions_pca = links_displacements_data['links_positions_transformed'],
            n_links_axis        = n_ax_points,
            duration            = duration_fraction,
        )

        traj_mse = MechProc.compute_trajectory_linearity(
            timestep      = timestep,
            com_positions = com_positions,
            sim_fraction  = sim_fraction
        )

        traj_curv = MechProc.compute_trajectory_curvature(
            timestep     = timestep,
            com_pos      = com_positions,
            sim_fraction = sim_fraction
        )

        torque = MechProc.sum_torques(
            joints_torques = joints_active_torques,
            sim_fraction   = sim_fraction
        )

        energy = MechProc.sum_energy(
            torques      = joints_active_torques,
            speeds       = joints_velocities,
            timestep     = timestep,
            sim_fraction = sim_fraction,
        )

        # Derived metrics
        lenght_ax         = self.snn_network.params.mechanics.mech_axial_length
        tail_beat_amp     = links_displacements_amp[-1]
        distance_fwd      = speed_fwd * duration_fraction
        cost_of_transport = energy / np.abs(distance_fwd) if distance_fwd != 0 else np.nan
        stride_length     = speed_fwd / freq_ax

        if self.snn_network.params.simulation.gait == 'swim':
            velocity      = np.amax([np.abs(speed_fwd), 1e-6])
            mu_kin_water  = 1.0034 * 1e-6

            strouhal_number   = freq_ax * tail_beat_amp / velocity
            swimming_number   = 2*np.pi*freq_ax * tail_beat_amp * lenght_ax / mu_kin_water
            reynolds_number   = velocity * lenght_ax / mu_kin_water

        else:
            strouhal_number   = np.nan
            swimming_number   = np.nan
            reynolds_number   = np.nan

        # Normalize metrics
        speed_fwd_bl     = speed_fwd / lenght_ax
        speed_lat_bl     = speed_lat / lenght_ax
        speed_abs_bl     = speed_abs / lenght_ax

        tail_beat_amp_bl = tail_beat_amp / lenght_ax
        stride_length_bl = stride_length / lenght_ax

        links_displacements_amp_bl  = links_displacements_amp / lenght_ax

        # Return metrics
        mech_metrics = {
            # Frequency
            'mech_freq_ax'         : freq_ax,
            'mech_freq_lb'         : freq_lb,
            'mech_freq_diff'       : freq_diff,

            # Speed
            'mech_speed_fwd'       : speed_fwd,
            'mech_speed_lat'       : speed_lat,
            'mech_speed_abs'       : speed_abs,
            'mech_stride_len'      : stride_length,

            # Phase lag
            'mech_ipl_ax_a'        : ipl_ax_a,
            'mech_ipl_ax_t'        : ipl_ax_t,
            'mech_wave_number_a'   : wave_number_a,
            'mech_wave_number_t'   : wave_number_t,
            'mech_ipl_lb_h'        : ipl_lb_h,
            'mech_ipl_lb_c'        : ipl_lb_c,
            'mech_ipl_lb_d'        : ipl_lb_d,

            # Neuro-muscolar phase lag
            'mech_n2m_lag_ax_all'  : n2m_lag_ax_all,
            'mech_n2m_lag_ax_trk'  : n2m_lag_ax_trk,
            'mech_n2m_lag_ax_tal'  : n2m_lag_ax_tal,
            'mech_n2m_lag_lb_all'  : n2m_lag_lb_all,

            # Torque and energy
            'mech_torque'          : torque,
            'mech_energy'          : energy,
            'mech_cot'             : cost_of_transport,

            # Trajectory
            'mech_traj_mse'        : traj_mse,
            'mech_traj_curv'       : traj_curv,

            # Tail beat
            'mech_tail_beat_amp'   : tail_beat_amp,

            # Dimensionless numbers
            'mech_strouhal_number' : strouhal_number,
            'mech_swimming_number' : swimming_number,
            'mech_reynolds_number' : reynolds_number,

            # Normalized metrics
            'mech_speed_fwd_bl'    : speed_fwd_bl,
            'mech_speed_lat_bl'    : speed_lat_bl,
            'mech_speed_abs_bl'    : speed_abs_bl,
            'mech_stride_len_bl'   : stride_length_bl,
            'mech_tail_beat_amp_bl': tail_beat_amp_bl,

            # Displacements
            'mech_joints_disp_mean': joints_displacements_mean.tolist(),
            'mech_joints_disp_amp' : joints_displacements_amp.tolist(),
            'mech_links_disp_mean' : links_displacements_mean.tolist(),
            'mech_links_disp_amp'  : links_displacements_amp.tolist(),

            # Displacements normalized
            'mech_links_disp_amp_bl'  : links_displacements_amp_bl.tolist(),
        }

        assert all( [key in MECH_METRICS for key in mech_metrics.keys()] ), 'Not all metrics are listed'
        assert all( [key in mech_metrics for key in MECH_METRICS] ),        'Not all metrics are computed'

        logging.info(f'MECHANICS METRICS: {json.dumps(mech_metrics, indent=4)}')

        self.all_metrics_data = {
            'mech_metrics'            : mech_metrics,
            'joints_commands'         : joints_commands,
            'joints_positions'        : joints_positions,
            'joints_velocities'       : joints_velocities,
            'joints_active_torques'   : joints_active_torques,
            'links_positions'         : links_positions,
            'links_velocities'        : links_velocities,
            'com_positions'           : com_positions,
            'joints_displacements'    : joints_displacements,
            'links_displacements'     : links_displacements,
            'links_displacements_data': links_displacements_data,
            'freq_joints'             : freq_joints,
            'n2m_lags_all'            : n2m_lags_all,
        }

        if self.save_all_metrics_data:
            os.makedirs(self.results_data_folder, exist_ok=True)
            filename = f'{self.results_data_folder}/mechanics_metrics.dill'
            with open(filename, 'wb') as outfile:
                logging.info(f'Saving all metrics data to {filename}')
                dill.dump(self.all_metrics_data, outfile)

        return mech_metrics

    def compute_wave_number(self, ipl_ax_a, ipl_ax_t):
        ''' Compute the wave number '''

        n_ax_joints   = self.snn_network.params.mechanics.mech_axial_joints
        lenght_ax     = self.snn_network.params.mechanics.mech_axial_length
        joints_pos_ax = np.array( self.snn_network.params.mechanics.mech_axial_joints_position )
        joints_pos_lb = np.array( self.snn_network.params.mechanics.mech_limbs_pairs_positions )

        p0_ax = joints_pos_ax[0]
        p1_ax = joints_pos_ax[-1]
        p0_tr = joints_pos_lb[0] if len(joints_pos_lb) > 0 else p0_ax
        p1_tr = joints_pos_lb[1] if len(joints_pos_lb) > 1 else p1_ax

        range_ax = p1_ax - p0_ax
        range_tr = p1_tr - p0_tr

        ind0_t      = np.argmax(joints_pos_ax >= p0_tr)
        ind1_t      = np.argmax(joints_pos_ax >= p1_tr) - 1
        n_tr_joints = ind1_t - ind0_t + 1

        # Wave number
        wave_number_a = ipl_ax_a * (n_ax_joints - 1) * lenght_ax / range_ax
        wave_number_t = ipl_ax_t * (n_tr_joints - 1) * lenght_ax / range_tr

        return wave_number_a, wave_number_t

    def plot_phase_evolution(
        self,
        joints_positions: np.ndarray,
        joint_index     : int,
    ):
        ''' Plot the phase evolution of a joint '''

        data     = self.sim_res_data
        timestep = data.timestep
        n_steps  = np.shape(data.sensors.links.array)[0]

        # Compute hilbert transform
        times       = np.arange(n_steps) * timestep
        signal      = joints_positions[:, joint_index]
        hilb_signal = hilbert(signal)
        inst_phases = np.unwrap(np.angle(hilb_signal))
        ph_min      = np.min(inst_phases)
        ph_max      = np.max(inst_phases)

        # Compute linear fit of the phase
        t_on          = times[-1] / 2
        times_0       = times[times < t_on]
        times_1       = times[times >= t_on]
        inst_phases_0 = inst_phases[times < t_on]
        inst_phases_1 = inst_phases[times >= t_on]

        slope_0, intercept_0, _, _, _ = linregress(times_0, inst_phases_0)
        slope_1, intercept_1, _, _, _ = linregress(times_1, inst_phases_1)

        plt.plot(times_0, inst_phases_0, 'b')
        plt.plot(times_1, inst_phases_1, 'c')
        plt.plot(times, slope_0*times + intercept_0, 'r', label='OFF', linestyle='--', linewidth=0.5)
        plt.plot(times, slope_1*times + intercept_1, 'g', label='ON',  linestyle='--', linewidth=0.5)
        plt.vlines(t_on, ph_min, ph_max, 'k')
        plt.ylim([ph_min, ph_max])
        plt.legend()
        plt.show()


    ## COM POSITION VS JOINT PHASE

    def _compute_frequency(
        self,
        signal  : np.ndarray,
        timestep: float,
        max_freq: float = 10,
    ):
        ''' Compute the frequency of a signal '''
        n_iterations      = signal.shape[0]
        max_freq_ind      = round(max_freq * n_iterations * timestep)
        fft_results       = np.fft.fft(signal)
        fft_dominant_inds = np.argmax(np.abs(fft_results[1:max_freq_ind])) + 1
        frequency         = fft_dominant_inds / (n_iterations * timestep)
        return frequency

    def _apply_filter_around_frequency(
        self,
        signal  : np.ndarray,
        timestep: float,
    ):
        ''' Apply a low-pass filter around the signal frequency '''
        fnyq        = 0.5 / timestep
        signal_freq = self._compute_frequency(signal, timestep)
        num, den    = butter(5, 2*signal_freq/fnyq, btype= 'lowpass' )
        signal_f    = filtfilt(num, den, signal, padtype= 'odd')
        return signal_f

    def _compute_angles_sum(
        self,
        x_signals   : np.ndarray,
        y_signals   : np.ndarray,
    ):
        ''' Compute the evolution of the sum of all angles '''

        n_iterations = self.sim_res_data.sensors.joints.size(0)
        n_steps      = np.shape(y_signals)[1]

        # Vectors
        vects_x = x_signals[1:, :] - x_signals[:-1, :]
        vects_y = y_signals[1:, :] - y_signals[:-1, :]

        # Dot products
        dot_products = (
            vects_x[1:] * vects_x[:-1] +
            vects_y[1:] * vects_y[:-1]
        )

        # Cross products
        cross_products = (
            vects_x[1:] * vects_y[:-1] -
            vects_y[1:] * vects_x[:-1]
        )

        # Angles
        angles                = np.arctan2(cross_products, dot_products)
        angles_sum            = np.zeros(n_iterations)
        angles_sum[-n_steps:] = np.sum(angles, axis=0)

        return angles_sum

    def _compute_trunk_tail_angle(
        self,
        x_signals   : np.ndarray,
        y_signals   : np.ndarray,
    ):
        ''' Compute the joint positions reference '''

        # Get signals
        n_iterations       = self.sim_res_data.sensors.joints.size(0)
        n_signals, n_steps = np.shape(y_signals)
        n_signals_trunk    = n_signals // 2

        trunk_x0 = x_signals[                  0]
        trunk_x1 = x_signals[n_signals_trunk - 1]
        tail_x0  = x_signals[    n_signals_trunk]
        tail_x1  = x_signals[                 -1]

        trunk_y0 = y_signals[                  0]
        trunk_y1 = y_signals[n_signals_trunk - 1]
        tail_y0  = y_signals[    n_signals_trunk]
        tail_y1  = y_signals[                 -1]

        # Compute angle between trunk and tail
        trunk_vector = np.array([trunk_x1 - trunk_x0, trunk_y1 - trunk_y0])
        tail_vector  = np.array([ tail_x1  - tail_x0,  tail_y1  - tail_y0])

        dot_product   = np.sum(trunk_vector * tail_vector, axis=0)
        cross_product = np.cross(trunk_vector, tail_vector, axis=0)

        angles_trunk_tail            = np.zeros(n_iterations)
        angles_trunk_tail[-n_steps:] = np.arctan2(cross_product, dot_product)
        # angles_trunk_tail[-n_steps:] /= np.amax(angles_trunk_tail)

        # step = n_steps // 2
        # plt.plot( x_signals[:, step], y_signals[:, step] )
        # plt.plot( [trunk_x0[step], trunk_x1[step]], [trunk_y0[step], trunk_y1[step]], '-o' )
        # plt.plot( [ tail_x0[step],  tail_x1[step]], [ tail_y0[step],  tail_y1[step]], '-o' )
        # plt.plot( [ 0, trunk_vector[0][step] ], [ 0, trunk_vector[1][step] ] )
        # plt.plot( [ 0,  tail_vector[0][step] ], [ 0,  tail_vector[1][step] ] )

        return angles_trunk_tail

    def _compute_target_angle_from_coordinates(
        self,
        x_signals   : np.ndarray,
        y_signals   : np.ndarray,
    ):
        ''' Compute the target joint positions from coordinates '''

        # target_angle = self._compute_trunk_tail_angle(
        #     x_signals,
        #     y_signals,
        # )

        target_angle = self._compute_angles_sum(
            x_signals,
            y_signals,
        )

        return target_angle

    def _get_reference_signal_from_parameters(
        self,
        target_freq: float,
    ):
        ''' Get the reference signal from the parameters '''

        # Simulatio parameters
        start_delay  = self.water_dynamics_options['delay_start']
        timestep     = self.sim_res_data.timestep
        n_iterations = self.sim_res_data.sensors.joints.size(0)

        n_delay      = round(start_delay / timestep)
        times        = np.arange(0, timestep*n_iterations, timestep)

        # Mechanical parameters
        mech_pars   = self.snn_network.params.mechanics
        body_length = mech_pars.mech_axial_length
        n_joints    = mech_pars.mech_axial_joints
        n_points    = n_joints + 2
        points_pos  = np.array( [0] + list(mech_pars.mech_axial_joints_position) + [body_length] ) / body_length

        ## NOTE: OLD METHOD
        # joints_pos  = np.array( mech_pars.mech_axial_joints_position ) / body_length
        # target_twl  = 0.95

        # # Joint positions reference
        # joint_positions_ref = np.sum(
        #     [
        #         np.sin(
        #             2*np.pi*target_freq* (times - start_delay) -
        #             2*np.pi*target_twl*joints_pos[joint_ind]
        #         )
        #         for joint_ind in target_joints
        #     ],
        #     axis = 0
        # )
        # joint_positions_ref[:n_delay] = 0
        # joint_positions_ref          /= np.amax(joint_positions_ref)

        # Build signals
        target_twl = 0.95
        c1, c2, c3 = [ +0.05, -0.13, +0.28 ]
        points_env = 0.6 * ( c1 + c2*points_pos + c3*points_pos**2 )

        x_signals_ref = np.zeros((n_points, n_iterations))
        y_signals_ref = np.zeros((n_points, n_iterations))

        for point_ind in range(n_points):
            point_pos = points_pos[point_ind]
            point_env = points_env[point_ind]

            # NOTE: Fish head is at 0, tail points negative
            x_signals_ref[point_ind] = point_pos * np.ones(n_iterations)
            y_signals_ref[point_ind] = np.zeros(n_iterations)
            y_signals_ref[point_ind] = point_env * (
                np.sin(
                    2*np.pi * target_freq * (times - start_delay) -
                    2*np.pi *  target_twl * point_pos
                )
            )

        # NOTE: Fish head is at 0, tail points negative
        x_signals_ref *= -1

        # Account for delay
        x_signals_ref = x_signals_ref[:, n_delay:]
        y_signals_ref = y_signals_ref[:, n_delay:]

        # Get reference signal
        joint_positions_ref = self._compute_target_angle_from_coordinates(
            x_signals_ref,
            y_signals_ref,
        )

        return joint_positions_ref

    def _get_reference_signal_from_file(
        self,
        target_signal_path: str,
    ):
        ''' Get the reference signal from a file '''

        # Simulatio parameters
        start_delay  = self.water_dynamics_options['delay_start']
        timestep     = self.sim_res_data.timestep
        n_iterations = self.sim_res_data.sensors.joints.size(0)

        n_delay      = round(start_delay / timestep)
        n_signal     = n_iterations - n_delay

        target_signal_file = f'{target_signal_path}/kinematics_signals.csv'
        target_signal      = pd.read_csv(target_signal_file)

        # Check timestep
        target_times    = target_signal['time']
        target_timestep = target_times[1] - target_times[0]

        ratio = timestep / target_timestep
        if np.isclose(ratio, round(ratio)):
            downsample_factor = round(ratio)
            target_signal     = target_signal.iloc[::downsample_factor].reset_index(drop=True)
            target_times      = target_signal['time']
            target_timestep   = target_times[1] - target_times[0]
        else:
            raise ValueError('Timestep mismatch')

        # Convert to arrays
        # Ex: time, x_Head, x_SC 1, y_Head, y_SC 1
        x_cols = [col for col in target_signal.columns if col.startswith('x_')]
        y_cols = [col for col in target_signal.columns if col.startswith('y_')]

        x_signals_ref = np.array([target_signal[col].values[:n_signal] for col in x_cols])
        y_signals_ref = np.array([target_signal[col].values[:n_signal] for col in y_cols])

        # NOTE: Fish head is at 0, tail points negative (-1)
        # NOTE: Saved signal inverts time and phases (-1)
        x_signals_ref *= -1
        y_signals_ref *= -1

        # Get reference signal
        joint_positions_ref = self._compute_target_angle_from_coordinates(
            x_signals_ref,
            y_signals_ref,
        )

        return joint_positions_ref

    def _get_reference_signal(
        self,
        times: np.ndarray,
    ):
        ''' Get the reference signal '''

        from lilytorch.body_loader import load_body_from_parameters

        n_steps   = len(times)
        time_step = times[1] - times[0]

        water_dynamics  = self.water_dynamics_options
        water_path      = water_dynamics['results_path']
        water_pars_path = f'{water_path}/parameters.yaml'

        fish_body    = load_body_from_parameters(water_pars_path)
        fish_props   = fish_body.zebrafish_properties

        (
            leader_x_evolution,
            leader_y_evolution,
        ) = fish_props.get_coordinates_evolution(
            times            = times,
            signal_amp_fun   = fish_body.signal_amp_fun,
            signal_phase_fun = fish_body.signal_phase_fun,
            normalize        = False,
        )

        # Transformations
        leader_y_evolution *= -1

        if water_dynamics['invert_x']:
            leader_x_evolution = -leader_x_evolution
        if water_dynamics['invert_y']:
            leader_y_evolution = -leader_y_evolution

        translation_x = water_dynamics['translation'][0]
        translation_y = water_dynamics['translation'][1]

        leader_x_evolution += translation_x
        leader_y_evolution += translation_y

        leader_dyn_pos_fun = water_dynamics['pos_offset_function']
        leader_dynamic_pos = np.array( [ leader_dyn_pos_fun(t) for t in times ] ) # leader_times

        leader_x_evolution = ( leader_x_evolution.T + leader_dynamic_pos[:, 0] ).T
        leader_y_evolution = ( leader_y_evolution.T + leader_dynamic_pos[:, 1] ).T

        # Delay
        delay_start     = water_dynamics['delay_start']
        delay_steps     = round(delay_start / time_step)
        delay_buffer_x  = np.array([leader_x_evolution[0, :]] * delay_steps) * 1.0
        delay_buffer_y  = np.array([leader_y_evolution[0, :]] * delay_steps) * 0.0 + translation_y

        if delay_steps:
            leader_x_evolution = np.concatenate(
                [
                    delay_buffer_x,
                    leader_x_evolution,
                ]
            )
            leader_y_evolution = np.concatenate(
                [
                    delay_buffer_y,
                    leader_y_evolution,
                ]
            )

        leader_x_evolution = leader_x_evolution[:n_steps]
        leader_y_evolution = leader_y_evolution[:n_steps]

        # Get reference signal
        joint_positions_ref = self._compute_target_angle_from_coordinates(
            leader_x_evolution.T,
            leader_y_evolution.T,
        )

        return joint_positions_ref

    def get_com_position_vs_joint_phase_data(
        self,
        points_positions: np.ndarray,
    ):
        ''' Save the COM position vs joint phase '''

        plotpars          : dict = self.snn_network.params.monitor.farms_simulation['plotpars']
        com_vs_angle_pars : dict = plotpars.get('com_position_joint_phase_relationship')

        if com_vs_angle_pars is None:
            return

        show_data : bool = com_vs_angle_pars.get('showit', False)
        save_data : bool = com_vs_angle_pars.get('save_data', False)

        self.data_com_pos_vs_joint_phase = None

        if not show_data and not save_data:
            return

        target_joints      = com_vs_angle_pars['target_joints']
        target_dim         = com_vs_angle_pars['target_dim']
        target_freq        = com_vs_angle_pars['target_freq']
        target_pos         = com_vs_angle_pars['target_pos']
        target_signal_path = com_vs_angle_pars.get('target_signal_path', None)

        target_offset_fun  = com_vs_angle_pars.get('target_pos_offset_fun', None)
        target_offset_args = com_vs_angle_pars.get('target_pos_offset_args', None)

        if target_offset_fun is None:
            target_offset_fun = lambda time: np.zeros(2)
        if target_offset_args is None:
            target_offset_args = []

        # Parameters
        start_delay  = self.water_dynamics_options['delay_start']
        body_length  = self.snn_network.params.mechanics.mech_axial_length
        timestep     = self.sim_res_data.timestep
        n_iterations = self.sim_res_data.sensors.joints.size(0)

        n_delay      = round(start_delay / timestep)
        n_signal     = n_iterations - n_delay
        times        = np.arange(0, timestep*n_iterations, timestep)

        ########################################################
        # Get Data #############################################
        ########################################################

        # Reference data
        # if target_signal_path:
        #     joint_positions_ref = self._get_reference_signal_from_file(
        #         target_signal_path = target_signal_path,
        #     )
        # else:
        #     message = 'WARNING: Defining reference signal from parameters'
        #     logging.warning(message)
        #     print(message)
        #     joint_positions_ref = self._get_reference_signal_from_parameters(
        #         target_freq = target_freq,
        #     )

        joint_positions_ref = self._get_reference_signal(times)

        # CoM positions reference
        com_positions_ref   = target_pos + np.array(
            [
                target_offset_fun(t, *target_offset_args)[target_dim]
                for t in times
            ]
        )

        # Positions of the target joint
        x_signals = points_positions[:, :, 0].T
        y_signals = points_positions[:, :, 1].T

        ## NOTE: OLD METHOD
        # joints_positions_all = np.array( self.sim_res_data.sensors.joints.positions_all() )
        # joint_positions      = np.sum( joints_positions_all[:, target_joints], axis=1 )

        joint_positions = self._compute_target_angle_from_coordinates(x_signals, y_signals)

        # Filter joint positions
        joint_positions     = self._apply_filter_around_frequency(joint_positions, timestep)
        joint_positions_ref = self._apply_filter_around_frequency(joint_positions_ref, timestep)

        # # Rescale joint positions reference
        # abs_peaks            = find_peaks( np.abs(joint_positions) )[0]
        # joint_amp            = np.mean( np.abs(joint_positions[abs_peaks]) )
        # joint_positions_ref *= joint_amp

        # Positions of the target COM dimension
        com_positions_all = np.array( self.sim_res_data.sensors.links.global_com_positions_all() )
        com_positions     = com_positions_all[:, target_dim]

        # Convert joint angles to phases
        hilb_joint_positions_ref = hilbert(joint_positions_ref)
        hilb_joint_positions     = hilbert(joint_positions)

        joint_phases_ref = np.unwrap(np.angle(hilb_joint_positions_ref))
        joint_phases     = np.unwrap(np.angle(hilb_joint_positions))

        # Adjust for delay
        joint_phases_ref[:n_delay] = 0

        # Align phases to second peak after delay
        peaks_ref = find_peaks( joint_positions_ref )[0]
        peaks     = find_peaks( joint_positions )[0]

        n_peaks_ref = len(peaks_ref)
        n_peaks     = len(peaks)

        if n_peaks_ref:
            peak_ind          = min(np.argmax(peaks_ref > n_delay) + 1, n_peaks_ref - 1)
            first_peak_ref    = peaks_ref[peak_ind]
            joint_phases_ref -= joint_phases_ref[first_peak_ref]

        if n_peaks:
            peak_ind      = min(np.argmax(peaks > n_delay) + 1, n_peaks - 1)
            first_peak    = peaks[peak_ind]
            joint_phases -= joint_phases[first_peak]

        # Convert COM positions to body length units
        com_positions_ref = com_positions_ref / body_length
        com_positions     = com_positions / body_length

        # Data to compare
        com_positions_diff = com_positions_ref - com_positions
        joint_phases_diff  = joint_phases_ref - joint_phases

        # Normalize phases
        joint_phases_diff = np.mod(joint_phases_diff, 2*np.pi)
        joint_phases_diff[joint_phases_diff > np.pi] -= 2*np.pi

        # Integral of the phase difference
        phi1                = joint_phases[n_delay:]
        phi0                = joint_phases_ref[n_delay:]
        n_cycles            = (phi1[-1] - phi1[0]) / (2*np.pi)
        n_cycles_ref        = (phi0[-1] - phi0[0]) / (2*np.pi)
        freq_lockking_ratio = n_cycles / n_cycles_ref

        logging.info(f"Frequency locking ratio: {freq_lockking_ratio * 100:.2f}%")

        # Plot data
        # phases_norm                               = np.mod(joint_phases, 2*np.pi)
        # phases_norm_ref                           = np.mod(joint_phases_ref, 2*np.pi)
        # phases_norm[phases_norm > np.pi]         -= 2*np.pi
        # phases_norm_ref[phases_norm_ref > np.pi] -= 2*np.pi
        # phases_norm                              /= 2*np.pi
        # phases_norm_ref                          /= 2*np.pi

        # plt.plot(phases_norm,     label='phase sim', c='b', lw=0.5, ls='--')
        # plt.plot(phases_norm_ref, label='phase ref', c='r', lw=0.5, ls='--')
        # plt.plot(joint_positions,       label='sim', c='b')
        # plt.plot(joint_positions_ref,   label='ref', c='r')
        # plt.legend()

        ########################################################
        # All Data #############################################
        ########################################################

        self.data_com_pos_vs_joint_phase = pd.DataFrame(
            {
                'times'              : times,
                'joint_positions'    : joint_positions,
                'joint_positions_ref': joint_positions_ref,
                'joint_phases'       : joint_phases,
                'joint_phases_ref'   : joint_phases_ref,
                'com_positions'      : com_positions,
                'com_positions_ref'  : com_positions_ref,
                'com_positions_diff' : com_positions_diff,
                'joint_phases_diff'  : joint_phases_diff,
            }
        )

        if save_data:
            current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
            file_name    = f'com_vs_angle_data_{current_time}.csv'

            self.data_com_pos_vs_joint_phase.to_csv(f'{self.results_data_folder}/{file_name}')

        return
