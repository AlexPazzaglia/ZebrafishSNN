import logging
import dill
import json
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert, butter, filtfilt
from network_modules.simulation.mechanical_simulation import MechSimulation
from network_modules.vortices.body_loader import load_body_from_parameters

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

    # External forces
    'mech_force_total_x',
    'mech_force_total_y',
    'mech_force_total_z',

    # Torque and energy
    'mech_torque_total',
    'mech_torque_active',
    'mech_energy_total',
    'mech_energy_active',
    'mech_power_active',
    'mech_power_total',
    'mech_cot',

    # Trajectory
    'mech_traj_mse',
    'mech_traj_curv',

    # Tail beat
    'mech_tail_beat_std',
    'mech_tail_beat_amp',
    'mech_tail_beat_amp_std',

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
    'mech_points_disp_mean',
    'mech_points_disp_amp',

    # Displacements normalized
    'mech_points_disp_amp_bl',
]

MECH_METRICS_VECT_JOINTS = [
    'mech_joints_disp_mean',
    'mech_joints_disp_amp',
]

MECH_METRICS_VECT_LINKS = [
    'mech_points_disp_mean',
    'mech_points_disp_amp',
    'mech_points_disp_amp_bl',
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

    def _add_joints_commands_arrays(
        self,
        data_arrays: dict[str, np.ndarray],
    ):

        if self.snn_network.control_type in [
            'position_control',
            'hybrid_position_control'
        ]:
            data_arrays['joints_commands']    = None
            data_arrays['joints_tau_act_sum'] = None
            data_arrays['joints_tau_act_dif'] = None
            return data_arrays

        mech_pars   = self.snn_network.params.mechanics
        n_ax_joints = mech_pars.mech_axial_joints
        n_lb_joints = mech_pars.mech_limbs_joints

        n_comm = 2 * (n_ax_joints + n_lb_joints)
        j_comm = np.array( self.sim_res_data.state.array )
        j_comm = j_comm[:, :n_comm]
        j_ang  = data_arrays['joints_positions']

        # NOTE:
        # tau_act         = alpha * (A1-A2) + beta * (A1 + A2) * th
        # tau_act(th = 0) = alpha * (A1-A2)

        act1 = j_comm[:, 1::2]
        act2 = j_comm[:, 0::2]

        muscles = self.animat_options['control']['muscles']
        alphas  = np.array( [ m['alpha'] for m in muscles ] )
        betas   = np.array( [ m['beta']  for m in muscles ] )

        tau_act_dif = + (act1 - act2) * alphas
        tau_act_sum = - (act1 + act2) * betas * j_ang

        assert np.allclose(
            tau_act_sum + tau_act_dif,
            data_arrays['joints_active_torques'],
            atol = 1e-12
        ), 'Active torques do not match the commands'

        data_arrays['joints_commands']    = j_comm
        data_arrays['joints_tau_act_sum'] = tau_act_sum
        data_arrays['joints_tau_act_dif'] = tau_act_dif

        spring = data_arrays['joints_spring_torques']
        damper = data_arrays['joints_damping_torques']
        total  = data_arrays['joints_torques']

        assert np.allclose(
            spring + damper + tau_act_sum + tau_act_dif,
            total,
            atol = 1e-12
        ), 'Total torques do not match the sum of spring, damper and active torques'

        return data_arrays

    def load_mechanical_simulation_arrays(self):
        ''' Load mechanical simulation arrays '''

        data        = self.sim_res_data.sensors
        mech_pars   = self.snn_network.params.mechanics
        n_ax_joints = mech_pars.mech_axial_joints
        n_lb_joints = mech_pars.mech_limbs_joints
        n_ax_links  = n_ax_joints + 1

        data_arrays = {
            'joints_positions'     : np.array( data.joints.positions_all() ),
            'joints_velocities'    : np.array( data.joints.velocities_all() ),

            'joints_torques'        : np.array( data.joints.cmd_torques() ),
            'joints_active_torques' : np.array( data.joints.active_torques() ),
            'joints_spring_torques' : np.array( data.joints.spring_torques() ),
            'joints_damping_torques': np.array( data.joints.damping_torques() ),

            'joints_powers'        : np.array( data.joints.mechanical_power() ),
            'joints_active_powers' : np.array( data.joints.mechanical_power_active() ),

            'urdf_positions'       : np.array( data.links.urdf_positions() ),
            'links_positions'      : np.array( data.links.com_positions() ),
            'links_velocities'     : np.array( data.links.com_lin_velocities() ),
            'links_ang_positions'  : np.array( data.links.com_orientations() ),
            'links_ang_veloticies' : np.array( data.links.com_ang_velocities() ),
            'links_forces'         : np.array( data.xfrc.forces() ),
        }

        # Add motor commands (if any)
        data_arrays = self._add_joints_commands_arrays(data_arrays)

        # Take care of last iteration
        trim_last = lambda arr: arr[:-1] if np.any(arr[-1]) else arr

        for arr_name, arr in data_arrays.items():
            if arr is None:
                continue
            data_arrays[arr_name] = trim_last(arr)

        n_steps = min([ len(arr) for arr in data_arrays.values() if arr is not None ])

        for arr_name, arr in data_arrays.items():
            if arr is None:
                continue
            data_arrays[arr_name] = arr[:n_steps]

        #  Compute center of mass position
        links_positions = data_arrays['links_positions']
        links_masses    = self.snn_network.params.mechanics.mech_links_masses
        links_masses    = np.array(links_masses)
        com_positions   = np.sum(
            links_positions[:, :n_ax_links] *
            links_masses[np.newaxis, :, np.newaxis],
            axis= 1
        ) / np.sum(links_masses)

        data_arrays['com_positions'] = com_positions

        # Extend points positions including the tail
        urdf_positions   = data_arrays['urdf_positions']
        lenght_tail_link = self.snn_network.params.mechanics.mech_axial_links_length[-1]

        tail_positions = MechProc._compute_tail_positions_from_points_positions(
            points_positions = urdf_positions,
            joints_positions = data_arrays['joints_positions'],
            length_tail_link = lenght_tail_link,
            n_links_axis     = n_ax_links,
        )
        points_positions = np.insert(
            urdf_positions,
            n_ax_links,
            tail_positions,
            axis = 1,
        )
        data_arrays['points_positions'] = points_positions

        # Extract arrays
        self.joints_positions      = data_arrays['joints_positions']
        self.joints_velocities     = data_arrays['joints_velocities']

        self.joints_torques        = data_arrays['joints_torques']
        self.joints_active_torques = data_arrays['joints_active_torques']
        self.joints_spring_torques = data_arrays['joints_spring_torques']
        self.joints_damping_torques= data_arrays['joints_damping_torques']

        self.joints_powers         = data_arrays['joints_powers']
        self.joints_active_powers  = data_arrays['joints_active_powers']

        self.urdf_positions        = data_arrays['urdf_positions']
        self.points_positions      = data_arrays['points_positions']
        self.links_positions       = data_arrays['links_positions']
        self.links_velocities      = data_arrays['links_velocities']
        self.links_ang_positions   = data_arrays['links_ang_positions']
        self.links_ang_veloticies  = data_arrays['links_ang_veloticies']
        self.links_forces          = data_arrays['links_forces']
        self.com_positions         = data_arrays['com_positions']

        self.joints_commands       = data_arrays['joints_commands']
        self.joints_tau_act_sum    = data_arrays['joints_tau_act_sum']
        self.joints_tau_act_dif    = data_arrays['joints_tau_act_dif']

        return data_arrays

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
        transient_time    = min( 5.0, duration / 2.0 )

        sim_fraction      = 1.0 - transient_time / duration
        n_steps_fraction  = round( n_steps * sim_fraction )
        duration_fraction = timestep * n_steps_fraction

        logging.info(f'Mech_metrics sim fraction: {sim_fraction:.3f}')
        logging.info(f'Mech_metrics sim duration: {duration_fraction:.3f}')

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
        mech_data_arrays = self.load_mechanical_simulation_arrays()

        # Get COM position vs joint phase
        self.get_com_position_vs_joint_phase_data(
            points_positions = self.points_positions,
        )

        # Compute metrics
        (
            joints_disp,
            joints_disp_mean,
            _joints_disp_std,
            joints_disp_amp,
        ) = MechProc.compute_joints_displacements_metrics(
            joints_positions = self.joints_positions,
            n_joints_axis    = n_ax_joints,
            sim_fraction     = sim_fraction,
        )

        (
            points_disp,
            points_disp_mean,
            points_disp_std,
            points_disp_amp,
            points_disp_amp_std,
            points_disp_data,
        ) = MechProc.compute_points_displacements_metrics(
            points_positions = self.points_positions,
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
            joints_positions = self.joints_positions,
            n_joints_axis    = n_ax_joints,
            n_joints_limbs   = n_lb_joints,
            timestep         = timestep,
            sim_fraction     = sim_fraction,
        )

        plot_joint_phase = False
        if plot_joint_phase:
            self.plot_phase_evolution(
                joints_positions = self.joints_positions,
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
            joints_angles     = self.joints_positions,
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

        (
            n2m_lag_ax_all,
            n2m_lag_ax_trk,
            n2m_lag_ax_tal,
            n2m_lag_lb_all,
            n2m_lags_all,
        ) = MechProc.compute_joints_neuro_muscolar_ipls(
            joints_commands = self.joints_commands,
            joints_angles   = self.joints_positions,
            joints_freqs    = freq_joints,
            n_joints_axis   = n_ax_joints,
            n_active_joints = n_ax_joints + n_lb_joints,
            limb_pairs_inds = inds_lb_pairs,
            timestep        = timestep,
        )

        speed_fwd, speed_lat, speed_abs = MechProc.compute_speed(
            points_positions_pca = points_disp_data['points_positions_transformed'],
            n_links_axis         = n_ax_points,
            duration             = duration_fraction,
        )

        traj_mse = MechProc.compute_trajectory_linearity(
            timestep      = timestep,
            com_positions = self.com_positions,
            sim_fraction  = sim_fraction
        )

        traj_curv = MechProc.compute_trajectory_curvature(
            timestep     = timestep,
            com_pos      = self.com_positions,
            sim_fraction = sim_fraction
        )

        force_total = MechProc.sum_forces(
            links_forces = self.links_forces,
            sim_fraction = sim_fraction,
        )

        torque_total = MechProc.sum_torques(
            joints_torques = self.joints_torques,
            sim_fraction   = sim_fraction
        )

        torque_active = MechProc.sum_torques(
            joints_torques = self.joints_active_torques,
            sim_fraction   = sim_fraction
        )

        energy_total = MechProc.sum_powers(
            joints_powers = self.joints_powers,
            timestep      = timestep,
            sim_fraction  = sim_fraction,
            positive_only = True,
        )

        energy_active = MechProc.sum_powers(
            joints_powers = self.joints_active_powers,
            timestep      = timestep,
            sim_fraction  = sim_fraction,
            positive_only = True,
        )

        # Derived metrics
        lenght_ax         = self.snn_network.params.mechanics.mech_axial_length
        tail_beat_std     = points_disp_std[-1]
        tail_beat_amp     = points_disp_amp[-1]
        tail_beat_amp_std = points_disp_amp_std[-1]
        distance_fwd      = speed_fwd * duration_fraction
        power_active      = energy_active / duration_fraction
        power_total       = energy_total / duration_fraction
        cost_of_transport = energy_total / np.abs( max(distance_fwd, np.finfo(float).eps))
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

        links_displacements_amp_bl  = points_disp_amp / lenght_ax

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

            # External forces
            'mech_force_total_x'   : force_total[0],
            'mech_force_total_y'   : force_total[1],
            'mech_force_total_z'   : force_total[2],

            # Torque and energy
            'mech_torque_total'    : torque_total,
            'mech_torque_active'   : torque_active,
            'mech_energy_total'    : energy_total,
            'mech_energy_active'   : energy_active,
            'mech_power_active'    : power_active,
            'mech_power_total'     : power_total,
            'mech_cot'             : cost_of_transport,

            # Trajectory
            'mech_traj_mse'        : traj_mse,
            'mech_traj_curv'       : traj_curv,

            # Tail beat
            'mech_tail_beat_std'    : tail_beat_std,
            'mech_tail_beat_amp'    : tail_beat_amp,
            'mech_tail_beat_amp_std': tail_beat_amp_std,

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
            'mech_joints_disp_mean': joints_disp_mean.tolist(),
            'mech_joints_disp_amp' : joints_disp_amp.tolist(),
            'mech_points_disp_mean' : points_disp_mean.tolist(),
            'mech_points_disp_amp'  : points_disp_amp.tolist(),

            # Displacements normalized
            'mech_points_disp_amp_bl'  : links_displacements_amp_bl.tolist(),
        }

        assert all( [key in MECH_METRICS for key in mech_metrics.keys()] ), 'Not all metrics are listed'
        assert all( [key in mech_metrics for key in MECH_METRICS] ),        'Not all metrics are computed'

        logging.info(f'MECHANICS METRICS: {json.dumps(mech_metrics, indent=4)}')

        self.all_metrics_data = mech_data_arrays |{
            'mech_metrics'             : mech_metrics,
            'joints_displacements'     : joints_disp,
            'points_displacements'     : points_disp,
            'points_displacements_data': points_disp_data,
            'freq_joints'              : freq_joints,
            'n2m_lags_all'             : n2m_lags_all,
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

        n_steps   = len(times)
        time_step = times[1] - times[0]

        water_dynamics_dict = self.water_dynamics_options
        water_callback      = water_dynamics_dict['callback_class']
        water_path          = water_dynamics_dict['results_path']
        water_pars_path     = f'{water_path}/parameters.yaml'

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
            steady_loop      = water_callback.steady_loop
        )

        # Transformations
        leader_y_evolution *= -1

        if water_dynamics_dict['invert_x']:
            leader_x_evolution = -leader_x_evolution
        if water_dynamics_dict['invert_y']:
            leader_y_evolution = -leader_y_evolution

        translation_x = water_dynamics_dict['translation'][0]
        translation_y = water_dynamics_dict['translation'][1]

        leader_x_evolution += translation_x
        leader_y_evolution += translation_y

        leader_dyn_pos_fun = water_dynamics_dict['pos_offset_function']
        leader_dynamic_pos = np.array( [ leader_dyn_pos_fun(t) for t in times ] ) # leader_times

        leader_x_evolution = ( leader_x_evolution.T + leader_dynamic_pos[:, 0] ).T
        leader_y_evolution = ( leader_y_evolution.T + leader_dynamic_pos[:, 1] ).T

        # Delay
        delay_start     = water_dynamics_dict['delay_start']
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

    def _linearize_phases(
        self,
        times  : np.ndarray,
        phases : np.ndarray,
        plot   : bool = False,
    ):
        ''' Linearize phases to cycles '''

        # Find crossings 2pi
        phases_wrapped   = phases % (2 * np.pi)
        cycle_crossings  = np.where(np.diff(phases_wrapped) < -np.pi)[0].astype(int)

        # Create linearized phases
        phases_lin = np.copy(phases)

        for i in range(len(cycle_crossings) - 1):

            # Get start and end indices
            idx0 = cycle_crossings[i]
            idx1 = cycle_crossings[i + 1]

            v0 = phases[idx0]
            v1 = phases[idx1]

            # Linear interpolation for each cycle
            # phases_lin[idx0:idx1] = (
            #     phases[idx0] +
            #     np.linspace(0, 2 * np.pi, idx1 - idx0, endpoint=False)
            # )

            phases_lin[idx0:idx1 + 1] = (
                v0 + (v1 - v0) * np.linspace(0, 1, idx1 - idx0 + 1)
            )


        if not plot:
            return phases_lin

        # Plot
        phases_lin_wrapped = phases_lin % (2 * np.pi)

        plt.figure()
        plt.plot(times, phases_wrapped, label='phases')
        plt.plot(times, phases_lin_wrapped, label='phases_lin')
        plt.plot(
            times[cycle_crossings],
            phases_wrapped[cycle_crossings],
            'o',
            color = 'red',
            label = 'crossings'
        )
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (rad)')
        plt.title('Phase Unwrapping Check')
        plt.legend()

        return phases_lin

    def _get_firing_rates(
        self,
        times : np.ndarray,
    ):
        ''' Get PS rate evolution '''

        t_tag = 'times_raw'
        c_tag = 'spike_count_raw'
        net_modules = self.snn_network.params.topology.network_modules

        target_modules = {
            'mn' : net_modules['mn']['axial']['mn'],
            'ps' : net_modules['ps']['axial']['ps'],
            'ex' : net_modules['cpg']['axial']['ex'],
            'in' : net_modules['cpg']['axial']['in'],
        }
        firing_rates = {}

        for mod_name, mod in target_modules.items():

            # Get spike count
            c_times = self.snn_network.smooth_activations[mod_name][t_tag]
            c_vals  = self.snn_network.smooth_activations[mod_name][c_tag]

            # Convert to firing rate
            timestep   = float( c_times[1] - c_times[0] )
            rates_raw  = c_vals / timestep / mod.n_pool_side

            # Smooth over time window
            sigma_t  = 0.010
            sigma_n  = sigma_t / timestep
            rates    = gaussian_filter1d(rates_raw, sigma=sigma_n)
            c_vals_f = gaussian_filter1d(c_vals, sigma=sigma_n)

            # Left and right rates
            rate_raw_l = np.mean(rates_raw[0::2], axis=0)
            rate_raw_r = np.mean(rates_raw[1::2], axis=0)
            rate_l     = np.mean(rates[0::2], axis=0)
            rate_r     = np.mean(rates[1::2], axis=0)

            # Compute left-right difference
            rate_diff = rate_l - rate_r
            rate_diff = np.interp(times, c_times, rate_diff)

            # Store results
            firing_rates[mod_name] = {
                'times'     : c_times,
                'counts'    : c_vals,
                'counts_f'  : c_vals_f,
                'rates_raw' : rates_raw,
                'rates'     : rates,
                'rate_l'    : rate_l,
                'rate_r'    : rate_r,
                'rate_raw_l': rate_raw_l,
                'rate_raw_r': rate_raw_r,
                'rate_diff' : rate_diff,
            }

        # DEBUG PLOT
        # mod      = 'mn'
        # key      = 'counts'
        # c_vals   = firing_rates[mod][key]
        # c_vals_l = np.sum( c_vals[0::2], axis=0)
        # c_vals_r = np.sum( c_vals[1::2], axis=0)

        # c_vals_l_f = gaussian_filter1d(c_vals_l, sigma=sigma_n)
        # c_vals_r_f = gaussian_filter1d(c_vals_r, sigma=sigma_n)

        # plt.plot(c_vals_l)
        # plt.plot(c_vals_r)
        # plt.plot(c_vals_l_f, color='0.5', linewidth=1.0)
        # plt.plot(c_vals_r_f, color='0.5', linewidth=1.0)

        return firing_rates

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

        # Linearize phases
        linearize = True

        if linearize:
            joint_phases_ref = self._linearize_phases(times, joint_phases_ref)
            joint_phases     = self._linearize_phases(times, joint_phases)

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
        if n_delay < n_iterations:
            phi1            = joint_phases[n_delay:]
            phi0            = joint_phases_ref[n_delay:]
            n_cycles        = (phi1[-1] - phi1[0]) / (2*np.pi)
            n_cycles_ref    = (phi0[-1] - phi0[0]) / (2*np.pi)
            freq_lock_ratio = n_cycles / n_cycles_ref

            logging.info(f"Frequency lock ratio: {freq_lock_ratio * 100:.2f}%")

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

        # Fluid-body coherence
        (
            coherence_pos_vel,
            coherence_vel_vel,
            coherence_acc_vel,
            coherence_ang_vor,
        ) = self.compute_fluid_body_coherence()

        # Propulsion forces
        propulsion_forces = self.compute_propulsion_forces()

        # Firing rate
        firing_rates   = self._get_firing_rates(times)

        # firing_rate_ex = firing_rates['ex']['rates']
        # firing_rate_in = firing_rates['in']['rates']
        # firing_rate_mn = firing_rates['mn']['rates']
        # firing_rate_ps = firing_rates['ps']['rates']

        # mod = 'ps'
        # seg = 10
        # plt.plot( firing_rates[mod]['rates'][2 * seg + 0] )
        # plt.plot( firing_rates[mod]['rates'][2 * seg + 1] )

        # mod = 'mn'
        # plt.plot( np.sum( firing_rates[mod]['rates'][0::2], axis=0 ) )
        # plt.plot( np.sum( firing_rates[mod]['rates'][1::2], axis=0 ) )

        ########################################################
        # All Data #############################################
        ########################################################

        def _counts_to_dict(tag, rates):
            return {
                f'{tag}_spike_count_{ind}' : rate
                for ind, rate in enumerate(rates)
            }

        def _joints_to_dict(tag, joints_array):
            n_steps, n_joints = joints_array.shape
            if n_steps == times.shape[0] - 1:
                joints_array = np.vstack([np.zeros(n_joints), joints_array])
            return {
                f'joint_{ind}_{tag}' : joints_array[:, ind]
                for ind in range(joints_array.shape[1])
            }

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

                'coherence_pos_vel'  : np.mean(coherence_pos_vel, axis=1),
                'coherence_vel_vel'  : np.mean(coherence_vel_vel, axis=1),
                'coherence_acc_vel'  : np.mean(coherence_acc_vel, axis=1),
                'coherence_ang_vor'  : coherence_ang_vor,

                'propulsion_fx'      : propulsion_forces[:, 0],
                'propulsion_fy'      : propulsion_forces[:, 1],
            } |

            # Add spike counts
            _counts_to_dict('ex', firing_rates['ex']['counts']) |
            _counts_to_dict('in', firing_rates['in']['counts']) |
            _counts_to_dict('mn', firing_rates['mn']['counts']) |
            _counts_to_dict('ps', firing_rates['ps']['counts']) |

            # Add joints quantities
            _joints_to_dict('pos', self.joints_positions) |
            _joints_to_dict('vel', self.joints_velocities) |
            _joints_to_dict('trq_spring', self.joints_spring_torques) |
            _joints_to_dict('trq_damping', self.joints_damping_torques) |
            _joints_to_dict('trq_tot', self.joints_torques) |
            _joints_to_dict('trq_act_sum', self.joints_tau_act_sum) |
            _joints_to_dict('trq_act_dif', self.joints_tau_act_dif)

        )

        if save_data:
            current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
            file_name    = f'com_vs_angle_data_{current_time}.csv'
            file_path    = f'{self.results_data_folder}/{file_name}'

            print(f'Saving COM position vs joint phase data to {file_path}')

            self.data_com_pos_vs_joint_phase.to_csv(file_path, index=False)

        # # Filtering coherence signal
        # nyquist       = 0.5 / timestep
        # normal_cutoff = 1.0 / nyquist
        # b, a = butter(2, normal_cutoff, btype='low')

        # coherence_vv = np.mean(coherence_vel_vel, axis=1)
        # coherence_vv = filtfilt(b, a, coherence_vv)

        # # Joints segmentation
        # frequency    = 3.36
        # steps_period = round(1 / (frequency * timestep))

        # n_steps = 10_000
        # step_on = 15_000

        # joints_off   = self.joints_positions[ step_on - 10_000 : step_on ]
        # joints_trans = self.joints_positions[ step_on : step_on + 2_000 ]
        # joints_on    = self.joints_positions[ step_on : step_on + n_steps ]


        # def segment_by_extrema(signal, steps_half_period):
        #     """
        #     Segment a 1D oscillatory signal based on both minima and maxima.
        #     Returns indices of extrema and corresponding segment boundaries.
        #     """
        #     # Find maxima
        #     maxima_idx, _ = find_peaks(signal, distance=steps_half_period)
        #     # Find minima
        #     minima_idx, _ = find_peaks(-signal, distance=steps_half_period)

        #     # Start with a minimum
        #     maxima_idx = maxima_idx[maxima_idx > min(minima_idx)]

        #     # Combine and sort
        #     extrema_idx = np.sort(np.concatenate([maxima_idx, minima_idx]))

        #     # Remove too-close extrema (e.g. flat noise)
        #     extrema_idx = np.unique(extrema_idx)


        #     # Build segments between consecutive extrema (half-periods)
        #     segments = [(extrema_idx[i], extrema_idx[i+1]) for i in range(len(extrema_idx)-1)]

        #     return extrema_idx, segments

        # def segment_all_joints(joints, steps_period):
        #     """
        #     Apply segmentation to each joint independently.
        #     Returns:
        #         extrema_all: list of extrema indices per joint
        #         segments_all: list of (start, end) index tuples per joint
        #     """
        #     steps_half_period = int(steps_period // 2)
        #     extrema_all, segments_all = [], []

        #     for j in range(joints.shape[1]):
        #         signal = joints[:, j]
        #         extrema_idx, segs = segment_by_extrema(signal, steps_half_period)
        #         extrema_all.append(extrema_idx)
        #         segments_all.append(segs)

        #     return extrema_all, segments_all

        # def plot_superimposed_periods(signal, segments, title, color="tab:blue", normalize_length=False):
        #     """
        #     Plot all detected periods (segments) superimposed, aligned at time 0.

        #     Parameters
        #     ----------
        #     signal : np.ndarray
        #         1D array of joint position values.
        #     segments : list of (start_idx, end_idx)
        #         List of segment boundaries. Each (start, end) defines one half- or full-period.
        #     title : str
        #         Title for the plot.
        #     color : str
        #         Line color for all periods.
        #     normalize_length : bool
        #         If True, resample each segment to the same length (for clean overlay).
        #     """

        #     for (start, end) in segments:
        #         segment = signal[start:end]

        #         if len(segment) < 3:
        #             continue  # skip too short segments

        #         # normalize length if desired
        #         if normalize_length:
        #             segment = np.interp(
        #                 np.linspace(0, 1, 200),  # common normalized time axis
        #                 np.linspace(0, 1, len(segment)),
        #                 segment
        #             )
        #             t = np.linspace(0, 1, len(segment))
        #         else:
        #             t = np.arange(len(segment))

        #         plt.plot(t, segment, color=color, alpha=0.3)

        #     plt.title(title)
        #     plt.xlabel("Normalized time (period start → end)")
        #     plt.ylabel("Joint position")
        #     plt.tight_layout()

        #     return

        # extrema_on , segments_on  = segment_all_joints(joints_on, steps_period)
        # extrema_off, segments_off = segment_all_joints(joints_off, steps_period)

        # # Get extrema and segments for one joint

        # joint_id = 0

        # signal_off = joints_off[:, joint_id]
        # signal_on  = joints_on[:, joint_id]

        # segments_off_joint = segments_off[joint_id]
        # segments_on_joint  = segments_on[joint_id]

        # # (Optional) Convert from half-periods to full periods (min→min or max→max)
        # # This makes the overlaid waveforms cleaner and symmetric
        # full_segments_off = [(segments_off_joint[2*i][0], segments_off_joint[2*i+1][1])
        #                     for i in range(len(segments_off_joint)//2)]
        # full_segments_on  = [(segments_on_joint[2*i][0], segments_on_joint[2*i+1][1])
        #                     for i in range(len(segments_on_joint)//2)]

        # # Plot superimposed periods
        # plt.figure(figsize=(8, 4))
        # plot_superimposed_periods(signal_off, full_segments_off, f"Joint {joint_id}", color="tab:red")
        # plot_superimposed_periods(signal_on,  full_segments_on,  f"Joint {joint_id}", color="tab:green")


        return

    def compute_fluid_body_coherence(self):
        ''' Compute the coherence between the fluid and the body '''

        # Parameters
        start_delay = self.water_dynamics_options['delay_start']
        timestep    = self.sim_res_data.timestep
        n_delay     = round(start_delay / timestep)
        n_iterations = self.sim_res_data.sensors.joints.size(0)

        if n_delay == n_iterations:
            n_delay = 0

        # Get data
        dx = dy = 0.0001
        velocity_cb      = self.water_dynamics_options['velocity_callback']
        links_velocities = self.links_velocities
        links_positions  = self.links_positions

        fluid_velocities  = np.zeros_like(links_velocities)
        fluid_vorticities = np.zeros(links_velocities.shape[:2])

        for t_ind in range(links_velocities.shape[0]):
            t = t_ind * self.sim_res_data.timestep

            for p_ind in range(links_velocities.shape[1]):

                x, y, z = links_positions[t_ind, p_ind]

                # Base velocity at link position
                fluid_velocities[t_ind, p_ind] = velocity_cb(t, x, y, z)

                # Base velocities at offset positions
                v_dx_pos = velocity_cb(t, x + dx, y, z)
                v_dx_neg = velocity_cb(t, x - dx, y, z)
                v_dy_pos = velocity_cb(t, x, y + dy, z)
                v_dy_neg = velocity_cb(t, x, y - dy, z)

                # Central difference derivatives
                dvy_dx = (v_dx_pos[1] - v_dx_neg[1]) / (2 * dx)
                dvx_dy = (v_dy_pos[0] - v_dy_neg[0]) / (2 * dy)

                # 2D vorticity component (z-component of curl)
                fluid_vorticities[t_ind, p_ind] = dvy_dx - dvx_dy


        # Remove speed offset
        fluid_vel_off     = velocity_cb(0, np.inf, np.inf, np.inf)
        fluid_velocities -= fluid_vel_off

        # Extract xy components (n_steps, n_links, 2)
        links_pos = links_positions[:, :, :2]
        links_vel = links_velocities[:, :, :2]
        fluid_vel = fluid_velocities[:, :, :2]

        links_ang = self.links_ang_veloticies[:, :, 2]
        fluid_vor = fluid_vorticities[:, :]

        links_acc     = np.zeros_like(links_vel)
        links_acc[1:] = np.diff(links_vel, axis=0) / timestep

        assert links_vel.shape == fluid_vel.shape, 'Arrays must have the same shape'
        assert links_acc.shape == fluid_vel.shape, 'Arrays must have the same shape'
        assert links_ang.shape == fluid_vor.shape, 'Arrays must have the same shape'

        # Pad array with zeros to match n_iterations
        links_pos = self._pad_array(links_pos, n_iterations)
        links_vel = self._pad_array(links_vel, n_iterations)
        links_acc = self._pad_array(links_acc, n_iterations)
        fluid_vel = self._pad_array(fluid_vel, n_iterations)

        links_ang = self._pad_array(links_ang, n_iterations)
        fluid_vor = self._pad_array(fluid_vor, n_iterations)

        # Average coherence
        mean_pos_vel, coherence_pos_vel = self._compute_coherence(
            arr1    = links_pos,
            arr2    = fluid_vel,
            n_delay = n_delay,
        )
        mean_vel_vel, coherence_vel_vel = self._compute_coherence(
            arr1    = links_vel,
            arr2    = fluid_vel,
            n_delay = n_delay,
        )
        mean_acc_vel, coherence_acc_vel = self._compute_coherence(
            arr1    = links_acc,
            arr2    = fluid_vel,
            n_delay = n_delay,
        )
        mean_ang_vor, coherence_ang_vor = self._compute_coherence(
            arr1    = links_ang,
            arr2    = fluid_vor,
            n_delay = n_delay,
        )

        logging.info(f'Coherence links pos - fluid vel: {mean_pos_vel:.4e}')
        logging.info(f'Coherence links vel - fluid vel: {mean_vel_vel:.4e}')
        logging.info(f'Coherence links acc - fluid vel: {mean_acc_vel:.4e}')
        logging.info(f'Coherence links ang - fluid vor: {mean_ang_vor:.4e}')

        # # Filtering coherence signal
        # nyquist       = 0.5 / timestep
        # normal_cutoff = 1.0 / nyquist
        # b, a = butter(2, normal_cutoff, btype='low')

        # c_signal   = coherence_vel_vel
        # # c_signal   = coherence_ang_vor

        # if len(c_signal.shape) == 2:
        #     c_signal = np.mean(c_signal, axis=1)
        # c_signal_f = filtfilt(b, a, c_signal)

        # plt.figure()
        # plt.plot(c_signal)
        # plt.plot(c_signal_f)
        # plt.xlabel('Time')
        # plt.ylabel('Coherence')
        # plt.title('Coherence')
        # plt.show()

        return (
            coherence_pos_vel, coherence_vel_vel,
            coherence_acc_vel, coherence_ang_vor,
        )

    def _compute_coherence(self, arr1, arr2, n_delay=0):
        ''' Compute the coherence between two arrays '''

        # Compute dot product (F · v) for each link and timestep
        dot_products = np.sum(arr1 * arr2, axis=-1)  # (n_steps, n_signals)

        # Compute magnitudes of F and v
        norm1 = np.linalg.norm(arr1, axis=-1)  # (n_steps, n_signals))
        norm2 = np.linalg.norm(arr2, axis=-1)  # (n_steps, n_signals))

        # Avoid division by zero
        coherence       = np.zeros_like(dot_products)
        mask            = (norm1 > 0) & (norm2 > 0)
        mask_norm       = norm1[mask] * norm2[mask]
        coherence[mask] = dot_products[mask] / mask_norm

        # Average coherence
        mean_coherence = np.mean(coherence[n_delay:])

        # v1 = np.copy(arr1)
        # v1[:, :, 0] /= norm1
        # v1[:, :, 1] /= norm1
        # v2 = np.copy(arr2)
        # v2[:, :, 1] /= norm2
        # v2[:, :, 0] /= norm2

        # step  = 20_000
        # point = 10

        # for step in np.arange(0, v1.shape[0], 1000):
        #     plt.plot(
        #         [0, v1[step, point, 0]],
        #         [0, v1[step, point, 1]],
        #         'b',
        #     )
        #     plt.plot(
        #         [0, v2[step, point, 0]],
        #         [0, v2[step, point, 1]],
        #         'r',
        #     )
        # plt.axis('equal')

        return mean_coherence, coherence

    def _pad_array(self, arr, n_iterations):
        ''' Pad array to match n_iterations '''
        n_pads = n_iterations - arr.shape[0]
        if n_pads > 0:
            pad = [ arr[0] ] * n_pads
            arr = np.vstack([pad, arr])
        elif n_pads < 0:
            raise ValueError('Len(arr) > n_iterations, cannot pad array')
        return arr

    def compute_propulsion_forces(self):
        ''' Compute the coherence between the fluid and the body '''

        # Parameters
        start_delay  = self.water_dynamics_options['delay_start']
        timestep     = self.sim_res_data.timestep
        n_delay      = round(start_delay / timestep)
        n_iterations = self.sim_res_data.sensors.joints.size(0)

        if n_delay == n_iterations:
            n_delay = 0

        # Get data
        links_forces = self.links_forces
        total_forces = np.sum(links_forces, axis=1)
        total_forces = self._pad_array(total_forces, n_iterations)

        # Average forces
        mean_force = np.mean(total_forces[n_delay:], axis=0)
        logging.info(f'Mean propulsion Fx: {mean_force[0]:.4e}')
        logging.info(f'Mean propulsion Fy: {mean_force[1]:.4e}')

        return total_forces
