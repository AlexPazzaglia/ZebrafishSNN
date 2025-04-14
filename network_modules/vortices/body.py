
import os
import torch
import numpy as np
import pandas as pd

from scipy.interpolate import interp2d

import network_modules.vortices.plot_fish as zebraplot

from network_modules.vortices.load_data import get_experimental_signal
from network_modules.vortices.extract_sinusoidal_signal import (
    get_sinusoidal_signal,
    get_real_time_phase_fun,
    get_real_time_amp_fun,
)

"""
Analitical SDFs
"""
def circle(x,y,xt=0,yt=60,r=25):
    return torch.sqrt((x-xt)**2+(y-yt)**2)-r

def box(x,y,xb=20,yb=20):
    qx=torch.abs(x)-xb
    qy=torch.abs(y)-yb
    return torch.sqrt(
        torch.maximum(qx,torch.zeros_like(x))**2 +
        torch.maximum(qy,torch.zeros_like(y))**2
    )+torch.minimum(torch.maximum(qx,qy),torch.zeros_like(x))


def body_from_yaml(device, x, y, body_pars, eps=0.05, costum_update=None, **kwargs):

    type = body_pars["type"]

    if type == "fish_analytical":
        control_pars = body_pars["control"]
        return ZebrafishAnalytical(
            device, x, y,
            eps           = eps,
            body_length   = control_pars["body_length"],
            amp_scaling   = control_pars["amp_scaling"],
            frequency     = control_pars["frequency"],
            wavefrequency = control_pars["wavefrequency"],
            xshift        = control_pars["xshift"],
            yshift        = control_pars["yshift"],
            timestep      = control_pars["timestep"],
            duration      = control_pars["duration"],
        )

    elif type == "fish_experimental_continuous":
        control_pars = body_pars["control"]
        return ZebrafishExperimentalContinuous(
            device, x, y,
            eps             = eps,
            body_length     = control_pars["body_length"],
            wave_number     = control_pars["wave_number"],
            folder_name     = control_pars["folder_name"],
            file_name       = control_pars["file_name"],
            save_data       = control_pars["save_data"],
            plot_data       = control_pars["plot_data"],
            target_fish     = control_pars["target_fish"],
            signal_name     = control_pars["signal_name"],
            start_recording = control_pars["start_recording"],
            end_recording   = control_pars["end_recording"],
            timestep        = control_pars["timestep"],
            total_duration  = control_pars["total_duration"],
            freq_scaling    = control_pars["freq_scaling"],
            amp_scaling     = control_pars["amp_scaling"],
            amp_modulation  = control_pars.get("amp_modulation", False),
            freq_min        = control_pars.get("freq_min", None),
            freq_max        = control_pars.get("freq_max", None),
            xshift          = control_pars["xshift"],
            yshift          = control_pars["yshift"],
        )

class Body:

    def __init__(self, device, x, y, eps=0.05, **kwargs):
        """

        """
        self.device=device
        self.dtype = x.dtype

        self.x   = x
        self.y   = y
        self.X, self.Y = torch.meshgrid(x,y,indexing="ij")
        self.nx  = len(x)
        self.ny  = len(y)
        self.dx = float(x[1]-x[0])
        self.dy = float(y[1]-y[0])
        self.eps = eps

        self.xflat = self.X.flatten()
        self.yflat = self.Y.flatten()
        self.stacked_xy = torch.stack((self.xflat,self.yflat))
        self.ones_stacked=torch.ones(self.nx*self.ny).to(self.device)

        self.oldpos_u = torch.zeros((self.nx,self.ny),device=self.device)
        self.oldpos_v = torch.zeros((self.nx,self.ny),device=self.device)

        # body velocities
        self.body_u = torch.zeros((self.nx,self.ny),device=self.device)
        self.body_v = torch.zeros((self.nx,self.ny),device=self.device)
        self.old_points = torch.stack((torch.zeros((self.nx,self.ny),device=self.device).flatten(),torch.zeros((self.nx,self.ny),device=self.device).flatten()))

        self.rad_conv = (torch.pi/180)


    def compute_sdf_properties(self, sdf_val):

        (gradx, grady) = torch.gradient(sdf_val, spacing=[self.dx, self.dy])

        norm = torch.sqrt(gradx**2+grady**2)

        # compute curvature
        numerator = (
            (grady**2)*torch.gradient(gradx, spacing=self.dx, axis=0)[0]+
            (gradx**2)*torch.gradient(grady, spacing=self.dy, axis=1)[0]+
            -2*gradx*grady*torch.gradient(grady, spacing=self.dx, axis=0)[0]
        )
        denominator = norm**3

        # numerator = torch.gradient(gradx, dim=0, spacing=self.dx)[0]+torch.gradient(grady, dim=1, spacing=self.dy)[0]
        # denominator = (gradx**2+grady**2)**1.5

        curvature = torch.where(denominator>0, numerator/denominator, 0)

        # normalize gradients
        gradx=torch.where(norm>0, gradx/norm, 0)
        grady=torch.where(norm>0, grady/norm, 0)

        return (
            sdf_val,
            gradx,
            grady,
            curvature,
        )

    def phi(self,d):
        return torch.where(
            torch.abs(d)<self.eps,
            ( 1 + torch.cos(torch.pi*d/self.eps) )/( 2*self.eps ),
            0
        )

    def mu_funcs(self, d):
        s=torch.sin(torch.pi*d/self.eps)
        c=torch.cos(torch.pi*d/self.eps)
        mu_0_eps = torch.where(
            d<=-self.eps,
            0,
            torch.where(
                d>=self.eps,
                1,
                0.5*( 1 + d/self.eps + s/torch.pi )
            )
        )

        mu_1_eps = torch.where(
            torch.abs(d)>=self.eps,
            0,
            self.eps*( 0.25 - (d/(2*self.eps))**2 - ( d*s/self.eps+(1+c)/torch.pi )/(2*torch.pi) )
        )
        return (mu_0_eps, mu_1_eps)



    def update_body(self, fun, theta, transl, dt=1):
        """
        Update sdf properties from analytical rototranslation map
        """
        theta = torch.tensor(theta*self.rad_conv, device=self.device).clone().detach()
        s = torch.sin(theta)
        c = torch.cos(theta)
        rot = torch.stack([torch.stack([c, s]),
                        torch.stack([-s, c])]).to(self.device)
        trans = torch.stack((transl[0]*self.ones_stacked, transl[1]*self.ones_stacked))

        # newpoints=rot.T@self.stacked_xy-trans

        newpoints=self.stacked_xy-trans
        newpoints=rot@newpoints

        # newpos = self.stacked_xy+trans
        # newpos = rot@newpos

        # newpos=rot@self.stacked_xy+trans

        vel = - rot.T @ (newpoints - self.old_points) / dt

        newpos_u = newpoints[0].reshape(self.nx, self.ny)
        newpos_v = newpoints[1].reshape(self.nx, self.ny)

        # self.body_uprev = self.body_u
        # self.body_vprev = self.body_v

        self.body_u= vel[0].reshape(self.nx, self.ny)
        self.body_v= vel[1].reshape(self.nx, self.ny)

        self.old_points = newpoints

        # self.oldpos_u = newpos_u
        # self.oldpos_v = newpos_v

        sdf_val = fun(newpos_u, newpos_v)

        return self.compute_sdf_properties(sdf_val)

# ZEBRAFISH PROPERTIES

class ZebrafishProperties():

    def __init__(
        self,
        device         = 'cpu',
        body_length    = 0.018,
        amp_scaling    = 1.0,
        body_type      = 'analytical',
        thickness_type = 'sketch',
    ):

        self.device      = device
        self.body_length = body_length
        self.amp_scaling = amp_scaling

        # Envelope (Di Santo et al. 2021 - All Fishes)
        # NOTE: Model defines the peal-to-peak amplitude         -> 0.5 factor
        # NOTE: max(envelope) = 0.20 while max(zebrafish) = 0.24 -> 1.2 factor
        self.c0 = +0.05 * 0.6 * self.amp_scaling
        self.c1 = -0.13 * 0.6 * self.amp_scaling
        self.c2 = +0.28 * 0.6 * self.amp_scaling

        # Body type
        # "fish_analytical"
        # "fish_experimental"
        # "fish_experimental_continuous"
        self.body_type = body_type

        # Thickness model
        if thickness_type == 'model':
            self.thickness_f      = zebraplot.get_fish_half_thickness_from_model
            self.thickness_spline = zebraplot._get_fish_half_thickness_spline_from_model(body_length)
            self.thickness_args   = (body_length, self.thickness_spline)

        elif thickness_type == 'gazzola':
            self.thickness_f      = zebraplot.get_fish_half_thickness_from_gazzola
            self.thickness_spline = None
            self.thickness_args   = (body_length,)

        elif thickness_type == 'sketch':
            self.thickness_f      = zebraplot.get_fish_half_thickness_from_sketch
            self.thickness_spline = zebraplot._get_fish_half_thickness_spline_from_sketch(body_length)
            self.thickness_args   = (body_length, self.thickness_spline)

        else:
            raise ValueError(f'Unknown thickness type: {thickness_type}')

        return

    # KINEMATICS

    def envelope(
        self,
        s_norm : torch.Tensor,
    ):
        '''
        Theoretical amplitude envelope of the fish (Di Santo et al. 2021 - All Fishes)
        NOTE: Arclengths are normalized to [0,1]
        '''
        return self.c0 + (self.c1 * s_norm) + (self.c2 * s_norm**2)

    def thk(
        self,
        s_norm : torch.Tensor,
    ):
        '''
        Fish thickness
        NOTE: Arclengths are normalized to [0,1]
        '''
        thk_vals = torch.tensor(
            self.thickness_f(s_norm, *self.thickness_args),
            dtype  = torch.float32,
            device = self.device
        )
        return thk_vals

    def sdf_fun(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
    ):
        ''' Fish signed distance function '''
        s      = x.clamp(0,self.body_length)
        sdf    = torch.sqrt((x-s)**2+y**2)
        s_norm = s / self.body_length
        return sdf-self.thk(s_norm)

    def update_xy_map(
        self,
        XC       : torch.Tensor,
        YC       : torch.Tensor,
        amp_fun  : callable,
        phase_fun: callable,
        time     : float,
    ):
        """
        Update sdf properties from analytical rototranslation map
        """
        s_norm = XC.clamp(0,self.body_length) / self.body_length
        new_x  = XC
        new_dy = (
            self.body_length *
            amp_fun(time, s_norm) *
            np.cos( phase_fun(time, s_norm) )
        )
        new_y = (
            YC +
            torch.tensor(
                new_dy.reshape(XC.shape),
                dtype  = torch.float32,
                device = self.device
            )
        )
        new_sdf = self.sdf_fun(new_x,new_y)
        return new_x, new_y, new_sdf

    # SIGNALS

    def get_amp_and_phase_func_analytical(
        self,
        signal_kwargs : dict,
    ):
        ''' Get the analytical sinusoidal signal '''

        # new_y = (
        #     self.YC +
        #     self.L * self.envelope(s_val) * torch.cos(
        #         2*torch.pi * self.wavefrequency * s_val -
        #         2*torch.pi * self.frequency * t
        #     )
        # )

        # Unpack the arguments
        frequency     = signal_kwargs['frequency']
        wavefrequency = signal_kwargs['wavefrequency']

        signal_temporal_amp_fun   = lambda t: np.ones_like(t)
        signal_spatial_amp_fun    = lambda s: self.envelope(s)
        signal_temporal_phase_fun = lambda t: 2 * np.pi * frequency * t
        signal_spatial_phase_fun  = lambda s: 2 * np.pi * wavefrequency * s

        # Combine the functions
        def signal_amp_fun(t, s):
            t_arr = np.array(t).flatten()
            s_arr = np.array(s).flatten()
            return (
                signal_temporal_amp_fun(t_arr) *
                signal_spatial_amp_fun(s_arr)
            )

        def signal_phase_fun(t, s):
            t_arr = np.array(t).flatten()
            s_arr = np.array(s).flatten()
            return (
                signal_temporal_phase_fun(t_arr) -
                signal_spatial_phase_fun(s_arr)
            )

        return signal_amp_fun, signal_phase_fun

    def get_amp_and_phase_func_experimental_continuous(
        self,
        signal_kwargs : dict,
    ):
        ''' Get the experimental sinusoidal signal '''

        # new_y = (
        #     self.YC +
        #     self.L * self.signal_amp_fun(t) * self.envelope(s_val) * torch.cos(
        #         2 * torch.pi * self.wave_number * s_val -
        #         self.signal_phase_fun(t)
        #     )
        # )

        # Unpack needed arguments
        modulate_amp  = signal_kwargs.get('modulate_amp', False)
        wavefrequency = signal_kwargs.pop('wavefrequency', 0.95)

        # Get the signal
        (
            times,
            signal_vals,
        ) = get_sinusoidal_signal(**signal_kwargs)

        # SPATIAL (Analytical)
        signal_spatial_amp_fun   = lambda s: self.envelope(s)
        signal_spatial_phase_fun = lambda s: 2 * np.pi * wavefrequency * s

        # TEMPORAL (From experimental signal)

        # Amplitude
        if modulate_amp:
            signal_temporal_amp_fun = get_real_time_amp_fun(times, signal_vals)
        else:
            signal_temporal_amp_fun = lambda t: np.ones_like(t)

        # Phase
        signal_temporal_phase_fun = get_real_time_phase_fun(
            times  = times,
            signal = signal_vals,
        )

        # Combine the functions
        def signal_amp_fun(t,s):
            t_arr = np.array(t).flatten()
            s_arr = np.array(s).flatten()
            return (
                signal_temporal_amp_fun(t_arr) *
                signal_spatial_amp_fun(s_arr)
            )

        def signal_phase_fun(t,s):
            t_arr = np.array(t).flatten()
            s_arr = np.array(s).flatten()
            return (
                signal_temporal_phase_fun(t_arr) -
                signal_spatial_phase_fun(s_arr)
            )

        return signal_amp_fun, signal_phase_fun

    def get_amp_and_phase_func_experimental(
        self,
        signal_kwargs : dict,
    ):
        ''' Get the analytical sinusoidal signal '''

        # Get the signal
        points_coords_df = get_experimental_signal(**signal_kwargs)

        # Normalized coordinates
        times       = points_coords_df['time'].values
        points_x    = points_coords_df.filter(regex='x_').values
        points_y    = points_coords_df.filter(regex='y_').values
        points_x[:] = np.mean(points_x, axis=0)

        # INTERPOLATION
        t_vals        = times
        s_vals        = points_x[0]
        sig_evolution = points_y

        amp_evolution   = []
        phase_evolution = []

        # Get amplitude and phase for each spatial coordinate
        for s_ind, s_val in enumerate(s_vals):
            sig_vals = sig_evolution[:, s_ind]

            amp_fun   = get_real_time_amp_fun(t_vals, sig_vals)
            phase_fun = get_real_time_phase_fun(t_vals, sig_vals)

            amp_vals   = amp_fun(t_vals)
            phase_vals = phase_fun(t_vals)

            amp_evolution.append(amp_vals)
            phase_evolution.append(phase_vals)

        amp_evolution   = np.array(amp_evolution)
        phase_evolution = np.array(phase_evolution)

        # Create bilinear interpolation functions
        amp_interp   = interp2d(t_vals, s_vals, amp_evolution, kind='linear')
        phase_interp = interp2d(t_vals, s_vals, phase_evolution, kind='linear')

        # Define the amplitude and phase functions
        def signal_amp_fun(t, s):
            t_arr    = np.array(t).flatten()
            s_arr    = np.array(s).flatten()
            int_vals = amp_interp(t_arr, s_arr).flatten()
            return int_vals

        def signal_phase_fun(t, s):
            t_arr    = np.array(t).flatten()
            s_arr    = np.array(s).flatten()
            int_vals = phase_interp(t_arr, s_arr).flatten()
            return int_vals

        return signal_amp_fun, signal_phase_fun

    def get_amp_and_phase_func(
        self,
        signal_kwargs : dict,
    ):
        ''' Get the amplitude and phase functions '''
        if self.body_type == 'fish_analytical':
            return self.get_amp_and_phase_func_analytical(signal_kwargs)

        if self.body_type == 'fish_experimental':
            return self.get_amp_and_phase_func_experimental(signal_kwargs)

        if self.body_type == 'fish_experimental_continuous':
            return self.get_amp_and_phase_func_experimental_continuous(signal_kwargs)

        raise ValueError(f'Unknown signal type: {self.signal_type}')


    # SAVING

    def get_coordinates_evolution(
        self,
        times,
        signal_amp_fun,
        signal_phase_fun,
        n_points = 10,
        normalize = True,
    ):
        ''' Get the evolution of the coordinates '''
        n_steps               = len(times)
        s_vals                = np.linspace(0, 1, n_points)
        positions_x_evolution = np.array([s_vals] * n_steps)
        positions_y_evolution = np.array(
            [
                signal_amp_fun(times, s_val) *
                np.cos( signal_phase_fun(times, s_val) )
                for s_val in s_vals
            ]
        ).T

        if not normalize:
            positions_x_evolution *= self.body_length
            positions_y_evolution *= self.body_length

        return positions_x_evolution, positions_y_evolution

    def save_signal(
        self,
        folder_name     : str,
        times           : np.ndarray,
        signal_amp_fun  : callable,
        signal_phase_fun: callable,
    ):
        '''
        Save the signal to a csv file
        NOTE: Saved coordinates are normalized to body length
        '''

        (
            positions_x_evolution,
            positions_y_evolution,
        ) = self.get_coordinates_evolution(
            times           = times,
            signal_amp_fun  = signal_amp_fun,
            signal_phase_fun= signal_phase_fun,
        )

        points_names = [
            'Head', 'Hindbrain',
            'SC 1', 'SC 2', 'SC 3', 'SC 4',
            'SC 5', 'SC 6', 'SC 7', 'SC 8',
        ]

        # Create a dictionary with the data
        points_coords_dict = {'time': times}
        for i, point_name in enumerate(points_names):
            points_coords_dict[f'x_{point_name}'] = positions_x_evolution[:, i]
            points_coords_dict[f'y_{point_name}'] = positions_y_evolution[:, i]

        # Create the DataFrame
        points_coords_df = pd.DataFrame(points_coords_dict)

        # Order columns to have time, then x, then y
        columns = ['time'] + [
            f'{coord}_{point_name}'
            for point_name in points_names
            for coord in ['x', 'y']
        ]
        points_coords_df = points_coords_df[columns]

        # Save the DataFrame
        points_coords_df.to_csv(
            os.path.join(folder_name, 'kinematics_signals.csv'),
            index = False,
        )

        return

# ZEBRAFISH BODIES

class ZebrafishBody(Body):

    def __init__(
        self,
        device,
        x,
        y,
        body_length    = 0.018,
        amp_scaling    = 1.0,
        timestep       = 0.001,
        duration       = 30.0,
        body_type      = 'fish_analytical',
        thickness_type = 'sketch',
        xshift         = 0.0,
        yshift         = 0.0,
        eps            = 0.05,
    ):
        super().__init__(device, x, y, eps=eps)
        """

        """
        self.L              = body_length
        self.amp_scaling    = amp_scaling

        # Body type
        # 'fish_analytical', 'fish_experimental', 'fish_experimental_continuous'
        self.body_type = body_type

        # Thickness type
        # 'model', 'gazzola', 'sketch'
        self.thickness_type = thickness_type

        self.timestep = timestep
        self.duration = duration
        self.n_steps  = (self.duration // self.timestep) + 1
        self.times    = np.arange(self.n_steps) * self.timestep

        self.XC = self.X-xshift
        self.YC = self.Y-yshift

        # Zebrafish properties
        self.zebrafish_properties = ZebrafishProperties(
            device,
            body_length    = self.L,
            amp_scaling    = self.amp_scaling,
            body_type      = body_type,
            thickness_type = self.thickness_type,
        )

        # PHASE AND AMPLITUDE FUNCTIONS
        self.signal_amp_fun   : callable = None
        self.signal_phase_fun: callable  = None

        self.bodies = [self]

        return

    def update(self, t, dt=1):
        """
        Update sdf properties from rototranslation map
        """

        new_x, new_y, new_sdf = self.zebrafish_properties.update_xy_map(
            XC        = self.XC,
            YC        = self.YC,
            amp_fun   = self.signal_amp_fun,
            phase_fun = self.signal_phase_fun,
            time      = t,
        )

        self.body_u    = 0
        self.body_v    = -(new_y-self.oldpos_v)/dt
        self.oldpos_v  = new_y
        sdf_properties = self.compute_sdf_properties(new_sdf)

        return [sdf_properties]

    def initialize(self, initial_time=0):
        """ Initialize sdf properties """
        return self.update(initial_time)

    def save_signal(self, folder_name):
        ''' Save the signal to a csv file '''
        self.zebrafish_properties.save_signal(
            folder_name      = folder_name,
            times            = self.times,
            signal_amp_fun   = self.signal_amp_fun,
            signal_phase_fun = self.signal_phase_fun,
        )

class ZebrafishAnalytical(ZebrafishBody):

    def __init__(
        self,
        device,
        x,
        y,
        body_length   = 0.018,
        amp_scaling   = 1.0,
        frequency     = 3.5,
        wavefrequency = 0.95,
        timestep      = 0.001,
        duration      = 30.0,
        xshift        = 0.0,
        yshift        = 0.0,
        eps           = 0.05,
    ):
        super().__init__(
            device, x, y,
            body_length = body_length,
            amp_scaling = amp_scaling,
            timestep    = timestep,
            duration    = duration,
            body_type   = 'fish_analytical',
            xshift      = xshift,
            yshift      = yshift,
            eps         = eps
        )

        # PHASE AND AMPLITUDE FUNCTIONS
        self.frequency     = frequency
        self.wavefrequency = wavefrequency

        signal_kwargs = {
            'frequency'     : self.frequency,
            'wavefrequency' : self.wavefrequency,
        }

        (
            self.signal_amp_fun,
            self.signal_phase_fun,
        ) = self.zebrafish_properties.get_amp_and_phase_func_analytical(signal_kwargs)

        return

class ZebrafishExperimentalContinuous(ZebrafishBody):

    def __init__(
        self,
        device,
        x,
        y,
        body_length,
        wave_number,
        folder_name,
        file_name,
        save_data,
        plot_data,
        target_fish,
        signal_name,
        start_recording,
        end_recording,
        timestep,
        total_duration,
        freq_scaling,
        amp_scaling   = 1.0,
        amp_modulation= False,
        freq_min      = None,
        freq_max      = None,
        xshift        = -0.0,
        yshift        = 0.0,
        eps           = 0.05
    ):
        super().__init__(
            device, x, y,
            body_length = body_length,
            amp_scaling = amp_scaling,
            timestep    = timestep,
            duration    = total_duration,
            body_type   = 'fish_experimental_continuous',
            xshift      = xshift,
            yshift      = yshift,
            eps         = eps
        )

        # PHASE AND AMPLITUDE FUNCTIONS
        self.wavefrequency = wave_number

        signal_kwargs = {
            'folder_name'    : folder_name,
            'file_name'      : file_name,
            'target_fish'    : target_fish,
            'start_recording': start_recording,
            'end_recording'  : end_recording,
            'timestep'       : timestep,
            'total_duration' : total_duration,
            'freq_scaling'   : freq_scaling,
            'save_data'      : save_data,
            'plot_data'      : plot_data,
            'sig_name'       : signal_name,
            'modulate_amp'   : amp_modulation,
            'min_freq'       : freq_min,
            'max_freq'       : freq_max,
            'verbose'        : True,
            'wavefrequency'  : wave_number,
        }

        (
            self.signal_amp_fun,
            self.signal_phase_fun,
        ) = self.zebrafish_properties.get_amp_and_phase_func_experimental_continuous(signal_kwargs)

        return














