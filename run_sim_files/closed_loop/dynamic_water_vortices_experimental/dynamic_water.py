import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from farms_mujoco.swimming.drag import WaterProperties

from lilytorch.util.yaml_operations import yaml2pyobject

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

        # Boundary conditions
        self.constant_vx = sim_pars['boundary_conditions']['BC_values_u'][0]
        self.constant_vy = sim_pars['boundary_conditions']['BC_values_v'][0]

        self._apply_boundary_conditions_transformation()

        # Saved fields
        self.iterations_saved = sim_pars["output"]["save_every"]

        # Last uploaded field
        self.last_loaded_iteration = None
        self.last_loaded_vx_field  = None
        self.last_loaded_vy_field  = None


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
    def _get_grid_coordinates(
        self,
        x_pos       : float,
        y_pos       : float,
        x_pos_offset: float = 0.0,
        y_pos_offset: float = 0.0,
    ):
        ''' Get quantized coordinates '''

        # Get quantized coordinates
        dx = (self.xmax - self.xmin) / self.grid_n
        dy = (self.ymax - self.ymin) / self.grid_n

        xq = round( (x_pos - self.xmin - x_pos_offset) / dx)
        yq = round( (y_pos - self.ymin - y_pos_offset) / dy)

        # Handle out of bounds
        if xq < 0 or xq >= self.grid_n:
            xq = np.nan
        if yq < 0 or yq >= self.grid_n:
            yq = np.nan

        return xq, yq

    def _load_all_velocities_iteration(
        self,
        time: float,
    ):
        ''' Load velocity field '''
        iteration  = int(time / self.time_step)
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
    # PLOTTING FUNCTIONS ######################################################
    ###########################################################################

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
            time_text.set_text(f"time: {100 * anim_step / self.iterations :.1f} %")

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
        "2024-10-25T17:25:17.952503_implicit/"
    )

    # Water dynamics
    water_dynamics = WaterDynamicsCallback(
        results_path = results_path,
        invert_x     = True,
        invert_y     = False,
        translation  = np.array([-0.0472, 0.0]),
    )


    # Animate velocity field
    fig  = plt.figure()
    anim = water_dynamics.animate_velocity_vector_field(
        fig             = fig,
        anim_step_jump  = 1,
        remove_constant = False,
    )

    plt.show()


if __name__ == "__main__":
    main()
