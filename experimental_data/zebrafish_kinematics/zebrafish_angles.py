'''
Interpolate kinematics data with model data
Alessandro Pazzaglia 09/02/2024
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

# PLOTTING
def plot_joints_angles_evolution(
    joints_angles: np.ndarray,
    color        : str,
    label        : str,
):
    ''' Plot the evolution of the joints angles '''
    n_angles          = joints_angles.shape[1]
    fig_joints_angles = plt.figure(f'Joints angles - {label}')
    plt.plot(
        joints_angles + np.arange(n_angles) * np.amax(joints_angles),
        color     = color,
        linewidth = 1.0,
        linestyle = '--',
        marker    = 'o',
    )
    plt.xlabel('Timestep')
    plt.ylabel('Joint angles')
    plt.grid()
    plt.title(f'Joint angles - {label}')
    return fig_joints_angles

def plot_joint_coordinates_and_body_axis(
    joint_coordinates_ref: np.ndarray[float],
    joint_coordinates_new: np.ndarray[float],
    joint_coordinates_spl: np.ndarray[float],
):
    ''' Animate joint coordinates '''

    plt.plot(
        joint_coordinates_ref[:, 0],
        joint_coordinates_ref[:, 1],
        color     = 'b',
        linewidth = 2.0,
        linestyle = '-',
        marker    = 'o',
    )
    plt.plot(
        joint_coordinates_new[:, 0],
        joint_coordinates_new[:, 1],
        color     = 'g',
        linewidth = 2.0,
        linestyle = '-',
        marker    = 'D',
    )
    plt.plot(
        joint_coordinates_spl[:, 0],
        joint_coordinates_spl[:, 1],
        color     = 'k',
        linewidth = 1.0,
        linestyle = '--',
        marker    = None,
    )
    return

def animate_joint_coordinates_and_body_axis(
    coordinates_ref: np.ndarray[float],
    coordinates_new: np.ndarray[float],
    coordinates_spl: np.ndarray[float],
    axis_length    : float = None,
):
    ''' Animate joint coordinates '''

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Joint coordinates')

    ax.set_xlim( [- 0.1 * axis_length, 1.1 * axis_length ] )
    ax.set_ylim( [- 0.6 * axis_length, 0.6 * axis_length ] )

    # Create the line object for the kinematic chain
    ax.axis('equal')

    line_joints_ref, = ax.plot(
        [], [],
        color           = 'b',
        linewidth       = 2.0,
        linestyle       = 'None',
        marker          = 'o',
        markersize      = 15,
        fillstyle       = 'none',
        label           = 'Reference',
    )

    line_joints_new, = ax.plot(
        [], [],
        color     = 'g',
        linewidth = 2.0,
        linestyle = 'None',
        marker    = 'D',
        label     = 'New',
    )

    line_joints_spl, = ax.plot(
        [], [],
        color     = 'k',
        linewidth = 1.0,
        linestyle = '--',
        marker    = None,
        label     = 'Spline',
    )

    ax.legend()

    # Define the update function for the animation
    def _update_plot(ind):
        ''' Update the line data with the coordinates for the current frame '''

        line_joints_ref.set_data(
            coordinates_ref[ind, :, 0],
            coordinates_ref[ind, :, 1]
        )
        line_joints_new.set_data(
            coordinates_new[ind, :, 0],
            coordinates_new[ind, :, 1]
        )
        line_joints_spl.set_data(
            coordinates_spl[ind, :, 0],
            coordinates_spl[ind, :, 1]
        )

        return line_joints_ref, line_joints_new, line_joints_spl

    update_func = lambda frame: _update_plot(frame)

    # Create the animation
    animation = FuncAnimation(
        fig,
        update_func,
        frames   = coordinates_ref.shape[0],
        interval = 10,
        blit     = False,
    )

    return fig, animation

# KINEMATICS
def compute_angles_from_coordinates(
    coordinates_xy: np.ndarray[float],
):
    ''' Compute the angles from the coordinates '''

    n_points = len(coordinates_xy)
    angles   = np.zeros(n_points-2)

    for point in range(1, n_points - 1):

        p0 = coordinates_xy[point - 1]
        p1 = coordinates_xy[point]
        p2 = coordinates_xy[point + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        v21_parall = np.dot(v1, v2)
        v21_orthog = np.cross(v1, v2)

        angle = np.arctan2(v21_orthog, v21_parall)

        if abs(angle) > np.pi:
            angle = - (2*np.pi - angle)

        angles[point-1] = angle

    return angles

def compute_coordinates_from_angles(
    links_lengths: list[float],
    joints_angles: np.ndarray[float],
):
    '''
    Analyze joint data.
    Computes the Cartesian coordinates of each joint in a kinematic chain.
    '''

    num_points = len(links_lengths) + 1
    num_joints = len(joints_angles)

    assert num_points == num_joints + 2, 'Number of joints and link lengths do not match'

    # Compute the cumulative angles for each joint at the current step
    sum_angles = np.array(
        [
            np.sum( joints_angles[:joint] )
            for joint in range(0, num_points)
        ]
    )

    # Compute the Cartesian coordinates for each joint using the link lengths and cumulative angles
    joint_pos = np.zeros((num_points, 2))
    for i in range(1, num_points):
        joint_pos[i] = [
            joint_pos[i - 1, 0] + links_lengths[i - 1] * np.cos(sum_angles[i - 1]),
            joint_pos[i - 1, 1] + links_lengths[i - 1] * np.sin(sum_angles[i - 1]),
        ]

    return joint_pos

def compute_coordinate_along_axis(
    joint_coordinates: np.ndarray[float],
    axis_positons    : np.ndarray[float],
):
    ''' Compute the coordinate along the axis '''

    links_lengths = np.array(
        [
            np.linalg.norm(joint_coordinates[i + 1] - joint_coordinates[i])
            for i in range(len(joint_coordinates) - 1)
        ]
    )

    axis_coordinates = np.zeros((len(axis_positons), 2))
    sum_links        = np.concatenate( [ [0], np.cumsum(links_lengths) ] )

    for axis_index, axis_positon in enumerate(axis_positons):

        # Find the link that contains the coordinate
        link_index = np.argmax(sum_links > axis_positon) - 1

        # Compute the coordinate
        prev_link = joint_coordinates[link_index]
        prev_sum  = sum_links[link_index]

        link_dir = joint_coordinates[link_index + 1] - prev_link
        link_dir = link_dir / np.linalg.norm(link_dir)

        axis_coordinates[axis_index] = prev_link + ( axis_positon - prev_sum ) * link_dir

    return axis_coordinates

def compute_distance_to_reference_coordinates(
    links_coordinates_ref: np.ndarray[float],
    links_coordinates_new: np.ndarray[float],
):
    ''' Compute the distance between two sets of coordinates '''

    links_lengths_ref = np.array(
        [
            np.linalg.norm(links_coordinates_ref[i + 1] - links_coordinates_ref[i])
            for i in range(len(links_coordinates_ref) - 1)
        ]
    )

    # Compute the position of each reference link along the axis of the new links
    axis_positions_ref   = np.concatenate( [ [0], np.cumsum(links_lengths_ref) ] )
    axis_coordinates_new = compute_coordinate_along_axis(
        links_coordinates_new,
        axis_positions_ref,
    )

    # Compute the distance between the reference and new coordinates
    distance = np.linalg.norm(axis_coordinates_new - links_coordinates_ref)

    return distance

# SPLINE INTERPOLATION
def compute_cubic_spline(
    coordinates_xy : np.ndarray
):
    ''' Returns a cubic spline object from a list of points. '''

    # Compute arc lengths
    links_lengths = np.array(
        [
            np.linalg.norm( coordinates_xy[point] - coordinates_xy[point - 1] )
            for point in range(1, len(coordinates_xy))
        ]
    )
    arc_lengths = np.concatenate( [ [0], np.cumsum(links_lengths) ] ) / np.sum(links_lengths)

    # Separate x and y coordinates from the points
    x_coords = coordinates_xy[:, 0]
    y_coords = coordinates_xy[:, 1]

    # Create a cubic spline object
    cubic_spline_x = CubicSpline(arc_lengths, x_coords)
    cubic_spline_y = CubicSpline(arc_lengths, y_coords)

    return cubic_spline_x, cubic_spline_y

def spline_curve_length(
    spline_curve_x: CubicSpline,
    spline_curve_y: CubicSpline,
    s_start       : float,
    s_end         : float,
    num_samples   : int =100
):
    ''' Returns the length of the spline curve between x_start and x_end. '''

    s_samples = np.linspace(s_start, s_end, num_samples)
    samples_x = spline_curve_x(s_samples)
    samples_y = spline_curve_y(s_samples)

    # Integrate
    arc_length = np.sum(
        np.sqrt(
            np.diff(samples_x)**2 + np.diff(samples_y)**2
        )
    )

    return arc_length

def compute_coordinates_from_arc_lengths(
    spline_curve_x: CubicSpline,
    spline_curve_y: CubicSpline,
    arc_lengths   : np.ndarray[float],
):
    ''' Returns the point on the spline curve with the desired normalized arc length.'''
    return np.array(
        [
            [ spline_curve_x(arc_length), spline_curve_y(arc_length) ]
            for arc_length in arc_lengths
        ]
    )

# MAIN
def main():

    ############################### [ PARAMETERS ] ###############################

    # Mechanical coordinates
    links_coordinates = np.array(
        [
            0.0,
            0.003000000026077032,
            0.004000000189989805,
            0.004999999888241291,
            0.006000000052154064,
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
            0.017000000923871994,
            0.018,
        ]
    )

    axis_length_all   = links_coordinates[-1]
    links_lengths_all = np.diff(links_coordinates)

    # Remove tail links
    # NOTE: The tail links are not considered in the interpolation
    n_tail_links   = 2
    n_active_links = len(links_lengths_all) - n_tail_links

    axis_length   = np.sum(links_lengths_all[:n_active_links])
    links_lengths = links_lengths_all[:n_active_links]

    # Reference
    n_links_ref       = 9
    links_lengths_ref = np.ones(n_links_ref) * axis_length / n_links_ref
    joints_amps_ref   = np.array([0.06618, 0.05, 0.05, 0.06324, 0.08235, 0.10735, 0.16471, 0.22059])

    # New
    n_links_new       = len(links_lengths)
    links_lengths_new = np.copy(links_lengths)

    ############################### [ SIGNAL GENERATION ] ###############################

    # Build travelling wave
    frequency_temporal = 10.0
    frequency_spatial  = 1.0 / axis_length

    n_steps   = 500
    times     = np.arange(n_steps) * 0.001
    positions = np.cumsum(links_lengths_ref[:-1])

    phases_spatial  = 2 * np.pi * frequency_spatial * positions
    phases_temporal = 2 * np.pi * frequency_temporal * times

    # Calculate joints_angles
    joints_angles_ref = np.zeros((n_steps, n_links_ref-1))
    for step_index in range(n_steps):
        joints_angles_ref[step_index] = (
            joints_amps_ref * np.cos( phases_temporal[step_index] - phases_spatial)
        )

    ############################### [ ANGLES COMPUTATION ] ###############################

    arc_lengths_new  = np.concatenate([ [0], np.cumsum(links_lengths_new) ]) / axis_length
    joint_angles_new = np.zeros((n_steps, n_links_new-1))

    coordinates_ref = np.zeros((n_steps, n_links_ref + 1, 2))
    coordinates_new = np.zeros((n_steps, n_links_new + 1, 2))
    coordinates_spl = np.zeros((n_steps,             101, 2))

    for step_index in range(n_steps):

        coordinates_ref[step_index] = compute_coordinates_from_angles(
            links_lengths_ref,
            joints_angles_ref[step_index],
        )

        # Compute the spline curve interpolation
        spline_x, spline_y = compute_cubic_spline(coordinates_ref[step_index])

        coordinates_spl[step_index] = compute_coordinates_from_arc_lengths(
            spline_curve_x = spline_x,
            spline_curve_y = spline_y,
            arc_lengths    = np.linspace(0, 1, 101)
        )

        coordinates_new[step_index] = compute_coordinates_from_arc_lengths(
            spline_curve_x = spline_x,
            spline_curve_y = spline_y,
            arc_lengths    = arc_lengths_new,
        )

        # Compute the joint angles
        joint_angles_new[step_index] = compute_angles_from_coordinates(
            coordinates_new[step_index]
        )

    # Compute joints angles amplitude
    joints_amps_new = np.amax(joint_angles_new, axis=0)

    ############################### [ PLOTTING ] ##############################

    fig_animation, animation = animate_joint_coordinates_and_body_axis(
        coordinates_ref = coordinates_ref,
        coordinates_new = coordinates_new,
        coordinates_spl = coordinates_spl,
        axis_length     = axis_length,
    )

    # JOINTS ANGLES
    fig_joints_angles_ref = plot_joints_angles_evolution(
        joints_angles = joints_angles_ref,
        color         = 'b',
        label         = 'Ref'
    )

    fig_joints_angles_new = plot_joints_angles_evolution(
        joints_angles = joint_angles_new,
        color         = 'g',
        label         = 'New'
    )

    # JOINTS AMPLITUDES
    fig_joints_amps = plt.figure(figsize=(18.5, 10.5))

    phases_spatial_ref = np.cumsum(links_lengths_ref[:-1]) / axis_length
    phases_spatial_new = np.cumsum(links_lengths_new[:-1]) / axis_length

    plt.plot(
        phases_spatial_ref,
        joints_amps_ref,
        color     = 'b',
        linewidth = 1.0,
        linestyle = '--',
        marker    = 'o',
        label    = 'Reference',
    )
    plt.plot(
        phases_spatial_new,
        joints_amps_new,
        color     = 'g',
        linewidth = 1.0,
        linestyle = '--',
        marker    = 'D',
        label    = 'New',
    )
    plt.xlabel('Normalized joint position')
    plt.ylabel('Joint amplitude')
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()
    plt.title('Joint amplitudes')

    plt.show()

    ############################### [ SAVING ] ################################
    data_path = 'experimental_data/zebrafish_kinematics'

    print(f'Saving Joints amplitudes in {data_path}/joints_amps_new.npy ...', end=' ', flush=True)

    np.save(
        f'{data_path}/joints_amps_new.npy',
        joints_amps_new
    )
    np.savetxt(
        f'{data_path}/joints_amps_new.txt',
        joints_amps_new,
        delimiter = ',',
        fmt       = '%.5f',
    )

    print('Done')
    print(f'Saving animation in {data_path}/zebrafish_angles.mp4 ...', end=' ', flush=True)

    animation.save(
        f'{data_path}/zebrafish_angles.mp4',
        writer = 'ffmpeg',
        fps    = 30
    )

    print('Done')

    return

if __name__ == '__main__':
    main()