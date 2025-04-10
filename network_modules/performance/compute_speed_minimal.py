import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _compute_body_linear_fit(
    coordinates_xy: np.ndarray,
    n_links_pca   : int = None,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the PCA of the links positions at all steps '''

    coordinates_pca_xy = coordinates_xy[:n_links_pca]
    cov_mat = np.cov(
        [
            coordinates_pca_xy[:, 0],
            coordinates_pca_xy[:, 1],
        ]
    )

    eig_values, eig_vecs = np.linalg.eig(cov_mat)
    largest_index        = np.argmax(eig_values)
    direction_fwd        = eig_vecs[:, largest_index]

    # Align the direction with the start-finish axis
    p_tail2head    = coordinates_xy[0] - coordinates_xy[-1]
    direction_sign = np.sign( np.dot( p_tail2head, direction_fwd ) )
    direction_fwd  = direction_sign * direction_fwd

    direction_left = np.cross(
        [0,0,1],
        [direction_fwd[0], direction_fwd[1], 0]
    )[:2]

    return direction_fwd, direction_left

def _compute_head_orientation(
    coordinates_xy: np.ndarray[float],
):
    ''' Compute the angles from the coordinates '''

    n_steps = coordinates_xy.shape[0]
    angles  = np.zeros(n_steps)

    for step in range(n_steps):
        p_neck2head = coordinates_xy[step, 0, :] - coordinates_xy[step, 1, :]
        angle       = np.arctan2(p_neck2head[1], p_neck2head[0])

        if abs(angle) > np.pi:
            angle = - (2*np.pi - angle)

        angles[step] = angle

    return angles

def _compute_coordinates_from_angles(
    position_head   : np.ndarray[float],
    orientation_head: float,
    joints_angles   : np.ndarray[float],
    links_lengths   : list[float],
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
            - np.sum( joints_angles[:joint] ) + orientation_head
            for joint in range(0, num_points)
        ]
    )

    # Compute the Cartesian coordinates for each joint using the link lengths and cumulative angles
    joint_pos    = np.zeros((num_points, 2))
    joint_pos[0] = position_head

    for i in range(1, num_points):
        joint_pos[i] = [
            joint_pos[i - 1, 0] - links_lengths[i - 1] * np.cos(sum_angles[i - 1]),
            joint_pos[i - 1, 1] - links_lengths[i - 1] * np.sin(sum_angles[i - 1]),
        ]

    return joint_pos

def compute_speed(
    duration        : float,
    points_positions: np.ndarray,
    joints_angles   : np.ndarray,
    n_points_axis   : int,
    sim_fraction    : float = 1.0,
) -> dict:
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    speed_metrics       = {}
    n_steps             = points_positions.shape[0]
    duration_considered = duration * sim_fraction
    n_steps_considered  = round(n_steps * sim_fraction)

    points_positions = points_positions[-n_steps_considered::, :n_points_axis, :2]
    joints_angles    = joints_angles[-n_steps_considered::]

    # Compute links lengths
    links_lengths = np.zeros(n_points_axis - 1)
    for i in range(n_points_axis - 1):
        links_lengths[i] = np.linalg.norm(
            points_positions[0, i + 1] - points_positions[0, i]
        )

    # Compute average speed
    com_positions = np.mean(points_positions, axis=1)
    com_speed     = ( com_positions[-1] - com_positions[0] ) / duration_considered

    # Compute average head orientation
    head_orientation      = _compute_head_orientation(points_positions)
    head_orientation_mean = np.mean(head_orientation)

    # Compute average body configuration
    joints_angles_mean = np.mean(joints_angles, axis=0)

    # Reconstruct average body configuration
    body_configuration_mean = _compute_coordinates_from_angles(
        position_head    = np.zeros(2),
        orientation_head = head_orientation_mean,
        joints_angles    = joints_angles_mean,
        links_lengths    = links_lengths,
    )

    # Compute principal components of the average body configuration
    direction_fwd, direction_left = _compute_body_linear_fit(
        coordinates_xy = body_configuration_mean,
        n_links_pca    = n_points_axis,
    )

    # Compute forward and lateral speed
    speed_metrics['speed_fwd'] = np.dot(com_speed, direction_fwd)
    speed_metrics['speed_lat'] = np.dot(com_speed, direction_left)

    # Compute absolute speed
    speed_metrics['speed_abs'] = np.linalg.norm(com_speed)

    # Optional data
    speed_metrics['com_positions']           = com_positions
    speed_metrics['com_speed']               = com_speed
    speed_metrics['head_orientation_mean']   = head_orientation_mean
    speed_metrics['body_configuration_mean'] = body_configuration_mean
    speed_metrics['direction_fwd']           = direction_fwd
    speed_metrics['direction_left']          = direction_left

    return speed_metrics

###############################################################################
###################### TEST ###################################################
###############################################################################

def plot_speed_metrics(
    speed_metrics: dict,
    axis_length  : float,
):
    ''' Plot speed, body configuration and forward/lateral direction'''

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.set_xlim([ - 1.5 * axis_length, + 1.5 * axis_length ])
    ax.set_ylim([ - 1.5 * axis_length, + 1.5 * axis_length ])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    ax.plot(
        speed_metrics['body_configuration_mean'][:, 0],
        speed_metrics['body_configuration_mean'][:, 1],
        color      = 'b',
        linewidth  = 2.0,
        marker     = 'o',
        markersize = 5,
        label      = 'Body configuration',
    )

    # Forward direction
    ax.quiver(
        speed_metrics['body_configuration_mean'][0, 0],
        speed_metrics['body_configuration_mean'][0, 1],
        speed_metrics['direction_fwd'][0] * axis_length,
        speed_metrics['direction_fwd'][1] * axis_length,
        color = 'r',
        label = 'Forward direction',
        scale = 10 / axis_length
    )

    # Lateral direction
    ax.quiver(
        speed_metrics['body_configuration_mean'][0, 0],
        speed_metrics['body_configuration_mean'][0, 1],
        speed_metrics['direction_left'][0] * axis_length,
        speed_metrics['direction_left'][1] * axis_length,
        color = 'g',
        label = 'Lateral direction',
        scale = 10 / axis_length
    )

    # Speed
    com_speed_unit = speed_metrics['com_speed'] / np.linalg.norm(speed_metrics['com_speed'])
    ax.quiver(
        speed_metrics['body_configuration_mean'][0, 0],
        speed_metrics['body_configuration_mean'][0, 1],
        com_speed_unit[0] * axis_length,
        com_speed_unit[1] * axis_length,
        color = 'k',
        label = 'Speed',
        scale = 10 / axis_length
    )

    ax.legend()

    return fig

def animate_joint_coordinates_and_body_axis(
    coordinates_xy : np.ndarray[float],
    axis_length    : float = None,
):
    ''' Animate joint coordinates '''

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Joint coordinates')

    # Create the line object for the kinematic chain
    ax.axis('equal')

    line_joints, = ax.plot(
        [], [],
        color           = 'b',
        linewidth       = 2.0,
        marker          = 'o',
        markersize      = 5,
    )

    # Define the update function for the animation
    def _update_plot(ind):
        ''' Update the line data with the coordinates for the current frame '''

        line_joints.set_data(
            coordinates_xy[ind, :, 0],
            coordinates_xy[ind, :, 1]
        )

        ax.set_xlim(
            [
                coordinates_xy[ind, 0, 0] - 3 * axis_length,
                coordinates_xy[ind, 0, 0] + 3 * axis_length
            ]
        )
        ax.set_ylim(
            [
                coordinates_xy[ind, 1, 0] - 3 * axis_length,
                coordinates_xy[ind, 1, 0] + 3 * axis_length
            ]
        )

        return line_joints

    update_func = lambda frame: _update_plot(frame)

    # Create the animation
    animation = FuncAnimation(
        fig,
        update_func,
        frames   = coordinates_xy.shape[0],
        interval = 10,
        blit     = False,
    )

    return fig, animation

def main():

    # Simulation parameters
    duration = 4.0
    timestep = 0.01
    times    = np.arange(0, duration, timestep)
    n_steps  = len(times)

    # Body parameters
    n_joints_axis = 15
    n_links_axis  = 16
    n_points_axis = 17

    links_lengths = [0.1] * n_links_axis
    axis_length   = sum(links_lengths)

    # Joint angles parameters
    frequency   = 1.0
    amplitude   = 2 * np.pi / n_joints_axis
    wave_number = 0.4

    joints_amps = np.ones(n_joints_axis) * amplitude
    joints_ipls = np.linspace(0, wave_number, n_joints_axis)

    # Generate joint angles
    joints_angles = np.zeros((n_steps, n_joints_axis))
    for i, t in enumerate(times):
        joints_angles[i] = joints_amps * np.sin( 2*np.pi * ( frequency * t + joints_ipls ) )

    # Head trajectory parameters
    speed_head    = np.array([-2.0, -2.0])
    position_head = np.zeros((n_steps, 2))
    for i, t in enumerate(times):
        position_head[i] = speed_head * t

    orientation_head = np.linspace(0, 1*np.pi, n_steps)

    # Compute joint coordinates
    joint_coordinates = np.zeros((n_steps, n_points_axis, 2))
    for i, t in enumerate(times):
        joint_coordinates[i] = _compute_coordinates_from_angles(
            position_head    = position_head[i],
            orientation_head = orientation_head[i],
            joints_angles    = joints_angles[i],
            links_lengths    = links_lengths,
        )

    # Compute speed
    speed_metrics = compute_speed(
        duration        = duration,
        points_positions= joint_coordinates,
        joints_angles   = joints_angles,
        n_points_axis   = n_points_axis,
    )

    print(f"Forward speed: {speed_metrics['speed_fwd']:.2f} m/s")
    print(f"Lateral speed: {speed_metrics['speed_lat']:.2f} m/s")
    print(f"Absolute speed: {speed_metrics['speed_abs']:.2f} m/s")

    # Plot joint coordinates
    fig, animation = animate_joint_coordinates_and_body_axis(
        joint_coordinates,
        axis_length = axis_length,
    )

    # Plot speed, body configuration and forward/lateral direction
    fig_speed = plot_speed_metrics(
        speed_metrics = speed_metrics,
        axis_length   = axis_length,
    )

    plt.show()



if __name__ == '__main__':
    main()