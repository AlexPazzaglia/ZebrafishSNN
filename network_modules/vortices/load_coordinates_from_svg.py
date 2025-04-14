import numpy as np
import matplotlib.pyplot as plt

from svgpathtools import svg2paths
from scipy.interpolate import CubicSpline, interp1d

def _load_coordinates_from_svg(
    svg_file : str,
    path_id  : str,
    plot     : bool = False,
):
    ''' Load x and y coordinates from an SVG file. '''

    # Load the SVG file
    paths, attributes = svg2paths(svg_file)

    # Find the path with the specific id if needed
    path_data = None
    for path, attr in zip(paths, attributes):
        if 'id' in attr and attr['id'] == path_id:
            path_data = path
            break

    if not path_data:
        ValueError("Path with specified ID not found.")

    # Extract points along the path
    points = []
    for segment in path_data:
        # Discretize each segment for more points
        segment_points = [segment.point(t) for t in [0, 1]]
        points.extend(segment_points)

    # Extract x and y coordinates
    x_coords = + np.array( [point.real for point in points] )
    y_coords = - np.array( [point.imag for point in points] )

    if plot:
        plt.figure(f'Original sketch {path_id}')
        plt.plot(x_coords, y_coords, 'o', ls='-', lw=2)
        plt.axis('equal')

    return x_coords, y_coords

def _compute_fish_boundaries_and_midline(
    x_coords    : np.ndarray,
    y_coords    : np.ndarray,
    x_mid_coords: np.ndarray = None,
    y_mid_coords: np.ndarray = None,
    n_vals      : int = 1000,
):
    ''' Compute the fish boundaries and midline. '''

    # Tail tip
    rightmost_idx = np.argmax(x_coords)

    # Dorsal points
    dorsal_x = x_coords[: rightmost_idx + 1]
    dorsal_y = y_coords[: rightmost_idx + 1]

    # Ventral_points
    ventral_x = x_coords[rightmost_idx :][::-1]
    ventral_y = y_coords[rightmost_idx :][::-1]

    # Compute spline interpolations
    s_coords_dorsal  = np.linspace(0, 1, len(dorsal_x))
    s_coords_ventral = np.linspace(0, 1, len(ventral_x))
    dorsal_x_spline  = interp1d(s_coords_dorsal, dorsal_x, kind='linear')
    dorsal_y_spline  = interp1d(s_coords_dorsal, dorsal_y, kind='linear')
    ventral_x_spline = interp1d(s_coords_ventral, ventral_x, kind='linear')
    ventral_y_spline = interp1d(s_coords_ventral, ventral_y, kind='linear')

    # Interpolation points
    s_vals = np.linspace(0, 1, n_vals)
    x_drs  = dorsal_x_spline(s_vals)
    y_drs  = dorsal_y_spline(s_vals)
    x_vtr  = ventral_x_spline(s_vals)
    y_vtr  = ventral_y_spline(s_vals)

    # Generate midline at spline points
    if x_mid_coords is None or y_mid_coords is None:

        # Calculate raw midline
        x_mid_raw = 0.5 * (x_drs + x_vtr)
        y_mid_raw = 0.5 * (y_drs + y_vtr)

        # Calculate the polynomial midline
        poly_x     = np.polyfit(s_vals, x_mid_raw, 4)
        poly_y     = np.polyfit(s_vals, y_mid_raw, 5)

        x_mid_poly = np.polyval(poly_x, s_vals)
        y_mid_poly = np.polyval(poly_y, s_vals)

        # Corrections at the endpoints
        dx_start = x_mid_raw[0] - x_mid_poly[0]
        dx_end   = x_mid_raw[-1] - x_mid_poly[-1]
        dy_start = y_mid_raw[0] - y_mid_poly[0]
        dy_end   = y_mid_raw[-1] - y_mid_poly[-1]

        s_blend = np.linspace(1, 0, len(s_vals))  # Weight for start correction
        e_blend = np.linspace(0, 1, len(s_vals))  # Weight for end correction

        x_mid   = x_mid_poly + dx_start * s_blend + dx_end * e_blend
        y_mid   = y_mid_poly + dy_start * s_blend + dy_end * e_blend

    else:

        # Remove duplicate points
        x_mid_coords = np.concatenate(
            [
                [x_mid_coords[0]],
                x_mid_coords[1:-1:2],
                [x_mid_coords[-1]],
            ]
        )
        y_mid_coords = np.concatenate(
            [
                [y_mid_coords[0]],
                y_mid_coords[1:-1:2],
                [y_mid_coords[-1]],
            ]
        )

        lengths_midline = np.cumsum( np.sqrt( np.diff(x_mid_coords)**2 + np.diff(y_mid_coords)**2 ) )
        lengths_midline = np.concatenate(([0], lengths_midline))
        s_midline       = lengths_midline / lengths_midline[-1]

        # Cubic spline interpolation
        x_mid_spline = CubicSpline(s_midline, x_mid_coords)
        y_mid_spline = CubicSpline(s_midline, y_mid_coords)

        x_mid = x_mid_spline(s_vals)
        y_mid = y_mid_spline(s_vals)

    fish_curves = {
        's_vals': s_vals,
        'x_drs' : x_drs,
        'y_drs' : y_drs,
        'x_vtr' : x_vtr,
        'y_vtr' : y_vtr,
        'x_mid' : x_mid,
        'y_mid' : y_mid,
    }

    return fish_curves

def _find_ray_tracing_thickness(
    x_curve      : np.ndarray,
    y_curve      : np.ndarray,
    x_start      : np.ndarray,
    y_start      : np.ndarray,
    nx           : np.ndarray,
    ny           : np.ndarray,
    max_dist     : float = 0.2,
    search_radius: int = 10,
):
    """ Find exact intersection of a ray with a curve """

    #########
    # STEP 1. Find the approximate closest point to narrow down the search area
    #########

    # Project the ray out to max_dist
    ray_end_x = x_start + max_dist * nx
    ray_end_y = y_start + max_dist * ny

    # Calculate distances from each curve point to the ray
    x1, y1 = x_start, y_start
    x2, y2 = ray_end_x, ray_end_y

    # Vector from ray start to each curve point
    vx = x_curve - x1
    vy = y_curve - y1

    # Ray direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Normalized projection of point onto ray (parameter t)
    dot_prod = dx * vx + dy * vy
    len_sq   = dx * dx + dy * dy
    t        = np.clip(dot_prod / len_sq, 0, 1)

    # Closest point on ray to each curve point
    px = x1 + t * dx
    py = y1 + t * dy

    # Distance from each curve point to closest point on ray
    distances = np.sqrt((px - x_curve)**2 + (py - y_curve)**2)

    # Find index of minimum distance
    closest_idx = np.argmin(distances)

    #########
    # STEP 2: Calculate exact intersection
    #########

    start_idx = max(0, closest_idx - search_radius)
    end_idx   = min(len(x_curve) - 1, closest_idx + search_radius)

    # Initialize best result
    best_dist = float('inf')
    best_intersection = None

    # Check intersection with nearby line segments
    for i in range(start_idx, end_idx):
        if i + 1 >= len(x_curve):
            continue

        # Line segment endpoints
        x1, y1 = x_curve[i], y_curve[i]
        x2, y2 = x_curve[i+1], y_curve[i+1]

        # Line segment direction vector
        dx = x2 - x1
        dy = y2 - y1

        # Solve the linear system for intersection
        denominator = dx * (-ny) - dy * (-nx)

        # Check if lines are parallel
        if abs(denominator) < 1e-10:
            continue

        # Compute intersection parameters
        s = ((x_start - x1) * (-ny) - (y_start - y1) * (-nx)) / denominator

        # If s is outside [0,1], intersection is outside line segment
        if s < 0 or s > 1:
            continue

        # Compute t (ray parameter)
        if abs(nx) > abs(ny):
            t = (x1 + s * dx - x_start) / nx
        else:
            t = (y1 + s * dy - y_start) / ny

        # Check if intersection is valid
        if t >= 0 and t <= max_dist and t < best_dist:
            # Compute intersection point
            x_intersect = x_start + t * nx
            y_intersect = y_start + t * ny

            # Save best result
            best_dist = t
            best_intersection = (x_intersect, y_intersect)

    # Use the closest point on the curve as fallback
    if best_intersection is None:
        ray_t = np.dot([x_curve[closest_idx] - x_start, y_curve[closest_idx] - y_start], [nx, ny])
        return (x_start + ray_t * nx, y_start + ray_t * ny), ray_t

    return best_intersection, best_dist

def _compute_fish_thickness(
    x_coords    : np.ndarray,
    y_coords    : np.ndarray,
    x_mid_coords: np.ndarray,
    y_mid_coords: np.ndarray,
    n_vals      : int = 1000,
    plot        : bool = False,
):
    """
    Compute the thickness of a fish body as a function of normalized axial position.
    """

    # Convert to numpy arrays if they aren't already
    x_coords     = np.asarray(x_coords)
    y_coords     = np.asarray(y_coords)
    x_mid_coords = np.asarray(x_mid_coords)
    y_mid_coords = np.asarray(y_mid_coords)

    # Rescale in 0-1 range
    max_range = max(
        x_coords.max() - x_coords.min(),
        y_coords.max() - y_coords.min(),
    )
    x_min = x_coords.min()
    y_min = y_coords.min()

    x_coords     = (x_coords - x_min) / max_range
    y_coords     = (y_coords - y_min) / max_range
    x_mid_coords = (x_mid_coords - x_min) / max_range
    y_mid_coords = (y_mid_coords - y_min) / max_range

    # Compute fish boundaries and midline
    fish_curves = _compute_fish_boundaries_and_midline(
        x_coords     = x_coords,
        y_coords     = y_coords,
        x_mid_coords = x_mid_coords,
        y_mid_coords = y_mid_coords,
        n_vals       = n_vals,
    )

    # Extract the curves
    s_vals = fish_curves['s_vals']
    x_drs  = fish_curves['x_drs']
    y_drs  = fish_curves['y_drs']
    x_vtr  = fish_curves['x_vtr']
    y_vtr  = fish_curves['y_vtr']
    x_mid  = fish_curves['x_mid']
    y_mid  = fish_curves['y_mid']

    # Calculate midline derivatives using polynomial coefficients
    dx_ds = np.gradient(x_mid, s_vals)
    dy_ds = np.gradient(y_mid, s_vals)

    # Normalize the normal vectors
    normal_lengths = np.sqrt(dx_ds**2 + dy_ds**2)
    normal_x       = -dy_ds / normal_lengths
    normal_y       = dx_ds / normal_lengths

    # Initialize arrays for results
    n_vals       = len(s_vals)
    thk_drs      = np.zeros(n_vals)
    thk_vtr      = np.zeros(n_vals)
    x_inters_drs = np.zeros(n_vals)
    y_inters_drs = np.zeros(n_vals)
    x_inters_vtr = np.zeros(n_vals)
    y_inters_vtr = np.zeros(n_vals)

    # Project rays to find intersections
    for i in range(n_vals):
        # Get the midline point and normal vector
        x0, y0 = x_mid[i], y_mid[i]
        nx, ny = normal_x[i], normal_y[i]

        # Find dorsal intersection (positive normal direction)
        (dx, dy), d_dist = _find_ray_tracing_thickness(
            x_curve = x_drs,
            y_curve = y_drs,
            x_start = x0,
            y_start = y0,
            nx      = nx,
            ny      = ny,
        )
        thk_drs[i]      = d_dist
        x_inters_drs[i] = dx
        y_inters_drs[i] = dy

        # Find ventral intersection (negative normal direction)
        (vx, vy), v_dist = _find_ray_tracing_thickness(
            x_curve = x_vtr,
            y_curve = y_vtr,
            x_start = x0,
            y_start = y0,
            nx      = -nx,
            ny      = -ny,
        )
        thk_vtr[i]      = v_dist
        x_inters_vtr[i] = vx
        y_inters_vtr[i] = vy

    fish_curves['thk_drs']      = thk_drs
    fish_curves['thk_vtr']      = thk_vtr
    fish_curves['x_inters_drs'] = x_inters_drs
    fish_curves['y_inters_drs'] = y_inters_drs
    fish_curves['x_inters_vtr'] = x_inters_vtr
    fish_curves['y_inters_vtr'] = y_inters_vtr

    if not plot:
        return fish_curves

    # Plot the results
    plt.figure('Original sketch with ray tracing')
    plt.axis('equal')
    plt.plot(x_drs, y_drs)
    plt.plot(x_vtr, y_vtr)
    plt.plot(x_mid, y_mid)

    for i in range(n_vals):
        plt.plot( [x_mid[i], x_inters_drs[i]], [y_mid[i], y_inters_drs[i]], lw=0.5, c='k')
        plt.plot( [x_mid[i], x_inters_vtr[i]], [y_mid[i], y_inters_vtr[i]], lw=0.5, c='k')


    plt.figure('Thickness profile')
    plt.axis('equal')
    body_length = np.sum( np.sqrt( np.diff(x_mid)**2 + np.diff(y_mid)**2 ) )
    plt.plot(s_vals, +thk_drs / body_length)
    plt.plot(s_vals, -thk_vtr / body_length)

    return fish_curves

def _get_fish_half_thickness_spline_from_sketch(
    body_length: float,
    n_vals     : int = 1000,
    plot       : bool = False,
):
    '''
    Fish thickness model spline
    Derived from the mechanical model of the zebrafish body
    '''

    svg_file = (
        'network_modules/vortices/data/zebrafish_sketch.svg'
    )
    path_id0 = 'path13'
    path_id1 = 'path5298'

    # Load x and y coordinates from an SVG file
    x_coords, y_coords = _load_coordinates_from_svg(
        svg_file = svg_file,
        path_id  = path_id0,
        plot     = plot,
    )

    x_mid, y_mid = _load_coordinates_from_svg(
        svg_file = svg_file,
        path_id  = path_id1,
        plot     = plot,
    )

    # Compute thicness
    fish_curves = _compute_fish_thickness(
        x_coords     = x_coords,
        y_coords     = y_coords,
        x_mid_coords = x_mid,
        y_mid_coords = y_mid,
        n_vals       = n_vals,
        plot         = plot,
    )

    # Rescale
    s_vals = fish_curves['s_vals']
    t_vals = fish_curves['thk_drs']

    s_min, s_max = min(s_vals), max(s_vals)
    t_min, t_max = min(t_vals), max(t_vals)

    scale = body_length / (s_max - s_min)

    s_coords = (s_vals - s_min) * scale
    t_coords = (t_vals - t_min) * scale

    # Sort according to x coordinates
    sort_indices = np.argsort(s_coords)
    s_coords = s_coords[sort_indices]
    t_coords = t_coords[sort_indices]

    # Remove duplicates
    s_coords, indices = np.unique(s_coords, return_index=True)
    t_coords = t_coords[indices]

    # Remove points with y = 0 for x > 0 and x < BODY_LENGTH
    target_inds = (
        (s_coords > 0) & (s_coords < body_length) & (t_coords == 0)
    )
    s_coords = s_coords[~target_inds]
    t_coords = t_coords[~target_inds]

    # Compute thickness spline
    # thickness_spline = CubicSpline(s_coords, t_coords)
    thickness_spline = interp1d(s_coords, t_coords, kind='linear')

    if plot:
        plt.figure('Rescaled thickness profile')
        plt.plot(s_coords, +t_coords, 'ro')
        plt.plot(s_coords, -t_coords, 'ro')

    return thickness_spline

def main():
    ''' Main function '''

    # Fish thickness model spline
    # body_length = 0.018
    body_length = 0.018

    # Compute thickness spline
    thickness_spline = _get_fish_half_thickness_spline_from_sketch(
        body_length = body_length,
        n_vals      = 1000,
        plot        = True,
    )

    # Plot the coordinates
    s_vals = np.linspace(0, body_length, 1000)
    t_vals = thickness_spline(s_vals)

    plt.plot(s_vals, +t_vals, 'b-')
    plt.plot(s_vals, -t_vals, 'b-')

    plt.fill_between(s_vals, -t_vals, +t_vals, color='gray', alpha=0.5)
    plt.axis('equal')
    plt.show()

    return

if __name__ == "__main__":
    main()

