
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, interp1d
from matplotlib.animation import FuncAnimation

from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path

from network_modules.vortices.load_coordinates_from_svg import _get_fish_half_thickness_spline_from_sketch

###############################################################################
# SPLINE ######################################################################
###############################################################################

def compute_cubic_spline(
    coordinates_x : np.ndarray,
    coordinates_y : np.ndarray,
):
    ''' Returns a cubic spline object from a list of points. '''

    # Combine x and y coordinates
    coordinates_xy = np.stack((coordinates_x, coordinates_y), axis=-1)

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

###############################################################################
# FISH BOUNDARIES #############################################################
###############################################################################

def _get_fish_half_thickness_spline_from_model(
    fish_length: float
):
    '''
    Fish thickness model spline
    Derived from the mechanical model of the zebrafish body
    '''
    s_coords = np.array(
        [
            0.00000000e+00, 1.85375688e-05, 2.59863671e-04, 3.75253149e-04,
            6.61936962e-04, 9.68398441e-04, 1.58240186e-03, 1.83324935e-03,
            2.67368424e-03, 3.67668830e-03, 4.70110425e-03, 6.06038261e-03,
            6.71571663e-03, 7.27757934e-03, 9.24973080e-03, 1.08033482e-02,
            1.23470686e-02, 1.34687836e-02, 1.40113190e-02, 1.41112078e-02,
            1.48027059e-02, 1.53315544e-02, 1.62264019e-02, 1.74619712e-02,
            1.80000000e-02,
        ]
    )
    y_coords = np.array(
        [
            0.00000000e+00, 9.15521925e-05, 2.58186783e-04, 3.88766211e-04,
            4.69319851e-04, 6.04953773e-04, 6.78922684e-04, 7.72618010e-04,
            8.85581041e-04, 9.20080493e-04, 9.46684097e-04, 9.85263519e-04,
            1.03368572e-03, 1.02831663e-03, 7.72560102e-04, 5.57949118e-04,
            3.94504830e-04, 2.58659862e-04, 2.68833330e-04, 2.52653865e-04,
            2.47203824e-04, 2.26617545e-04, 1.24073370e-04, 1.29116038e-04,
            0.00000000e+00,
        ]
    )

    # Substitute points 0:9 with an ellipse
    ellipse_a = s_coords[9] - s_coords[0]
    ellipse_b = y_coords[9] - y_coords[0]
    ellipse_x0 = s_coords[9]
    ellipse_y0 = y_coords[0]

    ellipse_s = np.linspace(s_coords[0], s_coords[9], 20)
    ellipse_y = ellipse_y0 + ellipse_b * np.sqrt(1 - ((ellipse_s - ellipse_x0) ** 2) / ellipse_a ** 2)

    ellipse_min   = np.min(ellipse_y)
    ellipse_max   = np.max(ellipse_y)
    ellipse_range = ellipse_max - ellipse_min
    ellipse_y     = ( ellipse_y - ellipse_min ) / ellipse_range
    ellipse_y     = ellipse_min + ellipse_range * ellipse_y ** (0.75)

    s_coords = np.concatenate([ellipse_s, s_coords[10:]])
    y_coords = np.concatenate([ellipse_y, y_coords[10:]])

    # Rescale
    scaling   = fish_length / s_coords[-1]
    s_coords *= scaling
    y_coords *= scaling

    # plt.axis('equal')
    # plt.plot(s_coords, +y_coords, 'k')
    # plt.plot(s_coords, -y_coords, 'k')
    # plt.plot(ellipse_s, +ellipse_y, 'r-')
    # plt.plot(ellipse_s, -ellipse_y, 'r-')

    # Compute thickness spline
    # thickness_spline = CubicSpline(x_coords_scaled, y_coords_scaled)
    thickness_spline = interp1d(s_coords, y_coords, kind='linear')

    return thickness_spline

def get_fish_half_thickness_from_model(
    normalized_arclengths: np.ndarray,
    fish_length          : float,
    thickness_spline     : callable,
):
    '''
    Fish thickness model
    Derived from the mechanical model of the zebrafish body
    # NOTE: Equation defines the half-thickness
    '''

    if thickness_spline is None:
        thickness_spline = _get_fish_half_thickness_spline_from_model(fish_length)

    # Compute thickness
    arclength        = normalized_arclengths * fish_length
    thickness_values = thickness_spline(arclength)

    return  thickness_values

def get_fish_half_thickness_from_gazzola(
    normalized_arclengths: np.ndarray,
    length               : float = 0.018,
):
    """
    Fish thickness
    See Gazzola et al. 2011
    # NOTE: Equation defines the half-thickness
    """

    L = length
    s = normalized_arclengths * L
    sb, st, wh, wt = 0.07*L ,0.95*L ,0.07*L ,0.01*L

    # Conditions
    cond1 = ( 0 <= s) & (s <  sb)
    cond2 = (sb <= s) & (s <  st)
    cond3 = (st <= s) & (s <=  L)

    thickness = np.zeros_like(s)
    thickness[cond1] = np.sqrt(2*wh*s[cond1]-s[cond1]**2)
    thickness[cond2] = wh-(wh-wt)*(((s[cond2]-sb)/(st-sb))**2)
    thickness[cond3] = wt*(L-s[cond3])/(L-st)

    return thickness

def get_fish_half_thickness_from_sketch(
    normalized_arclengths: np.ndarray,
    fish_length          : float,
    thickness_spline     : callable = None,
):
    '''
    Fish thickness model
    Derived from the mechanical model of the zebrafish body
    '''

    if thickness_spline is None:
        thickness_spline, _, _ = _get_fish_half_thickness_spline_from_sketch(fish_length)

    # Compute thickness
    arclength        = normalized_arclengths * fish_length
    thickness_values = thickness_spline(arclength)

    return thickness_values

def get_fish_boundaries(
    positions_x   : np.ndarray,
    positions_y   : np.ndarray,
    thickness_f   : callable,
    thickness_args: tuple = None,
    n_points      : int = 100,
):
    ''' Returns the fish body shape given the midline points and thickness function. '''

    if thickness_args is None:
        thickness_args = ()

    # Compute spline
    (
        spline_x,
        spline_y,
    ) = compute_cubic_spline(
        coordinates_x = positions_x,
        coordinates_y = positions_y,
    )

    # Generate midline points and thickness
    s_values         = np.linspace(0, 1, n_points)
    x_midline        = spline_x(s_values)
    y_midline        = spline_y(s_values)
    thickness_values = thickness_f(s_values, *thickness_args)

    # Calculate the fish body shape by offsetting along a perpendicular vector
    x_upper = np.zeros(n_points)
    y_upper = np.zeros(n_points)
    x_lower = np.zeros(n_points)
    y_lower = np.zeros(n_points)

    for i in range(n_points - 1):
        # Tangent vector
        dx = x_midline[i+1] - x_midline[i]
        dy = y_midline[i+1] - y_midline[i]

        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length

        # Perpendicular vector
        perp_x = -dy
        perp_y = dx

        # Calculate upper and lower points by offsetting along the perpendicular
        thickness_v = thickness_values[i]
        x_upper[i] = (x_midline[i] + perp_x * thickness_v)
        y_upper[i] = (y_midline[i] + perp_y * thickness_v)
        x_lower[i] = (x_midline[i] - perp_x * thickness_v)
        y_lower[i] = (y_midline[i] - perp_y * thickness_v)

    # Close the fish body shape by connecting the endpoints
    x_upper[-1] = (x_midline[-1] + perp_x * thickness_values[-1])
    y_upper[-1] = (y_midline[-1] + perp_y * thickness_values[-1])
    x_lower[-1] = (x_midline[-1] - perp_x * thickness_values[-1])
    y_lower[-1] = (y_midline[-1] - perp_y * thickness_values[-1])

    return (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    )

###############################################################################
# FISH PLOTTING ###############################################################
###############################################################################

def _get_single_band(
    x_vals     : np.ndarray,
    y_dw       : np.ndarray,
    y_up       : np.ndarray,
    thk_mask_dw: np.ndarray = None,
    thk_mask_up: np.ndarray = None,
):
    if thk_mask_dw is None or thk_mask_up is None:
        n_points = x_vals.shape[0]
        inds   = np.arange(n_points)
        radius = n_points / 5
        inds_up = np.clip((inds - inds.min()) / radius, 0, 1)
        inds_dw = np.clip((inds.max() - inds) / radius, 0, 1)
        ramp_up   = np.sqrt(1 - (1 - inds_up) ** 2)
        ramp_dw   = np.sqrt(1 - (1 - inds_dw) ** 2)
        ramp_mask = np.minimum(ramp_up, ramp_dw)
    if thk_mask_dw is None:
        thk_mask_dw = ramp_mask
    if thk_mask_up is None:
        thk_mask_up = ramp_mask
    y_mid  = np.mean([y_dw, y_up], axis=0)
    thk_dw = ( y_mid -  y_dw ) * thk_mask_dw
    thk_up = (  y_up - y_mid ) * thk_mask_up
    y_lower_new = y_mid - thk_dw
    y_upper_new = y_mid + thk_up
    return y_lower_new, y_upper_new

def _plot_single_band(
    axis     : plt.Axes,
    x_band      : np.ndarray,
    y_band_dw: np.ndarray,
    y_band_up: np.ndarray,
    color    : str,
    alpha    : float = 0.7,
    zorder   : int = 2,
    faded    : bool = False,
):
    ''' Plot a single band. '''

    if not faded:
        fill_kwargs = { 'color': color, 'alpha': alpha, 'linewidth': 0.0, 'zorder': zorder }
        fish_band   = axis.fill_between(x_band, y_band_dw , y_band_up, **fill_kwargs)
        return [fish_band]

    n_aphas = 10
    alphas  = np.linspace(alpha, 0, n_aphas)
    i_jmp   = len(x_band) // n_aphas

    fish_bands = []
    for i_alpha in range(n_aphas):
        x_m_alpha       = x_band[i_alpha * i_jmp : (i_alpha + 1) * i_jmp + 1]
        y_band_dw_alpha = y_band_dw[i_alpha * i_jmp : (i_alpha + 1) * i_jmp + 1]
        y_band_up_alpha = y_band_up[i_alpha * i_jmp : (i_alpha + 1) * i_jmp + 1]

        fill_kwargs = { 'color': color, 'alpha': alphas[i_alpha], 'linewidth': 0.0, 'zorder': zorder }
        fish_band   = axis.fill_between(x_m_alpha, y_band_dw_alpha , y_band_up_alpha, **fill_kwargs)
        fish_bands.append(fish_band)

    return fish_bands

def _plot_bands(
    lines_dict: dict,
    axis      : plt.Axes,
    frac_0    : float,
    frac_1    : float,
    i0        : int,
    i1        : int,
    color     : str,
    alpha     : float = 0.7,
    thk_dw    : np.ndarray = None,
    thk_up    : np.ndarray = None,
    zorder    : int = 2,
    faded     : bool = False,
):
    x_midline = lines_dict['x_midline']
    y_midline = lines_dict['y_midline']
    x_lower   = lines_dict['x_lower']
    y_lower   = lines_dict['y_lower']
    x_upper   = lines_dict['x_upper']
    y_upper   = lines_dict['y_upper']

    x_m, x_l, x_u = x_midline[i0:i1], x_lower[i0:i1], x_upper[i0:i1]
    y_m, y_l, y_u = y_midline[i0:i1], y_lower[i0:i1], y_upper[i0:i1]
    frac_0, frac_1 = np.sort([frac_0, frac_1])
    fish_bands = []

    # First band (positive fractions)
    x_band    = x_m + 0.5 * (frac_0 + frac_1) * (x_u - x_m)
    y_band_dw = y_m + frac_0 * (y_u - y_m)
    y_band_up = y_m + frac_1 * (y_u - y_m)
    y_band_dw, y_band_up = _get_single_band(x_band, y_band_dw, y_band_up, thk_dw, thk_up)

    band_parts = _plot_single_band(
        axis     = axis,
        x_band      = x_band,
        y_band_dw= y_band_dw,
        y_band_up= y_band_up,
        color    = color,
        alpha    = alpha,
        zorder   = zorder,
        faded    = faded,
    )
    fish_bands.extend(band_parts)

    if frac_0 == -frac_1:
        return fish_bands

    # Second band (negative fractions)
    x_band    = x_m + 0.5 * (frac_0 + frac_1) * (x_l - x_m)
    y_band_up = y_m + frac_0 * (y_l - y_m)
    y_band_dw = y_m + frac_1 * (y_l - y_m)
    y_band_dw, y_band_up = _get_single_band(x_band, y_band_dw, y_band_up, thk_dw, thk_up)

    band_parts = _plot_single_band(
        axis     = axis,
        x_band   = x_band,
        y_band_dw= y_band_dw,
        y_band_up= y_band_up,
        color    = color,
        alpha    = alpha,
        zorder   = zorder,
        faded    = faded,
    )
    fish_bands.extend(band_parts)

    return fish_bands

def _arclength_to_coordinates(
    s,
    v,
    x_midline,
    y_midline,
    x_lower,
    y_lower,
    x_upper,
    y_upper
):
    """
    Convert param (s, v) -> (x, y) for the fish:
      s in [0,1] along the fish length (head->tail),
      v in [-1,1] across the fish width (lower->upper).
    This simple version picks the nearest integer index
    in [0, N-1] for s. For smoother results, do real interpolation.
    """
    N = len(x_midline)

    # Convert s to float index in [0, N-1]
    i_float = s * (N - 1)
    i_int = np.array(
        np.clip( np.round(i_float), 0, N - 1),
        dtype=int
    )

    # Map v in [-1,1] -> fraction f in [0,1]
    f = (v + 1) / 2.0

    # Grab the lower/upper edges at index i
    x_l = x_lower[i_int]
    x_u = x_upper[i_int]
    y_l = y_lower[i_int]
    y_u = y_upper[i_int]

    # Linear interpolation in the cross section
    x_out = x_l + f * (x_u - x_l)
    y_out = y_l + f * (y_u - y_l)

    return x_out, y_out

def _generate_param_hex_centers(
    n_columns: int = 100,
    n_rows   : int = 15,
):
    """
    Generate (s,v) centers in a hex-lattice fashion, with row spacing
    that shrinks as s goes from 0->1.  The range of s is [0,1].
    The range of v is [-1,1].
    """

    centers_s = np.linspace(0, 1, n_columns)
    centers_v = np.linspace(-1, 1, n_rows)
    centers_v = np.sign(centers_v) * np.abs(centers_v) ** 0.75

    centers          = np.zeros((n_columns, n_rows, 2))
    centers[:, :, 0] = np.tile(centers_s, (n_rows, 1)).T
    centers[:, :, 1] = np.tile(centers_v, (n_columns, 1))
    centers          = centers.reshape(-1, 2)
    return centers

def _fill_body_with_exagons(
    axis    : plt.Axes,
    x_midline: np.ndarray,
    y_midline: np.ndarray,
    x_lower  : np.ndarray,
    y_lower  : np.ndarray,
    x_upper  : np.ndarray,
    y_upper  : np.ndarray,
    head_n   : int,
    body_n   : int,
    tail_n   : int,
):

    # Create fish polygon for clipping hexagons.
    inds_fwd = np.arange(head_n, body_n - tail_n)
    inds_bwd = np.arange(body_n - tail_n - 1, head_n - 1, -1)

    cut_l        = 0.4
    cut_h        = 1 - cut_l
    cut_i0       = head_n
    cut_i1       = round(1.5 * head_n)
    inds_fwd_cut = np.arange(cut_i0, cut_i1)
    inds_bwd_cut = np.arange(cut_i1 - 1, cut_i0 - 1, -1)

    polygon_coords = np.concatenate(
        [
            np.column_stack( ( x_lower[inds_fwd], y_lower[inds_fwd] ) ),
            np.column_stack( ( x_upper[inds_bwd], y_upper[inds_bwd] ) ),

            # Add a midpoint
            np.column_stack(
                (
                    cut_l * x_lower[inds_fwd_cut] + cut_h * x_upper[inds_fwd_cut],
                    cut_l * y_lower[inds_fwd_cut] + cut_h * y_upper[inds_fwd_cut],
                )
            ),
            np.column_stack(
                (
                    cut_h * x_lower[inds_bwd_cut] + cut_l * x_upper[inds_bwd_cut],
                    cut_h * y_lower[inds_bwd_cut] + cut_l * y_upper[inds_bwd_cut],
                )
            ),
        ]
    )
    fish_path       = Path(polygon_coords)
    fish_body_patch = PathPatch(fish_path, facecolor='none', edgecolor='none', zorder=0)
    axis.add_patch(fish_body_patch)

    # 3) Generate param-based hex centers
    #    (You can generate once globally if you have multiple frames)
    n_columns = 100
    n_rows    = 15

    centers_arclength    = _generate_param_hex_centers(
        n_columns = n_columns,
        n_rows    = n_rows,
    )
    centers_x, centers_y = _arclength_to_coordinates(
        s         = centers_arclength[:, 0],
        v         = centers_arclength[:, 1],
        x_midline = x_midline,
        y_midline = y_midline,
        x_lower   = x_lower,
        y_lower   = y_lower,
        x_upper   = x_upper,
        y_upper   = y_upper,
    )
    centers_coords = np.column_stack((centers_x, centers_y))

    centers_coords_stack = centers_coords.reshape(n_columns, n_rows, 2)

    # 4) For each (s,v) center, compute (x,y) and draw a hex
    for column_ind in range(1, n_columns-1, 2):
        for row_ind in range(1, n_rows-1, 2):

            x_c, y_c = centers_coords_stack[column_ind, row_ind]

            if not fish_path.contains_point((x_c, y_c)):
                continue

            # Define neighbor cell centers
            x_c_l, y_c_l = centers_coords_stack[column_ind-1, row_ind]   # left vertex
            x_c_r, y_c_r = centers_coords_stack[column_ind+1, row_ind]   # right vertex
            x_c_u, y_c_u = centers_coords_stack[column_ind, row_ind+1]   # top edge midpoint
            x_c_d, y_c_d = centers_coords_stack[column_ind, row_ind-1]   # bottom edge midpoint

            # Create numpy arrays for easier math
            v_left  = np.array([x_c_l, y_c_l])
            v_right = np.array([x_c_r, y_c_r])
            m_top   = np.array([x_c_u, y_c_u])
            m_bot   = np.array([x_c_d, y_c_d])

            # Compute unit vector from left to right and distance d
            vec = v_right - v_left
            d = np.linalg.norm(vec)
            if d == 0:
                continue
            u = vec / d

            # Compute side length s (for a regular flat‐top hexagon, d = √3 * s)
            s = d / np.sqrt(3)

            # Top vertices
            v0 = m_top - (s/2) * u
            v1 = m_top + (s/2) * u
            v2 = v_right
            # Bottom vertices
            v3 = m_bot + (s/2) * u
            v4 = m_bot - (s/2) * u
            v5 = v_left

            # Check if all vertices are inside the fish body
            contained_points = [
                fish_path.contains_point((v[0], v[1]))
                for v in [v0, v1, v2, v3, v4, v5]
            ]

            if not all(contained_points):
                continue

            # Draw hexagon
            hexagon_verts = [v0, v1, v2, v3, v4, v5]

            hexagon = Polygon(
                hexagon_verts,
                closed    = True,
                facecolor = 'none',
                edgecolor = '0.60',
                linewidth = 0.25,
                alpha     = 0.50,
                zorder    = 2
            )
            hexagon.set_clip_path(fish_body_patch)
            axis.add_patch(hexagon)

    return axis

def _plot_fish_polygon(
    axis       : plt.Axes,
    x_lower    : np.ndarray,
    y_lower    : np.ndarray,
    x_upper    : np.ndarray,
    y_upper    : np.ndarray,
    plot_pars  : dict = None,
):
    ''' Plot fish polygon '''
    if plot_pars is None:
        plot_pars = {}

    body_color = plot_pars.get('body_color', np.array([252, 246, 220]) / 255)
    alpha      = plot_pars.get('alpha', 1.0) # 0.7
    edgecolor  = plot_pars.get('edgecolor', '0.2')
    linewidth  = plot_pars.get('linewidth', 1)
    zorder     = plot_pars.get('zorder', 1)

    # Create polygon vertices
    x_vertices = np.concatenate([x_upper, x_lower[::-1]])
    y_vertices = np.concatenate([y_upper, y_lower[::-1]])
    vertices   = np.column_stack([x_vertices, y_vertices])

    # Create and add white background polygon (same as the first fill_between)
    background_polygon = Polygon(
        vertices,
        closed = True,
        color  = '1.0', # White
        zorder = zorder
    )
    axis.add_patch(background_polygon)

    # Create and add the main fish polygon (same as the second fill_between)
    fish_polygon = Polygon(
        vertices,
        closed    = True,
        facecolor = body_color,
        alpha     = alpha,
        edgecolor = edgecolor,
        linewidth = linewidth,
        zorder    = zorder
    )
    axis.add_patch(fish_polygon)

    return (background_polygon, fish_polygon)

def _plot_fish_eyes(
    fish_plots: list,
    lines_dict: dict,
    axis      : plt.Axes,
    head_n    : int,
):
    ''' Plot fish eyes '''

    eye_kwargs = {
        'lines_dict': lines_dict,
        'axis'      : axis,
        'alpha'     : 1.0,
        'zorder'    : 3,
    }

    center_eye  = 0.43
    outer_eye_l = 0.16
    inner_eye_l = 0.14

    inner_eye = np.array( [ center_eye - inner_eye_l, center_eye + inner_eye_l ] )
    outer_eye = np.array( [ center_eye - outer_eye_l, center_eye + outer_eye_l ] )

    # Outer eye
    eye_start = round( head_n * outer_eye[0] )
    eye_end   = round( head_n * outer_eye[1] )
    frac_0, frac_1 = [0.80, +1.00]
    ind_0, ind_1   = [eye_start, eye_end]
    fish_parts     = _plot_bands(
        frac_0 = frac_0,
        frac_1 = frac_1,
        i0     = ind_0,
        i1     = ind_1,
        color  = '0.4',
        **eye_kwargs
    )
    fish_plots = fish_plots + fish_parts

    # Inner eye
    eye_start = round( head_n * inner_eye[0] )
    eye_end   = round( head_n * inner_eye[1] )
    frac_0, frac_1 = [0.90, +1.00]
    ind_0, ind_1   = [eye_start, eye_end]
    fish_parts     = _plot_bands(
        frac_0 = frac_0,
        frac_1 = frac_1,
        i0     = ind_0,
        i1     = ind_1,
        color  = '0.1',
        **eye_kwargs
    )
    fish_plots = fish_plots + fish_parts

    return fish_plots

def _plot_zebrafish_sketch(
    axis    : plt.Axes,
    x_midline: np.ndarray,
    y_midline: np.ndarray,
    x_lower  : np.ndarray,
    y_lower  : np.ndarray,
    x_upper  : np.ndarray,
    y_upper  : np.ndarray,
):
    ''' Plot zebrafish sketch with realistic eyes and a clipped hexagon skin pattern.
        Adjusted drawing order (zorder) ensures the hexagon pattern remains visible,
        and hexagons are clipped to the fish body boundaries. '''


    lines_dict = {
        'x_midline': x_midline,
        'y_midline': y_midline,
        'x_lower'  : x_lower,
        'y_lower'  : y_lower,
        'x_upper'  : x_upper,
        'y_upper'  : y_upper,
    }

    # Define region indices
    n_points = x_midline.shape[0]
    head_n   = n_points // 5
    trunk_n  = n_points // 2
    tail_n   = n_points // 6
    body_n   = n_points

    # fig, axis = plt.subplots(figsize=(10, 5))
    # plt.axis('equal')

    fish_plots = []

    # 1. Draw fish body shape (base fill) at zorder 1.
    body_polygons = _plot_fish_polygon(
        axis      = axis,
        x_lower   = x_lower,
        y_lower   = y_lower,
        x_upper   = x_upper,
        y_upper   = y_upper,
    )
    fish_plots = fish_plots + list(body_polygons)

    # 2. Draw tail and body bands at zorder 2.
    bands_kwargs = {
        'lines_dict': lines_dict,
        'axis'      : axis,
    }

    frac_0, frac_1 = [-1.00, 1.00]
    ind_0, ind_1   = [body_n - tail_n, body_n]
    fish_parts     = _plot_bands(
        frac_0 = frac_0,
        frac_1 = frac_1,
        i0     = ind_0,
        i1     = ind_1,
        color  = '0.9',
        zorder = 2,
        **bands_kwargs
    )
    fish_plots     = fish_plots + fish_parts

    frac_0, frac_1 = [0.40, 0.60]
    ind_0, ind_1   = [body_n - tail_n, body_n]
    fish_parts     = _plot_bands(
        frac_0 = frac_0,
        frac_1 = frac_1,
        i0     = ind_0,
        i1     = ind_1,
        color  = '0.7',
        zorder = 3,
        **bands_kwargs
    )
    fish_plots     = fish_plots + fish_parts

    band_color = np.array([122, 128, 170]) / 255
    frac_0, frac_1 = [0.50, 0.75]
    ind_0, ind_1   = [head_n, head_n + trunk_n]
    fish_parts     = _plot_bands(
        frac_0 = frac_0,
        frac_1 = frac_1,
        i0     = ind_0,
        i1     = ind_1,
        color  = band_color,
        faded  = True,
        zorder = 2,
        **bands_kwargs
    )
    fish_plots     = fish_plots + fish_parts

    # 3. Add hexagon skin pattern on top of the fish body,
    #    clipping each hexagon to the fish shape and drawing at zorder 3.
    axis = _fill_body_with_exagons(
        axis      = axis,
        x_midline = x_midline,
        y_midline = y_midline,
        x_lower   = x_lower,
        y_lower   = y_lower,
        x_upper   = x_upper,
        y_upper   = y_upper,
        head_n    = head_n,
        body_n    = body_n,
        tail_n    = tail_n,
    )

    # 4) Draw eyes
    fish_plots = _plot_fish_eyes(
        fish_plots = fish_plots,
        lines_dict = lines_dict,
        axis       = axis,
        head_n     = head_n,
    )

    return fish_plots

def plot_fish_configuration(
    axis          : plt.Axes,
    positions_x   : np.ndarray,
    positions_y   : np.ndarray,
    axis_lines    : list = None,
    thickness_type: str = 'model',
):
    ''' Plot fish configuration '''

    # Get thickness spline
    body_length = 0.018
    n_points    = 1000

    if thickness_type == 'model':
        thickness_f      = get_fish_half_thickness_from_model
        thickness_spline = _get_fish_half_thickness_spline_from_model(body_length)
        thickness_args   = (body_length, thickness_spline)

    elif thickness_type == 'gazzola':
        thickness_f      = get_fish_half_thickness_from_gazzola
        thickness_spline = None
        thickness_args   = (body_length,)

    elif thickness_type == 'sketch':
        thickness_f      = get_fish_half_thickness_from_sketch
        thickness_spline = _get_fish_half_thickness_spline_from_sketch(body_length)
        thickness_args   = (body_length, thickness_spline)

    (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    ) = get_fish_boundaries(
        positions_x    = positions_x,
        positions_y    = positions_y,
        thickness_f    = thickness_f,
        thickness_args = thickness_args,
        n_points       = n_points,
    )

    # Plot the fish body shape
    fish_plot_kwargs = {
        'axis'      : axis,
        'x_midline' : x_midline,
        'y_midline' : y_midline,
        'x_lower'   : x_lower,
        'y_lower'   : y_lower,
        'x_upper'   : x_upper,
        'y_upper'   : y_upper,
    }

    # axis.clear()
    fish_plots = _plot_zebrafish_sketch(**fish_plot_kwargs)
    axis_lines = fish_plots
    axis.set_aspect('equal')

    # if axis_lines is None:
    #     body_points = None
    #     midline     = None
    #     # body_points   , = axis.plot(positions_x, positions_y, 'o', label="Fish Body Points", markersize=2, color='red')
    #     # midline       , = axis.plot(x_midline, y_midline, 'k--', label="Midline")
    #     upper_boundary, = axis.plot(x_upper, y_upper, 'k-', label="Upper Boundary")
    #     lower_boundary, = axis.plot(x_lower, y_lower, 'k-', label="Lower Boundary")
    #     # fish_body       = axis.fill_between(x_midline, y_lower, y_upper, color='white', label="Fish Body")
    #     fish_plots = _plot_zebrafish_sketch_2(**fish_plot_kwargs)
    # else:
    #     body_points, midline, upper_boundary, lower_boundary = axis_lines[:4]
    #     fish_plots = axis_lines[4:]
    #     # body_points.set_data(positions_x, positions_y)
    #     # midline.set_data(x_midline, y_midline)
    #     upper_boundary.set_data(x_upper, y_upper)
    #     lower_boundary.set_data(x_lower, y_lower)
    #     # fish_body.remove()
    #     # fish_body = axis.fill_between(x_midline, y_lower, y_upper, color='white', label="Fish Body")

    #     for fish_plot in fish_plots:
    #         fish_plot.remove()
    #     fish_plots = _plot_zebrafish_sketch_2(**fish_plot_kwargs)

    # axis_lines = [body_points, midline, upper_boundary, lower_boundary] + fish_plots

    # if decorate:
    #     axis.legend()

    return axis_lines

def plot_fish_animation(
    times                : np.ndarray,
    positions_x_evolution: np.ndarray,
    positions_y_evolution: np.ndarray,
    thickness_type       : str = 'sketch',
    frame_step           : int = 1,
):
    ''' Animation of fish configuration '''

    # Generate frame indices with skipping
    frame_indices = np.arange(0, len(times), frame_step)

    # Plot fish configuration
    fig, ax = plt.subplots(figsize=(10, 5))

    axis_lines = plot_fish_configuration(
        axis           = ax,
        positions_x    = positions_x_evolution[0],
        positions_y    = positions_y_evolution[0],
        thickness_type = thickness_type,
    )

    def update(frame_idx):
        ax.clear()
        nonlocal axis_lines

        frame      = frame_indices[frame_idx]
        axis_lines = plot_fish_configuration(
            axis           = ax,
            positions_x    = positions_x_evolution[frame],
            positions_y    = positions_y_evolution[frame],
            axis_lines     = axis_lines,
            thickness_type = thickness_type,
        )
        ax.set_title(f"Time: {times[frame]:.2f} s")

        return axis_lines

    ani = FuncAnimation(
        fig    = fig,
        func   = update,
        frames = len(frame_indices),
        repeat = True,
    )

    return fig, ani

###############################################################################
# EXAMPLE #####################################################################
###############################################################################
def main():

    freq_t      = 3.0
    freq_x      = 1.0
    times       = np.linspace(0, 10, 1000)

    # Amplitude envelope
    bl = 0.018
    c2 = +0.28 * bl
    c1 = -0.13 * bl
    c0 = +0.05 * bl
    x  = np.linspace(0, bl, 16)
    s  = x / bl

    amp_envelope = c2*(s**2)+c1*s+c0

    # Evolve x and y positions
    positions_x_evolution = np.array( [ x for t in times ])
    positions_y_evolution = np.array(
        [
            amp_envelope * np.sin( 2*np.pi*(freq_x*s - freq_t * t))
            for t in times
        ]
    )

    # Plot fish configuration
    fig, ani = plot_fish_animation(
        times                 = times,
        positions_x_evolution = positions_x_evolution,
        positions_y_evolution = positions_y_evolution,
        thickness_type        = 'sketch',
        frame_step            = 1,
    )

    plt.show()

if __name__ == "__main__":
    main()
