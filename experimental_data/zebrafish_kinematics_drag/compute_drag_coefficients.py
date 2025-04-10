import numpy as np
import pickle

N_LINKS_AXIS = 16
LENGTH_AXIS  = 0.018

# LENGTH
LINKS_X_EXTENSIONS = np.array(
    [
        2.56,
        1.70,
        1.80,
        2.00,
        2.00,
        2.00,
        2.50,
        1.47,
        1.47,
        1.47,
        1.47,
        1.10,
        1.00,
        1.00,
        1.00,
        1.80,
    ]
) / 1000
LINKS_Y_EXTENSIONS = np.array(
    [
        1.77,
        1.84,
        2.00,
        2.00,
        2.00,
        2.00,
        2.05,
        1.80,
        1.60,
        1.30,
        1.00,
        0.80,
        0.50,
        0.50,
        0.35,
        0.35,
    ]
) / 1000
LINKS_Z_EXTENSIONS = np.array(
    [
        2.45,
        2.77,
        3.03,
        3.30,
        3.30,
        3.30,
        3.80,
        4.12,
        4.12,
        4.12,
        4.12,
        2.20,
        1.60,
        2.10,
        3.20,
        4.00,
    ]
) / 1000

# MASS
LINKS_MASSES = np.array(
    [
        3.2734375504999984e-06,
        4.0521463491666665e-06,
        4.854872365333333e-06,
        5.157439432833333e-06,
        5.498348945e-06,
        5.479874165666666e-06,
        5.159457690333333e-06,
        4.249855453833333e-06,
        3.4060886698333332e-06,
        2.650774316333333e-06,
        1.7336472188333334e-06,
        1.1791477706666665e-06,
        8.259051296666667e-07,
        9.194377328333334e-07,
        9.366918884999999e-07,
        8.46120548e-07,
    ]
)

BODY_MASS = np.sum(LINKS_MASSES)

# PROJECTED SURFACE
LINKS_X_EXT_SEMI = LINKS_X_EXTENSIONS / 2
LINKS_Y_EXT_SEMI = LINKS_Y_EXTENSIONS / 2
LINKS_Z_EXT_SEMI = LINKS_Z_EXTENSIONS / 2

LINKS_XY_FACE_AREA = np.pi * LINKS_X_EXT_SEMI * LINKS_Y_EXT_SEMI
LINKS_XZ_FACE_AREA = np.pi * LINKS_X_EXT_SEMI * LINKS_Z_EXT_SEMI
LINKS_YZ_FACE_AREA = np.pi * LINKS_Y_EXT_SEMI * LINKS_Z_EXT_SEMI

LINKS_WET_SUFACE = np.pi * ( LINKS_Y_EXT_SEMI + LINKS_Z_EXT_SEMI ) * LINKS_X_EXTENSIONS
BODY_AREA        = np.sum(LINKS_WET_SUFACE)

# ESTIMATE FROM LITERATURE
def estimate_parameters_from_literature():
    '''
    Estimate the drag coefficients from the literature
    McHenry and Lauder (2006)
    '''

    lengt_axis_mm = LENGTH_AXIS * 1000
    mass_body_g   = 4.14 * 1e-6 * lengt_axis_mm ** +3.17
    area_body_mm2 = 3.06 * 1e-1 * lengt_axis_mm ** +2.16

    mass_body = mass_body_g   * 1e-3
    area_body = area_body_mm2 * 1e-6
    c_inert   = 1.44 * 1e+2 * lengt_axis_mm ** -2.34

    # Compute error
    error_mass = (BODY_MASS - mass_body) / BODY_MASS
    error_area = (BODY_AREA - area_body) / BODY_AREA

    print(f'Mass error: {error_mass * 100:.2f}%')
    print(f'Area error: {error_area * 100:.2f}%')

    return mass_body, area_body, c_inert

# SAVING
def save_coefficients_to_pickle(
    coeff_x                 : np.ndarray,
    coeff_y                 : np.ndarray,
    coeff_z                 : np.ndarray,
    overall_linear_coeff    : np.ndarray,
    overall_rotational_coeff: np.ndarray,
    filename                : str,
):
    ''' Save the drag coefficients to a pickle file '''

    # Pickle file
    with open(f'{filename}.pickle', 'wb') as outfile:
        pickle.dump(
            {
                'coeff_x' : coeff_x,
                'coeff_y' : coeff_y,
                'coeff_z' : coeff_z,

                'linear_coeff_x' : overall_linear_coeff[:, 0],
                'linear_coeff_y' : overall_linear_coeff[:, 1],
                'linear_coeff_z' : overall_linear_coeff[:, 2],

                'overall_linear_coeff' : overall_linear_coeff,

                'angular_coeff_x' : overall_rotational_coeff[:, 0],
                'angular_coeff_y' : overall_rotational_coeff[:, 1],
                'angular_coeff_z' : overall_rotational_coeff[:, 2],

                'overall_rotational_coeff' : overall_rotational_coeff,

            },
            outfile
        )

    return

def save_coefficients_to_txt(
    coeff_x                 : np.ndarray,
    coeff_y                 : np.ndarray,
    coeff_z                 : np.ndarray,
    overall_linear_coeff    : np.ndarray,
    overall_rotational_coeff: np.ndarray,
    filename                : str,
):
    ''' Save the drag coefficients to a txt file '''

    overall_coefficients = np.concatenate(
        [overall_linear_coeff, overall_rotational_coeff],
        axis = 1
    )

    linear_coeff_x = overall_linear_coeff[:, 0]
    linear_coeff_y = overall_linear_coeff[:, 1]
    linear_coeff_z = overall_linear_coeff[:, 2]

    angular_coeff_x = overall_rotational_coeff[:, 0]
    angular_coeff_y = overall_rotational_coeff[:, 1]
    angular_coeff_z = overall_rotational_coeff[:, 2]

    # Get with scientific notation
    get_str_vec = lambda vec : ', '.join([f'{val:.4e}' for val in vec])
    with open(f'{filename}.txt', 'w') as outfile:

        outfile.write('\n')
        outfile.write(f'coeff_x = [ {get_str_vec(coeff_x)} ]\n')
        outfile.write(f'coeff_y = [ {get_str_vec(coeff_y)} ]\n')
        outfile.write(f'coeff_z = [ {get_str_vec(coeff_z)} ]\n')

        outfile.write('\n')
        outfile.write(f'linear_coeff_x = [ {get_str_vec(linear_coeff_x)} ]\n')
        outfile.write(f'linear_coeff_y = [ {get_str_vec(linear_coeff_y)} ]\n')
        outfile.write(f'linear_coeff_z = [ {get_str_vec(linear_coeff_z)} ]\n')


        outfile.write('\n')
        outfile.write(f'angular_coeff_x = [ {get_str_vec(angular_coeff_x)} ]\n')
        outfile.write(f'angular_coeff_y = [ {get_str_vec(angular_coeff_y)} ]\n')
        outfile.write(f'angular_coeff_z = [ {get_str_vec(angular_coeff_z)} ]\n')

        outfile.write('\n')
        outfile.write('Overall linear coefficients: \n')
        outfile.write(
            '\n'.join(
                [
                    f'-{drag_coeff:.4e}'
                    for drag_coeff in overall_linear_coeff.reshape(-1)
                ]
            )
        )
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('Overall angular coefficients: \n')
        outfile.write(
            '\n'.join(
                [
                    f'-{drag_coeff:.4e}'
                    for drag_coeff in overall_rotational_coeff.reshape(-1)
                ]
            )
        )
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('Overall coefficients: \n')
        outfile.write(
            '\n'.join(
                [
                    f'-{drag_coeff:.4e}'
                    for drag_coeff in overall_coefficients.reshape(-1)
                ]
            )
        )

    return

# COEFFICIENTS
def get_linear_farms_drag_coefficients(
    coeff_x      : np.ndarray,
    coeff_y      : np.ndarray,
    coeff_z      : np.ndarray,
):
    ''' Compute the linear drag coefficients for the model '''
    water_density   = 1000.0
    overall_coeff_x = coeff_x * 0.5 * water_density * LINKS_XY_FACE_AREA
    overall_coeff_y = coeff_y * 0.5 * water_density * LINKS_XZ_FACE_AREA
    overall_coeff_z = coeff_z * 0.5 * water_density * LINKS_YZ_FACE_AREA

    return overall_coeff_x, overall_coeff_y, overall_coeff_z

def get_rotational_farms_drag_coefficients(
    coeff_x      : np.ndarray,
    coeff_y      : np.ndarray,
    coeff_z      : np.ndarray,
):
    ''' Compute the rotational drag coefficients for the model '''
    water_density   = 1000.0

    radius_x = ( LINKS_X_EXT_SEMI + LINKS_Y_EXT_SEMI ) / 2
    radius_y = ( LINKS_X_EXT_SEMI + LINKS_Z_EXT_SEMI ) / 2
    radius_z = ( LINKS_Y_EXT_SEMI + LINKS_Z_EXT_SEMI ) / 2

    overall_coeff_x = coeff_x * np.pi * water_density * radius_x**4 * LINKS_X_EXTENSIONS
    overall_coeff_y = coeff_y * np.pi * water_density * radius_y**4 * LINKS_Y_EXTENSIONS
    overall_coeff_z = coeff_z * np.pi * water_density * radius_z**4 * LINKS_Z_EXTENSIONS

    return overall_coeff_x, overall_coeff_y, overall_coeff_z

def get_farms_drag_coefficients(
    coeff_x_lin,   # 0.05
    coeff_y_lin,   # 0.70
    coeff_z_lin,   # 0.70

    coeff_x_rot,   # 0.05
    coeff_y_rot,   # 0.70
    coeff_z_rot,   # 0.70

    save_data = True,
):
    ''' Compute the drag coefficients for the model '''

    # McHenry and Lauder (2006)
    mass_body, area_body, c_inert = estimate_parameters_from_literature()

    # Overall drag coefficients
    water_density = 1000.0

    linear_coeff_x_est = c_inert * 0.5 * water_density * BODY_AREA
    linear_coeff_x, linear_coeff_y, linear_coeff_z = get_linear_farms_drag_coefficients(
        coeff_x = coeff_x_lin,
        coeff_y = coeff_y_lin,
        coeff_z = coeff_z_lin,
    )

    overall_linear_coeff = np.array([linear_coeff_x, linear_coeff_y, linear_coeff_z]).T

    # Rotation drag coefficients
    angular_coeff_x, angular_coeff_y, angular_coeff_z = get_rotational_farms_drag_coefficients(
        coeff_x = coeff_x_rot,
        coeff_y = coeff_y_rot,
        coeff_z = coeff_z_rot,
    )

    overall_rotational_coeff = np.array([angular_coeff_x, angular_coeff_y, angular_coeff_z]).T

    if not save_data:
        return overall_linear_coeff, overall_rotational_coeff

    pre_path = 'experimental_data/zebrafish_kinematics_drag/'

    # PICKLE file
    filename = f'{pre_path}/drag_coefficients'

    save_coefficients_to_pickle(
        coeff_x                 = coeff_x_lin,
        coeff_y                 = coeff_y_lin,
        coeff_z                 = coeff_z_lin,
        overall_linear_coeff    = overall_linear_coeff,
        overall_rotational_coeff= overall_rotational_coeff,
        filename                = filename,
    )

    # TXT file
    save_coefficients_to_txt(
        coeff_x                 = coeff_x_lin,
        coeff_y                 = coeff_y_lin,
        coeff_z                 = coeff_z_lin,
        overall_linear_coeff    = overall_linear_coeff,
        overall_rotational_coeff= overall_rotational_coeff,
        filename                = filename,
    )

    return overall_linear_coeff, overall_rotational_coeff

if __name__ == '__main__':

    CX = 0.05
    CY = 0.70
    CZ = 0.70

    COEFF_X_LIN      = CX * np.ones(N_LINKS_AXIS)
    COEFF_X_LIN[1:] *= 0.3

    COEFF_Y_LIN = CY * np.ones(N_LINKS_AXIS)
    COEFF_Z_LIN = CZ * np.ones(N_LINKS_AXIS)

    COEFF_X_ROT = CX * np.ones(N_LINKS_AXIS)
    COEFF_X_ROT[:] *= 100

    COEFF_Y_ROT = CY * np.ones(N_LINKS_AXIS)
    COEFF_Z_ROT = CZ * np.ones(N_LINKS_AXIS)

    get_farms_drag_coefficients(
        coeff_x_lin = COEFF_X_LIN,
        coeff_y_lin = COEFF_Y_LIN,
        coeff_z_lin = COEFF_Z_LIN,
        coeff_x_rot = COEFF_X_ROT,
        coeff_y_rot = COEFF_Y_ROT,
        coeff_z_rot = COEFF_Z_ROT,
    )