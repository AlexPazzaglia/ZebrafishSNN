'''
Analysis utilities for the vortex matching simulations (phases, frequencies, fits).
'''


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
# FITTING ######################################################################
################################################################################

def save_fit_results(
    results   : dict,
    csv_path  : str,
    metadata  : dict = None,
):
    ''' Compute the fit performance and save it to a CSV file. '''

    results = results.copy()

    # Include any optional metadata (e.g., file name, trial number)
    if metadata:
        results.update(metadata)

    # Convert numpy arrays (e.g., params) to strings
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            results[key] = np.array2string(
                val,
                separator      = ' ',
                precision      = 8,
                suppress_small = True
            )

    # Single-row DataFrame
    new_df = pd.DataFrame([results])

    if os.path.exists(csv_path):
        # Load existing data
        existing_df = pd.read_csv(csv_path)

        # Concatenate and drop duplicates
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
    else:
        combined_df = new_df

    # Save full de-duplicated data
    combined_df.to_csv(csv_path, index=False)

    return

def load_fit_results(csv_path: str) -> pd.DataFrame:
    ''' Load the fit performance data from a CSV file. '''
    try:
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates()
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return pd.DataFrame()  # Return empty DataFrame on failure
    except pd.errors.EmptyDataError:
        print(f"File is empty: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()

###############################################################################
# PERFORMANCE #################################################################
###############################################################################

def load_performance(
    spawn_x_vals     : np.ndarray,
    spawn_y_vals     : np.ndarray,
    seed_vals        : np.ndarray,
    file_path        : str,
    target_quantities: list[str],
):
    ''' Plot the performance data '''

    # Parameters
    n_spawn_x = len(spawn_x_vals)
    n_spawn_y = len(spawn_y_vals)
    n_seeds   = len(seed_vals)

    parameters_vals_list = [
        [
            spawn_x_vals[spawn_x_ind],
            spawn_y_vals[spawn_y_ind],
            seed_vals[seed_ind],
        ]
        for spawn_x_ind in range(n_spawn_x)
        for spawn_y_ind in range(n_spawn_y)
        for seed_ind in range(n_seeds)
    ]
    parameters_vals = np.array(parameters_vals_list)

    # Load performance data
    performance_data = pd.read_csv(file_path)

    # Select the relevant columns
    performance_data = performance_data[target_quantities]

    # Add parameters to the DataFrame
    performance_data['spawn_x'] = parameters_vals[:, 0]
    performance_data['spawn_y'] = parameters_vals[:, 1]
    performance_data['seed']    = parameters_vals[:, 2]

    return performance_data

def compute_performance_differences_by_mode(
    performance_all  : pd.DataFrame,
    target_quantities: list[str],
    target_diff_types: list[str] = None,
):
    '''
    Compute differences closed_loop - open_loop (relative or absolute)
    for each target quantity, across all matching data points.
    '''

    if target_diff_types is None:
        target_diff_types = ['abs'] * len(target_quantities)

    assert len(target_quantities) == len(target_diff_types), \
        "target_quantities and target_diff_types must have the same length."

    # Pivot the dataframe to get one row per (FN, x, y, seed) with both open/closed values
    pivot_cols = ['muscle_FN', 'spawn_x', 'spawn_y', 'seed']
    wide_df = performance_all.pivot_table(
        index   = pivot_cols,
        columns = 'mode_tag',
        values  = target_quantities
    )

    # Compute relative differences: (closed_loop - open_loop) / open_loop
    diff_data = {}
    for quantity, diff_type in zip(target_quantities, target_diff_types):

        # Compute difference
        open_vals   = wide_df[(quantity, 'open_loop')]
        closed_vals = wide_df[(quantity, 'closed_loop')]

        if diff_type == 'abs':
            quantity_diff = ( closed_vals - open_vals )
        elif diff_type == 'rel':
            quantity_diff = (closed_vals - open_vals) / open_vals * 100.0
        else:
            raise ValueError(f"Unknown difference type: {diff_type}")

        # Add to the result
        diff_data[quantity] = quantity_diff

    # Assemble result into a dataframe
    diff_df = pd.DataFrame(diff_data, index=wide_df.index).reset_index()

    return diff_df

def _plot_difference_distribution_by_muscle(
    performance_diff   : pd.DataFrame,
    target_quantity    : str,
    relative_difference: bool = False,
):
    ''' Plot the distribution of paired differences (CL - OL) '''

    # Compute paired differences
    if relative_difference:
        tag = 'Relative Difference (%)'
    else:
        tag = 'Absolute Difference'

    # Plot
    fig = plt.figure(
        f'Distribution of Differences: {target_quantity}',
        figsize=(8, 6)
    )

    # Set up seaborn styling
    sns.set_theme(
        style   = "whitegrid",
        palette = "colorblind",
    )

    # Plot grouped boxplots (or violinplot)
    ax = sns.boxplot(
        data       = performance_diff,
        x          = 'muscle_FN',
        y          = target_quantity,
        width      = 0.6,
        showfliers = False,
        linewidth  = 1.0,
    )

    quantity_str = target_quantity.replace('_', ' ').capitalize()
    ax.set_title(f'{quantity_str} vs Muscle_FN')
    ax.set_xlabel('Muscle FN')
    ax.set_ylabel(tag)
    plt.tight_layout()

    return

def plot_differences_distribution_by_muscle(
    performance_all  : pd.DataFrame,
    target_quantities: list[str],
    target_diff_types: list[str],
):
    ''' Plot the distribution of paired differences (CL - OL) '''

    # Compute paired differences
    performance_diff = compute_performance_differences_by_mode(
        performance_all   = performance_all,
        target_quantities = target_quantities,
        target_diff_types = target_diff_types,
    )

    # Plot the distribution of differences
    for quantity, diff_type in zip(target_quantities, target_diff_types):

        _plot_difference_distribution_by_muscle(
            performance_diff   = performance_diff,
            target_quantity    = quantity,
            relative_difference= (diff_type == 'rel'),
        )
    return