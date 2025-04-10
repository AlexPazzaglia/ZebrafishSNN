''' Analyze results of an optimization '''
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import dill
import logging

import numpy as np
import matplotlib.pyplot as plt

import json

def load_optimization_parameters(
    opt_results_path : str,
):
    ''' Save parameters of the optimization '''

    filename = f'{opt_results_path}/parameters_optimization.dill'
    logging.info('Loading parameters_optimization data from %s', filename)

    pars_optimization = {}
    with open(filename, 'rb') as infile:
        pars_optimization['pars_names_list'] : list[str]   = dill.load(infile)
        pars_optimization['pars_units_list'] : list[str]   = dill.load(infile)
        pars_optimization['pars_l_list']     : list[float] = dill.load(infile)
        pars_optimization['pars_u_list']     : list[float] = dill.load(infile)

    return pars_optimization

def load_optimization_results(
    opt_results_path : str,
):
    ''' Load optimization results '''


    # Results across generations
    gen = 0
    results_optimization = []
    while True:
        filename = f'{opt_results_path}/generation_{gen}.dill'

        if not os.path.isfile(filename):
            break
        logging.info('Loading generation %i data from %s', gen, filename)
        with open(filename, 'rb') as infile:
            results_optimization.append( dill.load(infile) )

        gen+= 1

    return results_optimization

def main(gen=-1):

    data_path = 'experimental_data/zebrafish_neural_data_processing/neuron_parameters_optimization'
    results_path = f'{data_path}/optimization_results'

    # Determine the run folder to use
    if len(sys.argv) > 2:
        idx_run = int(sys.argv[2])
        run_folder = f'run_{idx_run}'
    else:
        # Find the latest run number if no specific run is provided
        existing_runs = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
        run_numbers = sorted([int(run.split('_')[1]) for run in existing_runs if run.startswith("run_")])
        if not run_numbers:
            print("No existing run folders found.")
            sys.exit(1)
        latest_run_number = run_numbers[-1]
        run_folder = f'run_{latest_run_number}'

    current_run_folder = os.path.join(results_path, run_folder)
    print(f"Using the run folder: {current_run_folder}")

    # # Load optimization parameters
    # pars_optimization = load_optimization_parameters(
    #     opt_results_path = results_path,
    # )

    # Load optimization results
    results_optimization = load_optimization_results(
        opt_results_path = current_run_folder,
    )

    n_gen = len(results_optimization)

    opt_inputs = np.array(
        [
            results_optimization[gen]['inputs']
            for gen in range(n_gen)
        ]
    )

    opt_outputs = np.array(
        [
            results_optimization[gen]['outputs']
            for gen in range(n_gen)
        ]
    )

    data = opt_inputs

    # Calculate the mean across neurons for each generation and parameter
    mean_data = np.mean(data, axis=1)

    # Create a figure with subplots
    fig, axs = plt.subplots(7, 1, figsize=(10, 20))


    # Read from a JSON file
    with open(f'{results_path}/vars_optimization.json', 'r') as file:
        loaded_vars_optimization = json.load(file)
    # Parameter information: name, min, and max values
    params_info = loaded_vars_optimization



    ####### Plot mean values of the parameters across generations #######

    for i, (name, min_val, max_val, _) in enumerate(params_info):
        axs[i].plot(mean_data[:, i])
        axs[i].set_title(name)
        axs[i].set_xlabel('Generation')
        axs[i].set_ylabel('Value')



        # Add horizontal lines for min and max values
        axs[i].axhline(y=min_val, color='r', linestyle='--')
        axs[i].axhline(y=max_val, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(current_run_folder, 'Mean_results.png'))
    plt.show()


    ######### Plot histograms of the parameters #########
    # Number of bins for the histogram
    n_bins = 20

    # Create a figure with subplots - one for each parameter
    fig, axs = plt.subplots(len(params_info), 1, figsize=(10, 20))

    for i, (name, min_val, max_val, _) in enumerate(params_info):
        # Parameter values for all neurons in the selected generation
        param_values = opt_inputs[gen, :, i]

        # Calculate the mean of the parameter values
        mean_val = np.mean(param_values)

        # Plot the histogram for the parameter values
        axs[i].hist(param_values, bins=n_bins, color='blue', alpha=0.7)

        # Add a vertical line for the mean value
        axs[i].axvline(x=mean_val, color='green', linestyle='-', label=f'Mean: {mean_val:.2f}')

        # Add horizontal lines for min and max values
        axs[i].axhline(y=min_val, color='r', linestyle='--', label='Min Value')
        axs[i].axhline(y=max_val, color='r', linestyle='--', label='Max Value')

        # Set title and labels
        axs[i].set_title(f'Parameter: {name}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

        # Add legend
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(current_run_folder, 'Histograms_results.png'))
    plt.show()

    return

if __name__ == '__main__':
    idx_generation = int(sys.argv[1])-1

    main(idx_generation)
