# 🧠🐟 ZebrafishSNN

**ZebrafishSNN** is a simulation framework for modeling the sensorimotor control of swimming in zebrafish using spiking neural networks (SNNs). This repository integrates biologically inspired neural models, mechanical simulations, and experimental data to analyze locomotor behaviors in virtual zebrafish within static and dynamic aquatic environments.


## 🧪 Features

- Biologically inspired spiking neural networks

- Integration with FARMS for physics-based swimming

- Data-driven neuronal parameters optimization

- Position control and Torque control

- Closed-loop sensorimotor analyses


## 📦 Installation

- Download and install Python 3.10+

- Create a folder for your project (e.g. ProjectSNN)

- Create and activate a virtual environment within the ProjectSNN folder:

    ```bash
    python -m venv snnenv
    snnenv\Scripts\activate     # Windows
    source snnenv/bin/activate  # Linux
    ```

- Clone ZebrafishSNN repository within the ProjectSNN folder:

    ```bash
    git clone git@gitlab.com:alessandro.pazzaglia/ZebrafishSNN.git
    ```

- Enter ZebrafishSNN folder and install requirements:

    ```bash
    pip install -r requirements.txt
    ```

- Enter farms folder and install farms packages:

    ```bash
    pip install -e farms_core
    pip install -e farms_mujoco
    pip install -e farms_sim
    pip install -e farms_amphibious
    ```

## 📁 Project Structure

### `experimental_data/`
Contains real and synthetic data used for calibration, validation, and analysis.

- `zebrafish_kinematics/` – Compute and simulate body angles during position control.
- `zebrafish_kinematics_drag/` – Study drag effects across different swimming frequencies.
- `zebrafish_kinematics_muscles/` – Optimize muscle parameters using dynamic simulations and genetic algorithms.
- `zebrafish_neural_data_processing/` – Analyze intrinsic neuronal properties and optimize neuron models.

---

### `farms_experiments/`
Experiments integrating the [FARMS simulator](https://github.com/epfl-lcn/farms) with zebrafish models.

- `experiments/` – YAML configurations for SNN vs. position-controlled swimming.
- `maps/` – Flow map generation for swimming arenas.
- `models/` – SDF models for zebrafish and environments.

---

### `network_experiments/`
Experiments for neural network simulation, training, and sensitivity analysis.

- SNN training with evolutionary optimization
- Logging and profiling tools
- Signal simulation and analysis utilities

---

### `network_implementations/`
SNN models for open-loop and closed-loop simulations.

---

### `network_modules/`
Core SNN framework for zebrafish neural modeling.

- `build/` – Network assembly
- `connectivity/` – Custom wiring strategies
- `core/` – Core logic and utilities
- `equations/` – Neuron and synapse model definitions
- `parameters/` – Setup for neurons, synapses, drives, and mechanics
- `performance/` – Simulation metrics and signal analysis
- `plotting/` – Visualization tools
- `simulation/` – Orchestration and callbacks
- `vortices/` – Vortex signal extraction and analysis

---

### `network_parameters/`
Modular YAML files for network, simulation, and mechanical configurations.

- Neuron/synapse topologies
- Simulation setup
- Mechanical/environment settings

---

### `neuron_experiments/`
Experiments for characterizing neuron and synapse behavior.

- Gain functions
- Current step responses
- Visualization of individual neuron dynamics

---

### `run_sim_files/`
Scripts for launching and analyzing full zebrafish simulations.

- `open_loop/` – Run the spiking neural network in isolation without mechanical model
- `position_control/` – Run the mechanical model in isolation with the desired kinematics
- `signal_driven/` – Run the mechanical model in isolation with the desired muscles activation
- `hybrid_position_control/` – Run the neural network receiving sensory feedback from the position-controlled mechanical body
- `closed_loop/` – Run the neural network controlling the mechanical body via muscles.

---


## 📚 References

- Pazzaglia, Alessandro, et al. "Balancing central control and sensory feedback produces adaptable and robust locomotor patterns in a spiking, neuromechanical model of the salamander spinal cord." PLOS Computational Biology 21.1 (2025): e1012101.

    Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012101

    Repository: https://github.com/AlexPazzaglia/SalamandraSNN

- Arreguit, Jonathan, Shravan Tata Ramalingasetty, and Auke Ijspeert. "FARMS: framework for animal and robot modeling and simulation." BioRxiv (2025): 2023-09.

    Paper: https://www.biorxiv.org/content/10.1101/2023.09.25.559130v3

    Repository: https://github.com/farmsim


## 📄 License

This project is licensed under the MIT License.

## ✍️ Author

Alessandro Pazzaglia, PhD student, Biorobotics Laboratory, EPFL
