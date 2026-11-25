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
    git clone git@github.com:AlexPazzaglia/ZebrafishSNN.git
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

## ▶️ Running simulations

All scripts are launched **from the repository root**, e.g.

```bash
python run_sim_files/ex_vivo_simulations/static_bending/run_static_bending.py
```

Output locations are defined in `network_experiments/snn_paths.py` and can be
changed with environment variables:

| Variable | Content | Default |
|---|---|---|
| `ZEBRAFISH_SNN_RESULTS` | Raw simulation outputs | `<repository>/simulation_results` |
| `ZEBRAFISH_SNN_ANALYSIS` | Curated results read by the post-processing | `$ZEBRAFISH_SNN_RESULTS/analysis` |
| `ZEBRAFISH_SNN_WATER_DYNAMICS` | Pre-computed leader-wake flow fields | `$ZEBRAFISH_SNN_RESULTS/fluid_solver/saved_simulations` |


## 📄 Reproducing *A spinal circuit for collective coordination*

The simulations and analyses of the computational part of
Picton, Madrid, Pazzaglia et al., *A spinal circuit for collective coordination*
are listed, figure by figure, in [`run_sim_files/README.md`](run_sim_files/README.md):

- `model_calibration/` – Calibration of the network weights and of the motor output gains
- `ex_vivo_simulations/` – In silico counterpart of the ex vivo (paralysed) preparations:
  static bending, pacing (Fig. 3F–H) and fictive schooling (Fig. 3I–J)
- `in_vivo_simulations/` – In silico counterpart of the freely swimming fish:
  proprioception ON vs OFF, vortex phase matching (Fig. 4A–F), energy consumption
  (Fig. 4L–M) and vortex phase matching across muscle stiffness

The vortex-matching simulations require the pre-computed wake of the leader
fish (see [`run_sim_files/README.md`](run_sim_files/README.md#vortex-matching-simulations)).


## 📁 Project Structure

### `experimental_data/`
Contains real and synthetic data used for calibration, validation, and analysis.

- `zebrafish_kinematics_drag/` – Calibrate the drag coefficients by replaying the zebrafish kinematics in position control.
- `zebrafish_kinematics_muscles/` – Optimize the muscle parameters for a target natural frequency, damping ratio and zero-frequency gain.
- `zebrafish_neural_data_processing/` – Fit the neuron models (V2a, V0d, MN, PS) to the measured intrinsic properties.

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
Simulations and analyses of the paper (see [`run_sim_files/README.md`](run_sim_files/README.md)).

- `model_calibration/` – Isolated network (`open_loop_network/`) and motor output gains (`motor_output_gains/`)
- `ex_vivo_simulations/` – The muscles are silenced and the body is bent in position control, while the network receives sensory feedback (`static_bending/`, `in_silico_pacing/`, `in_silico_schooling/`)
- `in_vivo_simulations/` – The network controls the body through the muscles and receives sensory feedback (`proprioception_on_off/`, `vortex_matching/`)

---


## 📚 References

- Picton, L.D., Madrid, D., Pazzaglia, A., et al. "A spinal circuit for collective coordination." (2026).

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
