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
Picton, Madrid, Pazzaglia et al., [*A spinal circuit for collective coordination*](https://arxiv.org/abs/2608.25909)
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

- `zebrafish_kinematics_muscles/` – Optimize the muscle parameters for a target natural frequency, damping ratio and zero-frequency gain.
- `zebrafish_neural_data_processing/` – Fit the neuron models (V2a, V0d, MN, PS) to the measured intrinsic properties.

---

### `farms_experiments/`
Experiments integrating the [FARMS simulator](https://github.com/epfl-lcn/farms) with zebrafish models.

- `experiments/` – YAML configurations for SNN vs. position-controlled swimming.
- `models/` – SDF models for zebrafish and environments.

---

### `network_experiments/`
Simulation setup and execution of the neuromechanical model.

- Default parameters of the zebrafish model for each control mode
- Simulation runners, post-processing and logging
- Paths of the simulation results (`snn_paths.py`)

---

### `network_implementations/`
Reference network implementations and connectivity schemes of the zebrafish model.

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
Single-neuron simulations used to fit the neuron models to the experimental
intrinsic properties (used by `experimental_data/zebrafish_neural_data_processing/`).

---

### `run_sim_files/`
Simulations and analyses of the paper (see [`run_sim_files/README.md`](run_sim_files/README.md)).

- `model_calibration/` – Isolated network (`open_loop_network/`) and motor output gains (`motor_output_gains/`)
- `ex_vivo_simulations/` – The muscles are silenced and the body is bent in position control, while the network receives sensory feedback (`static_bending/`, `in_silico_pacing/`, `in_silico_schooling/`)
- `in_vivo_simulations/` – The network controls the body through the muscles and receives sensory feedback (`proprioception_on_off/`, `vortex_matching/`)

---


## 📚 References

- Picton, Laurence, David Madrid, Alessandro Pazzaglia, et al. "A spinal circuit for collective coordination." arXiv preprint arXiv:2608.25909 (2026).

    Paper: https://arxiv.org/abs/2608.25909

- Pazzaglia, Alessandro, et al. "Balancing central control and sensory feedback produces adaptable and robust locomotor patterns in a spiking, neuromechanical model of the salamander spinal cord." PLOS Computational Biology 21.1 (2025): e1012101.

    Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012101

    Repository: https://github.com/AlexPazzaglia/SalamandraSNN

- Arreguit, Jonathan, Shravan Tata Ramalingasetty, and Auke Ijspeert. "FARMS: framework for animal and robot modeling and simulation." BioRxiv (2025): 2023-09.

    Paper: https://www.biorxiv.org/content/10.1101/2023.09.25.559130v3

    Repository: https://github.com/farmsim


## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{picton2026spinal,
  title         = {A spinal circuit for collective coordination},
  author        = {Picton, Laurence and Madrid, David and Pazzaglia, Alessandro and Wang, Yutong and
                   Bertuzzi, Maria and Ferrario, Andrea and Anastasiadis, Alexandros and Arreguit, Jonathan and
                   Fontanel, Pierre and Huang, Chun-Xiao and Mulleners, Karen and Song, Jianren and
                   Ijspeert, Auke Jan and El Manira, Abdel},
  year          = {2026},
  eprint        = {2608.25909},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.NC},
  doi           = {10.48550/arXiv.2608.25909},
  url           = {https://arxiv.org/abs/2608.25909},
}
```


## 📄 License

This project is licensed under the MIT License.

## ✍️ Author

Alessandro Pazzaglia, PhD student, Biorobotics Laboratory, EPFL
