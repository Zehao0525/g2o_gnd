# GGD(Or just none gaussian in general) G2O experiment

This a hand coded simulator based on the refactored ORB_SLAM2 repository. All the third party library are the same as : https://github.com/UCL/COMP0249_24-25_ORB_SLAM2.git. The refactored OrbSLAM also have better documentation, so if you have any probelm building this project, consider consulting the documentation of that repository. 

## User-visible changes from the original ORB-SLAM2:
1.  All executables are installed under `Build/Debug/experiments/...`. `incsim_test` (COMP0250-style incremental 2D) lives in `comp0250_reimplement`. `tutorial_slam2d` is a refactor of the g2o tutorial to smoke-test the CMake wiring.

2. To run it, use `Build/Debug/experiments/comp0250_reimplement/incsim_test` from THIS directory level. Config paths are relative to the repo root.

3. You can modify the setup of `incsim_test` via the JSON files under `experiments/comp0250_reimplement/config/`.

4. The outputs from the simulator and the slam_system are stored in `trajectory_before.g2o`, `trajectory_after.g2o`, `trajectory_gt.g2o`. The `trajectory_after.g2o` and `trajectory_before.g2o` files can be read back into the optimizer to set itself up (not exercised here). See `python/evaluators/comp0250_reimplement/visualizer.py` for visualization.


## Build instructions:

### Prerequisites

You can clone this repository using https://github.com/Zehao0525/g2o_ggd.git

It depends on a few widely-available libraries:

1. eigen3
2. boost
3. OpenCV (either 3.x or 4.x)
4. Suite sparse
5. GLEW
6. unzip
7. cmake (version 3.20 or above)

The ships with matched versions of DLib and DBoW2 (for the bag of words for data association), g2o (both front and backend optimization) and pangolin (GUI).

The build instructions are deliberately designed to be similar on all supported operating systems.

The line above came from OrbSLAM2 refactored. Even though I wish that was true, since my priority is not multysystem accessability, i only tested it on Linux. Specifically native Ubuntu 22.04. So it might not work on other machines. 

### Linux (and WSL2?) build instructions

Install the dependencies:

`sudo apt install cmake build-essential libeigen3-dev libboost-dev libboost-filesystem-dev libblas-dev liblapack-dev libepoxy-dev libopencv-dev libglew-dev mesa-utils libgl1-mesa-glx unzip`

Build by running:

`./Build.sh`

to build the release version. To build a debug version, type:

`./Build.sh Debug`

#### Installing cmake 3.20:

If your version of cmake is older than 3.20, you will need to install it manually:

`wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -`

`sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'`

`sudo apt update`

`sudo apt install cmake`

#### Display issues:
(This section is also from OrbSLAM2_refactored. I never got this issue, but if you did, consult this. )

You can get errors of the form `terminate called after throwing an instance of 'std::runtime_error' what():  Pangolin X11: Failed to open X display`. To fix (at least in our case) set:

`export DISPLAY=:0`


### Mac (Intel and Apple Silicon) build instructions

(This section is from OrbSLAM@_refactored. I never tried this.)

We use `homebrew` (https://brew.sh/) and build using the XCode command line tools. Please ensure that both have been installed.

Install the dependencies:

`brew install eigen boost suitesparse opencv glew`

You should be able to build the release by by running:

`./Build.sh`

To build a debug version, type:

`./Build.sh Debug`

If you want to avoid typing `./Install/bin` everywhere, run this command from the command line:

`set PATH=$PATH:$PWD/Install/bin`

### Windows 10/11 build (does not work; do NOT use)

(I deleted the .bat files at the start. But hey it turns out I didn't need to modify the .sh files after all. So if you want, replace the "Scripts" folder wth the one from OrbSLAM2_refactored, read the build instrcution for that and give this a shot. Might work, who knows.)


## Repository Overview

(over view)

### Tutorial_slam2d (Executable)

A chaotic first subdirectory filled with coding patterns of a person who had clearly never worked with such a large repository. This will be organised in the future.

* **tutorial_slam2d.cpp**
  *This is pure test work, not very interesting.* Functionally identical to `tutorial_slam2d.cpp` from the g2o base package; this was mainly something I wrote to familiarise myself with cross-subdirectory linking.

* **cauchy_edge_validity_test(2/3).cpp**
  *This is pure test work, not very interesting.* Tests the validity of the Cauchy edges using David Rosen’s formulation, as well as the convergence behaviour of the GGD kernels. It probably should have been placed in a unit test folder instead.

* **tutorial_w_bearing.cpp**
  Code used for the correlated absolute position test from the paper. Worth checking out. The design logic is that bearing and GPS information are absolute measurements with correlated noise. Previous tests showed that GGD priors optimise poorly, so this explores using absolute data first. Everything is hard-coded, and the bearing poses are currently turned off.

* **incsim_test.cpp**
  A C++ spinoff of the MATLAB code from [COMP0249 Coursework 1](https://github.com/UCL/COMP0249_24-25). Configurable using `simulator_config.json`, `slam_system_config.json`, and `view_config.json`. Everything works pretty much the same as the MATLAB code, except that the C++ version has prettier visualisation and faster execution. When in doubt, consult the COMP0249 material first; it is very well documented.

* **multibot_concept_test.cpp**
  I honestly cannot remember what this is for, but judging from the name, it is probably not important.

* **multibot_full_graph.cpp**
  Reads in the full multibot dataset from *Glenn Shimoda* and optimises the full graph as one.

* **multirobot_incsim_test.cpp**
  Reads in the factor graph data from Glenn Shimoda, and read it in vertex by vertex as it it was from an incremental simulator. The configs are in the `multirobot_configs` subdirectory. 



### Multidrone_slam (Executable)
A far more developed repository for multidrone experiment from simulated data. 
* **experiment.cpp**
  Single experiment with visulaisation: The parameters are controlled by json files in the `conifg` repository.

* **batch_experiment.cpp**
  Run the exteriment many times (equal to the number of data in the input folder), and output to the results to the output directory.

* `batch_experiment_config.json` 
  Controls where the input and export are for the batch_experiment, overrides the input and export path of single experimnets

* `experiment_base_config.json`
  Controls the config of single experiment. (The verbose field of the configs are rather chaotic, so you can ignore that.)

* `slam_system_config.json`
  Configuration of the SLAM system

* `topology`
  communication topology of the experiment

* `view_config`
  confog for the view
  

### UTISA_slam (Executable)
This repository is structurally similar to "Multidrone_slam". please reference discription of that. 

Also note that the covariance values of the UTISA test came from the following paper, with covariance set with the "velosity measurment principles" mentioned and $\sigma_{k}$ set to 0.1 as did the paper for the UTIAS tests. i.e: 
- $\sigma_{odom-v}$ = $\sqrt{2} / 2 * \sigma_{k}$ = 0.0707
- $\sigma_{odom-\omega}$ = $\sqrt{2} / a * \sigma_{k}$ (I used a = 0.258 for irobot rombas as described in the manual) = 0.548
- $\sigma_{r} = 0.5$
- $\sigma_{\theta}$ = 3 degrees = 0.0524

"Y. Huang, C. Xue, F. Zhu, W. Wang, Y. Zhang and J. A. Chambers, "Adaptive Recursive Decentralized Cooperative Localization for Multirobot Systems With Time-Varying Measurement Accuracy," in IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1-25, 2021, Art no. 8501525, doi: 10.1109/TIM.2021.3054005. keywords: {Robots;Covariance matrices;Noise measurement;Multi-robot systems;Location awareness;Estimation;Adaptive systems;Adaptive filter;decentralized cooperative localization;extended Kalman filter;multirobot systems;variational Bayesian},"

### Repo layout

| Folder | Role |
|--------|------|
| `src/types/tutorial_slam2d/` | Tutorial SE2 verts/edges/params (CMake target `G2O_Graph`) |
| `src/fght/` | GGD kernels + `GGDEdges/` (CMake target `GGD_Core`) |
| `tools/event_based/runtime/` | Event-driven simulators + SLAM frontends (`reimplement_comp0250`, `glenn_multirobot`, `multidrone_simulator`, `utisa_simulator`, `slam_system_base`) |
| `tools/event_based/viz/` | Pangolin views (CMake target `Incremental_Visualizer`) |
| `tools/offline/oneshot_simulator/` | Build-then-optimize simulator (CMake target `Oneshot_Simulator`) |
| `experiments/` | Runnable demos (`comp0250_reimplement`, `multirobot/`, `pilots/`) |
| `unit_tests/` | Parallel test drivers mirroring `experiments/` |
| `thirdparty/` | g2o, Pangolin, DBoW2, DLib (built via `Build_ThirdParty.sh`) |

### Oneshot_Simulator (`tools/offline/oneshot_simulator/`)
Simulator recreation for g2o_tutorial2d and its permutations.

### Event-based runtime (`tools/event_based/runtime/`)
Code supporting recreation of [COMP0249 Coursework 1](https://github.com/UCL/COMP0249_24-25), as well as simulation and SLAM system for data from *Glenn Shimoda*. `slam_system`, `system_model`, `platform_controller`, `incremental_simulator` are for COMP0249, in this case the simulator simulates data real time. `slam_system_base`. Anything with prefix "File" supports *Glenn Shimoda*'s data, in which case simulator reads in data line by line and parse them into events. `ordered_event_queue`, `events.h` are general purposed based class used across both experiments, as well as other subdirectories.

### Multidrone simulator (`tools/event_based/runtime/multidrone_simulator/`)
Code simulating SLAM system and communication from using data generated by `python/simulators/multidrone`. each agent contains *simulator* and *slam_system*. Data is read in by the simulator and parsed into `md_events`. agents communicate via `messages` and are managed by `agent_manager`. `stamp_map` help slam systems keep track of their factorgraph nodes.

### UTISA simulator (`tools/event_based/runtime/utisa_simulator/`)
Very similar implementation to `multidrone_simulator`.

### python/

Top-level layout:

| Folder | Role |
|--------|------|
| `simulators/` | Data-generation worlds (`multidrone/`) |
| `evaluators/` | Offline metrics / plots (`multidrone/`, `utisa/`, `ggd_studies/`, …) |
| `diagram_plotters/` | Paper/thesis diagrams (robust-kernel illustrations, etc.) |

### python/simulators/multidrone

Python multidrone data layer: generate scenes, run `WorldSim`, write GT/message logs for the C++ Multidrone stack. See **`python/simulators/multidrone/README.md`**.

**Main pipeline** (from repo root, with `PYTHONPATH=python`):

* **`batch_config_writer.py`** — inject shared sensor/controller defaults into `config/sim_config_batch.json`
* **`generate_batch_scenes.py`** — stage 1: scene folders with `trajectories.json` (+ optional `landmarks.json`)
* **`batch_simulate_from_scenes.py`** — stage 2: simulate each scene → `gt_log_*` / `msg_log_*` / `bot_ids.txt`

**Library:** `core/`. **Optional:** `tools/`. **Evaluation:** `python/evaluators/multidrone/`.

### python/evaluators/utisa

Offline tools for UTIAS MR.CLAM (UTISA). See **`python/evaluators/utisa/README.md`**.

* **`common.py`** — shared helpers (GT/TUM load, time crop, APE/ATE)
* **`plot_trajectory_comparison.py`** — GT vs pre/post trajectories + metrics
* **`compare_batch_ape.py`** — batch APE across result trees
* **`plot_single_robot_observations.py`** — one robot + observation rays
* **`plot_unit_experiment.py`** — unit-experiment result layout
* **`visualize_dataset.py`** — GT-only dataset viewer
* **`generate_mrclam_gt_dataset.py`** — synthesize GT-consistent odom/meas for debugging

### python/diagram_plotters

Paper figures (e.g. GGD vs robust kernels): `plot_ggd_robust_comparison.py`.

## Other various evaluators

Under `python/evaluators/`:

* **`glenn_multirobot/`** — Glenn Shimoda multi-robot (`visualizer_multibot_real*.py`, helpers)
* **`correlated_data_sim/`** — Tutorial_slam2d correlated GPS-like batches (`batch_visualizer*.py`)
* **`comp0250_reimplement/`** — COMP0249/0250-style 2D landmark runs (`visualizer.py`)

Root leftovers (not experiment evaluators): `samplerTest.py`, `tmp_data_processor.py`.




## Typical Workflow
Multidrone_slam and Tutorial_slam2d can generally be run as-is given that the configs are properly set up. What needs more of a workflow note is the multidrone pipeline.

Typical workflow: edit or generate configs → **`python/simulators/multidrone/generate_batch_scenes.py`** → **`batch_simulate_from_scenes.py`** → run C++ multidrone SLAM → use **`python/evaluators/multidrone/`** to compare trajectories against GT.




# To look:
- Cauchy distribution
- Map prior examples




