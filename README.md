# GND(Or just none gaussian in general) G2O experiment

This a hand coded simulator based on the refactored ORB_SLAM2 repository. All the third party library are the same as : https://github.com/UCL/COMP0249_24-25_ORB_SLAM2.git. The refactored OrbSLAM also have better documentation, so if you have any probelm building this project, consider consulting the documentation of that repository. 

## User-visible changes from the original ORB-SLAM2:
1.  All executables are installed in `Build/Debug/Source/Examples/Tutorial_slam2d". "incsim_test" is the one used for this project. "tutorial_slamed" is just a refactor of the g2o example to test the CMake files. 

2. To run the executable, run "Build/Debug/Source/Examples/Tutorial_slam2d/incsim_test" from THIS directory level. I used relative path for the json enties in the executable, so trying to execute this in another directory level may or may not work. 

3. You can modify the setup of "incsim_test" by modifying the json files at "Source/Examples/Tutorial_slam2d". The names are quite intuitive so I won't elaborate on what each of them do. 

4. The outputs from the simulator and the slam_system are stored in `trajectory_before.g2o`, `trajectory_after.g2o`, `trajectory_gt.g2o`. The `trajectory_after.g2o` and `trajectory_before.g2o` files can be read back into the optimizer to set itself up (not exercised here). `visualizer.py` can visualize the trajectory files.


## Build instructions:

### Prerequisites

You can clone this repository using https://github.com/Zehao0525/g2o_gnd.git

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
  *This is pure test work, not very interesting.* Tests the validity of the Cauchy edges using David Rosen’s formulation, as well as the convergence behaviour of the GND kernels. It probably should have been placed in a unit test folder instead.

* **tutorial_w_bearing.cpp**
  Code used for the correlated absolute position test from the paper. Worth checking out. The design logic is that bearing and GPS information are absolute measurements with correlated noise. Previous tests showed that GND priors optimise poorly, so this explores using absolute data first. Everything is hard-coded, and the bearing poses are currently turned off.

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

### G2O_Graph (Library)
Content largely for 2D experiments. The most note worthy implementation is `gnd_kernel.h` and `gnd_kernel.cpp`, which is mostly used in the rest of the repository.

### Oneshot_Simulator
Simulator recreation for g2o_tutorial2d and its premutations

### Incremental_Simulator
Code supporting recreation of [COMP0249 Coursework 1](https://github.com/UCL/COMP0249_24-25), as well as simulation and SLAM system for data from *Glenn Shimoda*. `slam_system`, `system_model`, `platform_controller`, `incremental_simulator` are for COMP0249, in this case the simulator simulates data real time. `slam_system_base`. Anything with prefix "File" supports *Glenn Shimoda*'s data, in which case simulator reads in data line by line and parse them into events. `ordered_event_queue`, `events.h` are general purposed based class used accorss both experiments, as well as other subdirectories.

### Multidrone_Simulator
Code simulating SLAM system ans communication from using data generated by `python/multirobot_simulator`. each agent contains *simulator* and *slam_system*. Data is read in by the simulator and parsed into `md_events`. agents commuicate via `messages` and are managed by `agent_manager`. `stamp_map` help slam systems keep track of their factorgraph nodes.


### UTISA_Simulator
Very similar implementation to `Multidrone_Simulator`

### python/multirobot_simulator

This directory is the **Python multidrone data and tooling layer**: it defines synthetic worlds (planned paths, optional random landmarks), runs the Python `WorldSim` (`simulator.py`) to produce per-robot **ground-truth logs** and **message logs** consumed by the C++ `Multidrone_Simulator`, and ships small **evaluator** and **visualizer** scripts for inspecting batches of runs. JSON under `config/` (`sim_config.json`, `sim_config_batch.json`, `landmarks.json`, …) controls bots, sensors, noise, and batch layout. 

The following are meant to be **run as scripts** (from the repository root unless noted); several also expose importable functions for notebooks or other drivers.

* **`batch_config_writer.py`** — Run: `python python/multirobot_simulator/batch_config_writer.py`. Reads `config/sim_config_batch.json` (creating a minimal skeleton if missing), injects a shared default block per bot (sensors, controller, initialization), and writes the merged JSON back. Use this when you want every bot entry to carry the same sensor/controller template after changing the bot list or when the batch config file is incomplete.

* **`generate_batch_scenes.py`** — Run: `python python/multirobot_simulator/generate_batch_scenes.py`. **Stage 1** of the two-stage batch pipeline: creates `N_TRAJECTORIES` scene subfolders under a configured `trajectories_root`, each with `trajectories.json` and, when `n_landmarks > 0`, a matching `landmarks.json`. Does **not** execute the simulator.

* **`batch_simulate_from_scenes.py`** — Run: `python python/multirobot_simulator/batch_simulate_from_scenes.py`. **Stage 2**: for every immediate subdirectory of `trajectories_root` that contains `trajectories.json`, instantiates `WorldSim`, steps until destinations are reached or `max_steps`, and writes `gt_log_*`, `msg_log_*`, and `bot_ids.txt` into a parallel tree under `batch_root`. Copies trajectory (and landmark) JSON into each run folder for downstream tools.

* **`landmark_generator.py`** — Run: `python python/multirobot_simulator/landmark_generator.py` (default writes `config/landmarks.json`), or `from multirobot_simulator.landmark_generator import landmark_generation` to sample `X` random 3D landmarks inside a box and save them as JSON.

Core library code used by the above includes **`simulator.py`**, **`trajectory_generator.py`**, and **`controller.py`**. **`visualize_bounded_rv.py`** (package root) plots histograms of the bounded noise helper against reference PDFs; run: `python python/multirobot_simulator/visualize_bounded_rv.py` or `python -m multirobot_simulator.visualize_bounded_rv`.

#### `evaluator/`

* **`plot_trajectory_comparison.py`** — Run: `python python/multirobot_simulator/evaluator/plot_trajectory_comparison.py`. Loads GT (`gt_log` / project GT format) and TUM pre/post trajectories, aligns timestamps, plots 3D paths, and reports APE-style error. Edit paths and options at the top of the script.

* **`compare_batch_ape.py`** — Run: `python python/multirobot_simulator/evaluator/compare_batch_ape.py`. For paired batch result directories (e.g. with and without GND) and a shared GT batch root, computes and compares APE over the first *N* numeric runs common to all inputs.

#### `visualizer/`

* **`log_replay_visualizer.py`** — Run: `python python/multirobot_simulator/visualizer/log_replay_visualizer.py`. Matplotlib animation replaying `gt_log_*.txt` / `msg_log_*.txt`: GT path, odometry-integrated estimate, optional robot and landmark observation rays, optional `landmarks.json`. Behavior is controlled by constants in the file (no command-line interface).

* **`controller_viser.py`** — Run: `python python/multirobot_simulator/visualizer/controller_viser.py`. 3D animation of a single velocity-controlled quad model (path, pose, heading) for debugging the controller in isolation.

* **`multi_controller_vizer.py`** — Run: `python python/multirobot_simulator/visualizer/multi_controller_vizer.py`. Same idea as `controller_viser.py` for multiple drones driven by the shared simulator/controller stack.

### python/utisa

This directory holds **offline Python tools for the UTIAS MR.CLAM (UTISA) 2D pipeline**: it reads dataset files (`Robot*_Groundtruth.dat`, `Robot*_Odometry.dat`, `Robot*_Measurement.dat`, `Landmark_Groundtruth.dat`, `Barcodes.dat`) and C++ **TUM** exports under a results folder, supports cropping by **simulation-relative time** (display window vs metric window), and can **synthesize** a GT-consistent MR.CLAM-like dataset for debugging. It complements the `UTISA_slam` executables and `UTISA_Simulator` in C++; nothing here runs the SLAM graph in Python.

* **`mrclam_eval_common.py`** — Not a standalone tool: shared helpers (GT loading, TUM XY, time alignment, duration derivation, pre-opt path resolution) imported by the scripts below.

* **`plot_trajectory_comparison_utisa.py`** — Run: `python python/utisa/plot_trajectory_comparison_utisa.py`. 2D plot of MR.CLAM GT vs pre-opt vs post-opt trajectories with ATE/APE; optional GT landmark markers. Globals at the top set `DATASET_DIR`, `RESULTS_DIR`, `DISPLAY_DURATION_SEC`, `SIM_DURATION_SEC`, and optional automatic duration derivation.

* **`compare_batch_ape_utisa.py`** — Run: `python python/utisa/compare_batch_ape_utisa.py`. Batch-style absolute pose error over multiple robots/runs for UTISA result trees, using the same time-window conventions as the trajectory comparison script.

* **`plot_single_robot_observations_utisa.py`** — Run: `python python/utisa/plot_single_robot_observations_utisa.py`. One robot: GT / pre / post trajectories in green tones, GT landmarks, post-optimization estimated landmarks if exported, and range–bearing observation rays from the **post-opt** pose with per-landmark colors and separate styling for robot–robot observations.

* **`generate_mrclam_gt_dataset.py`** — Run: `python python/utisa/generate_mrclam_gt_dataset.py` (see `--help` for paths). Copies MR.CLAM GT and barcode/landmark files into a new folder tree and **rewrites** odometry and measurement files so they are **exactly** consistent with ground truth (sanity-check pipeline).

* **`visualize_utisa_dataset.py`** — Run: `python python/utisa/visualize_utisa_dataset.py`. Quick 2D matplotlib view of robot GT trajectories from a dataset directory with a configurable time window.

## Other variouse evaluators:
* `batch_visualizer.py`: vuslizes and evaluates Tutorial_slam2d premutations (simulated correlated absolute position data)

* `visualizer_multibot_real.py`: visualise and evaluate the results from *Glenn Shimoda*'s data. 




## Typical Workflow
Multidrone_slam and Tutorial_slam2d can genreally be ran as if given that the configs are properly setup. What need more of a workflow note is the multirobot_slam.

Typical workflow: edit or generate configs → **`generate_batch_scenes.py`** to write many `trajectories.json` (+ optional `landmarks.json`) scene folders → **`batch_simulate_from_scenes.py`** to replay each scene and write logs under a batch output tree → run C++ multidrone SLAM on those logs → use **`evaluator/`** scripts to compare trajectories against GT.




