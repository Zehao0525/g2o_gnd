# GND(Or just none gaussian in general) G2O experiment

This a hand coded simulator based on the refactored ORB_SLAM2 repository. All the third party library are the same as : https://github.com/UCL/COMP0249_24-25_ORB_SLAM2.git. The refactored OrbSLAM also have better documentation, so if you have any probelm building this project, consider consulting the documentation of that repository. 

## User-visible changes from the original ORB-SLAM2:
1.  All executables are installed in `Build/Debug/Source/Examples/Tutorial_slam2d". "incsim_test" is the one used for this project. "tutorial_slamed" is just a refactor of the g2o example to test the CMake files. 

2. To run the executable, run "Build/Debug/Source/Examples/Tutorial_slam2d/incsim_test" from THIS directory level. I used relative path for the json enties in the executable, so trying to execute this in another directory level may or may not work. 

3. You can modify the setup of "incsim_test" by modifying the json files at "Source/Examples/Tutorial_slam2d". The names are quite intuitive so I won't elaborate on what each of them do. 
`.


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


