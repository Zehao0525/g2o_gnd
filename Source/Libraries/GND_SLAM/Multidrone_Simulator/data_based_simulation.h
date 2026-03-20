// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <map>
#include <vector>
#include <fstream>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"
#include "ordered_event_queue.hpp"

#include "g2o/core/hyper_graph.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

namespace g2o {
namespace tutorial {
namespace multibotsim{


/// Replay simulator for multidrone logs (Python `simulator.py`).
///
/// **World / body convention (matches Python):** right-handed; **+Z is up**; horizontal motion is in
/// the X–Y plane; **yaw** is a right-handed rotation about **+Z** (heading in the XY plane). Body-frame
/// odometry uses **+X forward**, **+Y lateral** (unused / zero), **+Z up** for vertical velocity.
class G2O_TUTORIAL_SLAM2D_API DataBasedSimulation {


 public:
  DataBasedSimulation(std::string id, const std::string& data_path, const std::string& gt_path);
  ~DataBasedSimulation();

  // =============================
  // Simulator methods
  // =============================

  /**
   * @brief returns the pose of the simulated vehicle
   */
  Isometry3 xTrue() const;


  void history(std::vector<double> &timeHistory, std::vector<g2o::Isometry3> & xTrueHistory) const;

  /**
   * @brief start the simulation
   * Start the simulator. This includes clearing any history
   * values, starting the platform controller and clearing the
   *  event queue. If the landmark configuration is random, the
   * landmark positions are also randomly drawn at this time.
   * 
   * The method also kick starts the simulation by scheduling a
   * call to the initialize handler.
   */
  void start();

  // Returns the scenario. We will hard code this for now. 
  // Scenario getScenario() const;

  /**
   * @brief return if the simulator should be stepped again
   */
  bool keepRunning() const;

  /**
   * @brief Stop the simulation. Sets carryOnRunning_ to false so that
   * step() will no longer advance. Does not close file streams.
   */
  void stop();

  std::vector<EventPtr> acquireEvents();



  // =============================
  // EventGenerator Methods
  // =============================
  /**
  * @brief Step the simulator
  */
  void step();

  void step(double dt);

  

protected:
  /**
  * @brief This is the first step made by the simulator. It initializes the platform state to its initial conditions
  * and emits an initialization event. Odometry is first set. Initial scheduling times to update odometry and simulate
  * all enabled sensors are also scheduled.
  * 
  * This handler should be the first thing called by the simulator.
  */
  void initialize();

  // We consult the system model and we give an event.
  /**
   * @brief Call the waypoint controller and work out the new control inputs for the simulator. Create a noise corrected odom
   * event which stores this value. 
   * 
   * This method will flag whether or not the simulation session is finished. (Outgoing events are emitted and the next call to this method is scheduled.)
  //  */
  // void updateOdometry(Edge& odom);

  // // /** 
  // //  * @brief Simulate SLAM observation and emit according event
  // // */
  // void updateObservation(Edge& obs);


  /**
   * @brief Store data into xTrueStore_ and timeStore_
   */
  void storeStepResults();

  // isWithinGPSOccluder, isDetectedByBearingSensors

public:
  // % Flag; if set to false the simulator will terminate at the next
  // % time step
  bool carryOnRunning_;

  // % Flag to show if debugging is enabled.
  // debug;
        
  // % Flag to show if the system has been initialized or not
  bool initialized_;

protected:
  // Platform State
  Isometry3 x_;

  // EventQueue
  OrderedEventQueue eventQueue_;

  std::vector<double> timeStore_;
  std::vector<Isometry3> xTrueStore_;

  // Vertex Edge Vectors
  int currentVtxNumber_;

  // Debug
  bool verbose_;

  // Communication
  std::string robotId_;

  double currentTime_;
  
  // File I/O for data-based simulation
  std::ifstream dataStream_;
  std::ifstream gtStream_;
  
  // Buffers for reading next lines
  struct GTBuffer {
    bool valid = false;
    double time = 0.0;
    Isometry3 pose = Isometry3::Identity();
  } gtBuffer_;
  
  enum class DataMsgType {
    None,
    Odom,
    RelPos
  };
  
  struct DataBuffer {
    bool valid = false;
    double time = 0.0;
    DataMsgType type = DataMsgType::None;
    Isometry3 odomPose = Isometry3::Identity();
    double odomOmegaZ = 0.0;
    std::string targetRobotId;
    Isometry3 relPose = Isometry3::Identity();
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
  } dataBuffer_;
  
  bool gtHasMore_;
  bool dataHasMore_;

  // Odom timestamps (for velocity->increment conversion)
  bool hasPrevOdomTime_ = false;
  double prevOdomTime_ = 0.0;
  
  // Internal helper methods
  bool readNextGT();
  bool readNextData();
  
};

}
}  // namespace tutorial
}  // namespace g2o
