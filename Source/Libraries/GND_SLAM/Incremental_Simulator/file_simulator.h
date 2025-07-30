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

#include <nlohmann/json.hpp>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "system_model.h"
#include "platform_controller.h"
#include "ordered_event_queue.hpp"
#include "sensor_data.h"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

namespace g2o {
namespace tutorial {


//
class G2O_TUTORIAL_SLAM2D_API FileSimulator {


protected:
  struct Vertex {
    int id;
    SE3Quat pose;          // stores rotation + translation

    Vertex(int vid, const SE3Quat& p)
      : id(vid), pose(p)
    {}
  };

  struct Edge {
    int v0, v1;
    SE3Quat delta;             // relative transform from v0 → v1
    Eigen::Matrix<double, 6, 6> info;    // full 6×6 information matrix

    Edge(int a, int b, const SE3Quat& d, const Eigen::Matrix<double, 6, 6>& I)
      : v0(a), v1(b), delta(d), info(I)
    {}
  };



 public:
  FileSimulator(const std::string& filename);
  ~FileSimulator();

  // =============================
  // Simulator methods
  // =============================
  /**
   * @brief returns the pose of the simulated viechle
   */
  SE2 xTrue2d() const;

  /**
   * @brief returns the pose of the simulated viechle
   */
  SE3Quat xTrue() const;


  void history(std::vector<double> &timeHistory, std::vector<g2o::SE3Quat> & xTrueHistory) const;

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
   * @brief return if the simulator should be stepped agian
   */
  bool keepRunning() const;


  std::vector<EventPtr> aquireEvents();



  // =============================
  // EventGenerator Methods
  // =============================
  /**
  * @brief Step the simulator
  */
  void step();



  void saveGroundTruth(const std::string& filename);

  

protected:
  /**
  * @brief This is the first step made by the simulator. It initializes the platform state to its initial conditions
  * and emits an initialization event. Odomotry is first set. Initial scheduling times to update odometry and simulate
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
   */
  void updateOdometry(Edge& odom);

  /** 
   * @brief Simulate SLAM observation and emmit according event
  */
  void updateObservation(Edge& obs);


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
  SE3Quat x_;

  // System Handles
  std::unique_ptr<SystemModel> systemModel_;
  OrderedEventQueue eventQueue_;
  std::unique_ptr<PlatformController> platformController_;
  // WaypointController controller_;

  std::vector<double> timeStore_;
  std::vector<SE3Quat> xTrueStore_;

  // Vertex Edge Vectors
  std::unordered_map<int,size_t> idToIndex_;
  std::vector<FileSimulator::Vertex> vertices_;
  std::unordered_map<int,std::vector<FileSimulator::Edge>> odomEdgesFrom_;
  std::unordered_map<int,std::vector<FileSimulator::Edge>> obsEdgesFrom_;
  int currentVtxNumber_;

  // Debug
  bool verbose_;
};

}  // namespace tutorial
}  // namespace g2o
