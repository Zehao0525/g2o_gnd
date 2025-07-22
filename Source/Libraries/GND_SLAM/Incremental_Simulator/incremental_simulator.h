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

#ifndef G2O_INCREMENTAL_SIMULATOR_H
#define G2O_INCREMENTAL_SIMULATOR_H

#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"
#include "system_model.h"
#include "platform_controller.h"
#include "ordered_event_queue.hpp"
#include "sensor_data.h"

namespace g2o {
namespace tutorial {


//
class G2O_TUTORIAL_SLAM2D_API IncrementalSimulator {
 public:
  enum G2O_TUTORIAL_SLAM2D_API MotionType { MO_LEFT, MO_RIGHT, MO_NUM_ELEMS };


 public:
  IncrementalSimulator(const std::string& filename);
  ~IncrementalSimulator();

  // =============================
  // Simulator methods
  // =============================
  /**
   * @brief returns the pose of the simulated viechle
   */
  SE2 xTrue() const;

  std::vector<Eigen::Vector2d> landmarkPosesTrue() const;

  std::vector<Eigen::Vector2d> waypointsTrue() const;

  /**
   * @brief returns the trajectory of the simulated viechle
   * @param timeHistory Timestamp
   * @param xTrueHistory poses
   */
  void history(std::vector<double> &timeHistory, std::vector<SE2> & xTrueHistory) const;

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
  * @brief Update the platform position to the next simulation time
  */
  void handlePredictForwards(double dT);

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
  void updateOdometry();
  //void predictGPSObservation();
  //void predictCompassObservation();
  //void predictBearingObservations();

  /** 
   * @brief Simulate SLAM observation and emmit according event
  */
  void predictSLAMObservations();


  /** 
   * @brief Simulate SLAM observation and emmit according event
  */
  void predictRangeBearingObservations();


  /** 
  * @brief Simulate SLAM observation and emmit according event
  */
  void predictGPSObservation();


  /** 
   * @brief Generate Heartbeat Event
  */
  void generateHeartbeat();

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
  SE2 x_;
  SE2 u_;
  Eigen::Matrix3d sigmaU_;
  Eigen::Matrix3d sigmaUSqrtm_;
  static thread_local std::unique_ptr<GaussianSampler<Eigen::Vector3d, Eigen::Matrix3d>> odomSampler_;

  // Scene info
  //Scenario scenario_;
  LandmarkPtrVector landmarks_;

  // System Handles
  std::unique_ptr<SystemModel> systemModel_;
  OrderedEventQueue eventQueue_;
  std::unique_ptr<PlatformController> platformController_;
  // WaypointController controller_;

  std::vector<double> timeStore_;
  std::vector<SE2> xTrueStore_;

  // Event Generator 
  int stepNumber_;
  int maximumStepNumber_;
  inline static double dT_ = 0.2;
  // // Ordered Event Queue
  //outgoingEvents;

  // // Whether the events in the queue are dispatched. If dispatched then it should be cleared.
  // // Which is always, after the simulator started
  // outgoingEventsDispatched;


  // EventBasedSimulator
  // The current time
  double currentTime_;

  // % Queue which stores the next event generator which is queued up
  // eventGeneratorQueue;

  // % Scale which can be applied to noise. Set to 0
  // % (no noise) or 1 (noise)
  double noiseScale_;
  
  // Event generator queue & Outgoing event Queue functionalities
  // We assume that the system is always in sync.
  // We'll deal with this later
  // So effectivly if the step number is divisible by period we make prediction.
  int odomPeriod_;
  int slamObsPeriod_;
  int rangBearingObsPeriod_;
  int gpsObsPeriod_;

  // TODO: Set up MainLoop.
  // TODO: Set up Events.
  // TODO: set up a SLAMSystem that takes events.

  // Debug
  bool verbose_;
};

}  // namespace tutorial
}  // namespace g2o

#endif
