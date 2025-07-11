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

#include "g2o_tutorial_slam2d_api.h"
#include "se2.h"

namespace g2o {
namespace tutorial {

//
class G2O_TUTORIAL_SLAM2D_API IncrementalSimulator {
 public:
  enum G2O_TUTORIAL_SLAM2D_API MotionType { MO_LEFT, MO_RIGHT, MO_NUM_ELEMS };

  /**
   * \brief simulated landmark
   */
  struct G2O_TUTORIAL_SLAM2D_API Landmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int id;
    Eigen::Vector2d truePose;
    Eigen::Vector2d simulatedPose;
    std::vector<int> seenBy;
    Landmark() : id(-1) {}
  };
  using LandmarkVector = std::vector<Landmark>;
  using LandmarkPtrVector = std::vector<Landmark*>;

  /**
   * simulated pose of the robot
   */
  struct G2O_TUTORIAL_SLAM2D_API GridPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int id;
    SE2 truePose;
    SE2 simulatorPose;
    LandmarkPtrVector landmarks;  ///< the landmarks observed by this node
  };
  using PosesVector = std::vector<GridPose>;

  /**
   * \brief odometry constraint
   */
  struct G2O_TUTORIAL_SLAM2D_API GridEdge {
    int from;
    int to;
    SE2 trueTransf;
    SE2 simulatorTransf;
    Eigen::Matrix3d information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };
  using GridEdgeVector = std::vector<GridEdge>;

  struct G2O_TUTORIAL_SLAM2D_API LandmarkEdge {
    int from;
    int to;
    Eigen::Vector2d trueMeas;
    Eigen::Vector2d simulatorMeas;
    Eigen::Matrix2d information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };
  using LandmarkEdgeVector = std::vector<LandmarkEdge>;

 public:
  IncrementalSimulator();
  ~IncrementalSimulator();

  const SE2& xTrue() const;
  void history() const;

  void start();
  Scenario getScenario() const;
  void processEvents(EventPtrVector events) const;

  // EventGenerator Methods
  // Step the simulatro
  void step();

protected:
  void processEvent(Event event);
  void registerEventHandler(EventType eventType, EventHandler eventHandler);
  void ignoreUnknownEventType();
  void handlePredictForwards(double dT);
  void initialize();
  // We consult the system model and we give an event.
  void updateOdometry();
  //void predictGPSObservation();
  //void predictCompassObservation();
  //void predictBearingObservations();
  void predictSLAMObservations();

  //void generateHeartbeat();
  void storeStepResults();

private:
  SE2 x_;
  Eigen::Vector3d u_;
  Eigen::Matrix3d sigmaU_;
  Eigen::Matrix3d sigmaUSqrtm_;

  Scenario scenario_;
  std::vector<Eigen::Vector2d> landmarks_;

  SystemModel systemModel_;
  WaypointController controller_;

  std::vector<double> timeStore_;
  std::vector<SE2> xTrueStore_;

  // Event Generator 
  int stepNumber_;
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
  // noiseScale;

  // % Flag; if set to false the simulator will terminate at the next
  // % time step
  // carryOnRunning;

  // % Flag to show if debugging is enabled.
  // debug;
        
  // % Flag to show if the system has been initialized or not
  bool initialized_;

  // 
  
  // Event generator queue & Outgoing event Queue functionalities
  // We assume that the system is always in sync.
  // We'll deal with this later
  // So effectivly if the step number is divisible by period we make prediction.
  int odomPeriod;
  int slamObsPeriod;

  // TODO: Set up MainLoop.
  // TODO: Set up Events.
  // TODO: set up a SLAMSystem that takes events.
};

}  // namespace tutorial
}  // namespace g2o

#endif
