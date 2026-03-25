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

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
// TODO Change to Levenburg
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
// #include "simulator.h"



#include "g2o_tutorial_slam2d_api.h"
#include "events.h"
#include "gnd_kernel.h"
#include "messages.hpp"
#include "stamp_map.hpp"

#include "types_tutorial_slam2d.h"
#include "GNDEdges/edge_platform_loc_prior_gnd.h"
#include "slam_system_base.h"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/edge_se3_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"



namespace g2o {
namespace tutorial {
namespace multibotsim{

using VertexContainer = g2o::OptimizableGraph::VertexContainer;


typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;


class G2O_TUTORIAL_SLAM2D_API MultiDroneSLAMSystem : public SlamSystemBase<VertexSE3, EdgeSE3> {


protected:
  using Base = SlamSystemBase<VertexSE3, EdgeSE3>;
  //using Base::verbose_;

  using Base::stepNumber_;
  using Base::currentTime_;
  using Base::initialized_;
  using Base::componentsReady_;

  using Base::optPeriod_;
  using Base::optCountProcess_;
  using Base::optCountStop_;
  using Base::optCountStopFix_;

  using Base::optimizer_;

  using Base::vertexId_;
  using Base::processModelEdges_;
  using Base::numProcessModelEdges_;
  using Base::unfixedTimeWindow_;


  using Base::x_;
  using Base::currentPlatformVertex_;


  using Base::platformVertices_;

public:
  using Base::optimize;

 public:
  struct Observation {
    g2o::EdgeSE3* observationPriorEdge;  // pointer, since EdgeSE3Prior is non-copyable
    g2o::EdgeSE3* observationEdge;  // pointer, since EdgeSE3 is non-copyable
    g2o::VertexSE3* observationVertex;  // pointer, since VertexSE3 is non-copyable
    std::string observerRobotId;                // ID of observer
    std::string observedRobotId;                // ID of robot being observed
    double observationTime;                     // timestamp of the observation
    int observationId;
    bool initialized;                   // If the data represented by the observation has been added to the graph

    // NOTE: at this moment, we are not using message Time. 
    Observation(std::string selfId, double obsTime, int obsId, std::string robotId, g2o::EdgeSE3* obsPriorEdge, g2o::EdgeSE3* obsEdge, g2o::VertexSE3* obsVtx)
      : observerRobotId(selfId),
        observationEdge(obsEdge),
        observationPriorEdge(obsPriorEdge),
        observationVertex(obsVtx),
        observedRobotId(robotId),
        observationTime(obsTime),
        observationId(obsId),
        initialized(false)
    {}
  };

  MultiDroneSLAMSystem(const std::string& id, const std::string& filename);
  ~MultiDroneSLAMSystem();


  void platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double,6,6>& P);

  void platformEstimate(Eigen::Isometry3d& pose) const;

  /**
   * @brief Get the trajectory as a vector of (timestamp, pose) pairs
   * @return Vector of pairs: (timestamp, Isometry3d pose)
   */
  std::vector<std::pair<double, Eigen::Isometry3d>> getTrajectory() const;

  /**
   * @brief Save trajectory to file in TUM format (timestamp x y z qx qy qz qw)
   * @param filename Output file path
   */
  void saveTrajectoryTUM(const std::string& filename) const;
  void setPreOptTrajectoryOutputDir(const std::string& output_dir);
  void setPreOptTrajectoryDumpEnabled(bool enabled);

  
  /**
   * @brief Initialize and start the SLAM system.
   */
  void start() override;


  /**
   * @brief stop the SLAM system and finalize result accumulation
   */
  void stop() override;

  /**
   * @brief trigger a request for data to line up observations
   */
  DSMessage broadcastDSMessage() const; 

  /**
   * @brief trigger a request for data to line up observations
   */
  void handleObservationSyncResponse(const DSMessage& msg); 


  DSMessage handleObservationSyncRequest(DSMessage& msg);

  void dumpPreOptTrajectory();


protected:
  /**
   * @brief process an event
   * @param event event to process
   */
  void processEvent(Event& event) override;


  /**
   * @brief handler for event types that are not yet registered with a handle (unused)
   */
  void ignoreUnknownEventType();

  /**
   * @brief event handler for initialisation events
   * @param event
   */
  void handleInitializationEvent(DataInitEvent event);

  /**
   * @brief event handler for odometry events
   * @param event
   */
  void handleOdometryEvent(DataOdomEvent event);

  /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void handleObservationEvent(DataObsEvent event);


  //void handleIntraObservationEvent(FileIntraObsEvent event);


  const std::string& getRobotId() const { return robotId_; }

public:
  bool gndActive_;

protected:

  // This mapping maps the vertex of the platform to platform vertices
  // This also maps the observed vertex to its location in "observations_"
  std::string robotId_;

  StampMap vertexStampMap_;

  int exVtxCount_;
  StampMap externalVertexStampMap_;
  //std::map<int, VertexSE3*> externalVertices_;
  //std::map<int, EdgeSE3*> externalVerticesPrior_;

  bool haveUninitializedObs_;
  std::vector<Observation> observations_;

  // Note, relative transforms' Isometry values are the inverse of the actual relative transforms.
  // (Transform to bot_i's frame) * (pose in bot_i's frame) == (Transform to observation source) * (noiseless observation measurement)
  std::map<std::string, VertexSE3*> relativeTransforms_;
  
  // For g2o we still need integer vertex ids; we assign them sequentially
  // per SLAM system when new relative-transform vertices are created.
  int nextRelativeTransformVtxId_ = 0;

  int obsCount_ = 0;
  bool graphChanged_;

  // For velocity-based odometry decoding: convert event velocity to displacement
  // using timestamp differences.
  bool hasLastOdomTime_ = false;
  double lastOdomTime_ = 0.0;
  std::string preOptTrajectoryOutputDir_;
  bool preOptTrajectoryDumpEnabled_ = false;

  // Throttle debug constraint-check prints (to avoid terminal spam).
  int se3PriorDiagPrinted_ = 0;
  static constexpr int kSe3PriorDiagPrintMax = 8;
};

}
}  // namespace tutorial
}  // namespace g2o
