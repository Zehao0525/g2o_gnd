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
#include "se2.h"
#include "events.h"
#include "sensor_data.h"
#include "gnd_kernel.h"

#include "types_tutorial_slam2d.h"
#include "GNDEdges/edge_platform_loc_prior_gnd.h"
#include "slam_system_base.h"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/edge_se3_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"



namespace g2o {
namespace tutorial {


using VertexContainer = g2o::OptimizableGraph::VertexContainer;


typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;


class G2O_TUTORIAL_SLAM2D_API FileSlamSystem : public SlamSystemBase<VertexSE3, EdgeSE3> {


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
    int observerRobotId;                // ID of observer
    int observerVertexId;               // ID of the vertex the obervation is from
    int observedRobotId;                // ID of robot being observed
    int observedVertexId;               // ID of the vertex observed
    bool initialized;                   // If the data represented by the observation has been added to the graph

    Observation(int selfId, int robotId, int selfVtxId, int vertexId, g2o::EdgeSE3* obsPriorEdge, g2o::EdgeSE3* obsEdge, g2o::VertexSE3* obsVtx)
      : observerRobotId(selfId),
        observationEdge(obsEdge),
        observationPriorEdge(obsPriorEdge),
        observationVertex(obsVtx),
        observedRobotId(robotId),
        observerVertexId(selfVtxId),
        observedVertexId(vertexId),
        initialized(false)
    {}
  };

  struct ObsSyncRequest {
    int observerRobotId;                // ID of observer
    int observerVertexId;               // ID of the vertex the obervation is from
    int observedRobotId;                // ID of robot being observed
    int observedVertexId;               // ID of the vertex observed
    Isometry3 observedVtxLocation;
    Eigen::Matrix<double,6,6> observedVtxInformation;
    ObsSyncRequest(int selfId, int robotId, int selfVtxId, int vertexId):
      observerRobotId(selfId),
      observedRobotId(robotId),
      observerVertexId(selfVtxId),
      observedVertexId(vertexId)
    {}
  };

  struct ObsSyncMessage {
    int sourceId;
    bool outGoing;
    std::vector<ObsSyncRequest> syncRequests;
    ObsSyncMessage(): sourceId(0), outGoing(false), syncRequests({}){}
    ObsSyncMessage(int sender, bool og, std::vector<ObsSyncRequest> syncReqs): sourceId(sender), outGoing(og), syncRequests(syncReqs){}
  };


  FileSlamSystem(int id,const std::string& filename);
  ~FileSlamSystem();

  void platformEstimate2d(Eigen::Vector3d& x, Eigen::Matrix2d& P);

  void platformEstimate2d(Eigen::Vector3d& x) const;

  void platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double,6,6>& P);

  /**
   * @brief Initialie and start the SLAM system.
   */
  void start() override;


  /**
   * @brief stop the SLAM system and finallise result accumulation
   */
  void stop() override;

  /**
   * @brief trigger a request for data to line up observations
   */
  ObsSyncMessage broadcastObsSyncMessage() const; 

  /**
   * @brief trigger a request for data to line up observations
   */
  void handleObservationSyncResponse(const ObsSyncMessage& msg); 


  ObsSyncMessage handleObservationSyncRequest(ObsSyncMessage& msg);



protected:
  /**
   * @brief process a event
   * @param events event to process
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
  void handleInitializationEvent(FileInitEvent event);

  /**
   * @brief event handler for odometry events
   * @param event
   */
  void handleOdometryEvent(FileOdomEvent event);

  /**
   * @brief event handler for landmark observation events
   * @param event
   */
  void handleObservationEvent(FileObsEvent event);


  void handleIntraObservationEvent(FileIntraObsEvent event);


  int getRobotId(){return robotId_;}

public:
  bool gndActive_;

protected:

  // This mapping maps the vertex of the platform to platform vertecies
  // This also mapps the observed vertex to its location in "observations_"
  std::map<int, int> VertexIdMap_;
  int fileVertexId_;

  int robotId_;
  std::map<int, VertexSE3*> externalVertices_;
  std::map<int, EdgeSE3*> externalVerticesPrior_;

  bool haveUninitializedObs_;
  std::vector<FileSlamSystem::Observation> observations_;

  // Note, relative trandforms' Isometry values are the inverse of the actual relative ransforms.
  // (Transform to bot_i's frame) * (pose in bot_i's frame) == (Transform to observation source) * (noiseless observation measurment)
  std::map<int, VertexSE3*> relativeTransforms_;

  int intraRobotCount_ = 0;
  bool graphChanged_;
};

}  // namespace tutorial
}  // namespace g2o
