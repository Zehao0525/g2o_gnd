// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "g2o_tutorial_slam2d_api.h"
#include "events_base.h"
#include "md_events.h"
#include "gnd_kernel.h"
#include "messages.hpp"
#include "stamp_map.hpp"
#include "types_tutorial_slam2d.h"
#include "GNDEdges/edge_platform_loc_prior_gnd.h"
#include "multibot_slam_system.hpp"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/edge_se3_pointxyz.h"
#include "g2o/types/slam3d/edge_se3_prior.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

/**
 * Multidrone SLAM system rebased onto MultibotSlamSystem / SlamSystemBase (new).
 * Parallel to MultiDroneSLAMSystem; does not replace it.
 */
class G2O_TUTORIAL_SLAM2D_API MultiDroneSLAMSystemNew
    : public ::g2o::MultibotSlamSystem<VertexSE3, EdgeSE3, std::string, DSMessage> {
 protected:
  using Base = ::g2o::MultibotSlamSystem<VertexSE3, EdgeSE3, std::string, DSMessage>;

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
  using Base::currentPlatformVertex_;
  using Base::platformVertices_;
  using Base::verbose_;
  using Base::robotId_;
  using Base::gndActiveConfig_;
  using Base::gndBound_;
  using Base::gndPower_;
  using Base::gndLnc_;
  using Base::gndTailPenaltyStd_;
  using Base::gndActiveAlwaysFalse_;
  using Base::pendingGndPriorEdges_;
  using Base::lmQueryEnabled_;
  using Base::robotQueryEnabled_;
  using Base::fixRelativetransform_;
  using Base::haveUninitializedObs_;
  using Base::graphChanged_;
  using Base::relativeTransforms_;
  using Base::relativeTransformInGraph_;
  using Base::nextRelativeTransformVtxId_;
  using Base::exVtxCount_;
  using Base::obsCount_;
  using Base::hasLastOdomTime_;
  using Base::lastOdomTime_;
  using Base::preOptTrajectoryOutputDir_;
  using Base::preOptTrajectoryDumpEnabled_;
  using Base::newPriorToggelableGndKernel;
  using Base::runStopOptimization;
  using Base::ignoreUnknownEventType;
  using Base::platformEstimate;
  using Base::platformEstimateMarginals;

 public:
  using Base::optimize;
  using Base::gndActive_;
  using EstimateType = Base::EstimateType;
  using CovarianceType = Base::CovarianceType;

  struct LandmarkEst {
    int lmId = 0;
    std::string observerId;
    EdgeSE3PointXYZ* estimatePriorEdge = nullptr;
    bool initialized = false;
  };

  struct Landmark {
    int lmId = 0;
    bool initialized = false;
    VertexPointXYZ* landmark = nullptr;
    std::map<std::string, LandmarkEst> landmarkEsts;
  };

  struct Observation {
    g2o::EdgeSE3* observationPriorEdge;
    g2o::EdgeSE3* observationEdge;
    g2o::VertexSE3* observationVertex;
    std::string observerRobotId;
    std::string observedRobotId;
    double observationTime;
    int observationId;
    bool initialized;

    Observation(std::string selfId, double obsTime, int obsId, std::string robotId,
                g2o::EdgeSE3* obsPriorEdge, g2o::EdgeSE3* obsEdge, g2o::VertexSE3* obsVtx)
        : observerRobotId(selfId),
          observationEdge(obsEdge),
          observationPriorEdge(obsPriorEdge),
          observationVertex(obsVtx),
          observedRobotId(robotId),
          observationTime(obsTime),
          observationId(obsId),
          initialized(false) {}
  };

  MultiDroneSLAMSystemNew(const std::string& id, const std::string& filename);
  ~MultiDroneSLAMSystemNew() override;

  void platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double, 6, 6>& P);
  void platformEstimate(Eigen::Isometry3d& pose) const;

  std::vector<std::pair<double, Eigen::Isometry3d>> getTrajectory() const;
  void saveTrajectoryTUM(const std::string& filename) const;
  void dumpPreOptTrajectory(const std::string& run_directory);

  void start() override;
  void stop() override;

  DSMessage broadcastSyncMessage() const override;
  DSMessage handleObservationSyncRequest(DSMessage& msg) override;
  void handleObservationSyncResponse(const DSMessage& msg) override;

  /** @deprecated Prefer broadcastSyncMessage(). */
  DSMessage broadcastDSMessage() const { return broadcastSyncMessage(); }

 protected:
  void processEvent(Event& event) override;
  void handleInitializationEvent(DataInitEvent event);
  void handleOdometryEvent(DataOdomEvent event);
  void handleObservationEvent(DataObsEvent event);
  void handleLMObservationEvent(DataLmObsEvent event);

  StampMap vertexStampMap_;
  StampMap externalVertexStampMap_;
  std::vector<Observation> observations_;
  std::map<int, Landmark> landmarks_;
  int nextLandmarkVertexSeq_ = 0;

  int se3PriorDiagPrinted_ = 0;
  static constexpr int kSe3PriorDiagPrintMax = 8;
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
