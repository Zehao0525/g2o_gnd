#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
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
#include "utisa_events.h"
#include "ggd_kernel.h"
#include "utisa_messages.hpp"
#include "utisa_stamp_map.hpp"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "edge_platform_pose_prior.h"
#include "edge_range_bearing.h"
#include "parameter_se2_offset.h"
#include "GGDEdges/edge_platform_loc_prior_ggd.h"
#include "multibot_slam_system.hpp"

namespace g2o {
namespace tutorial {
namespace multibotsim {

/** UTISA SLAM on MultibotSlamSystem / SlamSystemBase. */
class G2O_TUTORIAL_SLAM2D_API UTISASlamSystem
    : public ::g2o::MultibotSlamSystem<VertexSE2, EdgeSE2, std::string, UTSIAMessage> {
 protected:
  using Base = ::g2o::MultibotSlamSystem<VertexSE2, EdgeSE2, std::string, UTSIAMessage>;

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
  using Base::ggdActiveConfig_;
  using Base::ggdBound_;
  using Base::ggdPower_;
  using Base::ggdLnc_;
  using Base::ggdTailPenaltyStd_;
  using Base::ggdActiveAlwaysFalse_;
  using Base::pendingGgdPriorEdges_;
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
  using Base::newPriorToggelableGgdKernel;
  using Base::runStopOptimization;
  using Base::ignoreUnknownEventType;
  using Base::platformEstimate;
  using Base::platformEstimateMarginals;

 public:
  using Base::optimize;
  using Base::ggdActive_;
  using EstimateType = Base::EstimateType;
  using CovarianceType = Base::CovarianceType;

  struct LandmarkEst {
    int lmId = 0;
    std::string observerId;
    EdgeSE2PointXY* estimatePriorEdge = nullptr;
    bool initialized = false;
  };

  struct Landmark {
    int lmId = 0;
    bool initialized = false;
    VertexPointXY* landmark = nullptr;
    std::map<std::string, LandmarkEst> landmarkEsts;
  };

  struct Observation {
    EdgeSE2PointXY* observationPriorEdge;
    EdgeRangeBearing* observationEdge;
    VertexPointXY* observationVertex;
    std::string observerRobotId;
    std::string observedRobotId;
    double observationTime;
    int observationId;
    bool initialized;

    Observation(std::string selfId, double obsTime, int obsId, std::string robotId,
                EdgeSE2PointXY* obsPriorEdge, EdgeRangeBearing* obsEdge, VertexPointXY* obsVtx)
        : observerRobotId(std::move(selfId)),
          observationPriorEdge(obsPriorEdge),
          observationEdge(obsEdge),
          observationVertex(obsVtx),
          observedRobotId(std::move(robotId)),
          observationTime(obsTime),
          observationId(obsId),
          initialized(false) {}
  };

  UTISASlamSystem(const std::string& id, const std::string& filename);
  ~UTISASlamSystem() override;

  void platformEstimate(Eigen::Vector3d& x, Eigen::Matrix3d& P);
  void platformEstimate(Eigen::Vector3d& pose) const;

  std::vector<std::pair<double, SE2>> getTrajectory() const;
  void getRangeBearingObservationSegments(
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& landmarkSegs,
      std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& robotSegs) const;

  void saveTrajectoryTUM(const std::string& filename) const;
  void saveLandmarksXY(const std::string& filename) const;
  void dumpPreOptTrajectory(const std::string& run_directory);

  void start() override;
  void stop() override;

  UTSIAMessage broadcastSyncMessage() const override;
  UTSIAMessage handleObservationSyncRequest(UTSIAMessage& msg) override;
  void handleObservationSyncResponse(const UTSIAMessage& msg) override;

  /** @deprecated Prefer broadcastSyncMessage(). */
  UTSIAMessage broadcastUTSIAMessage() const { return broadcastSyncMessage(); }

 protected:
  void processEvent(Event& event) override;
  void handleInitializationEvent(UTISAInitEvent event);
  void handleOdometryEvent(UTISAOdomEvent event);
  void handleObservationEvent(UTISAObsEvent event);
  void handleLMObservationEvent(UTISALmObsEvent event);

  static constexpr int kPoseFrameParameterId = 0;
  static constexpr int kLandmarkSensorParameterId = 1;
  SE2 landmarkSensorOffset_;

  StampMap vertexStampMap_;
  StampMap externalVertexStampMap_;
  std::vector<Observation> observations_;
  std::map<int, Landmark> landmarks_;
  int nextLandmarkVertexSeq_ = 0;

  int se2PriorDiagPrinted_ = 0;
  static constexpr int kSe2PriorDiagPrintMax = 8;
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
