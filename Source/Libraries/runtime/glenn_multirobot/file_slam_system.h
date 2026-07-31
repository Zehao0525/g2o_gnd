#pragma once

#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "g2o_tutorial_slam2d_api.h"
#include "events_glenn.h"
#include "glenn_messages.hpp"
#include "gnd_kernel.h"
#include "multibot_slam_system.hpp"

#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/edge_se3_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/types_slam3d.h"

namespace g2o {
namespace tutorial {

/** Glenn multi-robot file SLAM on MultibotSlamSystem / SlamSystemBase. */
class G2O_TUTORIAL_SLAM2D_API FileSlamSystem
    : public ::g2o::MultibotSlamSystem<VertexSE3, EdgeSE3, int, FileObsSyncMessage> {
 protected:
  using Base = ::g2o::MultibotSlamSystem<VertexSE3, EdgeSE3, int, FileObsSyncMessage>;

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
  using Base::haveUninitializedObs_;
  using Base::graphChanged_;
  using Base::relativeTransforms_;
  using Base::gndActiveConfig_;
  using Base::runStopOptimization;
  using Base::ignoreUnknownEventType;
  using Base::platformEstimate;
  using Base::platformEstimateMarginals;

 public:
  using EstimateType = Base::EstimateType;
  using CovarianceType = Base::CovarianceType;
  using Base::optimize;
  using Base::gndActive_;

  struct Observation {
    g2o::EdgeSE3* observationPriorEdge;
    g2o::EdgeSE3* observationEdge;
    g2o::VertexSE3* observationVertex;
    int observerRobotId;
    int observerVertexId;
    int observedRobotId;
    int observedVertexId;
    bool initialized;

    Observation(int selfId, int robotId, int selfVtxId, int vertexId,
                g2o::EdgeSE3* obsPriorEdge, g2o::EdgeSE3* obsEdge,
                g2o::VertexSE3* obsVtx)
        : observationPriorEdge(obsPriorEdge),
          observationEdge(obsEdge),
          observationVertex(obsVtx),
          observerRobotId(selfId),
          observerVertexId(selfVtxId),
          observedRobotId(robotId),
          observedVertexId(vertexId),
          initialized(false) {}
  };

  using ObsSyncRequest = FileObsSyncRequest;
  using ObsSyncMessage = FileObsSyncMessage;

  FileSlamSystem(int id, const std::string& filename);
  ~FileSlamSystem() override;

  void platformEstimate2d(Eigen::Vector3d& x, Eigen::Matrix2d& P);
  void platformEstimate2d(Eigen::Vector3d& x) const;
  void platformEstimateSe3(EstimateType& x, CovarianceType& P);

  void start() override;
  void stop() override;

  FileObsSyncMessage broadcastSyncMessage() const override;
  FileObsSyncMessage handleObservationSyncRequest(FileObsSyncMessage& msg) override;
  void handleObservationSyncResponse(const FileObsSyncMessage& msg) override;

  /** @deprecated Prefer broadcastSyncMessage(). */
  ObsSyncMessage broadcastObsSyncMessage() const { return broadcastSyncMessage(); }

 protected:
  void processEvent(Event& event) override;
  void handleInitializationEvent(FileInitEvent event);
  void handleOdometryEvent(FileOdomEvent event);
  void handleObservationEvent(FileObsEvent event);
  void handleIntraObservationEvent(FileIntraObsEvent event);

  std::map<int, int> VertexIdMap_;
  int fileVertexId_ = -1;
  std::map<int, VertexSE3*> externalVertices_;
  std::map<int, EdgeSE3*> externalVerticesPrior_;
  std::vector<Observation> observations_;
  int intraRobotCount_ = 0;
};

}  // namespace tutorial
}  // namespace g2o
