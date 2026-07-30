// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "gnd_kernel.h"
#include "multibot_messages.hpp"
#include "slam_system_base_new.hpp"

namespace g2o {

/**
 * Shared middle layer for multi-robot SLAM systems built on SlamSystemBase.
 *
 * Holds robot identity, relative-transform / GND bookkeeping, and the
 * inter-robot observation sync API (broadcast → request → response).
 *
 * MessageType must derive from tutorial::multibotsim::SyncMessageBase
 * (e.g. DSMessage, UTSIAMessage, FileObsSyncMessage).
 */
template <typename VertexType, typename EdgeType, typename RobotId,
          typename MessageType>
class MultibotSlamSystem : public SlamSystemBase<VertexType, EdgeType> {
  static_assert(std::is_base_of_v<tutorial::multibotsim::SyncMessageBase, MessageType>,
                "MessageType must derive from SyncMessageBase");

 protected:
  using Base = SlamSystemBase<VertexType, EdgeType>;

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
  using Base::optimize;
  using Base::platformEstimate;
  using Base::platformEstimateMarginals;

 public:
  using EstimateType = typename Base::EstimateType;
  using CovarianceType = typename Base::CovarianceType;
  using RobotIdType = RobotId;
  using SyncMessageType = MessageType;

  MultibotSlamSystem(const RobotId& id, const std::string& filename);
  ~MultibotSlamSystem() override = default;

  // -------------------------------------------------------------------------
  // Inter-robot communications (observation sync)
  // -------------------------------------------------------------------------

  /** Emit pending observation (and optional landmark) sync queries. */
  virtual MessageType broadcastSyncMessage() const = 0;

  /** Answer another robot's sync queries (marginalize + fill poses / infos). */
  virtual MessageType handleObservationSyncRequest(MessageType& msg) = 0;

  /** Apply answered sync data into this robot's factor graph. */
  virtual void handleObservationSyncResponse(const MessageType& msg) = 0;

  void setLmQueryEnabled(bool enabled) { lmQueryEnabled_ = enabled; }
  void setRobotQueryEnabled(bool enabled) { robotQueryEnabled_ = enabled; }

  void setPreOptTrajectoryOutputDir(const std::string& output_dir) {
    preOptTrajectoryOutputDir_ = output_dir;
  }
  void setPreOptTrajectoryDumpEnabled(bool enabled) {
    preOptTrajectoryDumpEnabled_ = enabled;
  }

  static void resetPreOptTrajectoryBatchCounter() {
    preOptTrajectoryBatchCounter_ = 0;
  }
  static int takeNextPreOptTrajectoryBatchIndex() {
    return preOptTrajectoryBatchCounter_++;
  }

  bool gndActive_ = false;

 protected:
  /** Load GND / relative-transform keys from the same SLAM JSON as the base. */
  void loadMultibotConfig(const std::string& filename);

  /** Shared stop skeleton: optimize → optionally enable GND → optimize again. */
  void runStopOptimization();

  void onAfterOptimize() override;

  g2o::ToggelableGNDKernel* newPriorToggelableGndKernel();

  void ignoreUnknownEventType();

  const RobotId& getRobotId() const { return robotId_; }

  RobotId robotId_;

  bool gndActiveConfig_ = true;
  double gndBound_ = 3.0;
  double gndPower_ = 6.0;
  double gndLnc_ = 1e-3;
  double gndTailPenaltyStd_ = 5.0;
  bool gndActiveAlwaysFalse_ = false;
  std::vector<g2o::OptimizableGraph::Edge*> pendingGndPriorEdges_;

  bool lmQueryEnabled_ = true;
  bool robotQueryEnabled_ = true;
  bool fixRelativetransform_ = false;

  bool haveUninitializedObs_ = false;
  bool graphChanged_ = false;

  std::map<RobotId, VertexType*> relativeTransforms_;
  std::map<RobotId, bool> relativeTransformInGraph_;
  int nextRelativeTransformVtxId_ = 0;

  int exVtxCount_ = 0;
  int obsCount_ = 0;

  bool hasLastOdomTime_ = false;
  double lastOdomTime_ = 0.0;

  std::string preOptTrajectoryOutputDir_;
  bool preOptTrajectoryDumpEnabled_ = false;
  static int preOptTrajectoryBatchCounter_;
};

}  // namespace g2o

#include "multibot_slam_system.inl"
