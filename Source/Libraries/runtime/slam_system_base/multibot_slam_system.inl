// Implementation of MultibotSlamSystem (included from multibot_slam_system.hpp).
// Do not compile this translation unit on its own.

#include <iostream>

namespace g2o {

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
int MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::preOptTrajectoryBatchCounter_ =
    0;

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::MultibotSlamSystem(
    const RobotId& id, const std::string& filename)
    : Base(filename), robotId_(id) {
  loadMultibotConfig(filename);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::loadMultibotConfig(
    const std::string& filename) {
  std::ifstream f(filename);
  if (!f) {
    throw std::runtime_error("MultibotSlamSystem: cannot open SLAM config: " + filename);
  }
  nlohmann::json j;
  f >> j;

  gndActiveConfig_ = j.value("gndActive_", true);
  gndActive_ = gndActiveConfig_;
  gndBound_ = j.value("gndBound_", gndBound_);
  gndPower_ = j.value("gndPower_", gndPower_);
  gndLnc_ = j.value("gndLnc_", gndLnc_);
  gndTailPenaltyStd_ = j.value("gndTailPenaltyStd_", gndTailPenaltyStd_);
  fixRelativetransform_ = j.value("fixRelativetransform_", false);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::runStopOptimization() {
  optimize(optCountStop_);
  if (gndActiveConfig_) {
    gndActive_ = true;
  }
  optimize(optCountStopFix_);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::onAfterOptimize() {
  if (pendingGndPriorEdges_.empty()) {
    return;
  }
  for (auto* e : pendingGndPriorEdges_) {
    if (!e) {
      continue;
    }
    auto* rk = dynamic_cast<g2o::ToggelableGNDKernel*>(e->robustKernel());
    if (rk) {
      rk->setBoolPointer(&gndActive_);
    } else if (verbose_) {
      std::cerr << "[GND] Pending prior edge has no ToggelableGNDKernel; id="
                << e->id() << std::endl;
    }
  }
  pendingGndPriorEdges_.clear();
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
g2o::ToggelableGNDKernel*
MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::newPriorToggelableGndKernel() {
  return new g2o::ToggelableGNDKernel(gndBound_, gndPower_, gndLnc_,
                                      gndTailPenaltyStd_, &gndActiveAlwaysFalse_);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::ignoreUnknownEventType() {
  if (verbose_) {
    std::cerr << "[MultibotSlamSystem] Ignoring unknown event type\n";
  }
}

}  // namespace g2o
