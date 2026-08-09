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

  ggdActiveConfig_ = j.value("ggdActive_", true);
  ggdActive_ = ggdActiveConfig_;
  ggdBound_ = j.value("ggdBound_", ggdBound_);
  ggdPower_ = j.value("ggdPower_", ggdPower_);
  ggdLnc_ = j.value("ggdLnc_", ggdLnc_);
  ggdTailPenaltyStd_ = j.value("ggdTailPenaltyStd_", ggdTailPenaltyStd_);
  fixRelativetransform_ = j.value("fixRelativetransform_", false);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::runStopOptimization() {
  optimize(optCountStop_);
  if (ggdActiveConfig_) {
    ggdActive_ = true;
  }
  optimize(optCountStopFix_);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::onAfterOptimize() {
  if (pendingGgdPriorEdges_.empty()) {
    return;
  }
  for (auto* e : pendingGgdPriorEdges_) {
    if (!e) {
      continue;
    }
    auto* rk = dynamic_cast<g2o::ToggelableGGDKernel*>(e->robustKernel());
    if (rk) {
      rk->setBoolPointer(&ggdActive_);
    } else if (verbose_) {
      std::cerr << "[GGD] Pending prior edge has no ToggelableGGDKernel; id="
                << e->id() << std::endl;
    }
  }
  pendingGgdPriorEdges_.clear();
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
g2o::ToggelableGGDKernel*
MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::newPriorToggelableGgdKernel() {
  return new g2o::ToggelableGGDKernel(ggdBound_, ggdPower_, ggdLnc_,
                                      ggdTailPenaltyStd_, &ggdActiveAlwaysFalse_);
}

template <typename VertexType, typename EdgeType, typename RobotId, typename MessageType>
void MultibotSlamSystem<VertexType, EdgeType, RobotId, MessageType>::ignoreUnknownEventType() {
  if (verbose_) {
    std::cerr << "[MultibotSlamSystem] Ignoring unknown event type\n";
  }
}

}  // namespace g2o
