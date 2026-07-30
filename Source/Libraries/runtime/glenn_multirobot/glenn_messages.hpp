#pragma once

#include <vector>

#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"
#include "multibot_messages.hpp"
#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {

/** Glenn file-based sync: keyed by discrete vertex IDs rather than timestamps. */
struct FileObsSyncRequest {
  int observerRobotId = 0;
  int observerVertexId = 0;
  int observedRobotId = 0;
  int observedVertexId = 0;
  Isometry3 observedVtxLocation = Isometry3::Identity();
  Eigen::Matrix<double, 6, 6> observedVtxInformation =
      Eigen::Matrix<double, 6, 6>::Identity();

  FileObsSyncRequest() = default;

  FileObsSyncRequest(int selfId, int robotId, int selfVtxId, int vertexId)
      : observerRobotId(selfId),
        observerVertexId(selfVtxId),
        observedRobotId(robotId),
        observedVertexId(vertexId) {}
};

struct FileObsSyncMessage : multibotsim::SyncMessageBase {
  int sourceId = 0;
  /** True while this robot is advertising queries; false on answered payloads. */
  bool outGoing = false;
  std::vector<FileObsSyncRequest> syncRequests;

  FileObsSyncMessage() = default;

  FileObsSyncMessage(int sender, bool og, std::vector<FileObsSyncRequest> syncReqs)
      : sourceId(sender), outGoing(og), syncRequests(std::move(syncReqs)) {}
};

}  // namespace tutorial
}  // namespace g2o
