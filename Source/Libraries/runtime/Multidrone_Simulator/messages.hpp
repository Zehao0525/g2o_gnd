#pragma once

#include <string>
#include <vector>

#include "g2o_tutorial_slam2d_api.h"
#include "multibot_messages.hpp"
#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

using PoseStampEntry =
    PoseStampEntryT<Isometry3, Eigen::Matrix<double, 6, 6>>;
using LMPoseEntry = LMPoseEntryT<Isometry3, Eigen::Matrix<double, 6, 6>>;
using DSMessage = SyncMessageT<Isometry3, Eigen::Matrix<double, 6, 6>>;

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
