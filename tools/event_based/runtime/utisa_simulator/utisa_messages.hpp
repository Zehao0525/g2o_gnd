#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"
#include "multibot_messages.hpp"

namespace g2o {
namespace tutorial {
namespace multibotsim {

using PoseStampEntry = PoseStampEntryT<Eigen::Vector2d, Eigen::Matrix2d>;
using LMPoseEntry = LMPoseEntryT<Eigen::Vector2d, Eigen::Matrix2d>;
using UTSIAMessage = SyncMessageT<Eigen::Vector2d, Eigen::Matrix2d>;

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
