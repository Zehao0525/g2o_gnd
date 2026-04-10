#pragma once

#include <string>
#include <vector>

#include "g2o_tutorial_slam2d_api.h"
#include "sensor_data.h"

#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {
namespace multibotsim{

// Time-stamped pose (robot–robot sync)
struct PoseStampEntry {
    double time;                 // timestamp we care about
    std::string sourceId;        // who originated this query
    int observationId;           // local obs id (for matching on return)
    std::string subjectId;       // which robot / subject this pose refers to

    bool hasPose = false;        // false = request only; true = answer filled

    Isometry3 pose;              // full SE(3) pose
    Eigen::Matrix<double,6,6> information;  // info matrix of that pose

    PoseStampEntry(double t, std::string sid, int oid, std::string suid) :
        time(t), sourceId(sid), observationId(oid), subjectId(suid), hasPose(false){}
};

/// Landmark pose entry (e.g. for sync messages involving landmark IDs).
struct LMPoseEntry {
    int landmarkId = -1;
    std::string observerId;
    bool hasPose = false;
    Isometry3 pose = Isometry3::Identity();
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();

    LMPoseEntry(int lmId, std::string oid)
        : landmarkId(lmId),
          observerId(std::move(oid)),
          hasPose(false) {}
};

struct DSMessage {
    std::string sourceId;
    bool loaded;
    bool lm_query;
    std::vector<PoseStampEntry> poseEntries;
    std::vector<LMPoseEntry> lmEntries;

    DSMessage()
        : sourceId(""),
          loaded(false),
          lm_query(true),
          poseEntries(),
          lmEntries() {}

    /// Pass only pose entries, only landmark entries, or both (omit unused side with `{}`).
    DSMessage(std::string sender,
              bool loaded_,
              bool lm_query = true,
              std::vector<PoseStampEntry> pe = {},
              std::vector<LMPoseEntry> lm = {})
        : sourceId(std::move(sender)),
          loaded(loaded_),
          poseEntries(std::move(pe)),
          lmEntries(std::move(lm)) {}
};

};
}
}
