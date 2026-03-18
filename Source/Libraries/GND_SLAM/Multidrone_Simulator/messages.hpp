#pragma once

#include <map>
#include <vector>
#include <memory>
#include <set>
#include <string>
#include <cstdint>

#include "g2o_tutorial_slam2d_api.h"
#include "sensor_data.h"


#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {
namespace multibotsim{

// Time-stamped pose
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

enum class MessageKind : uint8_t {
    PoseQuery,
    PoseAnswer,
    // ...
};



struct DSMessage {
    std::string sourceId;
    bool loaded;
    std::vector<PoseStampEntry> poseEntries;
    DSMessage(): sourceId(""), loaded(false), poseEntries({}){}
    DSMessage(std::string sender, bool loaded, std::vector<PoseStampEntry> pe): sourceId(sender), loaded(loaded), poseEntries(pe){}
  };


};
}
}