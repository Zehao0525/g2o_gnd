#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

// Time-stamped pose (robot–robot sync): planar position in world frame.
struct PoseStampEntry {
  double time;
  std::string sourceId;
  int observationId;
  std::string subjectId;

  bool hasPose = false;

  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();

  PoseStampEntry(double t, std::string sid, int oid, std::string suid)
      : time(t),
        sourceId(std::move(sid)),
        observationId(oid),
        subjectId(std::move(suid)),
        hasPose(false) {}
};

struct LMPoseEntry {
  int landmarkId = -1;
  std::string observerId;
  bool hasPose = false;
  Eigen::Vector2d position = Eigen::Vector2d::Zero();
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();

  LMPoseEntry(int lmId, std::string oid)
      : landmarkId(lmId), observerId(std::move(oid)), hasPose(false) {}
};

struct UTSIAMessage {
  std::string sourceId;
  bool loaded;
  bool lm_query;
  std::vector<PoseStampEntry> poseEntries;
  std::vector<LMPoseEntry> lmEntries;

  UTSIAMessage()
      : sourceId(""),
        loaded(false),
        lm_query(true),
        poseEntries(),
        lmEntries() {}

  UTSIAMessage(std::string sender,
               bool loaded_,
               bool lm_query_in = true,
               std::vector<PoseStampEntry> pe = {},
               std::vector<LMPoseEntry> lm = {})
      : sourceId(std::move(sender)),
        loaded(loaded_),
        lm_query(lm_query_in),
        poseEntries(std::move(pe)),
        lmEntries(std::move(lm)) {}
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
