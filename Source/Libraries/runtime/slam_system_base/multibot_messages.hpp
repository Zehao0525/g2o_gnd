#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"

/**
To be honest, it's probbaly best if we have a message per edge. But oh well whatever.
*/

namespace g2o {
namespace tutorial {
namespace multibotsim {

/**
 * Polymorphic root for inter-robot sync payloads.
 * Concrete messages (DS / UTISA stamp-based, Glenn vertex-id based) derive from this.
 */
struct SyncMessageBase {
  virtual ~SyncMessageBase() = default;
};

/**
 * Time-stamped pose query / answer used by Multidrone and UTISA.
 * PoseT / InfoT differ: SE3+6x6 vs Vector2d+2x2.
 */
template <typename PoseT, typename InfoT>
struct PoseStampEntryT {
  double time = 0.0;
  std::string sourceId;
  int observationId = 0;
  std::string subjectId;

  bool hasPose = false;
  PoseT pose = PoseT();
  InfoT information = InfoT::Identity();

  PoseStampEntryT() = default;

  PoseStampEntryT(double t, std::string sid, int oid, std::string suid)
      : time(t),
        sourceId(std::move(sid)),
        observationId(oid),
        subjectId(std::move(suid)),
        hasPose(false) {}
};

/**
 * Landmark estimate query / answer. observerId is the marginalizing robot.
 */
template <typename PoseT, typename InfoT>
struct LMPoseEntryT {
  int landmarkId = -1;
  std::string observerId;
  bool hasPose = false;
  PoseT pose = PoseT();
  InfoT information = InfoT::Identity();

  LMPoseEntryT() = default;

  LMPoseEntryT(int lmId, std::string oid)
      : landmarkId(lmId), observerId(std::move(oid)), hasPose(false) {}
};

/**
 * Stamp-based sync envelope shared by Multidrone (`DSMessage`) and UTISA (`UTSIAMessage`).
 */
template <typename PoseT, typename InfoT>
struct SyncMessageT : SyncMessageBase {
  using PoseEntry = PoseStampEntryT<PoseT, InfoT>;
  using LMEntry = LMPoseEntryT<PoseT, InfoT>;

  std::string sourceId;
  bool loaded = false;
  bool lm_query = true;
  std::vector<PoseEntry> poseEntries;
  std::vector<LMEntry> lmEntries;

  SyncMessageT() = default;

  SyncMessageT(std::string sender,
               bool loaded_,
               bool lm_query_in = true,
               std::vector<PoseEntry> pe = {},
               std::vector<LMEntry> lm = {})
      : sourceId(std::move(sender)),
        loaded(loaded_),
        lm_query(lm_query_in),
        poseEntries(std::move(pe)),
        lmEntries(std::move(lm)) {}
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
