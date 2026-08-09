#pragma once

#include <Eigen/Core>

#include "events_base.h"
#include "g2o/types/slam3d/se3quat.h"

namespace g2o {
namespace tutorial {

enum class GlennEventType {
  FileInitialization = 0,
  FileOdometry = 1,
  FileObservation = 2,
  FileIntraObservation = 3,
};

struct G2O_TUTORIAL_SLAM2D_API GlennEventBase : public Event {
  explicit GlennEventBase(double t) : Event(t) {}
  virtual GlennEventType glennEventType() const = 0;
  int sortPriority() const override {
    return static_cast<int>(glennEventType());
  }
};

struct G2O_TUTORIAL_SLAM2D_API FileInitEvent : public GlennEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int vtxId;
  Isometry3 value;
  Eigen::Matrix<double, 6, 6> information;
  FileInitEvent(double eventTime, int vtxIdIn, const Isometry3& pos,
                const Eigen::Matrix<double, 6, 6>& info)
      : GlennEventBase(eventTime), vtxId(vtxIdIn), value(pos), information(info) {}
  GlennEventType glennEventType() const override {
    return GlennEventType::FileInitialization;
  }
};

struct G2O_TUTORIAL_SLAM2D_API FileOdomEvent : public GlennEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int vtxId;
  Isometry3 value;
  Eigen::Matrix<double, 6, 6> information;
  FileOdomEvent(double eventTime, int vtxIdIn, const Isometry3& pos,
                const Eigen::Matrix<double, 6, 6>& info)
      : GlennEventBase(eventTime), vtxId(vtxIdIn), value(pos), information(info) {}
  GlennEventType glennEventType() const override {
    return GlennEventType::FileOdometry;
  }
};

struct G2O_TUTORIAL_SLAM2D_API FileObsEvent : public GlennEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int robotIdFrom;
  int robotIdTo;
  int vtxIdFrom;
  int vtxIdTo;
  Isometry3 value;
  Eigen::Matrix<double, 6, 6> information;
  FileObsEvent(double eventTime, int robotId, int targetRobotId, int vtxId0,
               int vtxId1, const Isometry3& pos,
               const Eigen::Matrix<double, 6, 6>& info)
      : GlennEventBase(eventTime),
        robotIdFrom(robotId),
        robotIdTo(targetRobotId),
        vtxIdFrom(vtxId0),
        vtxIdTo(vtxId1),
        value(pos),
        information(info) {}
  GlennEventType glennEventType() const override {
    return GlennEventType::FileObservation;
  }
};

struct G2O_TUTORIAL_SLAM2D_API FileIntraObsEvent : public GlennEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int vtxIdFrom;
  int vtxIdTo;
  Isometry3 value;
  Eigen::Matrix<double, 6, 6> information;
  FileIntraObsEvent(double eventTime, int vtxId0, int vtxId1, const Isometry3& pos,
                    const Eigen::Matrix<double, 6, 6>& info)
      : GlennEventBase(eventTime),
        vtxIdFrom(vtxId0),
        vtxIdTo(vtxId1),
        value(pos),
        information(info) {}
  GlennEventType glennEventType() const override {
    return GlennEventType::FileIntraObservation;
  }
};

}  // namespace tutorial
}  // namespace g2o
