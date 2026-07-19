#pragma once

#include <string>

#include "events.h"

namespace g2o {
namespace tutorial {

enum class DataEventType {
  Initialization = 0,
  Odometry = 1,
  Observation = 2,
  LandmarkObservation = 3,
};

struct G2O_TUTORIAL_SLAM2D_API DataEventBase : public Event {
  explicit DataEventBase(double t) : Event(t) {}
  EventType type() const final { return EventType::Other; }
  virtual DataEventType dataEventType() const = 0;
  /// Same timestamp: Initialization < Odometry < Observation < LandmarkObservation.
  int sortPriority() const override {
    return static_cast<int>(dataEventType());
  }
};

struct G2O_TUTORIAL_SLAM2D_API DataInitEvent : public DataEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Isometry3 value;
  bool posFixed;
  Eigen::Matrix<double, 6, 6> information;
  DataInitEvent(const double eventTime, const bool fixed, const Isometry3& pos,
                const Eigen::Matrix<double, 6, 6>& info)
      : DataEventBase(eventTime), value(pos), posFixed(fixed), information(info) {}
  DataEventType dataEventType() const override {
    return DataEventType::Initialization;
  }
};

struct G2O_TUTORIAL_SLAM2D_API DataOdomEvent : public DataEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  /// Body-frame linear velocities in translation(): x = v_fwd, y = 0, z = v_z.
  Isometry3 value;
  /// Yaw rate about +Z (rad/s), stored explicitly.
  double omegaZ{0.0};
  Eigen::Matrix<double, 6, 6> information;
  DataOdomEvent(const double eventTime, const Isometry3& velBody,
                double omegaZ_rad_per_s,
                const Eigen::Matrix<double, 6, 6>& info)
      : DataEventBase(eventTime),
        value(velBody),
        omegaZ(omegaZ_rad_per_s),
        information(info) {}
  DataEventType dataEventType() const override { return DataEventType::Odometry; }
};

struct G2O_TUTORIAL_SLAM2D_API DataObsEvent : public DataEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::string robotIdTo;
  Isometry3 value;
  Eigen::Matrix<double, 6, 6> information;
  DataObsEvent(const double eventTime, const std::string robotId,
               const Isometry3& pos, const Eigen::Matrix<double, 6, 6>& info)
      : DataEventBase(eventTime),
        robotIdTo(robotId),
        value(pos),
        information(info) {}
  DataEventType dataEventType() const override {
    return DataEventType::Observation;
  }
};

/// Landmark relative-position observation from python `lmobs_rp`.
struct G2O_TUTORIAL_SLAM2D_API DataLmObsEvent : public DataEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int landmarkId = -1;
  Eigen::Vector3d relPos = Eigen::Vector3d::Zero();
  Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
  DataLmObsEvent(const double eventTime, int lmId,
                 const Eigen::Vector3d& relPosBody,
                 const Eigen::Matrix3d& info)
      : DataEventBase(eventTime),
        landmarkId(lmId),
        relPos(relPosBody),
        information(info) {}
  DataEventType dataEventType() const override {
    return DataEventType::LandmarkObservation;
  }
};

}  // namespace tutorial
}  // namespace g2o

