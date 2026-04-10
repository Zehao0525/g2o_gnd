#pragma once

#include <string>

#include <Eigen/Core>

#include "events.h"
#include "se2.h"

namespace g2o {
namespace tutorial {

enum class UTISAEventType {
  Initialization = 0,
  Odometry = 1,
  Observation = 2,
  LandmarkObservation = 3,
};

struct G2O_TUTORIAL_SLAM2D_API UTISAEventBase : public Event {
  explicit UTISAEventBase(double t) : Event(t) {}
  EventType type() const final { return EventType::Other; }
  virtual UTISAEventType utisaEventType() const = 0;
  /// Same timestamp: Initialization < Odometry < Observation < LandmarkObservation.
  int sortPriority() const override {
    return static_cast<int>(utisaEventType());
  }
};

struct G2O_TUTORIAL_SLAM2D_API UTISAInitEvent : public UTISAEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  SE2 value;
  bool posFixed;
  Eigen::Matrix3d information;
  UTISAInitEvent(const double eventTime, const bool fixed, const SE2& pose,
                 const Eigen::Matrix3d& info)
      : UTISAEventBase(eventTime), value(pose), posFixed(fixed), information(info) {}
  UTISAEventType utisaEventType() const override {
    return UTISAEventType::Initialization;
  }
};

struct G2O_TUTORIAL_SLAM2D_API UTISAOdomEvent : public UTISAEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Eigen::Vector3d velBody;
  Eigen::Matrix3d information;
  UTISAOdomEvent(const double eventTime, const Eigen::Vector3d& vel,
                 const Eigen::Matrix3d& info)
      : UTISAEventBase(eventTime), velBody(vel), information(info) {}
  UTISAEventType utisaEventType() const override { return UTISAEventType::Odometry; }
};

struct G2O_TUTORIAL_SLAM2D_API UTISAObsEvent : public UTISAEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::string robotIdTo;
  /// MR.CLAM: range [m], bearing [rad] in the sensor/body frame (same as log file).
  double range = 0.0;
  double bearing = 0.0;
  /// Information in (range, bearing) space; typically diagonal 1/σ_r², 1/σ_b².
  Eigen::Matrix2d information;
  UTISAObsEvent(const double eventTime, const std::string robotId, double range_m, double bearing_rad,
                const Eigen::Matrix2d& info)
      : UTISAEventBase(eventTime),
        robotIdTo(robotId),
        range(range_m),
        bearing(bearing_rad),
        information(info) {}
  UTISAEventType utisaEventType() const override {
    return UTISAEventType::Observation;
  }
};

struct G2O_TUTORIAL_SLAM2D_API UTISALmObsEvent : public UTISAEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int landmarkId = -1;
  double range = 0.0;
  double bearing = 0.0;
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  UTISALmObsEvent(const double eventTime, int lmId, double range_m, double bearing_rad,
                  const Eigen::Matrix2d& info)
      : UTISAEventBase(eventTime),
        landmarkId(lmId),
        range(range_m),
        bearing(bearing_rad),
        information(info) {}
  UTISAEventType utisaEventType() const override {
    return UTISAEventType::LandmarkObservation;
  }
};

}  // namespace tutorial
}  // namespace g2o
