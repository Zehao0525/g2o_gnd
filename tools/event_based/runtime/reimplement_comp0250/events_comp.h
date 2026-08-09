#pragma once

#include <utility>

#include <Eigen/Core>

#include "events_base.h"
#include "se2.h"
#include "sensor_data.h"

namespace g2o {
namespace tutorial {

enum class CompEventType {
  HeartBeat = 0,
  LandmarkObservations = 1,
  LMRangeBearingObservations = 2,
  GPSObservation = 3,
  Odometry = 4,
  Initialization = 5,
};

struct G2O_TUTORIAL_SLAM2D_API CompEventBase : public Event {
  explicit CompEventBase(double t) : Event(t) {}
  virtual CompEventType compEventType() const = 0;
  int sortPriority() const override {
    return static_cast<int>(compEventType());
  }
};

struct G2O_TUTORIAL_SLAM2D_API HeartBeat : public CompEventBase {
  explicit HeartBeat(double timestamp) : CompEventBase(timestamp) {}
  CompEventType compEventType() const override { return CompEventType::HeartBeat; }
};

struct G2O_TUTORIAL_SLAM2D_API LandmarkObservationsEvent : public CompEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  LandmarkObservationVector landmarkObservations;

  LandmarkObservationsEvent(double timestamp, LandmarkObservationVector observations)
      : CompEventBase(timestamp), landmarkObservations(std::move(observations)) {}
  CompEventType compEventType() const override {
    return CompEventType::LandmarkObservations;
  }
};

struct G2O_TUTORIAL_SLAM2D_API LMRangeBearingObservationsEvent : public CompEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  LMRangeBearingObservationVector landmarkObservations;

  LMRangeBearingObservationsEvent(double timestamp,
                                  LMRangeBearingObservationVector observations)
      : CompEventBase(timestamp), landmarkObservations(std::move(observations)) {}
  CompEventType compEventType() const override {
    return CompEventType::LMRangeBearingObservations;
  }
};

struct G2O_TUTORIAL_SLAM2D_API OdometryEvent : public CompEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  SE2 value;
  Eigen::Matrix3d covariance;
  OdometryEvent(double timestamp, const SE2& velocity, const Eigen::Matrix3d& cov)
      : CompEventBase(timestamp), value(velocity), covariance(cov) {}
  CompEventType compEventType() const override { return CompEventType::Odometry; }
};

struct G2O_TUTORIAL_SLAM2D_API InitializationEvent : public CompEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  SE2 pose;
  SE2 velocity;
  Eigen::Matrix3d covariance;
  Eigen::Matrix3d sigmaU;
  InitializationEvent(double timestamp, const SE2& pos, const SE2& vel,
                      const Eigen::Matrix3d& posCov, const Eigen::Matrix3d& sigmau)
      : CompEventBase(timestamp),
        pose(pos),
        velocity(vel),
        covariance(posCov),
        sigmaU(sigmau) {}
  CompEventType compEventType() const override {
    return CompEventType::Initialization;
  }
};

struct G2O_TUTORIAL_SLAM2D_API GPSObservationEvent : public CompEventBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  Eigen::Vector2d value;
  Eigen::Matrix2d covariance;
  GPSObservationEvent(double timestamp, const Eigen::Vector2d& pos,
                      const Eigen::Matrix2d& cov)
      : CompEventBase(timestamp), value(pos), covariance(cov) {}
  CompEventType compEventType() const override {
    return CompEventType::GPSObservation;
  }
};

}  // namespace tutorial
}  // namespace g2o
