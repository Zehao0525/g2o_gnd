#include <iostream>

#include "platform_controller.h"
#include "g2o/stuff/misc.h"  // for normalize_theta

namespace g2o {
namespace tutorial {

PlatformController::PlatformController() {
  off_ = false;
}

void PlatformController::setWaypoints(const std::vector<Eigen::Vector2d>& waypoints) {
  waypoints_ = waypoints;
  numWaypoints_ = static_cast<int>(waypoints_.size());
}

void PlatformController::setControllerParams(double minSpeed, double maxSpeed, double maxAccel,
                                             double maxDelta, double maxDeltaRate, double odomUpdatePeriod,
                                            double B, bool repeat) {
  minSpeed_ = minSpeed;
  maxSpeed_ = maxSpeed;
  maxAccel_ = maxAccel;
  maxDelta_ = maxDelta;
  maxDeltaRate_ = maxDeltaRate;
  odomUpdatePeriod_ = odomUpdatePeriod;
  B_ = B;
  repeatVisitingWaypoints_ = repeat;
}

void PlatformController::start() {
  waypointIndex_ = (numWaypoints_ > 0) ? 0 : -1;
  u_.setZero();
}

SE2 PlatformController::computeControlInputs(const SE2& x) {
  if (waypointIndex_ >= numWaypoints_){
    off_ = true;
    return SE2();
  }

  Eigen::Vector2d dx = waypoints_[waypointIndex_] - x.translation();
  double d = dx.norm();

  if (d < maxAcceptableDistanceFromWaypoint_) {
    if (waypointIndex_ + 1 >= numWaypoints_) {
      if (repeatVisitingWaypoints_) {
        waypointIndex_ = 0;
      } else {
        off_ = true;
        return SE2();  // zero-size vector indicates end
      }
    } else {
      waypointIndex_++;
    }
    dx = waypoints_[waypointIndex_] - x.translation();
    d = dx.norm();
  }

  double diffSpeed = 0.1 * d - u_[0];
  double maxDiffSpeed = maxAccel_ * odomUpdatePeriod_;
  diffSpeed = std::clamp(diffSpeed, -maxDiffSpeed, maxDiffSpeed);
  u_[0] = std::clamp(u_[0] + diffSpeed, minSpeed_, maxSpeed_);

  double headingError = std::atan2(dx.y(), dx.x()) - x.rotation().angle() - u_[1];
  double diffDelta = g2o::normalize_theta(headingError);
  double maxDiffDelta = maxDeltaRate_ * odomUpdatePeriod_;
  diffDelta = std::clamp(diffDelta, -maxDiffDelta, maxDiffDelta);
  u_[1] = std::clamp(u_[1] + diffDelta, -maxDelta_, maxDelta_);

  double psiDot = u_[0] * std::sin(u_[1]) / B_;
  if(verbose_){std::cout << " = Controller output: "<< u_[0] << "," << psiDot << std::endl;}
  return SE2(u_[0], 0.0, psiDot);
}

}  // namespace tutorial
}  // namespace g2o
