#ifndef G2O_INCSIN2D_PLATFORM_CONTROLLER_H
#define G2O_INCSIN2D_PLATFORM_CONTROLLER_H

#include <vector>
#include <Eigen/Core>
#include "se2.h"

namespace g2o {
namespace tutorial {

class PlatformController {
public:
  PlatformController();

  void setWaypoints(const std::vector<Eigen::Vector2d>& waypoints);
  void setControllerParams(double minSpeed, double maxSpeed, double maxAccel,
                           double maxDelta, double maxDeltaRate, double odomUpdatePeriod,
                            double B, bool repeat);
  void start();

  bool off() const{return off_;}

  SE2 computeControlInputs(const SE2& x);

private:
  std::vector<Eigen::Vector2d> waypoints_;
  size_t waypointIndex_ = 0;
  size_t numWaypoints_ = 0;
  bool repeatVisitingWaypoints_ = true;

  double maxAcceptableDistanceFromWaypoint_ = 1.0;
  Eigen::Vector2d u_ = Eigen::Vector2d::Zero(); // [speed, steering angle]

  bool off_;

  // Config
  double B_ = 1.0;
  double minSpeed_ = 0.0, maxSpeed_ = 1.0;
  double maxAccel_ = 1.0;
  double maxDelta_ = M_PI / 4;
  double maxDeltaRate_ = M_PI / 10;
  double odomUpdatePeriod_ = 0.2;
};

}  // namespace tutorial
}  // namespace g2o

#endif
