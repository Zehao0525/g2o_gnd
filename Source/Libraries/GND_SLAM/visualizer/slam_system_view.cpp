// SLAMSystemView.cpp
#include "slam_system_view.h"
#include <GL/gl.h>

namespace g2o {
namespace tutorial {
namespace viz {


SLAMSystemView::SLAMSystemView(SlamSystem* system, const Eigen::Vector3f& color)
    : View(color), slamSystem_(system) {}

void SLAMSystemView::update() {
    //std::lock_guard<std::mutex> lock(slamSystem_->mutex());
    Eigen::Vector3d x;
    Eigen::Matrix2d P;
    slamSystem_->platformEstimate(x,P);
    updateRobotPose(x);
    //updateRobotMarginals(5*Eigen::Matrix2d::Identity());
}

}}}
