#include "multi_drone_slam_system_view.h"
#include "drone_slam_system.h"
#include <GL/gl.h>
#include <fstream>
#include <cmath>

namespace g2o {
namespace tutorial {
namespace viz {

MultiDroneSLAMSystemView::MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const Eigen::Vector3f& color, bool visualise_path)
    : View(color), slamSystem_(system), currentPose3d_(Eigen::Isometry3d::Identity()), visualise_path_(visualise_path) {}

MultiDroneSLAMSystemView::MultiDroneSLAMSystemView(multibotsim::MultiDroneSLAMSystem* system, const std::string& filename, bool visualise_path)
    : View(Eigen::Vector3f(0.0f, 1.0f, 0.0f)), slamSystem_(system), currentPose3d_(Eigen::Isometry3d::Identity()), visualise_path_(visualise_path) {
    setView(filename);
}

void MultiDroneSLAMSystemView::setView(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open view config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    auto color = j["slam_system"]["platform"].value("color", std::vector<float>{0.0f, 1.0f, 0.0f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);
    platformSize_ = j["slam_system"]["platform"].value("size", 1.5);
}

void MultiDroneSLAMSystemView::update() {
    Eigen::Isometry3d pose;
    slamSystem_->platformEstimate(pose);

    // Update 3D state under lock only; release before calling base View methods
    // so we never hold dataMutex_ while calling appendToRobotPath/updateRobotPose
    // (they lock the same mutex → same-thread deadlock otherwise).
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        currentPose3d_ = pose;
        path3d_.push_back(pose);
    }

    // Convert 3D pose to 2D for path storage (x, y, yaw)
    Eigen::Vector3d translation = pose.translation();
    Eigen::Matrix3d rotation = pose.rotation();
    double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    Eigen::Vector3d pose2d(translation.x(), translation.y(), yaw);

    appendToRobotPath(pose2d);
    updateRobotPose(pose2d);
}

void MultiDroneSLAMSystemView::pause() {
    View::pause();
}

void MultiDroneSLAMSystemView::renderScene() const {
    // Use const_cast to lock mutex in const method
    std::lock_guard<std::mutex> lock(const_cast<MultiDroneSLAMSystemView*>(this)->dataMutex_);

    // [keep] World-frame SLAM path (semi-transparent dotted line)
    if (!path3d_.empty() && visualise_path_) {
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);  // dotted
        glLineWidth(2.0f);
        glColor4f(color_[0], color_[1], color_[2], 0.4f);
        glBegin(GL_LINE_STRIP);
        // Draw all historical points, but force the final point to the current pose
        // so the path endpoint is always attached to the current SLAM indicator.
        for (size_t i = 0; i < path3d_.size(); ++i) {
            const auto& T = (i + 1 == path3d_.size()) ? currentPose3d_ : path3d_[i];
            const auto& p = T.translation();
            glVertex3f(static_cast<float>(p.x()),
                       static_cast<float>(p.y()),
                       static_cast<float>(p.z()));
        }
        glEnd();
        glDisable(GL_LINE_STIPPLE);
    }
    // [undo] If you don’t want the SLAM path, delete the if(!path3d_.empty()) block above.

    // [keep] Current SLAM pose frame + cube
    glPushMatrix();

    Eigen::Matrix4d mat = currentPose3d_.matrix();
    glMultMatrixd(mat.data());

    // Coordinate axes at current pose
    glLineWidth(2.0f);
    glBegin(GL_LINES);

    // X axis - red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(platformSize_, 0.0f, 0.0f);

    // Y axis - green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, platformSize_, 0.0f);

    // Z axis - blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, platformSize_);

    glEnd();

    // Small cube body at current pose
    glColor3fv(color_.data());
    glPushMatrix();
    glScalef(platformSize_ * 0.3f, platformSize_ * 0.3f, platformSize_ * 0.3f);

    glBegin(GL_LINES);
    // Bottom face
    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);  glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);   glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);  glVertex3f(-0.5f, -0.5f, -0.5f);
    // Top face
    glVertex3f(-0.5f, -0.5f, 0.5f);  glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);   glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);   glVertex3f(-0.5f, -0.5f, 0.5f);
    // Vertical edges
    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);  glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);   glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);  glVertex3f(-0.5f, 0.5f, 0.5f);
    glEnd();

    glPopMatrix();  // cube
    glPopMatrix();  // pose
}

}}}
