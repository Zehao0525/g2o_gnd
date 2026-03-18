#include "data_based_simulation_view.h"
#include "data_based_simulation.h"
#include <GL/gl.h>
#include <fstream>
#include <cmath>

namespace g2o {
namespace tutorial {
namespace viz {

DataBasedSimulationView::DataBasedSimulationView(multibotsim::DataBasedSimulation* sim, const Eigen::Vector3f& color, bool visualise_path)
    : View(color), simulation_(sim), currentPose3d_(Eigen::Isometry3d::Identity()), visualise_path_(visualise_path) {}

DataBasedSimulationView::DataBasedSimulationView(multibotsim::DataBasedSimulation* sim, const std::string& filename, bool visualise_path)
    : View(Eigen::Vector3f(0.5f, 0.5f, 0.5f)), simulation_(sim), currentPose3d_(Eigen::Isometry3d::Identity()), visualise_path_(visualise_path) {
    setView(filename);
}

void DataBasedSimulationView::setView(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open view config file: " + filename);
    }
    nlohmann::json j;
    f >> j;

    auto color = j["slam_system"]["platform"].value("color", std::vector<float>{0.5f, 0.5f, 0.5f});
    color_ = Eigen::Vector3f(color[0], color[1], color[2]);
    platformSize_ = j["slam_system"]["platform"].value("size", 1.5);
}

void DataBasedSimulationView::update() {
    Eigen::Isometry3d pose = simulation_->xTrue();

    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        currentPose3d_ = pose;
        path3d_.push_back(pose);
    }

    Eigen::Vector3d translation = pose.translation();
    Eigen::Matrix3d rotation = pose.rotation();
    double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    Eigen::Vector3d pose2d(translation.x(), translation.y(), yaw);

    appendToRobotPath(pose2d);
    updateRobotPose(pose2d);
}

void DataBasedSimulationView::pause() {
    View::pause();
}

void DataBasedSimulationView::renderScene() const {
    std::lock_guard<std::mutex> lock(const_cast<DataBasedSimulationView*>(this)->dataMutex_);

    // [keep] World-frame ground-truth path (semi-transparent solid line)
    if (!path3d_.empty() && visualise_path_) {
        glLineWidth(2.0f);
        glColor4f(color_[0], color_[1], color_[2], 0.4f);
        glBegin(GL_LINE_STRIP);
        for (const auto& T : path3d_) {
            const auto& p = T.translation();
            glVertex3f(static_cast<float>(p.x()),
                       static_cast<float>(p.y()),
                       static_cast<float>(p.z()));
        }
        glEnd();
    }
    // [undo] If you don’t want the GT path, delete the entire if(!path3d_.empty()) block above.

    // [keep] Current pose frame + cube (drawn in robot frame)
    glPushMatrix();
    Eigen::Matrix4d mat = currentPose3d_.matrix();
    glMultMatrixd(mat.data());

    // Local XYZ axes at current pose
    glLineWidth(1.5f);
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(platformSize_, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, platformSize_, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, platformSize_);

    glEnd();

    // Small cube body at current pose
    glColor3fv(color_.data());
    glPushMatrix();
    glScalef(platformSize_ * 0.25f, platformSize_ * 0.25f, platformSize_ * 0.25f);

    glBegin(GL_LINES);
    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(0.5f, -0.5f, -0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);  glVertex3f(0.5f, 0.5f, -0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);   glVertex3f(-0.5f, 0.5f, -0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);  glVertex3f(-0.5f, -0.5f, -0.5f);

    glVertex3f(-0.5f, -0.5f, 0.5f);  glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, 0.5f);   glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, 0.5f);    glVertex3f(-0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, 0.5f);   glVertex3f(-0.5f, -0.5f, 0.5f);

    glVertex3f(-0.5f, -0.5f, -0.5f); glVertex3f(-0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, -0.5f, -0.5f);  glVertex3f(0.5f, -0.5f, 0.5f);
    glVertex3f(0.5f, 0.5f, -0.5f);   glVertex3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-0.5f, 0.5f, -0.5f);  glVertex3f(-0.5f, 0.5f, 0.5f);
    glEnd();

    glPopMatrix();  // cube
    glPopMatrix();  // pose
}

}}}
