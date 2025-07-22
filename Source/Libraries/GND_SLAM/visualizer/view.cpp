// view.cpp
#include "view.h"
//#include "epoxy/gl.h"
#include <GL/gl.h>

namespace g2o {
namespace tutorial {
namespace viz {

View::View(const Eigen::Vector3f& color) : color_(color), platformSize_(1.5), marginalsInitialized_(false),
                                            pose_(Eigen::Vector3d()){}

void View::renderRobotPose() const {
    //const Eigen::Vector3d& pose = path_.back();
    double x = pose_[0], y = pose_[1], theta = pose_[2];
    const double size = platformSize_;

    glPushMatrix();
    glTranslated(x, y, 0);
    glRotated(theta * 180.0 / M_PI, 0, 0, 1);
    glColor3f(color_[0], color_[1], color_[2]);

    glBegin(GL_TRIANGLES);
    glVertex2d(size, 0.0);
    glVertex2d(-size / 2, size / 2);
    glVertex2d(-size / 2, -size / 2);
    glEnd();

    glColor3f(0.0f, 0.0f, 0.0f);    // Set color to black
    glBegin(GL_LINES);
    glVertex2d(- size/4, 0.0);
    glVertex2d(size/4, 0.0);
    glVertex2d(0.0, size/4);
    glVertex2d(0.0, - size/4);
    glEnd();


    if(marginalsInitialized_){
        glColor3f(color_[0], color_[1], color_[2]);
        glBegin(GL_LINE_LOOP);

        for (int angle = 0; angle <= 360; angle += 10) {
            double theta = angle * M_PI / 180.0;
            Eigen::Vector2d unitCircle(std::cos(theta), std::sin(theta));

            // Apply sqrt covariance and scale
            Eigen::Vector2d pt = 2 * sqrtPoseMarginal2d_ * unitCircle;

            glVertex2d(pt[0], pt[1]);
        }

        glEnd();
    }

    glPopMatrix();
}

void View::renderRobotPath() const {
    glColor3f(color_[0], color_[1], color_[2]);
    glBegin(GL_LINE_STRIP);
    for (const auto& pose : path_) {
        glVertex2d(pose[0], pose[1]);
    }
    glEnd();
    std::cout<<"rndrobpose";
}

void View::renderScene() const {}
void View::setView(const std::string& filename){}

void View::updateRobotPath(const std::vector<Eigen::Vector3d>& path) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    path_ = path;
    pose_ = path_.back();
}

void View::appendToRobotPath(const Eigen::Vector3d& pose) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    path_.emplace_back(pose);
    pose_ = pose;
}

void View::updateRobotPose(const Eigen::Vector3d& pose) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    pose_ = pose;
}

void View::updateRobotMarginals(const Eigen::Matrix2d& marginals) {
    std::lock_guard<std::mutex> lock(dataMutex_);
    poseMarginal2d_ = marginals;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(marginals);
    if (eig.eigenvalues().minCoeff() < 0) {
        return;  // Invalid covariance, skip drawing
    }

    // Compute square root of covariance
    sqrtPoseMarginal2d_ = eig.operatorSqrt();
}

}
}
}
