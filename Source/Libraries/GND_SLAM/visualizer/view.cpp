// view.cpp
#include "view.h"
//#include "epoxy/gl.h"
#include <GL/gl.h>

namespace g2o {
namespace tutorial {
namespace viz {

View::View(const Eigen::Vector3f& color) : color_(color), platformSize_(1.5), running_(true),
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
    glVertex2d(0.0, 0.0);
    glVertex2d(-size *3/ 2, size / 2);
    glVertex2d(-size *3/ 2, -size / 2);
    glEnd();

    // glColor3f(0.0f, 0.0f, 0.0f);    // Set color to black
    // glBegin(GL_LINES);
    // glVertex2d(- size/4, 0.0);
    // glVertex2d(size/4, 0.0);
    // glVertex2d(0.0, size/4);
    // glVertex2d(0.0, - size/4);
    // glEnd();

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

void View::renderMeasurmentViz() {
    std::vector<std::shared_ptr<MeasurmentViz>> remaining;

    for (const auto& viz :  measVizVertex_) {
        if (!viz) continue;

        if(running_) viz->lifetimeRemaining_--;

        if (viz->lifetimeRemaining_ <= 0)
            continue;
            
        // Dispatch to specific draw function
        switch (viz->type()) {
            case MeasurmentViz::VizType::Covariance:
                renderCovariance(*std::static_pointer_cast<CovarianceViz>(viz));
                break;
            case MeasurmentViz::VizType::Circle:
                renderCircle(*std::static_pointer_cast<CircleViz>(viz));
                break;
            case MeasurmentViz::VizType::Line:
                renderLine(*std::static_pointer_cast<LineViz>(viz));
                break;
            default:
                break;
        }

        remaining.push_back(viz);  // keep it
    }

    measVizVertex_ = std::move(remaining);  // remove expired
}


void View::renderCovariance(const CovarianceViz& viz) {
    double alpha = 1.0;
    if (viz.lifetime_ > viz.fadeStart_) {
        alpha = std::min(1.0, double(viz.lifetimeRemaining_) / (viz.lifetime_ - viz.fadeStart_));
    }

    glColor4f(viz.color_[0], viz.color_[1], viz.color_[2], alpha);

    // Transform center based on attachment type
    Eigen::Vector2d center = viz.x_;
    if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Loc) {
        center += pose_.head<2>();
    } else if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Pose) {
        Eigen::Rotation2Dd R(pose_[2]);
        center = R * center + pose_.head<2>();
    }

    glPushMatrix();
    glTranslated(center[0], center[1], 0.0);

    glBegin(GL_LINE_LOOP);
    for (int angle = 0; angle <= 360; angle += 10) {
        double theta = angle * M_PI / 180.0;
        Eigen::Vector2d unitCircle(std::cos(theta), std::sin(theta));
        Eigen::Vector2d pt = 2.0 * (viz.covSqrt_ * unitCircle);
        glVertex2d(pt[0], pt[1]);
    }
    glEnd();

    glPopMatrix();
}

void View::renderCircle(const CircleViz& viz) {
    double alpha = 1.0;
    if (viz.lifetime_ > viz.fadeStart_) {
        alpha = std::min(1.0, double(viz.lifetimeRemaining_) / (viz.lifetime_ - viz.fadeStart_));
    }

    glColor4f(viz.color_[0], viz.color_[1], viz.color_[2], alpha);

    // Transform center based on attachment type
    Eigen::Vector2d center = viz.center_;
    if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Loc) {
        center += pose_.head<2>();
    } else if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Pose) {
        Eigen::Rotation2Dd R(pose_[2]);
        center = R * center + pose_.head<2>();
    }

    glBegin(GL_LINE_LOOP);
    for (int angle = 0; angle <= 360; angle += 10) {
        double theta = angle * M_PI / 180.0;
        glVertex2d(center[0] + viz.radius_ * std::cos(theta),
                   center[1] + viz.radius_ * std::sin(theta));
    }
    glEnd();
}

void View::renderLine(const LineViz& viz) {
    double alpha = 1.0;
    if (viz.lifetime_ > viz.fadeStart_) {
        alpha = std::min(1.0, double(viz.lifetimeRemaining_) / (viz.lifetime_ - viz.fadeStart_));
    }

    glColor4f(viz.color_[0], viz.color_[1], viz.color_[2], alpha);

    Eigen::Vector2d x1 = viz.x1_;
    Eigen::Vector2d x2 = viz.x2_;

    if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Loc) {
        x1 += pose_.head<2>();
        x2 += pose_.head<2>();
    } else if (viz.attachmentType_ == MeasurmentViz::AttachmentType::Pose) {
        Eigen::Rotation2Dd R(pose_[2]);
        x1 = R * x1 + pose_.head<2>();
        x2 = R * x2 + pose_.head<2>();
    }

    glBegin(GL_LINES);
    glVertex2d(x1[0], x1[1]);
    glVertex2d(x2[0], x2[1]);
    glEnd();
}

void View::pause(){
    running_ = false;
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

}
}
}
