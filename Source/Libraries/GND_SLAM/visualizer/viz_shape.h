#pragma once
#include <map>
#include <vector>
#include <memory>
#include <set>

#include "g2o_tutorial_slam2d_api.h"

namespace g2o {
namespace tutorial {
namespace viz {

struct G2O_TUTORIAL_SLAM2D_API MeasurmentViz {
    int lifetime_;
    int lifetimeRemaining_;
    int fadeStart_;
    Eigen::Vector3f color_;
    enum class VizType {
        Covariance,
        Circle,
        Line
    };

    enum class AttachmentType {
        World,
        Loc,
        Pose
    };

    MeasurmentViz::AttachmentType attachmentType_;

    MeasurmentViz(int lifetime, int fadeStart, AttachmentType attachmentType, Eigen::Vector3f color) : lifetime_(lifetime), lifetimeRemaining_(lifetime), fadeStart_(fadeStart), attachmentType_(attachmentType), color_(color) {};
    virtual ~MeasurmentViz() = default;
    virtual VizType type() const = 0;
};


// Covariance visualization struct
struct G2O_TUTORIAL_SLAM2D_API CovarianceViz : public MeasurmentViz {
    Eigen::Vector2d x_;
    Eigen::Matrix2d covSqrt_;

    CovarianceViz(const Eigen::Vector2d& x, const Eigen::Matrix2d& covSqrt,
                  int lifetime, int fadeStart,  AttachmentType attachmentType, Eigen::Vector3f color)
        : MeasurmentViz(lifetime, fadeStart, attachmentType, color), x_(x), covSqrt_(covSqrt) {}

    VizType type() const override { return VizType::Covariance; }
};

// Circle visualization struct
struct G2O_TUTORIAL_SLAM2D_API CircleViz : public MeasurmentViz {
    Eigen::Vector2d center_;
    double radius_;

    CircleViz(const Eigen::Vector2d& center, double radius,
              int lifetime, int fadeStart, AttachmentType attachmentType, Eigen::Vector3f color)
        : MeasurmentViz(lifetime, fadeStart, attachmentType, color), center_(center), radius_(radius) {}

    VizType type() const override { return VizType::Circle; }
};

// Line visualization struct
struct G2O_TUTORIAL_SLAM2D_API LineViz : public MeasurmentViz {
    Eigen::Vector2d x1_;
    Eigen::Vector2d x2_;

    LineViz(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2,
            int lifetime, int fadeStart, AttachmentType attachmentType, Eigen::Vector3f color)
        : MeasurmentViz(lifetime, fadeStart, attachmentType, color), x1_(x1), x2_(x2) {}

    VizType type() const override { return VizType::Line; }
};



}}}