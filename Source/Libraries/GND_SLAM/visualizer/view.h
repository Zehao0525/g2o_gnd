#pragma once
#include <mutex>
#include <Eigen/Core>
#include <vector>
#include <iostream>

#include <Eigen/Eigenvalues>

#include "viz_shape.h"

namespace g2o {
namespace tutorial {
namespace viz {
class View {
public:
    View(const Eigen::Vector3f& color);
    virtual ~View() = default;

    // Called to update internal data (e.g., query the SLAM system / simulator)
    virtual void update() = 0;

    // Rendering logic
    void renderRobotPose() const;
    void renderRobotPath() const;
    void renderMeasurmentViz();
    void renderCovariance(const CovarianceViz& viz);
    void renderCircle(const CircleViz& viz);
    void renderLine(const LineViz& viz);

    virtual void pause();


    virtual void renderScene() const;
    virtual void setView(const std::string& filename);

    // Temporarily used for synchronouse simulations
    void updateRobotPath(const std::vector<Eigen::Vector3d>& path);
    void appendToRobotPath(const Eigen::Vector3d& pose);
    void updateRobotPose(const Eigen::Vector3d& pose);

protected:
    Eigen::Vector3f color_;
    double platformSize_;
    std::vector<Eigen::Vector3d> path_;

    Eigen::Vector3d pose_;

    std::vector<std::shared_ptr<MeasurmentViz>> measVizVertex_;
    Eigen::Vector3f measColor_;
    bool running_;

    std::mutex dataMutex_;
};

}
}
}