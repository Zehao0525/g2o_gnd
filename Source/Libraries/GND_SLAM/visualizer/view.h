#pragma once
#include <mutex>
#include <Eigen/Core>
#include <vector>
#include <iostream>

#include <Eigen/Eigenvalues>

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
    virtual void renderScene() const;
    virtual void setView(const std::string& filename);

    // Temporarily used for synchronouse simulations
    void updateRobotPath(const std::vector<Eigen::Vector3d>& path);
    void appendToRobotPath(const Eigen::Vector3d& pose);
    void updateRobotPose(const Eigen::Vector3d& pose);
    void updateRobotMarginals(const Eigen::Matrix2d& margianls);

protected:
    Eigen::Vector3f color_;
    double platformSize_;
    std::vector<Eigen::Vector3d> path_;

    Eigen::Vector3d pose_;

    bool marginalsInitialized_;
    Eigen::Matrix2d poseMarginal2d_;
    Eigen::Matrix2d sqrtPoseMarginal2d_;

    std::mutex dataMutex_;
};

}
}
}