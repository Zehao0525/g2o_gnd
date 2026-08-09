#ifndef G2O_TUTORIAL_SLAM_VISUALIZER_H
#define G2O_TUTORIAL_SLAM_VISUALIZER_H

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>

namespace g2o {
namespace tutorial {
namespace viz {

enum class VizType {
    PoseOnly,
    Path
};

class Visualizer {
public:
    Visualizer();
    ~Visualizer();

    void start();
    void stop();

    void setVizType(std::string typeKey);

    void updateRobotPath(const std::vector<Eigen::Vector3d>& path);
    void appendToRobotPath(Eigen::Vector3d pose);
    void updateLandmarks(const std::vector<Eigen::Vector2d>& landmarks);

protected:
    void renderLoop();
    void renderRobotPose(const Eigen::Vector3d& currentPose);
    void renderRobotPath(const std::vector<Eigen::Vector3d>& robotPath);

    std::thread renderThread_;
    std::mutex dataMutex_;
    std::vector<Eigen::Vector3d> robotPath_;
    std::vector<Eigen::Vector2d> landmarks_;
    std::atomic<bool> running_;

    Viztype vizType_;
};

}  // namespace viz
}  // namespace tutorial
}  // namespace g2o

#endif
