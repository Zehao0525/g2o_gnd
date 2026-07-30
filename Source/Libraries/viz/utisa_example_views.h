#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#include "view.h"

#include "utisa_simulator_api.h"

namespace g2o {
namespace tutorial {
namespace viz {

/// SLAM trajectory view for UTISA slam systems (2D SE2 estimate → 3D display).
class UTISASlamSystemView : public View {
 public:
  UTISASlamSystemView(multibotsim::UTISASlamSystem* system, const Eigen::Vector3f& color,
                      bool visualise_path = true);
  UTISASlamSystemView(multibotsim::UTISASlamSystem* system, const std::string& filename,
                      bool visualise_path = true);

  UTISASlamSystemView(multibotsim::UTISASlamSystemNew* system, const Eigen::Vector3f& color,
                      bool visualise_path = true);
  UTISASlamSystemView(multibotsim::UTISASlamSystemNew* system, const std::string& filename,
                      bool visualise_path = true);

  void setView(const std::string& filename) override;
  void update() override;
  void pause() override;
  void renderScene() const override;

 private:
  multibotsim::UTISASlamSystem* slamSystem_ = nullptr;
  multibotsim::UTISASlamSystemNew* slamSystemNew_ = nullptr;
  Eigen::Isometry3d currentPose3d_;
  std::vector<Eigen::Isometry3d> path3d_;
  bool visualise_path_;
};

/// Ground-truth view for `UTISASimulator` (SE2 `xTrue()` → 3D display).
class UTISASimulationView : public View {
 public:
  UTISASimulationView(multibotsim::UTISASimulator* sim, const Eigen::Vector3f& color,
                      bool visualise_path = true);
  UTISASimulationView(multibotsim::UTISASimulator* sim, const std::string& filename,
                      bool visualise_path = true);

  void setView(const std::string& filename) override;
  void setLandmarkGroundtruthFile(const std::string& filename);
  void update() override;
  void pause() override;
  void renderScene() const override;

 private:
  multibotsim::UTISASimulator* simulation_;
  Eigen::Isometry3d currentPose3d_;
  std::vector<Eigen::Isometry3d> path3d_;
  bool visualise_path_;
  std::map<int, Eigen::Vector2d> landmarkGt_;
};

}  // namespace viz
}  // namespace tutorial
}  // namespace g2o
