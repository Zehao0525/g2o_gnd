#include "utisa_example_views.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include <GL/gl.h>
#include <nlohmann/json.hpp>

#include "se2.h"

namespace g2o {
namespace tutorial {
namespace viz {

namespace {

Eigen::Isometry3d pose2dVectorToIsometry3d(const Eigen::Vector3d& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() << p(0), p(1), 0.0;
  T.linear() = Eigen::AngleAxisd(p(2), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return T;
}

Eigen::Isometry3d se2ToIsometry3d(const ::g2o::tutorial::SE2& se2) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() << se2.translation().x(), se2.translation().y(), 0.0;
  T.linear() = Eigen::AngleAxisd(se2.rotation().angle(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return T;
}

}  // namespace

// --- UTISASlamSystemView ---

UTISASlamSystemView::UTISASlamSystemView(multibotsim::UTISASlamSystem* system, const Eigen::Vector3f& color,
                                       bool visualise_path)
    : View(color),
      slamSystem_(system),
      currentPose3d_(Eigen::Isometry3d::Identity()),
      visualise_path_(visualise_path) {}

UTISASlamSystemView::UTISASlamSystemView(multibotsim::UTISASlamSystem* system, const std::string& filename,
                                       bool visualise_path)
    : View(Eigen::Vector3f(0.0f, 1.0f, 0.0f)),
      slamSystem_(system),
      currentPose3d_(Eigen::Isometry3d::Identity()),
      visualise_path_(visualise_path) {
  setView(filename);
}

void UTISASlamSystemView::setView(const std::string& filename) {
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

void UTISASlamSystemView::update() {
  Eigen::Vector3d poseVec;
  slamSystem_->platformEstimate(poseVec);
  Eigen::Isometry3d pose = pose2dVectorToIsometry3d(poseVec);

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

void UTISASlamSystemView::pause() {
  View::pause();
}

void UTISASlamSystemView::renderScene() const {
  std::lock_guard<std::mutex> lock(const_cast<UTISASlamSystemView*>(this)->dataMutex_);
  if (visualise_path_) {
    renderRobotPath();
  }
  renderRobotPose2D();

  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> lmSegs;
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> rbSegs;
  slamSystem_->getRangeBearingObservationSegments(lmSegs, rbSegs);

  glLineWidth(1.0f);
  glColor4f(1.0f, 0.55f, 0.1f, 0.25f);  // landmark observations
  glBegin(GL_LINES);
  for (const auto& s : lmSegs) {
    glVertex2d(s.first.x(), s.first.y());
    glVertex2d(s.second.x(), s.second.y());
  }
  glEnd();

  glColor4f(0.1f, 0.8f, 1.0f, 0.25f);  // robot observations
  glBegin(GL_LINES);
  for (const auto& s : rbSegs) {
    glVertex2d(s.first.x(), s.first.y());
    glVertex2d(s.second.x(), s.second.y());
  }
  glEnd();
}

// --- UTISASimulationView ---

UTISASimulationView::UTISASimulationView(multibotsim::UTISASimulator* sim, const Eigen::Vector3f& color,
                                         bool visualise_path)
    : View(color),
      simulation_(sim),
      currentPose3d_(Eigen::Isometry3d::Identity()),
      visualise_path_(visualise_path) {}

UTISASimulationView::UTISASimulationView(multibotsim::UTISASimulator* sim, const std::string& filename,
                                       bool visualise_path)
    : View(Eigen::Vector3f(0.5f, 0.5f, 0.5f)),
      simulation_(sim),
      currentPose3d_(Eigen::Isometry3d::Identity()),
      visualise_path_(visualise_path) {
  setView(filename);
}

void UTISASimulationView::setView(const std::string& filename) {
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

void UTISASimulationView::setLandmarkGroundtruthFile(const std::string& filename) {
  landmarkGt_.clear();
  std::ifstream in(filename);
  if (!in) {
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    int sid = -1;
    double x = 0.0;
    double y = 0.0;
    if (iss >> sid >> x >> y) {
      landmarkGt_[sid] = Eigen::Vector2d(x, y);
    }
  }
}

void UTISASimulationView::update() {
  Eigen::Isometry3d pose = se2ToIsometry3d(simulation_->xTrue());

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

void UTISASimulationView::pause() {
  View::pause();
}

void UTISASimulationView::renderScene() const {
  std::lock_guard<std::mutex> lock(const_cast<UTISASimulationView*>(this)->dataMutex_);
  if (visualise_path_) {
    renderRobotPath();
  }
  renderRobotPose2D();

  if (!landmarkGt_.empty()) {
    glPointSize(5.0f);
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    for (const auto& kv : landmarkGt_) {
      glVertex2d(kv.second.x(), kv.second.y());
    }
    glEnd();
  }
}

}  // namespace viz
}  // namespace tutorial
}  // namespace g2o
