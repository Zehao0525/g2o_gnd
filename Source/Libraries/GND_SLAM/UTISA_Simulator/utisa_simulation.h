// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.

#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o_tutorial_slam2d_api.h"
#include "ordered_event_queue.hpp"
#include "utisa_events.h"

#include "g2o/core/hyper_graph.h"
#include "se2.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

/// Replay simulator for UTIAS MR.CLAM-style logs (see `test_data/utisa/...`).
///
/// **MR.CLAM (2009) layout** — one dataset folder per constructor overload:
/// - `Robot{N}_Groundtruth.dat`: `#` header lines, then `time [s]  x [m]  y [m]  orientation [rad]`
/// - `Robot{N}_Odometry.dat`: `time [s]  forward velocity [m/s]  angular velocity [rad/s]`
/// - `Robot{N}_Measurement.dat`: `time [s]  barcode  range [m]  bearing [rad]`
/// - `Barcodes.dat`: maps barcode → subject index (1–5 robots, 6–20 landmarks)
///
/// Range and bearing are passed through to the SLAM system as \((r,\phi)\) for `EdgeRangeBearing`.
///
class G2O_TUTORIAL_SLAM2D_API UTISASimulator {
 public:
  /// UTIAS MR.CLAM: separate `.dat` streams + barcode map (paths typically from one dataset directory).
  UTISASimulator(std::string id, const std::string& odometry_dat_path,
                 const std::string& measurement_dat_path, const std::string& groundtruth_dat_path,
                 const std::string& barcodes_dat_path);

  ~UTISASimulator();

  SE2 xTrue() const;

  void history(std::vector<double>& timeHistory, std::vector<SE2>& xTrueHistory) const;

  void start();

  bool keepRunning() const;

  void stop();

  std::vector<EventPtr> acquireEvents();

  void step();

  void step(double dt);
  void setDurationLimit(double duration_sec);
  void setOdomInformation(const Eigen::Matrix3d& info);
  void setRangeBearingInformation(const Eigen::Matrix2d& info);

 protected:
  void initialize();

  void initializeMrclam();

  void storeStepResults();

 public:
  bool carryOnRunning_;

  bool initialized_;

 protected:
  SE2 x_;

  OrderedEventQueue eventQueue_;

  std::vector<double> timeStore_;
  std::vector<SE2> xTrueStore_;

  int currentVtxNumber_;

  bool verbose_;

  std::string robotId_;

  double currentTime_;

  double timeOrigin_ = 0.0;
  bool hasTimeOrigin_ = false;
  double durationLimitSec_ = -1.0;  // <0 means unlimited

  /// MR.CLAM: odometry + measurement; ground truth shared with `gtStream_` below.
  std::ifstream odomStream_;
  std::ifstream measStream_;

  /// Ground truth stream (`RobotN_Groundtruth.dat`).
  std::ifstream gtStream_;

  std::unordered_map<int, int> barcodeToSubject_;

  int robotSubjectId_ = -1;

  struct GTBuffer {
    bool valid = false;
    double time = 0.0;
    SE2 pose;
  } gtBuffer_;

  enum class DataMsgType {
    None,
    Odom,
    RelPos,
    LmObs
  };

  struct DataBuffer {
    bool valid = false;
    double time = 0.0;
    DataMsgType type = DataMsgType::None;
    Eigen::Vector3d odomVel = Eigen::Vector3d::Zero();
    std::string targetRobotId;
    int landmarkId = -1;
    /// MR.CLAM measurement: range [m], bearing [rad] (sensor/body frame).
    double range = 0.0;
    double bearing = 0.0;
    Eigen::Matrix2d rbInformation = Eigen::Matrix2d::Identity();
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
  } dataBuffer_;

  bool gtHasMore_ = false;
  bool dataHasMore_ = false;

  struct MrclamOdomNext {
    bool valid = false;
    double t = 0.0;
    double v = 0.0;
    double omega = 0.0;
  } mrclamOdomNext_;

  struct MrclamMeasNext {
    bool valid = false;
    double t = 0.0;
    int barcode = 0;
    double range = 0.0;
    double bearing = 0.0;
  } mrclamMeasNext_;

  Eigen::Matrix3d odomInformation_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix2d rangeBearingInformation_ = Eigen::Matrix2d::Identity();

  bool readNextGT();

  bool readNextData();

  bool readNextDataMrclam();

  void loadBarcodes(const std::string& path);

  void mrclamPrefetchOdom();

  void mrclamPrefetchMeas();

  void pushEvent(EventPtr e);

  std::uint64_t nextEventTieOrder_{0};
};

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
