// data_based_simulation.cpp

#include "data_based_simulation.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <vector>

#include <Eigen/Geometry>

namespace g2o {
namespace tutorial {
namespace multibotsim{

// ======== Helper structs (internal to this .cpp) ========

// Buffer for next GT (ground-truth) line
struct GTBuffer {
  bool valid = false;
  double time = 0.0;
  Isometry3 pose = Isometry3::Identity();
};

// Type of data message in the data file
enum class DataMsgType {
  None,
  Odom,
  RelPos
};

// Buffer for next data line
struct DataBuffer {
  bool valid = false;
  double time = 0.0;
  DataMsgType type = DataMsgType::None;

  // For odom:
  Isometry3 odomPose = Isometry3::Identity();
  double odomOmegaZ = 0.0;

  // For relpos:
  std::string targetRobotId;
  Isometry3 relPose = Isometry3::Identity();
  Eigen::Matrix<double, 6, 6> information =
      Eigen::Matrix<double, 6, 6>::Identity();
};

// ======== IMPORTANT: required private members in the header ========
//
// In DataBasedSimulation (data_based_simulation.h), you should add:
//
//  #include <fstream>
//  #include <Eigen/Core>
//
//  protected:
//    std::ifstream dataStream_;
//    std::ifstream gtStream_;
//
//    GTBuffer gtBuffer_;
//    DataBuffer dataBuffer_;
//
//    bool gtHasMore_;
//    bool dataHasMore_;
//
// End of NOTE.
// ========================================================

// Small helper: replace commas with spaces so >> on doubles works even
// if the file has "0.0," etc.
static void sanitizeLine(std::string &line) {
  std::replace(line.begin(), line.end(), ',', ' ');
}

// Build an SE3 from position + quaternion (x,y,z,qx,qy,qz,qw)
static Isometry3 makeIsometryFromPosQuat(double px, double py, double pz,
                                         double qx, double qy, double qz,
                                         double qw) {
  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();
  Isometry3 T = Isometry3::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(px, py, pz);
  return T;
}

// Python simulator.py logs: "t pose x y z yaw, vx, vy, vz, r" (yaw about +Z, planar motion).
static Isometry3 makeIsometryFromPosYaw(double px, double py, double pz, double yaw) {
  Isometry3 T = Isometry3::Identity();
  T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T.translation() = Eigen::Vector3d(px, py, pz);
  return T;
}

// Body-frame linear velocities only. Keep rotation = Identity: angular velocity (rad/s) is carried
// separately on DataOdomEvent::omegaZ. Encoding ω as R_z(ω) would lose |ω| beyond 2π when decoded.
static Isometry3 makeIsometryFromOdomVel(double linear_fw, double linear_z) {
  Isometry3 T = Isometry3::Identity();
  T.translation() = Eigen::Vector3d(linear_fw, 0.0, linear_z);
  return T;
}

// ======== DataBasedSimulation implementation ========

DataBasedSimulation::DataBasedSimulation(std::string id,
                                         const std::string &data_path,
                                         const std::string &gt_path)
    : carryOnRunning_(true),
      initialized_(false),
      x_(Isometry3::Identity()),
      eventQueue_(),
      currentVtxNumber_(0),
      verbose_(false),
      robotId_(std::move(id)),
      currentTime_(0.0),
      gtHasMore_(false),
      dataHasMore_(false) {
  // Open files
  dataStream_.open(data_path);
  if (!dataStream_) {
    throw std::runtime_error("Failed to open data file: " + data_path);
  }

  gtStream_.open(gt_path);
  if (!gtStream_) {
    throw std::runtime_error("Failed to open GT file: " + gt_path);
  }
}

DataBasedSimulation::~DataBasedSimulation() {
  if (dataStream_.is_open()) dataStream_.close();
  if (gtStream_.is_open()) gtStream_.close();
}

// ----------- Simple accessors -----------

Isometry3 DataBasedSimulation::xTrue() const {
  return x_;
}

void DataBasedSimulation::history(std::vector<double> &timeHistory,
                                  std::vector<g2o::Isometry3> &xTrueHistory) const {
  timeHistory = timeStore_;
  xTrueHistory = xTrueStore_;
}

bool DataBasedSimulation::keepRunning() const {
  return carryOnRunning_;
}

std::vector<EventPtr> DataBasedSimulation::acquireEvents() {
  auto events = eventQueue_.orderedEvents();
  eventQueue_.clear();
  return events;
}

// ----------- Internal helpers to read next lines -----------

bool DataBasedSimulation::readNextGT() {
  gtBuffer_.valid = false;

  std::string line;
  while (std::getline(gtStream_, line)) {
    sanitizeLine(line);
    if (line.empty()) continue;

    std::istringstream iss(line);
    double t;
    std::string type;
    if (!(iss >> t >> type)) {
      continue;  // malformed, skip
    }
    if (type != "pose") {
      continue;  // ignore other types if any
    }

    double px, py, pz;
    if (!(iss >> px >> py >> pz)) {
      continue;
    }
    std::vector<double> tail;
    double v;
    while (iss >> v) {
      tail.push_back(v);
    }

    gtBuffer_.time = t;
    // Two formats:
    // 1) Python simulator: "t pose x y z yaw vx vy vz r"  -> 5 numbers after xyz (yaw + velocities).
    // 2) Legacy quaternion: "t pose x y z qx qy qz qw"    -> exactly 4 numbers after xyz.
    if (tail.size() == 4) {
      gtBuffer_.pose =
          makeIsometryFromPosQuat(px, py, pz, tail[0], tail[1], tail[2], tail[3]);
    } else if (tail.size() >= 5) {
      const double yaw = tail[0];
      gtBuffer_.pose = makeIsometryFromPosYaw(px, py, pz, yaw);
    } else {
      continue;
    }
    gtBuffer_.valid = true;
    return true;
  }

  return false;  // no more GT lines
}

bool DataBasedSimulation::readNextData() {
  dataBuffer_.valid = false;

  std::string line;
  while (std::getline(dataStream_, line)) {
    sanitizeLine(line);
    if (line.empty()) continue;

    std::istringstream iss(line);
    double t;
    std::string type;
    if (!(iss >> t >> type)) {
      continue;  // malformed
    }

    if (type == "init") {
      // Initialization lines are handled in initialize(); skip if encountered later.
      continue;
    } else if (type == "odom") {
      // Odom log stores velocities (v_fwd, v_z, omega), not pose increments.
      // Leave dt conversion to the SLAM system.
      double vel_fw, vel_z, omega;
      if (!(iss >> vel_fw >> vel_z >> omega)) {
        continue;
      }

      dataBuffer_.time = t;
      dataBuffer_.type = DataMsgType::Odom;
      dataBuffer_.odomPose = makeIsometryFromOdomVel(vel_fw, vel_z);
      dataBuffer_.odomOmegaZ = omega;

      // Optional: read appended uncertainty for odometry.
      // New format (Python): 3 variances for (x, z, yaw) [var_x var_z var_yaw]
      // Older format: 21 values for upper-triangle of 6x6 INFORMATION matrix.
      Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
      std::vector<double> infoVals;
      double v;
      while (iss >> v) {
        infoVals.push_back(v);
      }

      // Clamp information diagonal to satisfy isValidInformationMatrix limits.
      const double maxDiag = 9.0e9;
      const double eps = 1e-12;

      if (infoVals.size() >= 21) {
        // Interpret as upper triangle of INFORMATION directly.
        size_t idx = 0;
        for (int r = 0; r < 6; ++r) {
          for (int c = r; c < 6; ++c) {
            info(r, c) = infoVals[idx];
            info(c, r) = infoVals[idx];
            ++idx;
            if (idx >= infoVals.size()) break;
          }
          if (idx >= infoVals.size()) break;
        }
      } else if (infoVals.size() >= 3) {
        // Interpret first 3 values as velocity variances (v_fwd, v_z, omega).
        // Conversion to displacement information is handled in the SLAM system using dt.
        double var_x = infoVals[0];
        double var_z = infoVals[1];
        double var_yaw = infoVals[2];

        var_x = std::max(var_x, eps);
        var_z = std::max(var_z, eps);
        var_yaw = std::max(var_yaw, eps);

        info(0, 0) = 1.0 / var_x;
        info(2, 2) = 1.0 / var_z;
        info(5, 5) = 1.0 / var_yaw;
      }

      // Robustness: clamp extreme diagonal values to satisfy SLAM limits.
      for (int i = 0; i < 6; ++i) {
        if (std::isfinite(info(i, i)) && info(i, i) > maxDiag) {
          info(i, i) = maxDiag;
        }
      }
      dataBuffer_.information = info;

      dataBuffer_.valid = true;
      return true;
    } else if (type == "relpos") {
      int target_id;
      double px, py, pz;
      double qx, qy, qz, qw;
      if (!(iss >> target_id >> px >> py >> pz >> qx >> qy >> qz >> qw)) {
        continue;
      }

      dataBuffer_.time = t;
      dataBuffer_.type = DataMsgType::RelPos;
      dataBuffer_.targetRobotId = std::to_string(target_id);
      dataBuffer_.relPose = makeIsometryFromPosQuat(px, py, pz, qx, qy, qz, qw);

      // Initialize information as identity
      Eigen::Matrix<double, 6, 6> info =
          Eigen::Matrix<double, 6, 6>::Identity();

      // Read remaining doubles as covariance upper triangle if present
      std::vector<double> covVals;
      double v;
      while (iss >> v) {
        covVals.push_back(v);
      }
      if (covVals.size() >= 21) {
        // Fast path: diagonal-only.
        // IMPORTANT: Python's relpos sensor logs `error_std` values directly into
        // the 21 "matrix" entries (and uses them as std dev when generating noise).
        // Therefore those diagonal entries are STD, not VAR/COV.
        // Convert to information as: info = 1/var = 1/(std^2).
        //
        // Upper-triangle indexing for diagonal entries in a 6x6 matrix:
        // (0,0)->0, (1,1)->6, (2,2)->11, (3,3)->15, (4,4)->18, (5,5)->20
        static constexpr size_t kDiagIdx[6] = {0, 6, 11, 15, 18, 20};
        info.setZero();
        const double eps = 1e-12;
        const double maxDiag = 9.0e9;
        for (int i = 0; i < 6; ++i) {
          const double std_ii = covVals[kDiagIdx[i]];
          double info_ii = 1.0;
          if (std::isfinite(std_ii) && std_ii > eps) {
            const double var_ii = std_ii * std_ii;
            info_ii = 1.0 / var_ii;
          }
          if (info_ii > maxDiag) info_ii = maxDiag;
          info(i, i) = info_ii;
        }
      }
      dataBuffer_.information = info;

      dataBuffer_.valid = true;
      return true;
    } else {
      // Unknown type; skip
      continue;
    }
  }

  return false;  // no more data lines
}

// ----------- Core simulation methods -----------

void DataBasedSimulation::start() {
  // Reset state
  carryOnRunning_ = true;
  initialized_ = false;
  currentTime_ = 0.0;

  eventQueue_.clear();
  timeStore_.clear();
  xTrueStore_.clear();

  x_ = Isometry3::Identity();

  // Reset streams to beginning
  if (dataStream_.is_open()) {
    dataStream_.clear();
    dataStream_.seekg(0);
  }
  if (gtStream_.is_open()) {
    gtStream_.clear();
    gtStream_.seekg(0);
  }

  // Initialize from the first init line in the data stream (streaming, no full-file load)
  initialize();

  // Fill buffers after initialization so the data stream continues after the init line.
  gtHasMore_ = readNextGT();
  dataHasMore_ = readNextData();

  initialized_ = true;
}

void DataBasedSimulation::initialize() {
  // Defaults if no init line is found
  bool fixed = true;
  Isometry3 pose = Isometry3::Identity();
  // Convention: all logged matrices (relpos/odom/init) are upper-triangular entries
  // of a 6x6 *information* matrix (not covariance). Keep it consistent.
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
  bool foundInit = false;

  // Scan the data stream until we find an "init ..." line.
  // Format (Python): "init {id} {x} {y} {z} qx qy qz qw {fixed} [upper triangle information]"
  // Note: no timestamp on init; we treat it as time=0.0 for the simulator.
  if (dataStream_.is_open()) {
    std::string line;
    while (std::getline(dataStream_, line)) {
      sanitizeLine(line);
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string type;
      if (!(iss >> type)) continue;
      if (type != "init") {
        continue;
      }

      std::string id;
      double x, y, z, qx, qy, qz, qw;
      std::string fixedToken;
      if (!(iss >> id >> x >> y >> z >> qx >> qy >> qz >> qw >> fixedToken)) {
        break;  // malformed init; fall back to defaults
      }

      fixed = (fixedToken == "true" || fixedToken == "1" || fixedToken == "True");
      pose = makeIsometryFromPosQuat(x, y, z, qx, qy, qz, qw);

      // Read remaining doubles as information upper triangle if present
      std::vector<double> infoVals;
      double v;
      while (iss >> v) {
        infoVals.push_back(v);
      }
      if (infoVals.size() >= 21) {
        info.setZero();
        size_t idx = 0;
        for (int r = 0; r < 6; ++r) {
          for (int c = r; c < 6; ++c) {
            info(r, c) = infoVals[idx];
            info(c, r) = infoVals[idx];
            ++idx;
            if (idx >= infoVals.size()) break;
          }
          if (idx >= infoVals.size()) break;
        }
      }

      foundInit = true;
      break;  // found init
    }
  }

  // If no init line was found (older logs), rewind the data stream so odom/relpos
  // events are still available from the beginning.
  if (!foundInit && dataStream_.is_open()) {
    dataStream_.clear();
    dataStream_.seekg(0);
  }

  // Optional robustness: tiny jitter and clamp extreme diagonal values
  // to satisfy isValidInformationMatrix() limits in the SLAM system.
  if (info.allFinite()) {
    info.diagonal().array() += 1e-12;
    const double maxDiag = 9.0e9;
    for (int i = 0; i < 6; ++i) {
      if (info(i, i) > maxDiag) info(i, i) = maxDiag;
    }
  }

  // Update internal pose/time for history
  x_ = pose;
  currentTime_ = 0.0;

  auto initEvt = std::make_shared<DataInitEvent>(
      0.0,    // eventTime
      fixed,  // posFixed
      pose,   // value
      info);  // information

  eventQueue_.push(initEvt);

  // Also store in history as t=0 pose
  storeStepResults();
}

void DataBasedSimulation::step() {
  // Default step: do not advance time, just flush events up to currentTime_
  step(0.05);
}

void DataBasedSimulation::step(double dt) {
  if (!carryOnRunning_) {
    return;
  }

  // Advance simulation time
  currentTime_ += dt;

  // 1) Update ground truth from GT file up to currentTime_
  while (gtHasMore_ && gtBuffer_.valid && gtBuffer_.time <= currentTime_) {
    // Update the true pose
    x_ = gtBuffer_.pose;
    storeStepResults();

    // Read next GT line
    gtHasMore_ = readNextGT();
  }

  // 2) Emit Data* events from data file up to currentTime_
  //    (Data* events are for data_based_simulation; File* events are for file_simulator.)
  while (dataHasMore_ && dataBuffer_.valid &&
         dataBuffer_.time <= currentTime_) {
    if (dataBuffer_.type == DataMsgType::Odom) {
      auto evt = std::make_shared<DataOdomEvent>(
          dataBuffer_.time, dataBuffer_.odomPose, dataBuffer_.odomOmegaZ,
          dataBuffer_.information);
      eventQueue_.push(evt);
    } else if (dataBuffer_.type == DataMsgType::RelPos) {
      auto evt = std::make_shared<DataObsEvent>(
          dataBuffer_.time,
          dataBuffer_.targetRobotId,
          dataBuffer_.relPose,
          dataBuffer_.information);
      eventQueue_.push(evt);
    }

    // Read next data line
    dataHasMore_ = readNextData();
  }

  // 3) Decide whether to keep running
  if (!gtHasMore_ && !dataHasMore_) {
    // no more input; after we’ve flushed everything, we can stop
    carryOnRunning_ = false;
  }
}

void DataBasedSimulation::stop() {
  carryOnRunning_ = false;
}

// ----------- Stubs for legacy methods in the header -----------

// void DataBasedSimulation::updateOdometry(Edge & /*odom*/) {
//   // This method is kept as a stub to satisfy the header.
//   // In this data-based simulation, odometry events are generated
//   // directly from the data file in step().
// }

// void DataBasedSimulation::updateObservation(Edge & /*obs*/) {
//   // This method is kept as a stub to satisfy the header.
//   // Observation events are generated from "relpos" lines in step().
// }

void DataBasedSimulation::storeStepResults() {
  timeStore_.push_back(currentTime_);
  xTrueStore_.push_back(x_);
}

}
}  // namespace tutorial
}  // namespace g2o
