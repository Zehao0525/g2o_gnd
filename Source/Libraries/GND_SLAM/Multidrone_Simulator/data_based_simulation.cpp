// data_based_simulation.cpp

#include "data_based_simulation.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

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

// Build a pure yaw rotation + simple translation from odom fields
// odom format: time odom linear_fw linear_z yaw
static Isometry3 makeIsometryFromOdom(double linear_fw, double linear_z,
                                      double yaw) {
  Isometry3 T = Isometry3::Identity();
  // Interpret linear_fw as x, linear_z as z in body frame for this step.
  T.translation() = Eigen::Vector3d(linear_fw, 0.0, linear_z);
  Eigen::AngleAxisd yawRot(yaw, Eigen::Vector3d::UnitZ());
  T.linear() = yawRot.toRotationMatrix();
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
    double qx, qy, qz, qw;
    if (!(iss >> px >> py >> pz >> qx >> qy >> qz >> qw)) {
      continue;  // malformed pose
    }

    gtBuffer_.time = t;
    gtBuffer_.pose = makeIsometryFromPosQuat(px, py, pz, qx, qy, qz, qw);
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
      double linear_fw, linear_z, yaw;
      if (!(iss >> linear_fw >> linear_z >> yaw)) {
        continue;
      }

      dataBuffer_.time = t;
      dataBuffer_.type = DataMsgType::Odom;
      dataBuffer_.odomPose = makeIsometryFromOdom(linear_fw, linear_z, yaw);

      // Optional: read appended 6x6 information (upper-triangle, 21 values)
      Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
      std::vector<double> infoVals;
      double v;
      while (iss >> v) {
        infoVals.push_back(v);
      }
      if (infoVals.size() >= 21) {
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
        // fill symmetric 6x6 from upper triangle
        size_t idx = 0;
        for (int r = 0; r < 6; ++r) {
          for (int c = r; c < 6; ++c) {
            info(r, c) = covVals[idx];
            info(c, r) = covVals[idx];
            ++idx;
            if (idx >= covVals.size()) break;
          }
          if (idx >= covVals.size()) break;
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
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();

  // Scan the data stream until we find an "init ..." line.
  // Format (Python): "init {id} {x} {y} {z} qx qy qz qw {fixed} [upper triangle covariance]"
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

      // Read remaining doubles as covariance upper triangle if present
      std::vector<double> covVals;
      double v;
      while (iss >> v) {
        covVals.push_back(v);
      }
      if (covVals.size() >= 21) {
        cov.setZero();
        size_t idx = 0;
        for (int r = 0; r < 6; ++r) {
          for (int c = r; c < 6; ++c) {
            cov(r, c) = covVals[idx];
            cov(c, r) = covVals[idx];
            ++idx;
            if (idx >= covVals.size()) break;
          }
          if (idx >= covVals.size()) break;
        }
      }

      break;  // found init
    }
  }

  // Convert covariance -> information (robustly). If fixed, info isn't used much, but still set.
  Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
  if (cov.allFinite()) {
    // Add tiny diagonal jitter to avoid singular matrices.
    Eigen::Matrix<double, 6, 6> cov_reg = cov;
    cov_reg.diagonal().array() += 1e-12;
    info = cov_reg.inverse();
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
          dataBuffer_.time, dataBuffer_.odomPose, dataBuffer_.information);
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
