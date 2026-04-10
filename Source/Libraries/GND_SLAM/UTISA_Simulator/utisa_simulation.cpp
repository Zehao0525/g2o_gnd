#include "utisa_simulation.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>

#include <Eigen/Geometry>

namespace g2o {
namespace tutorial {
namespace multibotsim {

static void sanitizeLine(std::string& line) {
  std::replace(line.begin(), line.end(), ',', ' ');
}

static std::string trimStr(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

static bool isCommentOrEmpty(const std::string& line) {
  const std::string t = trimStr(line);
  return t.empty() || t[0] == '#';
}

static SE2 makeSE2FromPosYaw(double px, double py, double yaw) {
  return SE2(px, py, yaw);
}

/// Default planar odometry information (velocity → displacement scaling happens in SLAM).
static Eigen::Matrix3d defaultMrclamOdomInformation() {
  constexpr double var_v = 0.02 * 0.02;
  constexpr double var_vy = 1.0;
  constexpr double var_w = 0.05 * 0.05;
  Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
  info(0, 0) = 1.0 / var_v;
  info(1, 1) = 1.0 / var_vy;
  info(2, 2) = 1.0 / var_w;
  return info;
}

// MR.CLAM: diagonal information in native (range [m], bearing [rad]) measurement space.
static Eigen::Matrix2d defaultMrclamMeasInformation() {
  constexpr double sigma_r = 0.15;
  constexpr double sigma_b = 0.08;
  Eigen::Matrix2d info = Eigen::Matrix2d::Zero();
  info(0, 0) = 1.0 / (sigma_r * sigma_r);
  info(1, 1) = 1.0 / (sigma_b * sigma_b);
  return info;
}

UTISASimulator::UTISASimulator(std::string id, const std::string& odometry_dat_path,
                               const std::string& measurement_dat_path,
                               const std::string& groundtruth_dat_path,
                               const std::string& barcodes_dat_path)
    : carryOnRunning_(true),
      initialized_(false),
      x_(),
      eventQueue_(),
      currentVtxNumber_(0),
      verbose_(false),
      robotId_(std::move(id)),
      currentTime_(0.0) {
  odomInformation_ = defaultMrclamOdomInformation();
  rangeBearingInformation_ = defaultMrclamMeasInformation();
  try {
    robotSubjectId_ = std::stoi(robotId_);
  } catch (...) {
    robotSubjectId_ = -1;
  }

  odomStream_.open(odometry_dat_path);
  measStream_.open(measurement_dat_path);
  gtStream_.open(groundtruth_dat_path);
  if (!odomStream_) {
    throw std::runtime_error("UTISASimulator: cannot open MR.CLAM odometry: " + odometry_dat_path);
  }
  if (!measStream_) {
    throw std::runtime_error("UTISASimulator: cannot open MR.CLAM measurement: " + measurement_dat_path);
  }
  if (!gtStream_) {
    throw std::runtime_error("UTISASimulator: cannot open MR.CLAM groundtruth: " + groundtruth_dat_path);
  }
  loadBarcodes(barcodes_dat_path);
}

UTISASimulator::~UTISASimulator() {
  if (odomStream_.is_open()) odomStream_.close();
  if (measStream_.is_open()) measStream_.close();
  if (gtStream_.is_open()) gtStream_.close();
}

void UTISASimulator::loadBarcodes(const std::string& path) {
  barcodeToSubject_.clear();
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("UTISASimulator: cannot open Barcodes.dat: " + path);
  }
  std::string line;
  while (std::getline(in, line)) {
    sanitizeLine(line);
    if (isCommentOrEmpty(line)) continue;
    std::istringstream iss(line);
    int subject = 0;
    int barcode = 0;
    if (iss >> subject >> barcode) {
      barcodeToSubject_[barcode] = subject;
    }
  }
}

SE2 UTISASimulator::xTrue() const {
  return x_;
}

void UTISASimulator::history(std::vector<double>& timeHistory, std::vector<SE2>& xTrueHistory) const {
  timeHistory = timeStore_;
  xTrueHistory = xTrueStore_;
}

void UTISASimulator::setDurationLimit(double duration_sec) {
  durationLimitSec_ = duration_sec;
}

void UTISASimulator::setOdomInformation(const Eigen::Matrix3d& info) {
  if (info.allFinite()) {
    odomInformation_ = info;
  }
}

void UTISASimulator::setRangeBearingInformation(const Eigen::Matrix2d& info) {
  if (info.allFinite()) {
    rangeBearingInformation_ = info;
  }
}

bool UTISASimulator::keepRunning() const {
  return carryOnRunning_;
}

std::vector<EventPtr> UTISASimulator::acquireEvents() {
  auto events = eventQueue_.orderedEvents();
  eventQueue_.clear();
  return events;
}

void UTISASimulator::pushEvent(EventPtr e) {
  if (e) {
    e->tieOrder = nextEventTieOrder_++;
  }
  eventQueue_.push(std::move(e));
}

bool UTISASimulator::readNextGT() {
  gtBuffer_.valid = false;

  std::string line;
  while (std::getline(gtStream_, line)) {
    sanitizeLine(line);
    if (isCommentOrEmpty(line)) continue;

    std::istringstream iss(line);

    double t = 0.0;
    double px = 0.0;
    double py = 0.0;
    double th = 0.0;
    if (!(iss >> t >> px >> py >> th)) {
      continue;
    }
    gtBuffer_.time = hasTimeOrigin_ ? (t - timeOrigin_) : t;
    gtBuffer_.pose = makeSE2FromPosYaw(px, py, th);
    gtBuffer_.valid = true;
    return true;
  }

  return false;
}

void UTISASimulator::mrclamPrefetchOdom() {
  mrclamOdomNext_.valid = false;
  std::string line;
  while (std::getline(odomStream_, line)) {
    sanitizeLine(line);
    if (isCommentOrEmpty(line)) continue;
    std::istringstream iss(line);
    double t = 0.0;
    double v = 0.0;
    double w = 0.0;
    if (iss >> t >> v >> w) {
      mrclamOdomNext_.valid = true;
      mrclamOdomNext_.t = t;
      mrclamOdomNext_.v = v;
      mrclamOdomNext_.omega = w;
      return;
    }
  }
}

void UTISASimulator::mrclamPrefetchMeas() {
  mrclamMeasNext_.valid = false;
  std::string line;
  while (std::getline(measStream_, line)) {
    sanitizeLine(line);
    if (isCommentOrEmpty(line)) continue;
    std::istringstream iss(line);
    double t = 0.0;
    int barcode = 0;
    double r = 0.0;
    double b = 0.0;
    if (iss >> t >> barcode >> r >> b) {
      mrclamMeasNext_.valid = true;
      mrclamMeasNext_.t = t;
      mrclamMeasNext_.barcode = barcode;
      mrclamMeasNext_.range = r;
      mrclamMeasNext_.bearing = b;
      return;
    }
  }
}

bool UTISASimulator::readNextDataMrclam() {
  dataBuffer_.valid = false;

  if (!mrclamOdomNext_.valid) {
    mrclamPrefetchOdom();
  }
  if (!mrclamMeasNext_.valid) {
    mrclamPrefetchMeas();
  }

  const bool haveO = mrclamOdomNext_.valid;
  const bool haveM = mrclamMeasNext_.valid;
  if (!haveO && !haveM) {
    return false;
  }

  bool takeOdom = false;
  if (haveO && haveM) {
    takeOdom = mrclamOdomNext_.t <= mrclamMeasNext_.t;
  } else if (haveO) {
    takeOdom = true;
  }

  if (takeOdom) {
    dataBuffer_.time = hasTimeOrigin_ ? (mrclamOdomNext_.t - timeOrigin_) : mrclamOdomNext_.t;
    dataBuffer_.type = DataMsgType::Odom;
    dataBuffer_.odomVel = Eigen::Vector3d(mrclamOdomNext_.v, 0.0, mrclamOdomNext_.omega);
    dataBuffer_.information = odomInformation_;
    mrclamOdomNext_.valid = false;
    mrclamPrefetchOdom();
    dataBuffer_.valid = true;
    return true;
  }

  const double t_abs = mrclamMeasNext_.t;
  const double t = hasTimeOrigin_ ? (t_abs - timeOrigin_) : t_abs;
  const int barcode = mrclamMeasNext_.barcode;
  const double range = mrclamMeasNext_.range;
  const double bearing = mrclamMeasNext_.bearing;
  mrclamMeasNext_.valid = false;
  mrclamPrefetchMeas();

  const auto it = barcodeToSubject_.find(barcode);
  if (it == barcodeToSubject_.end()) {
    return readNextDataMrclam();
  }
  const int subject = it->second;
  if (robotSubjectId_ >= 0 && subject == robotSubjectId_) {
    return readNextDataMrclam();
  }

  const Eigen::Matrix2d info2 = rangeBearingInformation_;

  dataBuffer_.time = t;
  if (subject >= 6 && subject <= 20) {
    dataBuffer_.type = DataMsgType::LmObs;
    dataBuffer_.landmarkId = subject;
    dataBuffer_.range = range;
    dataBuffer_.bearing = bearing;
    dataBuffer_.rbInformation = info2;
  } else if (subject >= 1 && subject <= 5) {
    dataBuffer_.type = DataMsgType::RelPos;
    dataBuffer_.targetRobotId = std::to_string(subject);
    dataBuffer_.range = range;
    dataBuffer_.bearing = bearing;
    dataBuffer_.rbInformation = info2;
  } else {
    return readNextDataMrclam();
  }

  dataBuffer_.valid = true;
  return true;
}

bool UTISASimulator::readNextData() {
  return readNextDataMrclam();
}

void UTISASimulator::initializeMrclam() {
  gtStream_.clear();
  gtStream_.seekg(0);

  SE2 pose;
  Eigen::Matrix3d info = Eigen::Matrix3d::Identity() * 1e4;

  std::string line;
  while (std::getline(gtStream_, line)) {
    sanitizeLine(line);
    if (isCommentOrEmpty(line)) continue;
    std::istringstream iss(line);
    double t = 0.0;
    double px = 0.0;
    double py = 0.0;
    double th = 0.0;
    if (iss >> t >> px >> py >> th) {
      timeOrigin_ = t;
      hasTimeOrigin_ = true;
      pose = makeSE2FromPosYaw(px, py, th);
      break;
    }
  }

  x_ = pose;
  currentTime_ = 0.0;

  // Use a fixed first pose anchor for MR.CLAM runs.
  // This avoids adding an explicit pose-prior edge that depends on cache resolution.
  auto initEvt = std::make_shared<UTISAInitEvent>(0.0, true, pose, info);
  pushEvent(initEvt);

  storeStepResults();
}

void UTISASimulator::initialize() {
  initializeMrclam();
}

void UTISASimulator::start() {
  carryOnRunning_ = true;
  initialized_ = false;
  currentTime_ = 0.0;
  hasTimeOrigin_ = false;
  timeOrigin_ = 0.0;

  eventQueue_.clear();
  nextEventTieOrder_ = 0;
  timeStore_.clear();
  xTrueStore_.clear();

  x_ = SE2();

  if (odomStream_.is_open()) {
    odomStream_.clear();
    odomStream_.seekg(0);
  }
  if (measStream_.is_open()) {
    measStream_.clear();
    measStream_.seekg(0);
  }
  mrclamOdomNext_.valid = false;
  mrclamMeasNext_.valid = false;
  if (gtStream_.is_open()) {
    gtStream_.clear();
    gtStream_.seekg(0);
  }

  initialize();

  mrclamPrefetchOdom();
  mrclamPrefetchMeas();

  gtHasMore_ = readNextGT();
  dataHasMore_ = readNextData();

  initialized_ = true;
}

void UTISASimulator::step() {
  step(0.05);
}

void UTISASimulator::step(double dt) {
  if (!carryOnRunning_) {
    return;
  }

  const double targetTime = currentTime_ + dt;
  if (durationLimitSec_ >= 0.0) {
    currentTime_ = std::min(targetTime, durationLimitSec_);
  } else {
    currentTime_ = targetTime;
  }

  while (gtHasMore_ && gtBuffer_.valid && gtBuffer_.time <= currentTime_) {
    x_ = gtBuffer_.pose;
    storeStepResults();

    gtHasMore_ = readNextGT();
  }

  while (dataHasMore_ && dataBuffer_.valid && dataBuffer_.time <= currentTime_) {
    if (dataBuffer_.type == DataMsgType::Odom) {
      auto evt = std::make_shared<UTISAOdomEvent>(dataBuffer_.time, dataBuffer_.odomVel,
                                                    dataBuffer_.information);
      pushEvent(evt);
    } else if (dataBuffer_.type == DataMsgType::RelPos) {
      auto evt = std::make_shared<UTISAObsEvent>(dataBuffer_.time, dataBuffer_.targetRobotId,
                                                  dataBuffer_.range, dataBuffer_.bearing,
                                                  dataBuffer_.rbInformation);
      pushEvent(evt);
    } else if (dataBuffer_.type == DataMsgType::LmObs) {
      auto evt = std::make_shared<UTISALmObsEvent>(dataBuffer_.time, dataBuffer_.landmarkId,
                                                    dataBuffer_.range, dataBuffer_.bearing,
                                                    dataBuffer_.rbInformation);
      pushEvent(evt);
    }

    dataHasMore_ = readNextData();
  }

  if (!gtHasMore_ && !dataHasMore_) {
    carryOnRunning_ = false;
  }
  if (durationLimitSec_ >= 0.0 && currentTime_ >= durationLimitSec_) {
    carryOnRunning_ = false;
  }
}

void UTISASimulator::stop() {
  carryOnRunning_ = false;
}

void UTISASimulator::storeStepResults() {
  timeStore_.push_back(currentTime_);
  xTrueStore_.push_back(x_);
}

}  // namespace multibotsim
}  // namespace tutorial
}  // namespace g2o
