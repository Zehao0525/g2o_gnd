#include "utisa_agent_manager.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <iomanip>
#include <map>
#include <set>
#include <limits>

#include <nlohmann/json.hpp>
#include "events.h"

namespace g2o {
namespace tutorial {
namespace multibotsim {

using json = nlohmann::json;

static std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

/// MR.CLAM logs use files `Robot{N}_*.dat`. Dataset subject indices are 1..5 for robots.
static int parseMrclamSubjectIndex(const std::string& raw) {
  const std::string t = trim(raw);
  if (t.empty()) {
    throw std::runtime_error("UTISAAgentManager: empty robot id");
  }
  try {
    if (t.size() >= 5 && t.compare(0, 5, "Robot") == 0) {
      return std::stoi(t.substr(5));
    }
    return std::stoi(t);
  } catch (const std::exception& e) {
    throw std::runtime_error("UTISAAgentManager: invalid robot id '" + raw + "': " + e.what());
  }
}

/// SLAM / sync use the same string ids as MR.CLAM measurements: "1" .. "5", not "Robot1".
static std::string canonicalMrclamRobotId(const std::string& raw) {
  return std::to_string(parseMrclamSubjectIndex(raw));
}

static std::string mrclamRobotFilePrefix(const std::string& canonicalId) {
  return std::string("Robot") + canonicalId;
}

static Eigen::Matrix3d parseMat3(const json& j, const std::string& key, const Eigen::Matrix3d& fallback) {
  if (!j.contains(key)) return fallback;
  const auto& m = j.at(key);
  if (!m.is_array() || m.size() != 3) {
    throw std::runtime_error("UTISAAgentManager: '" + key + "' must be a 3x3 array");
  }
  Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
  for (int r = 0; r < 3; ++r) {
    if (!m[r].is_array() || m[r].size() != 3) {
      throw std::runtime_error("UTISAAgentManager: '" + key + "' must be a 3x3 array");
    }
    for (int c = 0; c < 3; ++c) {
      out(r, c) = m[r][c].get<double>();
    }
  }
  return out;
}

static Eigen::Matrix2d parseMat2(const json& j, const std::string& key, const Eigen::Matrix2d& fallback) {
  if (!j.contains(key)) return fallback;
  const auto& m = j.at(key);
  if (!m.is_array() || m.size() != 2) {
    throw std::runtime_error("UTISAAgentManager: '" + key + "' must be a 2x2 array");
  }
  Eigen::Matrix2d out = Eigen::Matrix2d::Zero();
  for (int r = 0; r < 2; ++r) {
    if (!m[r].is_array() || m[r].size() != 2) {
      throw std::runtime_error("UTISAAgentManager: '" + key + "' must be a 2x2 array");
    }
    for (int c = 0; c < 2; ++c) {
      out(r, c) = m[r][c].get<double>();
    }
  }
  return out;
}

UTISAAgentManager::UTISAAgentManager(const std::string& config_path,
                           const std::string& log_path,
                           const std::string& slam_config_path) : verbose_(false)
{
  json j;
  {
    std::ifstream in(config_path);
    if (!in) {
      throw std::runtime_error("UTISAAgentManager: cannot open config file: " + config_path);
    }
    in >> j;
  }

  // 1. Parse config JSON (dt, output, communication, robot list)
  if (j.contains("dt")) {
    dt_ = j["dt"].get<double>();
  }
  if (j.contains("verbose")) {
    verbose_ = j["verbose"].get<bool>();
    if (verbose_) {
      std::cout << "UTISAAgentManager: verbose = true" << std::endl;
    }
  }
  if (j.contains("output_path")) {
    outputPath_ = j["output_path"].get<std::string>();
  }
  const Eigen::Matrix3d odomCov =
      parseMat3(j, "odom_covariance", (Eigen::Vector3d(0.02 * 0.02, 1.0, 0.05 * 0.05)).asDiagonal());
  const Eigen::Matrix2d rangeBearingCov =
      parseMat2(j, "range_bearing_covariance", (Eigen::Vector2d(0.15 * 0.15, 0.08 * 0.08)).asDiagonal());
  const Eigen::Matrix3d odomInfo = odomCov.inverse();
  const Eigen::Matrix2d rangeBearingInfo = rangeBearingCov.inverse();

  if (!odomInfo.allFinite() || !rangeBearingInfo.allFinite()) {
    throw std::runtime_error(
        "UTISAAgentManager: covariance matrices are singular/invalid; cannot invert to information.");
  }
  const double durationSec = j.value("Duration", j.value("duration", -1.0));
  if (j.contains("debug_outputs")) {
    debugOutputs_ = j["debug_outputs"].get<bool>();
  }

  lmQueryEnabled_ = j.value("lm_query", true);

  if (j.contains("communication") && j["communication"].is_object()) {
    const auto& c = j["communication"];
    if (c.contains("enabled")) {
      communicationEnabled_ = c["enabled"].get<bool>();
    }
    if (c.contains("frequency_hz")) {
      communicationFrequencyHz_ = c["frequency_hz"].get<double>();
    } else if (c.contains("frequency")) {
      communicationFrequencyHz_ = c["frequency"].get<double>();
    }
    if (c.contains("end_rounds")) {
      communicationEndRounds_ = std::max(0, c["end_rounds"].get<int>());
    }
  }

  if (communicationEnabled_) {
    if (communicationFrequencyHz_ > 0.0 && dt_ > 0.0) {
      const double stepsPerComm = 1.0 / (communicationFrequencyHz_ * dt_);
      if (!std::isfinite(stepsPerComm) || stepsPerComm <= 1.0) {
        communicationPeriodSteps_ = 1;
      } else if (stepsPerComm >= static_cast<double>(std::numeric_limits<int>::max())) {
        communicationPeriodSteps_ = std::numeric_limits<int>::max();
      } else {
        communicationPeriodSteps_ = std::max(1, static_cast<int>(std::lround(stepsPerComm)));
      }
    } else {
      communicationPeriodSteps_ = 1;
    }
  }

  if (true) {
    std::cout << "UTISAAgentManager: communication.enabled=" << (communicationEnabled_ ? "true" : "false")
              << " frequency=";
    {
      const std::ios::fmtflags old_flags = std::cout.flags();
      const std::streamsize old_prec = std::cout.precision();
      std::cout << std::scientific << std::setprecision(6) << communicationFrequencyHz_;
      std::cout.flags(old_flags);
      std::cout.precision(old_prec);
    }
    std::cout << " periodSteps=" << communicationPeriodSteps_
              << " dt=" << dt_ << std::endl;
  }

  // 2. Robot ids: prefer RobotIds / robot_ids in experiment config (canonical "1".."5" for MR.CLAM).
  const json* id_array = nullptr;
  if (j.contains("RobotIds") && j["RobotIds"].is_array() && !j["RobotIds"].empty()) {
    id_array = &j["RobotIds"];
  } else if (j.contains("robot_ids") && j["robot_ids"].is_array() && !j["robot_ids"].empty()) {
    id_array = &j["robot_ids"];
  }

  if (id_array) {
    for (const auto& el : *id_array) {
      if (!el.is_string()) {
        continue;
      }
      robotIds_.push_back(canonicalMrclamRobotId(el.get<std::string>()));
    }
    if (verbose_) {
      std::cout << "UTISAAgentManager: robot ids from config:";
      for (const auto& id : robotIds_) {
        std::cout << " " << id;
      }
      std::cout << std::endl;
    }
  } else {
    const std::string bot_ids_file = log_path + "/bot_ids.txt";
    std::ifstream bin(bot_ids_file);
    if (!bin) {
      throw std::runtime_error(
          "UTISAAgentManager: set RobotIds in experiment config or provide bot_ids.txt at: " + bot_ids_file);
    }
    std::string line;
    while (std::getline(bin, line)) {
      const auto id = trim(line);
      if (!id.empty()) {
        robotIds_.push_back(canonicalMrclamRobotId(id));
      }
    }
  }

  if (robotIds_.empty()) {
    throw std::runtime_error("UTISAAgentManager: no robot ids (RobotIds in config or bot_ids.txt)");
  }

  // 3. For each robot id, create a UTISASimulator and a UTISASlamSystem
  sims_.reserve(robotIds_.size());
  slamSystems_.reserve(robotIds_.size());

  namespace fs = std::filesystem;
  const fs::path log_root(log_path);

  for (const auto& id : robotIds_) {
    const std::string prefix = mrclamRobotFilePrefix(id);
    const fs::path mrclam_odom = log_root / (prefix + "_Odometry.dat");
    const fs::path mrclam_meas = log_root / (prefix + "_Measurement.dat");
    const fs::path mrclam_gt = log_root / (prefix + "_Groundtruth.dat");
    const fs::path mrclam_barcodes = log_root / "Barcodes.dat";
    if (!fs::exists(mrclam_odom) || !fs::exists(mrclam_meas) || !fs::exists(mrclam_gt) ||
        !fs::exists(mrclam_barcodes)) {
      throw std::runtime_error("UTISAAgentManager: MR.CLAM files missing for robot " + id +
                               " under dataset path: " + log_root.string());
    }
    sims_.push_back(std::make_unique<UTISASimulator>(id, mrclam_odom.string(), mrclam_meas.string(),
                                                     mrclam_gt.string(), mrclam_barcodes.string()));
    sims_.back()->setOdomInformation(odomInfo);
    sims_.back()->setRangeBearingInformation(rangeBearingInfo);
    if (durationSec >= 0.0) {
      sims_.back()->setDurationLimit(durationSec);
    }
    slamSystems_.push_back(std::make_unique<UTISASlamSystem>(id, slam_config_path));
    slamSystems_.back()->setLmQueryEnabled(lmQueryEnabled_);
    slamSystems_.back()->setPreOptTrajectoryDumpEnabled(debugOutputs_);
    slamSystems_.back()->setPreOptTrajectoryOutputDir(outputPath_ + "/pre_opt_trajectories");

    if (verbose_) {
      std::cout << "UTISAAgentManager: created simulation and SlamSystem for robot " << id << std::endl;
    }
  }
}

UTISAAgentManager::~UTISAAgentManager() = default;

void UTISAAgentManager::platformEstimate(Eigen::Vector3d& x, Eigen::Matrix3d& P) {
  if (slamSystems_.empty()) {
    x.setZero();
    P.setZero();
    return;
  }
  slamSystems_.front()->platformEstimate(x, P);
}

/**
 * @brief Initialize and start the SLAM system.
 */
void UTISAAgentManager::start() {
  {
    std::filesystem::path pre_root = std::filesystem::path(outputPath_) / "pre_opt_trajectories";
    std::error_code ec;
    std::filesystem::remove_all(pre_root, ec);
    if (ec && verbose_) {
      std::cerr << "UTISAAgentManager: warning clearing pre_opt_trajectories: " << ec.message() << "\n";
    }
    std::filesystem::create_directories(pre_root, ec);
    if (ec) {
      throw std::runtime_error(
          "UTISAAgentManager: cannot recreate pre_opt_trajectories at '" + pre_root.string() + "': " +
          ec.message());
    }
    UTISASlamSystem::resetPreOptTrajectoryBatchCounter();
  }

  for (auto& sim : sims_) {
    sim->start();
  }
  for (auto& slam : slamSystems_) {
    slam->start();
  }
  if (verbose_) {
    std::cout << "UTISAAgentManager: started simulations and SlamSystems" << std::endl;
  }
}

/**
 * @brief stop the SLAM system and finalize result accumulation
 */
void UTISAAgentManager::stop() {
  if (communicationEnabled_) {
    const int rounds = std::max(0, communicationEndRounds_);
    for (int r = 0; r < rounds; ++r) {
      performCommunication();
    }
  }
  for (auto& sim : sims_) {
    sim->stop();
  }
  for (auto& slam : slamSystems_) {
    slam->stop();
  }
  if (verbose_) {
    std::cout << "UTISAAgentManager: stopped simulations and SlamSystems" << std::endl;
  }
}

void UTISAAgentManager::dumpPreOptTrajectories() {
  if (!debugOutputs_) {
    return;
  }
  const int batch = UTISASlamSystem::takeNextPreOptTrajectoryBatchIndex();
  std::filesystem::path run_dir =
      std::filesystem::path(outputPath_) / "pre_opt_trajectories" / std::to_string(batch);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(run_dir, mkdir_ec);
  if (mkdir_ec) {
    throw std::runtime_error(
        "Cannot create pre-opt run directory '" + run_dir.string() + "': " + mkdir_ec.message());
  }
  const std::string run_str = run_dir.string();
  for (auto& slam : slamSystems_) {
    slam->dumpPreOptTrajectory(run_str);
  }
}

/**
 * @brief trigger a request for data to line up observations
 */
void UTISAAgentManager::step() {
  step(dt_);
}

bool UTISAAgentManager::keepRunning() const {
  for (const auto& sim : sims_) {
    if (sim && sim->keepRunning()) {
      return true;
    }
  }
  return false;
}

void UTISAAgentManager::step(double dt) {
  // High-level intent:
  //  - advance each UTISASimulator by dt
  //  - collect generated events/messages
  //  - feed events into corresponding UTISASlamSystem

  for (std::size_t i = 0; i < sims_.size(); ++i) {
    auto* sim  = sims_[i].get();
    auto* slam = slamSystems_[i].get();

    sim->step(dt);
    auto events = sim->acquireEvents();
    slam->processEvents(events);
  }
  if (verbose_) {
    std::cout << "UTISAAgentManager: stepped simulations and SlamSystems" << std::endl;
  }

  // After all drones have processed their own data, perform communication if enabled/scheduled.
  ++stepCount_;
  if (communicationEnabled_ &&
      (communicationPeriodSteps_ <= 1 || (stepCount_ % communicationPeriodSteps_ == 0))) {
    performCommunication();
  }
}

void UTISAAgentManager::setTopology(const std::vector<std::pair<std::string, std::string>>& topology) {
  topology_.clear();
  topology_.reserve(topology.size() * 2);

  // Input topology edges are bidirectional, but we make that explicit:
  for (const auto& edge : topology) {
    const auto& a = edge.first;
    const auto& b = edge.second;

    // (a, b)
    topology_.push_back(edge);

    // (b, a) — avoid duplicate insertion if someone already gave both
    if (a != b &&
        std::find(topology_.begin(), topology_.end(), std::make_pair(b, a)) == topology_.end()) {
      topology_.emplace_back(b, a);
    }
  }
}

void UTISAAgentManager::setTopologyJson(const std::string& topology_path) {
  std::ifstream in(topology_path);
  if (!in) {
    throw std::runtime_error("UTISAAgentManager::setTopologyJson: cannot open topology file: " +
                             topology_path);
  }

  json j;
  in >> j;

  std::vector<std::pair<std::string, std::string>> edges;

  // JSON keys are bot ids, each maps to an array of neighbor bot ids
  // e.g. {
  //   "0": ["1", "2"],
  //   "1": ["0"],
  //   "2": ["0"]
  // }
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string src = it.key();
    const auto& neighbors = it.value();
    if (!neighbors.is_array()) {
      continue;
    }
    for (const auto& n : neighbors) {
      if (!n.is_string()) continue;
      std::string dst = n.get<std::string>();
      edges.emplace_back(src, dst);
    }
  }

  // Now normalize / make bidirectional with setTopology
  setTopology(edges);
}

void UTISAAgentManager::getPoses(std::vector<std::vector<double>>& poses) {
  // You said we can leave this for now; stub implementation.
  // In future: query each slamSystems_[i] for its current pose and fill poses.
  poses = poses_;  // or leave empty
}

//NOTE: Worth taking another look
void UTISAAgentManager::saveTrajectories(const std::string& output_dir, const std::string& format) const {
  std::filesystem::path dir_path(output_dir);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(dir_path, mkdir_ec);
  if (mkdir_ec) {
    throw std::runtime_error(
        "Cannot create trajectory output directory '" + output_dir + "': " + mkdir_ec.message());
  }

  // Ensure output directory string ends with / for concatenation below
  std::string dir = dir_path.string();
  if (!dir.empty() && dir.back() != '/') {
    dir += "/";
  }

  std::cout << "Saving trajectories to directory: " << dir << std::endl;
  std::cout << "Format: " << format << std::endl;

  for (size_t i = 0; i < slamSystems_.size() && i < robotIds_.size(); ++i) {
    const std::string& robotId = robotIds_[i];
    std::string filename;

    if (format == "tum") {
      filename = dir + "trajectory_" + robotId + ".txt";
      slamSystems_[i]->saveTrajectoryTUM(filename);
    } else if (format == "g2o") {
      filename = dir + "trajectory_" + robotId + ".g2o";
      std::cerr << "Warning: g2o format not yet implemented, saving as TUM instead" << std::endl;
      filename = dir + "trajectory_" + robotId + ".txt";
      slamSystems_[i]->saveTrajectoryTUM(filename);
    } else {
      throw std::runtime_error("Unknown trajectory format: " + format + ". Supported: tum, g2o");
    }
  }

  std::cout << "Saved trajectories for " << slamSystems_.size() << " robots" << std::endl;
}

void UTISAAgentManager::saveLandmarks(const std::string& output_dir) const {
  std::filesystem::path dir_path(output_dir);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(dir_path, mkdir_ec);
  if (mkdir_ec) {
    throw std::runtime_error(
        "Cannot create landmarks output directory '" + output_dir + "': " + mkdir_ec.message());
  }

  std::string dir = dir_path.string();
  if (!dir.empty() && dir.back() != '/') {
    dir += "/";
  }

  for (size_t i = 0; i < slamSystems_.size() && i < robotIds_.size(); ++i) {
    const std::string& robotId = robotIds_[i];
    const std::string filename = dir + "landmarks_" + robotId + ".txt";
    slamSystems_[i]->saveLandmarksXY(filename);
  }
  std::cout << "Saved estimated landmarks for " << slamSystems_.size() << " robots" << std::endl;
}

void UTISAAgentManager::performCommunication() {
  // Build adjacency from topology_ (already made bidirectional by setTopology()).
  std::map<std::string, std::set<std::string>> neighbors;
  for (const auto& id : robotIds_) {
    neighbors[id];  // ensure key exists
  }
  for (const auto& e : topology_) {
    neighbors[e.first].insert(e.second);
  }

  // 1) Each drone broadcasts its query message
  std::map<std::string, UTSIAMessage> broadcasts;
  for (size_t i = 0; i < slamSystems_.size() && i < robotIds_.size(); ++i) {
    broadcasts[robotIds_[i]] = slamSystems_[i]->broadcastUTSIAMessage();
  }

  // Helper: id -> index
  std::map<std::string, size_t> idToIndex;
  for (size_t i = 0; i < robotIds_.size(); ++i) {
    idToIndex[robotIds_[i]] = i;
  }

  // 2) For each drone, aggregate neighbor queries, answer them locally,
  //    then deliver the response to all connected drones.
  for (size_t i = 0; i < slamSystems_.size() && i < robotIds_.size(); ++i) {
    const std::string& receiverId = robotIds_[i];
    auto* receiver = slamSystems_[i].get();

    std::vector<PoseStampEntry> aggregated;
    bool lm_query = false;

    // Include self broadcast
    if (auto it = broadcasts.find(receiverId); it != broadcasts.end()) {
      const auto& entries = it->second.poseEntries;
      aggregated.insert(aggregated.end(), entries.begin(), entries.end());
      lm_query = lm_query || it->second.lm_query;
    }

    // Union in all neighbor broadcasts
    for (const auto& n : neighbors[receiverId]) {
      auto it = broadcasts.find(n);
      if (it == broadcasts.end()) continue;
      const auto& entries = it->second.poseEntries;
      aggregated.insert(aggregated.end(), entries.begin(), entries.end());
      lm_query = lm_query || it->second.lm_query;
    }

    UTSIAMessage reqMsg(receiverId, /*loaded=*/false, lm_query, std::move(aggregated));
    UTSIAMessage response = receiver->handleObservationSyncRequest(reqMsg);

    // Deliver response to self and neighbors; recipients filter by PoseStampEntry.sourceId.
    receiver->handleObservationSyncResponse(response);
    for (const auto& n : neighbors[receiverId]) {
      auto itIdx = idToIndex.find(n);
      if (itIdx == idToIndex.end()) continue;
      const size_t j = itIdx->second;
      if (j < slamSystems_.size()) {
        slamSystems_[j]->handleObservationSyncResponse(response);
      }
    }
  }
}

} // namespace multibotsim
} // namespace tutorial
} // namespace g2o
