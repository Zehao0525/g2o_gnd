#include "agent_manager.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <iomanip>
#include <map>
#include <set>

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

AgentManager::AgentManager(const std::string& config_path,
                           const std::string& log_path,
                           const std::string& slam_config_path) : verbose_(false)
{
  // 1. Parse config JSON (for dt, possibly bots info)
  {
    std::ifstream in(config_path);
    if (!in) {
      throw std::runtime_error("AgentManager: cannot open config file: " + config_path);
    }
    json j;
    in >> j;

    if (j.contains("dt")) {
      dt_ = j["dt"].get<double>();
    }
    if (j.contains("verbose")) {
      verbose_ = j["verbose"].get<bool>();
      if (verbose_) {
        std::cout << "AgentManager: verbose = true" << std::endl;
      }
    }

    // Communication config (defaults: enabled=true, frequency=0 -> every step)
    if (j.contains("communication") && j["communication"].is_object()) {
      const auto& c = j["communication"];
      if (c.contains("enabled")) {
        communicationEnabled_ = c["enabled"].get<bool>();
      }
      if (c.contains("frequency")) {
        communicationFrequencyHz_ = c["frequency"].get<double>();
      }
    }

    if (communicationEnabled_) {
      if (communicationFrequencyHz_ > 0.0 && dt_ > 0.0) {
        // frequency is in Hz; dt_ is seconds per step
        const double stepsPerComm = 1.0 / (communicationFrequencyHz_ * dt_);
        communicationPeriodSteps_ = std::max(1, static_cast<int>(std::lround(stepsPerComm)));
      } else {
        communicationPeriodSteps_ = 1;  // every step
      }
    }

    if (verbose_) {
      std::cout << "AgentManager: communication.enabled=" << (communicationEnabled_ ? "true" : "false")
                << " frequency=" << communicationFrequencyHz_
                << " periodSteps=" << communicationPeriodSteps_ << std::endl;
    }
  }

  // 2. Read robot IDs from bot_ids.txt in the log directory
  {
    std::string bot_ids_file = log_path + "/bot_ids.txt";
    std::ifstream in(bot_ids_file);
    if (!in) {
      throw std::runtime_error("AgentManager: cannot open bot_ids.txt at: " + bot_ids_file);
    }

    std::string line;
    while (std::getline(in, line)) {
      auto id = trim(line);
      if (!id.empty()) {
        robotIds_.push_back(id);
      }
    }
  }

  // 3. For each robot id, create a DataBasedSimulation and a MultiDroneSLAMSystem
  sims_.reserve(robotIds_.size());
  slamSystems_.reserve(robotIds_.size());

  for (const auto& id : robotIds_) {
    std::string gt_path   = log_path + "/gt_log_"  + id + ".txt";
    std::string data_path = log_path + "/msg_log_" + id + ".txt";

    sims_.push_back(std::make_unique<DataBasedSimulation>(id, data_path, gt_path));
    slamSystems_.push_back(std::make_unique<MultiDroneSLAMSystem>(id, slam_config_path));

    if (verbose_) {
      std::cout << "AgentManager: created simulation and SlamSystem for robot " << id << std::endl;
    }
  }
}

AgentManager::~AgentManager() = default;

void AgentManager::platformEstimate(Eigen::Isometry3d& x,
                                    Eigen::Matrix<double,6,6>& P)
{
  // You didn't specify how to combine multiple drones' platform estimates.
  // For now, ask the first SLAM system (if any) for its platform estimate,
  // or leave as identity / zero if not available.

  if (slamSystems_.empty()) {
    x.setIdentity();
    P.setZero();
    return;
  }

  // If MultiDroneSLAMSystem has its own platformEstimate method, prefer that:
  // slamSystems_.front().platformEstimate(x, P);
  // For now, just set identity as a placeholder:
  x.setIdentity();
  P.setIdentity();
}

/**
 * @brief Initialize and start the SLAM system.
 */
void AgentManager::start() {
  for (auto& sim : sims_) {
    sim->start();
  }
  for (auto& slam : slamSystems_) {
    slam->start();
  }
  if (verbose_) {
    std::cout << "AgentManager: started simulations and SlamSystems" << std::endl;
  }
}

/**
 * @brief stop the SLAM system and finalize result accumulation
 */
void AgentManager::stop() {
  performCommunication();
  for (auto& sim : sims_) {
    sim->stop();
  }
  for (auto& slam : slamSystems_) {
    slam->stop();
  }
  if (verbose_) {
    std::cout << "AgentManager: stopped simulations and SlamSystems" << std::endl;
  }
}

/**
 * @brief trigger a request for data to line up observations
 */
void AgentManager::step() {
  step(dt_);
}

bool AgentManager::keepRunning() const {
  for (const auto& sim : sims_) {
    if (sim && sim->keepRunning()) {
      return true;
    }
  }
  return false;
}

void AgentManager::step(double dt) {
  // High-level intent:
  //  - advance each DataBasedSimulation by dt
  //  - collect generated events/messages
  //  - feed events into corresponding MultiDroneSLAMSystem

  for (std::size_t i = 0; i < sims_.size(); ++i) {
    auto* sim  = sims_[i].get();
    auto* slam = slamSystems_[i].get();

    sim->step(dt);
    auto events = sim->acquireEvents();
    slam->processEvents(events);
  }
  if (verbose_) {
    std::cout << "AgentManager: stepped simulations and SlamSystems" << std::endl;
  }

  // After all drones have processed their own data, perform communication if enabled/scheduled.
  ++stepCount_;
  if (communicationEnabled_ &&
      (communicationPeriodSteps_ <= 1 || (stepCount_ % communicationPeriodSteps_ == 0))) {
    performCommunication();
  }
}

void AgentManager::setTopology(const std::vector<std::pair<std::string, std::string>>& topology) {
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

void AgentManager::setTopologyJson(const std::string& topology_path) {
  std::ifstream in(topology_path);
  if (!in) {
    throw std::runtime_error("AgentManager::setTopologyJson: cannot open topology file: " +
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

void AgentManager::getPoses(std::vector<std::vector<double>>& poses) {
  // You said we can leave this for now; stub implementation.
  // In future: query each slamSystems_[i] for its current pose and fill poses.
  poses = poses_;  // or leave empty
}

//NOTE: Worth taking another look
void AgentManager::saveTrajectories(const std::string& output_dir, const std::string& format) const {
  // Ensure output directory ends with /
  std::string dir = output_dir;
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

void AgentManager::performCommunication() {
  // Build adjacency from topology_ (already made bidirectional by setTopology()).
  std::map<std::string, std::set<std::string>> neighbors;
  for (const auto& id : robotIds_) {
    neighbors[id];  // ensure key exists
  }
  for (const auto& e : topology_) {
    neighbors[e.first].insert(e.second);
  }

  // 1) Each drone broadcasts its query message
  std::map<std::string, DSMessage> broadcasts;
  for (size_t i = 0; i < slamSystems_.size() && i < robotIds_.size(); ++i) {
    broadcasts[robotIds_[i]] = slamSystems_[i]->broadcastDSMessage();
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

    // Include self broadcast
    if (auto it = broadcasts.find(receiverId); it != broadcasts.end()) {
      const auto& entries = it->second.poseEntries;
      aggregated.insert(aggregated.end(), entries.begin(), entries.end());
    }

    // Union in all neighbor broadcasts
    for (const auto& n : neighbors[receiverId]) {
      auto it = broadcasts.find(n);
      if (it == broadcasts.end()) continue;
      const auto& entries = it->second.poseEntries;
      aggregated.insert(aggregated.end(), entries.begin(), entries.end());
    }

    DSMessage reqMsg(receiverId, /*loaded=*/false, std::move(aggregated));
    DSMessage response = receiver->handleObservationSyncRequest(reqMsg);

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
