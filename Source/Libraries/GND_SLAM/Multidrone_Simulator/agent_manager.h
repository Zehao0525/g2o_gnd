#pragma once

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "drone_slam_system.h"
#include "data_based_simulation.h"
#include "messages.hpp"

namespace g2o {
namespace tutorial {
namespace multibotsim {

class AgentManager {
public:
  AgentManager( const std::string& config_path,
                const std::string& log_path,
                const std::string& slam_config_path);
  ~AgentManager();

  void platformEstimate(Eigen::Isometry3d& x, Eigen::Matrix<double,6,6>& P);

  /**
   * @brief Initialize and start the SLAM system.
   */
  void start();

  /**
   * @brief stop the SLAM system and finalize result accumulation
   */
  void stop();
  void dumpPreOptTrajectories();

  /**
   * @brief trigger a request for data to line up observations
   */
  void step(); 

  /**
   * @brief trigger a request for data to line up observations
   */
  void step(double dt);

  /**
   * @brief Return true if any simulation still has data to process (should keep stepping).
   */
  bool keepRunning() const;

  /// Set topology from explicit list of bidirectional edges
  void setTopology(const std::vector<std::pair<std::string, std::string>>& topology);

  /// Load topology from JSON file (keys = bot ids, values = array of neighbor ids)
  void setTopologyJson(const std::string& topology_path);

  /// Query the bots in the simulation for poses (you said we can leave impl for now).
  void getPoses(std::vector<std::vector<double>>& poses);

  /**
   * @brief Save trajectories for all robots to files
   * @param output_dir Directory where trajectory files will be saved
   * @param format Format to use: "tum" for TUM format (default), "g2o" for g2o format
   */
  void saveTrajectories(const std::string& output_dir, const std::string& format = "tum") const;

/// Perform topology-based communication between drones.
  /// Broadcast queries, aggregate queries per neighbor set, answer them locally,
  /// then deliver the responses back to connected drones.
  void performCommunication();

public:
  // Public so you can inspect/use it if you want
  std::vector<std::pair<std::string, std::string>> topology_;

  // Accessor for robot IDs
  const std::vector<std::string>& getRobotIds() const { return robotIds_; }

  // Accessor for SLAM systems (for visualization)
  MultiDroneSLAMSystem* getSlamSystem(size_t index) {
    if (index >= slamSystems_.size()) {
      return nullptr;
    }
    return slamSystems_[index].get();
  }

  // Accessor for simulations (for visualization)
  DataBasedSimulation* getSimulation(size_t index) {
    if (index >= sims_.size()) {
      return nullptr;
    }
    return sims_[index].get();
  }

protected:
  // This mapping maps the vertex of the platform to platform vertices
  // This also maps the observed vertex to its location in "observations_"
  std::map<int, int> VertexIdMap_;
  int fileVertexId_ = 0;

  // List of all robot IDs (strings from bot_ids.txt)
  std::vector<std::string> robotIds_;

  // One sim and one SLAM system per robot (unique_ptr avoids move/copy requirements)
  std::vector<std::unique_ptr<DataBasedSimulation>> sims_;
  std::vector<std::unique_ptr<MultiDroneSLAMSystem>> slamSystems_;

  // Optional: aggregated poses, not really used yet
  std::vector<std::vector<double>> poses_;

  // Time step from config (defaults to 0.05)
  double dt_ = 0.05;

  // Communication config (from experiment config JSON)
  bool communicationEnabled_ = true;
  double communicationFrequencyHz_ = 0.0;
  int communicationPeriodSteps_ = 1;   // run every N steps when enabled
  int communicationEndRounds_ = 1;     // run this many rounds right before shutdown
  int stepCount_ = 0;

  std::string outputPath_ = "test_results/multidrone";
  bool debugOutputs_ = false;
  bool verbose_;

};

} // namespace multibotsim
} // namespace tutorial
} // namespace g2o
