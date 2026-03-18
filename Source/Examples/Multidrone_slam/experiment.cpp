#include <fstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>

#include <nlohmann/json.hpp>
#include "agent_manager.h"
#include "data_based_simulation.h"
#include "drone_slam_system.h"
#include "messages.hpp"
#include "events.h"
#include "view_manager.h"
#include "multi_drone_slam_system_view.h"
#include "data_based_simulation_view.h"

using namespace g2o::tutorial::multibotsim;
using namespace g2o::tutorial::viz;
using json = nlohmann::json;

// Main entry point
int main(int argc, char* argv[]) {
  try {
    const std::string base_config_path = "Source/Examples/Multidrone_slam/config/experiment_base_config.json";

    std::string slam_config_path;
    std::string input_path;
    std::string topology_path = "Source/Examples/Multidrone_slam/config/topology.json";

    // Read in base config data
    std::ifstream in(base_config_path);
    if (!in) {
      throw std::runtime_error("cannot open base config file: " + base_config_path);
    }
    json j;
    in >> j;

    // Extract config_path (for dt and other experiment settings)
    // If not in base config, use a default or create a minimal config
    std::string config_path = base_config_path;  // Use base config as experiment config
    if (j.contains("config_path")) {
      config_path = j["config_path"].get<std::string>();
    }

    // Extract slam_config_path
    if (j.contains("slam_config_path")) {
      slam_config_path = j["slam_config_path"].get<std::string>();
    } else {
      throw std::runtime_error("base_config_path has no slam_config_path field");
    }

    // Extract input_path (log_path)
    if (j.contains("input_path")) {
      input_path = j["input_path"].get<std::string>();
    } else {
      throw std::runtime_error("base_config_path has no input_path field");
    }

    // Extract topology_path if provided
    if (j.contains("topology_path")) {
      topology_path = j["topology_path"].get<std::string>();
    }

    // Extract verbose
    bool verbose = false;
    if (j.contains("verbose")) {
      verbose = j["verbose"].get<bool>();
      if (verbose) {
        std::cout << "Experiment: verbose = true" << std::endl;
      }
    }

    std::cout << "Initializing AgentManager..." << std::endl;
    std::cout << "  Config path: " << config_path << std::endl;
    std::cout << "  SLAM config path: " << slam_config_path << std::endl;
    std::cout << "  Input path: " << input_path << std::endl;
    std::cout << "  Topology path: " << topology_path << std::endl;

    // Create AgentManager
    // Constructor: AgentManager(config_path, log_path, slam_config_path)
    AgentManager manager(config_path, input_path, slam_config_path);

    // Set topology from JSON file
    std::cout << "Setting topology..." << std::endl;
    manager.setTopologyJson(topology_path);

    // Setup visualization and read view config (step_pause, etc.)
    std::string view_config_path = "Source/Examples/Multidrone_slam/config/view_config.json";
    std::ifstream view_in(view_config_path);
    if (!view_in) {
      throw std::runtime_error("cannot open view config file: " + view_config_path);
    }
    json view_j;
    view_in >> view_j;
    double step_pause_sec = view_j.value("step_pause", 0.0);

    // Extract visualisation settings
    bool visualise_slam_path = view_j.value("visualise_slam_path", true);
    bool visualise_sim_path = view_j.value("visualise_sim_path", true);

    std::cout << "Setting up visualization..." << std::endl;
    ViewManager viewManager(view_config_path);
    if (step_pause_sec > 0.0) {
      std::cout << "  step_pause: " << step_pause_sec << " s" << std::endl;
    }

    // Create views for each robot: SLAM estimate and ground truth
    std::vector<std::shared_ptr<MultiDroneSLAMSystemView>> slamViews;
    std::vector<std::shared_ptr<DataBasedSimulationView>> gtViews;
    const auto& robotIds = manager.getRobotIds();

    // Color palette for different robots (SLAM estimate)
    std::vector<Eigen::Vector3f> robotEstColors = {
        Eigen::Vector3f(1.0f, 0.0f, 0.0f),  // Red
        Eigen::Vector3f(0.0f, 1.0f, 0.0f),  // Green
        Eigen::Vector3f(0.0f, 0.0f, 1.0f),  // Blue
        Eigen::Vector3f(1.0f, 1.0f, 0.0f),  // Yellow
        Eigen::Vector3f(1.0f, 0.0f, 1.0f),  // Magenta
        Eigen::Vector3f(0.0f, 1.0f, 1.0f)   // Cyan
    };

    // Color palette for different robots (SLAM estimate)
    std::vector<Eigen::Vector3f> robotGTColors = {
      Eigen::Vector3f(1.0f, 0.3f, 0.3f),  // Red
      Eigen::Vector3f(0.3f, 1.0f, 0.3f),  // Green
      Eigen::Vector3f(0.3f, 0.3f, 1.0f),  // Blue
      Eigen::Vector3f(1.0f, 1.0f, 0.3f),  // Yellow
      Eigen::Vector3f(1.0f, 0.3f, 1.0f),  // Magenta
      Eigen::Vector3f(0.3f, 1.0f, 1.0f)   // Cyan
  };

    // Ground truth: dimmer version of robot color so GT is visible but distinct
    auto gtColorFromRobot = [](const Eigen::Vector3f& c) {
        return Eigen::Vector3f(c[0] * 0.5f + 0.3f, c[1] * 0.5f + 0.3f, c[2] * 0.5f + 0.3f);
    };

    for (size_t i = 0; i < robotIds.size(); ++i) {
        MultiDroneSLAMSystem* slamSystem = manager.getSlamSystem(i);
        DataBasedSimulation* sim = manager.getSimulation(i);

        std::string bot_view_config = "Source/Examples/Multidrone_slam/config/bot" + robotIds[i] + "_view.json";

        if (slamSystem) {
            std::shared_ptr<MultiDroneSLAMSystemView> view;

            std::ifstream test_file(bot_view_config);
            if (test_file.good()) {
                view = std::make_shared<MultiDroneSLAMSystemView>(slamSystem, bot_view_config, visualise_slam_path);
            } else {
                Eigen::Vector3f color = robotEstColors[i % robotEstColors.size()];
                view = std::make_shared<MultiDroneSLAMSystemView>(slamSystem, color, visualise_slam_path);
            }

            viewManager.addView(view);
            slamViews.push_back(view);
            std::cout << "  Added SLAM view for robot " << robotIds[i] << std::endl;
        }

        if (sim) {
            Eigen::Vector3f gtColor = gtColorFromRobot(robotGTColors[i % robotGTColors.size()]);
            auto gtView = std::make_shared<DataBasedSimulationView>(sim, gtColor, visualise_sim_path);
            viewManager.addView(gtView);
            gtViews.push_back(gtView);
            std::cout << "  Added ground-truth view for robot " << robotIds[i] << std::endl;
        }
    }

    // Start visualization
    std::cout << "Starting visualization..." << std::endl;
    viewManager.start();

    // Start the simulation and SLAM systems
    std::cout << "Starting simulation..." << std::endl;
    manager.start();

    // Main simulation loop: run until all sims have no more data or max_steps reached
    std::cout << "Running simulation..." << std::endl;
    int step_count = 0;
    const int max_steps = 100000;

    while (step_count < max_steps && manager.keepRunning()) {
      if (verbose) {
        std::cout << "Experiment: step " << step_count << std::endl;
      }
      manager.step();
      step_count++;

      if (step_pause_sec > 0.0) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(step_pause_sec));
      }

      if (step_count % 10 == 0) {
        if (verbose) {
          std::cout << "Experiment: updating views" << std::endl;
        }
        for (auto& view : slamViews) {
          view->update();
        }
        for (auto& view : gtViews) {
          view->update();
        }
      }

      if (step_count % 1000 == 0) {
        std::cout << "  Step " << step_count << " completed" << std::endl;
      }
    }

    if (!manager.keepRunning()) {
      std::cout << "  Simulation finished (no more data) after " << step_count << " steps" << std::endl;
    } else if (step_count >= max_steps) {
      std::cout << "  Maximum steps (" << max_steps << ") reached, stopping..." << std::endl;
    }

    // Stop the simulation and finalize
    std::cout << "Stopping simulation..." << std::endl;
    manager.stop();

    // Save trajectories
    std::cout << "Saving trajectories..." << std::endl;
    std::string trajectory_output_dir = "test_results/multidrone/trajectories";
    manager.saveTrajectories(trajectory_output_dir, "tum");

    // Stop visualization
    std::cout << "Stopping visualization..." << std::endl;
    viewManager.stop();

    std::cout << "Simulation complete!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
