#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>

#include <nlohmann/json.hpp>

#include "utisa_agent_manager.h"
#include "view_manager_3d.h"
#include "utisa_example_views.h"

using namespace g2o::tutorial::multibotsim;
using namespace g2o::tutorial::viz;
using json = nlohmann::json;

namespace g2o::tutorial {
void forceLinkTypesTutorialSlam2d();  // Forward declaration
}

namespace {
void printWallElapsed(std::chrono::steady_clock::time_point t0) {
  using namespace std::chrono;
  const auto t1 = steady_clock::now();
  const double w = duration<double>(t1 - t0).count();
  const long long sec_ll = static_cast<long long>(std::floor(w + 1e-9));
  const long long xh = sec_ll / 3600;
  const long long ym = (sec_ll % 3600) / 60;
  const long long zs = sec_ll % 60;
  std::cout << "Wall time: " << xh << " hr, " << ym << " mins, " << zs << " secs ("
            << std::fixed << std::setprecision(3) << w << " secs)" << std::endl;
}
}  // namespace

int main(int argc, char* argv[]) {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();
  (void)argc;
  (void)argv;

  const auto wall_clock_start = std::chrono::steady_clock::now();
  try {
    const std::string base_config_path = "Source/Examples/UTISA_slam/config/experiment_base_config.json";
    const std::string view_config_path = "Source/Examples/UTISA_slam/config/view_config.json";
    const std::string view_config_dir = "Source/Examples/UTISA_slam/config";

    std::ifstream in(base_config_path);
    if (!in) {
      throw std::runtime_error("cannot open base config file: " + base_config_path);
    }
    json j;
    in >> j;

    const std::string config_path = base_config_path;
    const std::string slam_config_path = j.at("slam_config_path").get<std::string>();
    const std::string input_path = j.at("input_path").get<std::string>();
    const std::string output_path = j.value("output_path", std::string("test_results/utisa"));
    const std::string topology_path =
        j.value("topology_path", std::string("Source/Examples/UTISA_slam/config/topology.json"));
    const bool verbose = j.value("verbose", false);

    std::cout << "Initializing UTISAAgentManager..." << std::endl;
    std::cout << "  Config path: " << config_path << std::endl;
    std::cout << "  SLAM config path: " << slam_config_path << std::endl;
    std::cout << "  Input path: " << input_path << std::endl;
    std::cout << "  Output path: " << output_path << std::endl;
    std::cout << "  Topology path: " << topology_path << std::endl;

    UTISAAgentManager manager(config_path, input_path, slam_config_path);
    manager.setTopologyJson(topology_path);

    std::ifstream view_in(view_config_path);
    if (!view_in) {
      throw std::runtime_error("cannot open view config file: " + view_config_path);
    }
    json view_j;
    view_in >> view_j;
    const double step_pause_sec = view_j.value("step_pause", 0.0);
    double platform_size = view_j.value("platform_size", 1.5);
    if (view_j.contains("slam_system") && view_j["slam_system"].contains("platform")) {
      platform_size = view_j["slam_system"]["platform"].value("size", platform_size);
    }
    const bool visualise_slam_path = view_j.value("visualise_slam_path", true);
    const bool visualise_sim_path = view_j.value("visualise_sim_path", true);
    const bool visualise_pose = view_j.value("visualise_pose", true);

    std::cout << "Setting up visualization..." << std::endl;
    ViewManager3D viewManager(view_config_path);
    std::cout << "  platform_size: " << platform_size << std::endl;
    if (step_pause_sec > 0.0) {
      std::cout << "  step_pause: " << step_pause_sec << " s" << std::endl;
    }

    std::vector<std::shared_ptr<UTISASlamSystemView>> slamViews;
    std::vector<std::shared_ptr<UTISASimulationView>> gtViews;
    const auto& robotIds = manager.getRobotIds();

    std::vector<Eigen::Vector3f> robotEstColors = {
        Eigen::Vector3f(1.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 1.0f), Eigen::Vector3f(1.0f, 1.0f, 0.0f),
        Eigen::Vector3f(1.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 1.0f, 1.0f)};

    std::vector<Eigen::Vector3f> robotGTColors = {
        Eigen::Vector3f(1.0f, 0.3f, 0.3f), Eigen::Vector3f(0.3f, 1.0f, 0.3f),
        Eigen::Vector3f(0.3f, 0.3f, 1.0f), Eigen::Vector3f(1.0f, 1.0f, 0.3f),
        Eigen::Vector3f(1.0f, 0.3f, 1.0f), Eigen::Vector3f(0.3f, 1.0f, 1.0f)};

    auto gtColorFromRobot = [](const Eigen::Vector3f& c) {
      return Eigen::Vector3f(c[0] * 0.5f + 0.3f, c[1] * 0.5f + 0.3f, c[2] * 0.5f + 0.3f);
    };

    for (size_t i = 0; i < robotIds.size(); ++i) {
      UTISASlamSystem* slamSystem = manager.getSlamSystem(i);
      UTISASimulator* sim = manager.getSimulation(i);
      const std::string bot_view_config = view_config_dir + "/bot" + robotIds[i] + "_view.json";

      if (slamSystem) {
        std::shared_ptr<UTISASlamSystemView> view;
        std::ifstream test_file(bot_view_config);
        if (test_file.good()) {
          view = std::make_shared<UTISASlamSystemView>(slamSystem, bot_view_config, visualise_slam_path);
        } else {
          const Eigen::Vector3f color = robotEstColors[i % robotEstColors.size()];
          view = std::make_shared<UTISASlamSystemView>(slamSystem, color, visualise_slam_path);
        }
        view->setPlatformSize(platform_size);
        view->setVisualisePose(visualise_pose);
        viewManager.addView(view);
        slamViews.push_back(view);
      }

      if (sim) {
        const Eigen::Vector3f gtColor = gtColorFromRobot(robotGTColors[i % robotGTColors.size()]);
        auto gtView = std::make_shared<UTISASimulationView>(sim, gtColor, visualise_sim_path);
        gtView->setLandmarkGroundtruthFile(input_path + "/Landmark_Groundtruth.dat");
        gtView->setPlatformSize(platform_size);
        gtView->setVisualisePose(visualise_pose);
        viewManager.addView(gtView);
        gtViews.push_back(gtView);
      }
    }

    std::cout << "Starting visualization..." << std::endl;
    viewManager.start();

    std::cout << "Starting simulation..." << std::endl;
    manager.start();

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
        std::this_thread::sleep_for(std::chrono::duration<double>(step_pause_sec));
      }

      if (step_count % 10 == 0) {
        for (auto& view : slamViews) {
          view->update();
        }
        for (auto& view : gtViews) {
          view->update();
        }
      }
    }

    manager.dumpPreOptTrajectories();
    manager.stop();
    for (auto& view : slamViews) {
      view->update();
    }
    for (auto& view : gtViews) {
      view->update();
    }

    std::cout << "Saving trajectories..." << std::endl;
    manager.saveTrajectories(output_path + "/trajectories", "tum");
    manager.saveLandmarks(output_path + "/landmarks");

    std::cout << "Stopping visualization..." << std::endl;
    viewManager.stop();

    std::cout << "Simulation complete!" << std::endl;
    printWallElapsed(wall_clock_start);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    printWallElapsed(wall_clock_start);
    return 1;
  }
}
