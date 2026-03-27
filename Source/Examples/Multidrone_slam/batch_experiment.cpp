#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

#include <filesystem>
#include <nlohmann/json.hpp>

#include "agent_manager.h"
#include "data_based_simulation.h"
#include "drone_slam_system.h"
#include "view_manager_3d.h"
#include "multi_drone_slam_system_view.h"
#include "data_based_simulation_view.h"
#include <Eigen/Core>

using namespace g2o::tutorial::multibotsim;
using namespace g2o::tutorial::viz;
using json = nlohmann::json;

namespace {

static bool isAllDigits(const std::string& name) {
  if (name.empty()) return false;
  return std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c); });
}

/// List subdirectories of `root` whose names are non-negative integers (e.g. 0, 1, 2), sorted numerically.
static std::vector<std::string> listNumericBatchRuns(const std::filesystem::path& root) {
  std::vector<std::pair<int, std::string>> named;
  if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    return {};
  }
  for (const auto& e : std::filesystem::directory_iterator(root)) {
    if (!e.is_directory()) continue;
    const std::string name = e.path().filename().string();
    if (!isAllDigits(name)) continue;
    named.emplace_back(std::stoi(name), name);
  }
  std::sort(named.begin(), named.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<std::string> out;
  out.reserve(named.size());
  for (const auto& p : named) {
    out.push_back(p.second);
  }
  return out;
}

static void writeMergedConfig(const std::filesystem::path& path, const json& merged) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write merged experiment config: " + path.string());
  }
  out << merged.dump(2);
}

/**
 * Same flow as experiment.cpp for one dataset: AgentManager + loop + dump pre-opt + stop + save trajectories.
 * @param enable_visualization if false, skip Pangolin and views (suitable for batch).
 */
static int runOneBatchItem(
    const std::string& merged_config_path,
    const std::string& input_run_path,
    const std::string& output_run_path,
    const std::string& slam_config_path,
    const std::string& topology_path,
    bool verbose,
    bool enable_visualization) {
  std::cout << "  Input (logs):  " << input_run_path << std::endl;
  std::cout << "  Output base:   " << output_run_path << std::endl;

  AgentManager manager(merged_config_path, input_run_path, slam_config_path);
  manager.setTopologyJson(topology_path);

  const std::string view_config_path = "Source/Examples/Multidrone_slam/config/view_config.json";
  std::ifstream view_in(view_config_path);
  if (!view_in) {
    throw std::runtime_error("cannot open view config file: " + view_config_path);
  }
  json view_j;
  view_in >> view_j;
  const double step_pause_sec = view_j.value("step_pause", 0.0);
  const bool visualise_slam_path = view_j.value("visualise_slam_path", true);
  const bool visualise_sim_path = view_j.value("visualise_sim_path", true);
  const bool visualise_pose = view_j.value("visualise_pose", true);

  std::unique_ptr<ViewManager3D> viewManager;
  std::vector<std::shared_ptr<MultiDroneSLAMSystemView>> slamViews;
  std::vector<std::shared_ptr<DataBasedSimulationView>> gtViews;

  if (enable_visualization) {
    std::cout << "  Visualization: on" << std::endl;
    viewManager = std::make_unique<ViewManager3D>(view_config_path);
    if (step_pause_sec > 0.0) {
      std::cout << "  step_pause: " << step_pause_sec << " s" << std::endl;
    }

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

    const auto& robotIds = manager.getRobotIds();
    for (size_t i = 0; i < robotIds.size(); ++i) {
      MultiDroneSLAMSystem* slamSystem = manager.getSlamSystem(i);
      DataBasedSimulation* sim = manager.getSimulation(i);
      std::string bot_view_config =
          "Source/Examples/Multidrone_slam/config/bot" + robotIds[i] + "_view.json";
      if (slamSystem) {
        std::shared_ptr<MultiDroneSLAMSystemView> view;
        std::ifstream test_file(bot_view_config);
        if (test_file.good()) {
          view = std::make_shared<MultiDroneSLAMSystemView>(slamSystem, bot_view_config, visualise_slam_path);
        } else {
          Eigen::Vector3f color = robotEstColors[i % robotEstColors.size()];
          view = std::make_shared<MultiDroneSLAMSystemView>(slamSystem, color, visualise_slam_path);
        }
        view->setVisualisePose(visualise_pose);
        viewManager->addView(view);
        slamViews.push_back(view);
      }
      if (sim) {
        Eigen::Vector3f gtColor = gtColorFromRobot(robotGTColors[i % robotGTColors.size()]);
        auto gtView = std::make_shared<DataBasedSimulationView>(sim, gtColor, visualise_sim_path);
        gtView->setVisualisePose(visualise_pose);
        viewManager->addView(gtView);
        gtViews.push_back(gtView);
      }
    }
    viewManager->start();
  } else {
    std::cout << "  Visualization: off (batch)" << std::endl;
  }

  manager.start();

  int step_count = 0;
  const int max_steps = 100000;
  while (step_count < max_steps && manager.keepRunning()) {
    if (verbose) {
      std::cout << "  step " << step_count << std::endl;
    }
    manager.step();
    step_count++;

    if (enable_visualization && step_pause_sec > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(step_pause_sec));
    }

    if (enable_visualization && step_count % 10 == 0) {
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

  if (enable_visualization) {
    for (auto& view : slamViews) {
      view->update();
    }
    for (auto& view : gtViews) {
      view->update();
    }
  }

  const std::string trajectory_output_dir = output_run_path + "/trajectories";
  manager.saveTrajectories(trajectory_output_dir, "tum");

  if (enable_visualization && viewManager) {
    viewManager->stop();
  }

  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  try {
    const std::string batch_config_path =
        "Source/Examples/Multidrone_slam/config/batch_experiment_config.json";
    const std::string default_base_path =
        "Source/Examples/Multidrone_slam/config/experiment_base_config.json";

    std::ifstream batch_in(batch_config_path);
    if (!batch_in) {
      throw std::runtime_error("cannot open batch config: " + batch_config_path);
    }
    json batch_j;
    batch_in >> batch_j;

    const std::string batch_data_root = batch_j.at("input_path").get<std::string>();
    const std::string batch_output_root = batch_j.at("output_path").get<std::string>();
    const std::string base_config_path =
        batch_j.value("experiment_base_config", default_base_path);
    const bool batch_verbose = batch_j.value("verbose", false);
    const bool enable_visualization = batch_j.value("enable_visualization", false);

    std::ifstream base_in(base_config_path);
    if (!base_in) {
      throw std::runtime_error("cannot open experiment base config: " + base_config_path);
    }
    json base_j;
    base_in >> base_j;

    std::string slam_config_path = base_j.at("slam_config_path").get<std::string>();
    std::string topology_path = "Source/Examples/Multidrone_slam/config/topology.json";
    if (base_j.contains("topology_path")) {
      topology_path = base_j["topology_path"].get<std::string>();
    }


    const std::string merged_parent = batch_output_root + "/.batch_merged_configs";
    std::filesystem::create_directories(merged_parent);

    const auto run_dirs = listNumericBatchRuns(std::filesystem::path(batch_data_root));
    if (run_dirs.empty()) {
      std::cerr << "No numeric batch subdirectories found under: " << batch_data_root << std::endl;
      return 1;
    }

    std::cout << "Batch experiment: " << run_dirs.size() << " run(s)" << std::endl;
    std::cout << "  Batch data root:    " << batch_data_root << std::endl;
    std::cout << "  Batch output root:  " << batch_output_root << std::endl;
    std::cout << "  Base config:        " << base_config_path << std::endl;

    for (const std::string& dirName : run_dirs) {
      const std::string input_run = batch_data_root + "/" + dirName;
      const std::string output_run = batch_output_root + "/" + dirName;

      json merged = base_j;
      merged["output_path"] = output_run;
      if (batch_j.contains("verbose")) {
        merged["verbose"] = batch_verbose;
      }

      const std::string merged_path =
          merged_parent + "/experiment_run_" + dirName + ".json";
      writeMergedConfig(std::filesystem::path(merged_path), merged);

      std::cout << "\n=== Batch run " << dirName << " ===" << std::endl;
      const bool verbose = merged.value("verbose", false);
      runOneBatchItem(merged_path, input_run, output_run, slam_config_path, topology_path, verbose,
                      enable_visualization);
    }

    std::cout << "\nBatch complete (" << run_dirs.size() << " runs)." << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
