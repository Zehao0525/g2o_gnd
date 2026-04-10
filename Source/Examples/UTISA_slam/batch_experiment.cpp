#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

#include <filesystem>
#include <nlohmann/json.hpp>

#include "utisa_agent_manager.h"
#include "view_manager_3d.h"
#include "utisa_example_views.h"  // from Incremental_Visualizer
#include <Eigen/Core>

using namespace g2o::tutorial::multibotsim;
using namespace g2o::tutorial::viz;
using json = nlohmann::json;

namespace g2o::tutorial {
  void forceLinkTypesTutorialSlam2d();  // Forward declaration
}

namespace {

void renderProgressBar(const std::string& label, size_t current, size_t total, double elapsed_sec) {
  const size_t width = 30;
  const double ratio = (total > 0) ? static_cast<double>(current) / static_cast<double>(total) : 0.0;
  const size_t filled = static_cast<size_t>(std::round(ratio * static_cast<double>(width)));
  const size_t clamped_filled = std::min(filled, width);

  std::cout << "\r" << label << " [";
  for (size_t i = 0; i < clamped_filled; ++i) std::cout << "=";
  for (size_t i = clamped_filled; i < width; ++i) std::cout << " ";
  std::cout << "] " << current << "/" << total << " (" << std::setw(3)
            << static_cast<int>(std::round(ratio * 100.0)) << "%)"
            << " elapsed " << std::fixed << std::setprecision(1) << elapsed_sec << "s";
  std::cout.flush();
  if (current >= total) {
    std::cout << std::endl;
  }
}

void renderSpinner(const std::string& label, int step_count, double elapsed_sec) {
  static const char kSpinChars[] = {'|', '/', '-', '\\'};
  const char spin = kSpinChars[(step_count / 10) % 4];
  std::cout << "\r" << label << " " << spin << " steps=" << step_count
            << " elapsed " << std::fixed << std::setprecision(1) << elapsed_sec << "s";
  std::cout.flush();
}

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

/// True if `dir` looks like a UTIAS MR.CLAM log folder (matches `UTISAAgentManager` detection).
static bool looksLikeUtisaMrclamDataset(const std::filesystem::path& dir) {
  return std::filesystem::exists(dir / "Robot1_Odometry.dat") &&
         std::filesystem::exists(dir / "Barcodes.dat");
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
 * Same flow as experiment.cpp for one dataset: UTISAAgentManager + loop + dump pre-opt + stop + save trajectories.
 * @param enable_visualization if false, skip Pangolin and views (suitable for batch).
 */
static int runOneBatchItem(const std::string& merged_config_path, const std::string& input_run_path,
                           const std::string& output_run_path, const std::string& slam_config_path,
                           const std::string& topology_path, const std::string& view_config_path,
                           const std::string& view_config_dir, bool verbose, bool enable_visualization,
                           bool show_progress) {
  std::cout << "  Input (logs):  " << input_run_path << std::endl;
  std::cout << "  Output base:   " << output_run_path << std::endl;

  UTISAAgentManager manager(merged_config_path, input_run_path, slam_config_path);
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

  std::unique_ptr<ViewManager3D> viewManager;
  std::vector<std::shared_ptr<UTISASlamSystemView>> slamViews;
  std::vector<std::shared_ptr<UTISASimulationView>> gtViews;

  if (enable_visualization) {
    std::cout << "  Visualization: on" << std::endl;
    std::cout << "  platform_size: " << platform_size << std::endl;
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
      UTISASlamSystem* slamSystem = manager.getSlamSystem(i);
      UTISASimulator* sim = manager.getSimulation(i);
      std::string bot_view_config = view_config_dir + "/bot" + robotIds[i] + "_view.json";
      if (slamSystem) {
        std::shared_ptr<UTISASlamSystemView> view;
        std::ifstream test_file(bot_view_config);
        if (test_file.good()) {
          view = std::make_shared<UTISASlamSystemView>(slamSystem, bot_view_config, visualise_slam_path);
        } else {
          Eigen::Vector3f color = robotEstColors[i % robotEstColors.size()];
          view = std::make_shared<UTISASlamSystemView>(slamSystem, color, visualise_slam_path);
        }
        view->setPlatformSize(platform_size);
        view->setVisualisePose(visualise_pose);
        viewManager->addView(view);
        slamViews.push_back(view);
      }
      if (sim) {
        Eigen::Vector3f gtColor = gtColorFromRobot(robotGTColors[i % robotGTColors.size()]);
        auto gtView = std::make_shared<UTISASimulationView>(sim, gtColor, visualise_sim_path);
        gtView->setLandmarkGroundtruthFile(input_run_path + "/Landmark_Groundtruth.dat");
        gtView->setPlatformSize(platform_size);
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
  const auto run_start = std::chrono::steady_clock::now();

  int step_count = 0;
  while (manager.keepRunning()) {
    if (verbose) {
      std::cout << "  step " << step_count << std::endl;
    }
    manager.step();
    step_count++;

    if (show_progress && !verbose && !enable_visualization && (step_count % 200 == 0)) {
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
      renderSpinner("  Running", step_count, elapsed);
    }

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

  if (show_progress && !verbose && !enable_visualization) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    renderSpinner("  Running", step_count, elapsed);
    std::cout << std::endl;
  }

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
  manager.saveLandmarks(output_run_path + "/landmarks");

  if (enable_visualization && viewManager) {
    viewManager->stop();
  }

  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();
  (void)argc;
  (void)argv;
  const auto wall_clock_start = std::chrono::steady_clock::now();
  try {
    const std::string batch_config_path = "Source/Examples/UTISA_slam/config/batch_experiment_config.json";
    const std::string default_base_path = "Source/Examples/UTISA_slam/config/experiment_base_config.json";

    std::ifstream batch_in(batch_config_path);
    if (!batch_in) {
      throw std::runtime_error(
          "cannot open batch config: " + batch_config_path +
          " (paths are relative to the repository root; run from g2o_example/)");
    }
    json batch_j;
    batch_in >> batch_j;

    const std::string batch_data_root = batch_j.at("input_path").get<std::string>();
    const std::string batch_output_root = batch_j.at("output_path").get<std::string>();
    const std::string base_config_path = batch_j.value("experiment_base_config", default_base_path);
    const bool batch_verbose = batch_j.value("verbose", false);
    const bool enable_visualization = batch_j.value("enable_visualization", false);
    const bool single_dataset = batch_j.value("single_dataset", false);
    const bool show_progress = batch_j.value("show_progress", true);

    const std::string view_config_path = batch_j.value(
        "view_config_path", std::string("Source/Examples/UTISA_slam/config/view_config.json"));
    const std::string view_config_dir = batch_j.value("view_config_dir", std::string("Source/Examples/UTISA_slam/config"));

    std::ifstream base_in(base_config_path);
    if (!base_in) {
      throw std::runtime_error(
          "cannot open experiment base config: " + base_config_path +
          " (run from g2o_example/ so Source/... paths resolve)");
    }
    json base_j;
    base_in >> base_j;

    std::string slam_config_path = base_j.at("slam_config_path").get<std::string>();
    std::string topology_path = "Source/Examples/UTISA_slam/config/topology.json";
    if (base_j.contains("topology_path")) {
      topology_path = base_j["topology_path"].get<std::string>();
    }

    const std::string merged_parent = batch_output_root + "/.batch_merged_configs";
    std::filesystem::create_directories(merged_parent);

    std::vector<std::string> run_dirs;
    if (single_dataset) {
      run_dirs.push_back("single");
    } else {
      run_dirs = listNumericBatchRuns(std::filesystem::path(batch_data_root));
      if (run_dirs.empty() && looksLikeUtisaMrclamDataset(std::filesystem::path(batch_data_root))) {
        run_dirs.push_back("single");
      }
    }

    if (run_dirs.empty()) {
      std::cerr << "No batch runs: add numeric subdirs under input_path, or set \"single_dataset\": true, or place "
                   "MR.CLAM files (Robot1_Odometry.dat, Barcodes.dat) in input_path.\n"
                << "  input_path: " << batch_data_root << std::endl;
      printWallElapsed(wall_clock_start);
      return 1;
    }

    std::cout << "UTISA batch experiment: " << run_dirs.size() << " run(s)" << std::endl;
    std::cout << "  Batch data root:    " << batch_data_root << std::endl;
    std::cout << "  Batch output root:  " << batch_output_root << std::endl;
    std::cout << "  Base config:        " << base_config_path << std::endl;

    const auto batch_start = std::chrono::steady_clock::now();
    if (show_progress) {
      renderProgressBar("Batch", 0, run_dirs.size(), 0.0);
    }

    for (size_t run_idx = 0; run_idx < run_dirs.size(); ++run_idx) {
      const std::string& dirName = run_dirs[run_idx];
      std::string input_run;
      std::string output_run;
      if (dirName == "single") {
        input_run = batch_data_root;
        output_run = batch_output_root;
      } else {
        input_run = batch_data_root + "/" + dirName;
        output_run = batch_output_root + "/" + dirName;
      }

      json merged = base_j;
      merged["output_path"] = output_run;
      if (batch_j.contains("verbose")) {
        merged["verbose"] = batch_verbose;
      }

      const std::string merged_path =
          merged_parent + "/experiment_run_" + (dirName == "single" ? "single" : dirName) + ".json";
      writeMergedConfig(std::filesystem::path(merged_path), merged);

      std::cout << "\n=== Batch run " << (dirName == "single" ? "(single dataset)" : dirName) << " ===" << std::endl;
      const bool verbose = merged.value("verbose", false);
      runOneBatchItem(merged_path, input_run, output_run, slam_config_path, topology_path, view_config_path,
                      view_config_dir, verbose, enable_visualization, show_progress);

      if (show_progress) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start).count();
        renderProgressBar("Batch", run_idx + 1, run_dirs.size(), elapsed);
      }
    }

    std::cout << "\nBatch complete (" << run_dirs.size() << " runs)." << std::endl;
    printWallElapsed(wall_clock_start);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    printWallElapsed(wall_clock_start);
    return 1;
  }
}
