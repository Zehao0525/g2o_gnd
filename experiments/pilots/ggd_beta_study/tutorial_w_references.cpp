// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "edge_se2.h"
#include "edge_se2_pointxy.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/stuff/sampler.h"

#include "simulator.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "edge_se2_pointxy.h"
#include "ggd_kernel.h"
#include "edge_platform_loc_prior.h"

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;
using json = nlohmann::json;

namespace g2o::tutorial {
void forceLinkTypesTutorialSlam2d();  // Forward declaration
}  // namespace g2o::tutorial

namespace {

enum class GpsKernelKind {
  Gaussian,
  GGD,
  Huber,
  Tukey,
  Cauchy,
  GemanMcClure,
  Welsch,
  Saturated,
  DCS,
  PseudoHuber,
  Fair,
};

struct GpsKernelSpec {
  GpsKernelKind kind = GpsKernelKind::Gaussian;
  std::string name;
  std::string outputSuffix;
};

struct KernelParams {
  double ggdBound = 3.0;
  double ggdPower = 6.0;
  double ggdLnc = 1e-3;
  double ggdTailStd = 2.0;
  double robustDelta = 3.0;
};

struct SimulationConfig {
  int numNodes = 300;
  int seed = 37;
  int numTests = 30;
  SE2 sensorOffset = SE2(0.2, 0.1, -0.1);
  double transitionProb = 0.95;
  double gpsStd = 2.5;
  int gpsPeriod = 30;
  bool incrementGpsPeriod = false;
};

struct OptimizationConfig {
  int optIterations = 20;
  int ggdWarmupIterations = 10;
  int ggdActiveIterations = 10;
  bool verbose = false;
};

struct ExperimentConfig {
  SimulationConfig simulation;
  OptimizationConfig optimization;
  KernelParams kernels;
  std::string outputBaseDir = "test_results/ggd_beta_study/exp1_test_results_3";
  std::vector<GpsKernelSpec> kernelSpecs;
};

GpsKernelKind parseKernelKind(const std::string& type) {
  if (type == "gaussian") return GpsKernelKind::Gaussian;
  if (type == "ggd") return GpsKernelKind::GGD;
  if (type == "huber") return GpsKernelKind::Huber;
  if (type == "tukey") return GpsKernelKind::Tukey;
  if (type == "cauchy") return GpsKernelKind::Cauchy;
  if (type == "geman_mcclure") return GpsKernelKind::GemanMcClure;
  if (type == "welsch") return GpsKernelKind::Welsch;
  if (type == "saturated") return GpsKernelKind::Saturated;
  if (type == "dcs") return GpsKernelKind::DCS;
  if (type == "pseudo_huber") return GpsKernelKind::PseudoHuber;
  if (type == "fair") return GpsKernelKind::Fair;
  throw std::runtime_error("unknown kernel type: " + type);
}

ExperimentConfig loadExperimentConfig(const std::string& configPath) {
  std::ifstream in(configPath);
  if (!in) {
    throw std::runtime_error("cannot open config: " + configPath);
  }
  json j;
  in >> j;

  ExperimentConfig cfg;

  if (j.contains("simulation")) {
    const auto& s = j["simulation"];
    cfg.simulation.numNodes = s.value("num_nodes", cfg.simulation.numNodes);
    cfg.simulation.seed = s.value("seed", cfg.simulation.seed);
    cfg.simulation.numTests = s.value("num_tests", cfg.simulation.numTests);
    cfg.simulation.transitionProb = s.value("transition_prob", cfg.simulation.transitionProb);
    cfg.simulation.gpsStd = s.value("gps_std", cfg.simulation.gpsStd);
    cfg.simulation.gpsPeriod = s.value("gps_period", cfg.simulation.gpsPeriod);
    cfg.simulation.incrementGpsPeriod = s.value("increment_gps_period", cfg.simulation.incrementGpsPeriod);
    if (s.contains("sensor_offset") && s["sensor_offset"].is_array() &&
        s["sensor_offset"].size() >= 3) {
      cfg.simulation.sensorOffset = SE2(s["sensor_offset"][0], s["sensor_offset"][1],
                                        s["sensor_offset"][2]);
    }
  }

  if (j.contains("optimization")) {
    const auto& o = j["optimization"];
    cfg.optimization.optIterations = o.value("opt_iterations", cfg.optimization.optIterations);
    cfg.optimization.ggdWarmupIterations =
        o.value("ggd_warmup_iterations", cfg.optimization.ggdWarmupIterations);
    cfg.optimization.ggdActiveIterations =
        o.value("ggd_active_iterations", cfg.optimization.ggdActiveIterations);
    cfg.optimization.verbose = o.value("verbose", cfg.optimization.verbose);
  }

  if (j.contains("kernels")) {
    const auto& k = j["kernels"];
    cfg.kernels.ggdBound = k.value("ggd_bound", cfg.kernels.ggdBound);
    cfg.kernels.ggdPower = k.value("ggd_power", cfg.kernels.ggdPower);
    cfg.kernels.ggdLnc = k.value("ggd_lnc", cfg.kernels.ggdLnc);
    cfg.kernels.ggdTailStd = k.value("ggd_tail_std", cfg.kernels.ggdTailStd);
    cfg.kernels.robustDelta = k.value("robust_delta", cfg.kernels.robustDelta);
  }
  if (cfg.kernels.robustDelta <= 0.0) {
    cfg.kernels.robustDelta = cfg.kernels.ggdBound;
  }

  if (j.contains("output")) {
    cfg.outputBaseDir = j["output"].value("base_dir", cfg.outputBaseDir);
  }

  if (!j.contains("kernel_specs") || !j["kernel_specs"].is_array()) {
    throw std::runtime_error("config must contain kernel_specs array");
  }
  for (const auto& entry : j["kernel_specs"]) {
    GpsKernelSpec spec;
    spec.name = entry.at("name").get<std::string>();
    spec.outputSuffix = entry.value("output_suffix", spec.name);
    spec.kind = parseKernelKind(entry.at("type").get<std::string>());
    cfg.kernelSpecs.push_back(std::move(spec));
  }
  if (cfg.kernelSpecs.empty()) {
    throw std::runtime_error("kernel_specs must not be empty");
  }

  return cfg;
}

RobustKernel* makeGpsKernel(const GpsKernelSpec& spec, const KernelParams& params,
                            bool* ggdActivePtr) {
  switch (spec.kind) {
    case GpsKernelKind::Gaussian:
      return nullptr;
    case GpsKernelKind::GGD:
      return new ToggelableGGDKernel(params.ggdBound, params.ggdPower, params.ggdLnc,
                                     params.ggdTailStd, ggdActivePtr);
    case GpsKernelKind::Huber: {
      auto* k = new RobustKernelHuber();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::Tukey: {
      auto* k = new RobustKernelTukey();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::Cauchy: {
      auto* k = new RobustKernelCauchy();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::GemanMcClure: {
      auto* k = new RobustKernelGemanMcClure();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::Welsch: {
      auto* k = new RobustKernelWelsch();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::Saturated: {
      auto* k = new RobustKernelSaturated();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::DCS: {
      auto* k = new RobustKernelDCS();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::PseudoHuber: {
      auto* k = new RobustKernelPseudoHuber();
      k->setDelta(params.robustDelta);
      return k;
    }
    case GpsKernelKind::Fair: {
      auto* k = new RobustKernelFair();
      k->setDelta(params.robustDelta);
      return k;
    }
  }
  return nullptr;
}

void assignGpsKernels(SparseOptimizer& optimizer, const GpsKernelSpec& spec,
                      const KernelParams& params, bool* ggdActivePtr) {
  for (auto& edgePair : optimizer.edges()) {
    auto* gps = dynamic_cast<EdgePlatformLocPrior*>(edgePair);
    if (!gps) {
      continue;
    }
    gps->setRobustKernel(makeGpsKernel(spec, params, ggdActivePtr));
  }
}

void resetEstimates(SparseOptimizer& optimizer, const Simulator& simulator) {
  for (const auto& p : simulator.poses()) {
    if (auto* robot = dynamic_cast<VertexSE2*>(optimizer.vertex(p.id))) {
      robot->setEstimate(p.simulatorPose);
    }
  }
  for (const auto& l : simulator.landmarks()) {
    if (auto* landmark = dynamic_cast<VertexPointXY*>(optimizer.vertex(l.id))) {
      landmark->setEstimate(l.simulatedPose);
    }
  }
}

double meanPoseTranslationError(const SparseOptimizer& optimizer, const Simulator& simulator) {
  if (simulator.poses().empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto& p : simulator.poses()) {
    const auto* robot = dynamic_cast<const VertexSE2*>(optimizer.vertex(p.id));
    if (!robot) {
      continue;
    }
    sum += (robot->estimate().translation() - p.truePose.translation()).norm();
  }
  return sum / static_cast<double>(simulator.poses().size());
}

void runKernelOptimization(SparseOptimizer& optimizer, const GpsKernelSpec& spec,
                           const OptimizationConfig& optCfg, bool& ggdActive) {
  if (spec.kind == GpsKernelKind::GGD) {
    ggdActive = false;
    optimizer.initializeOptimization();
    optimizer.optimize(optCfg.ggdWarmupIterations);
    ggdActive = true;
    optimizer.initializeOptimization();
    optimizer.optimize(optCfg.ggdActiveIterations);
    return;
  }

  optimizer.initializeOptimization();
  optimizer.optimize(optCfg.optIterations);
}

}  // namespace

int main(int argc, char* argv[]) {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();

  const std::string defaultConfigPath =
      "experiments/pilots/ggd_beta_study/config/tutorial_w_references_config.json";
  const std::string configPath = (argc > 1) ? argv[1] : defaultConfigPath;
  const ExperimentConfig cfg = loadExperimentConfig(configPath);

  const SimulationConfig& simCfg = cfg.simulation;
  const OptimizationConfig& optCfg = cfg.optimization;
  const KernelParams& kernelParams = cfg.kernels;

  double gps_degree = Sampler::uniformRand(-M_PI, M_PI);
  int gps_period = simCfg.gpsPeriod;

  for (int testIdx = 0; testIdx < simCfg.numTests; ++testIdx) {
    Simulator simulator = Simulator(simCfg.seed + testIdx);
    simulator.simulate(simCfg.numNodes, simCfg.sensorOffset);
    Sampler::seedRand(simCfg.seed + testIdx);

    typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
    typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);
    OptimizationAlgorithmLevenberg* solver =
        new OptimizationAlgorithmLevenberg(
            std::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    ParameterSE2Offset* sensorOffset = new ParameterSE2Offset;
    sensorOffset->setOffset(simCfg.sensorOffset);
    sensorOffset->setId(0);
    optimizer.addParameter(sensorOffset);

    cerr << "Optimization: Adding robot poses ... ";
    for (size_t i = 0; i < simulator.poses().size(); ++i) {
      const Simulator::GridPose& p = simulator.poses()[i];
      const SE2& t = p.simulatorPose;
      VertexSE2* robot = new VertexSE2;
      robot->setId(p.id);
      robot->setEstimate(t);
      optimizer.addVertex(robot);

      if (static_cast<int>(i) % gps_period == 0) {
        EdgePlatformLocPrior* gps = new EdgePlatformLocPrior;
        gps->vertices()[0] = robot;

        if (Sampler::uniformRand(0.0, 1.0) > simCfg.transitionProb) {
          gps_degree = Sampler::uniformRand(-M_PI, M_PI);
        }
        Eigen::Vector2d gps_noise =
            Eigen::Vector2d(simCfg.gpsStd * sqrt(2) * cos(gps_degree),
                            simCfg.gpsStd * sqrt(2) * sin(gps_degree));

        Eigen::Matrix<double, 2, 2> cov;
        cov.fill(0.);
        cov(0, 0) = simCfg.gpsStd * simCfg.gpsStd;
        cov(1, 1) = simCfg.gpsStd * simCfg.gpsStd;
        gps->setInformation(cov.inverse());
        gps->setMeasurement((p.truePose * simCfg.sensorOffset).toVector().head<2>() + gps_noise);
        gps->setParameterId(0, sensorOffset->id());
        optimizer.addEdge(gps);
      }
    }
    cerr << "Number of poses added: " << simulator.poses().size() << endl;
    cerr << "done." << endl;

    cerr << "Optimization: Adding odometry measurements ... ";
    for (size_t i = 0; i < simulator.odometry().size(); ++i) {
      const Simulator::GridEdge& simEdge = simulator.odometry()[i];

      EdgeSE2* odometry = new EdgeSE2;
      odometry->vertices()[0] = optimizer.vertex(simEdge.from);
      odometry->vertices()[1] = optimizer.vertex(simEdge.to);
      odometry->setMeasurement(simEdge.simulatorTransf);
      odometry->setInformation(simEdge.information);
      optimizer.addEdge(odometry);
    }
    cerr << "Number of measurements added: " << simulator.odometry().size() << endl;
    cerr << "done." << endl;

    cerr << "Optimization: add landmark vertices ... ";
    for (size_t i = 0; i < simulator.landmarks().size(); ++i) {
      const Simulator::Landmark& l = simulator.landmarks()[i];
      VertexPointXY* landmark = new VertexPointXY;
      landmark->setId(l.id);
      landmark->setEstimate(l.simulatedPose);
      optimizer.addVertex(landmark);
    }
    cerr << "Number of landmarks added: " << simulator.landmarks().size() << endl;
    cerr << "done." << endl;

    cerr << "Optimization: add landmark observations ... ";
    for (size_t i = 0; i < simulator.landmarkObservations().size(); ++i) {
      const Simulator::LandmarkEdge& simEdge = simulator.landmarkObservations()[i];
      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      landmarkObservation->vertices()[0] = optimizer.vertex(simEdge.from);
      landmarkObservation->vertices()[1] = optimizer.vertex(simEdge.to);
      landmarkObservation->setMeasurement(simEdge.simulatorMeas);
      landmarkObservation->setInformation(simEdge.information);
      landmarkObservation->setParameterId(0, sensorOffset->id());
      optimizer.addEdge(landmarkObservation);
    }
    cerr << "Number of observations added: " << simulator.landmarkObservations().size() << endl;
    cerr << "done." << endl;

    std::stringstream dirStream;
    dirStream << cfg.outputBaseDir << "/test_" << testIdx;
    const std::string testDir = dirStream.str();
    std::filesystem::create_directories(testDir);

    VertexSE2* firstRobotPose = dynamic_cast<VertexSE2*>(optimizer.vertex(0));
    firstRobotPose->setFixed(true);
    optimizer.setVerbose(optCfg.verbose);

    optimizer.save((testDir + "/twb_before.g2o").c_str());
    simulator.saveGroundTruth((testDir + "/twb_gt.g2o").c_str());

    std::ofstream summary(testDir + "/kernel_summary.csv");
    summary << "kernel,chi2,mean_pose_translation_error\n";

    bool ggdActive = false;

    for (const GpsKernelSpec& spec : cfg.kernelSpecs) {
      resetEstimates(optimizer, simulator);
      assignGpsKernels(optimizer, spec, kernelParams,
                       spec.kind == GpsKernelKind::GGD ? &ggdActive : nullptr);

      cerr << "Optimizing test " << testIdx << " kernel " << spec.name << endl;
      runKernelOptimization(optimizer, spec, optCfg, ggdActive);

      const double chi2 = optimizer.activeChi2();
      const double meanErr = meanPoseTranslationError(optimizer, simulator);
      summary << spec.name << "," << chi2 << "," << meanErr << "\n";

      const std::string outPath = testDir + "/twb_" + spec.outputSuffix + ".g2o";
      optimizer.save(outPath.c_str());
      cerr << "  " << spec.name << ": chi2=" << chi2 << " mean_pose_err=" << meanErr << endl;
    }

    summary.close();
    optimizer.clear();

    if (simCfg.incrementGpsPeriod) {
      gps_period += 1;
    }
  }

  return 0;
}
