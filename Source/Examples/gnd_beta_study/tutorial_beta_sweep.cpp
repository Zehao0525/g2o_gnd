// GND beta parameter sweep on the tutorial_w_references correlated-GPS problem.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "edge_se2.h"
#include "edge_se2_pointxy.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/sparse_optimizer_terminate_action.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/stuff/sampler.h"

#include "simulator.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "edge_se2_pointxy.h"
#include "gnd_kernel.h"
#include "edge_platform_loc_prior.h"

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;
using json = nlohmann::json;

namespace g2o::tutorial {
void forceLinkTypesTutorialSlam2d();
}  // namespace g2o::tutorial

namespace {

struct KernelParams {
  double gndBound = 3.0;
  double gndLnc = 1e-3;
  double gndTailStd = 2.0;
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
  int gndWarmupIterations = 0;
  int maxIterations = 200;
  double chi2RelTol = 1e-5;
  bool verbose = false;
};

struct BetaSweepConfig {
  SimulationConfig simulation;
  OptimizationConfig optimization;
  KernelParams kernels;
  std::vector<double> betaValues;
  bool includeGaussianReference = true;
  std::string outputBaseDir = "test_results/gnd_beta_study/correlated_gps";
};

struct PhaseMetrics {
  int outerIters = 0;
  int lmInners = 0;
  double timeS = 0.0;
  double chi2Start = 0.0;
  double chi2End = 0.0;
  int nonMonotoneSteps = 0;
  int itersToTolerance = -1;
  int chi2SignFlips = 0;
  double maxChi2Spike = 0.0;
  double chi2Range = 0.0;
  bool converged = false;
  bool hitMaxIters = false;
  bool solverFailed = false;
};

struct RunMetrics {
  std::string variant;
  double beta = 0.0;
  bool isGaussianReference = false;
  double chi2Initial = 0.0;
  double chi2AfterWarmup = 0.0;
  double chi2Final = 0.0;
  PhaseMetrics warmup;
  PhaseMetrics active;
  double meanPoseErr = 0.0;

  bool solverFailed() const { return warmup.solverFailed || active.solverFailed; }

  int totalOuterIters() const { return warmup.outerIters + active.outerIters; }

  int totalLmInners() const { return warmup.lmInners + active.lmInners; }

  double totalTimeS() const { return warmup.timeS + active.timeS; }

  double chi2GainWarmup() const { return chi2Initial - chi2AfterWarmup; }

  double chi2GainActive() const { return chi2AfterWarmup - chi2Final; }
};

BetaSweepConfig loadConfig(const std::string& configPath) {
  std::ifstream in(configPath);
  if (!in) {
    throw std::runtime_error("cannot open config: " + configPath);
  }
  json j;
  in >> j;

  BetaSweepConfig cfg;

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
    cfg.optimization.gndWarmupIterations =
        o.value("gnd_warmup_iterations", cfg.optimization.gndWarmupIterations);
    cfg.optimization.maxIterations =
        o.value("max_iterations", o.value("gnd_active_iterations", cfg.optimization.maxIterations));
    cfg.optimization.chi2RelTol = o.value("chi2_rel_tol", cfg.optimization.chi2RelTol);
    cfg.optimization.verbose = o.value("verbose", cfg.optimization.verbose);
  }

  if (j.contains("kernels")) {
    const auto& k = j["kernels"];
    cfg.kernels.gndBound = k.value("gnd_bound", cfg.kernels.gndBound);
    cfg.kernels.gndLnc = k.value("gnd_lnc", cfg.kernels.gndLnc);
    cfg.kernels.gndTailStd = k.value("gnd_tail_std", cfg.kernels.gndTailStd);
  }

  if (j.contains("output")) {
    cfg.outputBaseDir = j["output"].value("base_dir", cfg.outputBaseDir);
  }

  cfg.includeGaussianReference = j.value("include_gaussian_reference", cfg.includeGaussianReference);

  if (!j.contains("beta_values") || !j["beta_values"].is_array() || j["beta_values"].empty()) {
    throw std::runtime_error("config must contain non-empty beta_values array");
  }
  for (const auto& b : j["beta_values"]) {
    cfg.betaValues.push_back(b.get<double>());
  }

  return cfg;
}

void assignGndKernels(SparseOptimizer& optimizer, double beta, const KernelParams& params,
                      bool* gndActivePtr) {
  for (auto& edgePair : optimizer.edges()) {
    auto* gps = dynamic_cast<EdgePlatformLocPrior*>(edgePair);
    if (!gps) {
      continue;
    }
    gps->setRobustKernel(new ToggelableGNDKernel(params.gndBound, beta, params.gndLnc,
                                                 params.gndTailStd, gndActivePtr));
  }
}

void clearGpsRobustKernels(SparseOptimizer& optimizer) {
  for (auto& edgePair : optimizer.edges()) {
    auto* gps = dynamic_cast<EdgePlatformLocPrior*>(edgePair);
    if (gps) {
      gps->setRobustKernel(nullptr);
    }
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

double currentChi2(SparseOptimizer& optimizer) {
  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  return optimizer.activeChi2();
}

bool relativeChi2GainBelowTol(double chi2Before, double chi2After, double relTol);

void writeTraceHeader(std::ostream& out);

void writeTraceRows(std::ostream& out, int testIdx, const std::string& variant, double beta,
                    const std::string& phase, double chi2Start,
                    const BatchStatisticsContainer& stats);

void accumulateBatchStats(const BatchStatisticsContainer& stats, double chi2Start, double relTol,
                          PhaseMetrics& metrics);

PhaseMetrics runOptimizationPhase(SparseOptimizer& optimizer, int maxIters, double chi2RelTol,
                                  bool collectStats, std::ostream* traceOut, int testIdx,
                                  const std::string& variant, double beta,
                                  const std::string& phase) {
  PhaseMetrics metrics;
  metrics.chi2Start = currentChi2(optimizer);
  if (maxIters <= 0) {
    metrics.chi2End = metrics.chi2Start;
    return metrics;
  }

  if (collectStats) {
    optimizer.setComputeBatchStatistics(true);
  }

  SparseOptimizerTerminateAction terminateAction;
  terminateAction.setGainThreshold(chi2RelTol);
  terminateAction.setMaxIterations(maxIters);
  optimizer.addPostIterationAction(&terminateAction);

  optimizer.initializeOptimization();

  const auto start = std::chrono::steady_clock::now();
  const int completed = optimizer.optimize(maxIters);
  metrics.timeS =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  optimizer.removePostIterationAction(&terminateAction);

  metrics.outerIters = completed;
  if (completed <= 0) {
    metrics.solverFailed = true;
    if (completed < 0) {
      metrics.outerIters = 0;
    }
  } else {
    metrics.hitMaxIters = completed >= maxIters;
  }

  if (collectStats) {
    const BatchStatisticsContainer& stats = optimizer.batchStatistics();
    accumulateBatchStats(stats, metrics.chi2Start, chi2RelTol, metrics);
    if (traceOut != nullptr) {
      writeTraceRows(*traceOut, testIdx, variant, beta, phase, metrics.chi2Start, stats);
    }
    optimizer.setComputeBatchStatistics(false);
  }

  metrics.chi2End = currentChi2(optimizer);
  metrics.converged =
      !metrics.solverFailed && metrics.itersToTolerance > 0 && !metrics.hitMaxIters;
  return metrics;
}

bool relativeChi2GainBelowTol(double chi2Before, double chi2After, double relTol) {
  if (chi2After <= 0.0) {
    return false;
  }
  const double gain = (chi2Before - chi2After) / chi2After;
  return gain >= 0.0 && gain < relTol;
}

void writeTraceHeader(std::ostream& out) {
  out << "test_idx,variant,beta,phase,iter,chi2,rel_gain,lm_inners,chi2_increased\n";
}

void writeTraceRows(std::ostream& out, int testIdx, const std::string& variant, double beta,
                    const std::string& phase, double chi2Start,
                    const BatchStatisticsContainer& stats) {
  auto writeRow = [&](int iter, double chi2, double relGain, int lmInners, bool increased) {
    out << testIdx << "," << variant << ",";
    if (variant == "gaussian") {
      out << ",";
    } else {
      out << beta << ",";
    }
    out << phase << "," << iter << "," << chi2 << ",";
    if (std::isfinite(relGain)) {
      out << relGain;
    }
    out << "," << lmInners << "," << (increased ? 1 : 0) << "\n";
  };

  writeRow(0, chi2Start, std::numeric_limits<double>::quiet_NaN(), 0, false);

  double prevChi2 = chi2Start;
  for (size_t i = 0; i < stats.size(); ++i) {
    const double chi2 = stats[i].chi2;
    const double relGain = (prevChi2 - chi2) / std::max(chi2, 1e-12);
    const bool increased = chi2 > prevChi2 * (1.0 + 1e-9);
    writeRow(static_cast<int>(i) + 1, chi2, relGain, stats[i].levenbergIterations, increased);
    prevChi2 = chi2;
  }
}

void accumulateBatchStats(const BatchStatisticsContainer& stats, double chi2Start, double relTol,
                          PhaseMetrics& metrics) {
  double prevChi2 = chi2Start;
  double minChi2 = chi2Start;
  double maxChi2 = chi2Start;
  int prevDeltaSign = 0;

  for (size_t i = 0; i < stats.size(); ++i) {
    const double chi2 = stats[i].chi2;
    minChi2 = std::min(minChi2, chi2);
    maxChi2 = std::max(maxChi2, chi2);
    metrics.lmInners += stats[i].levenbergIterations;

    if (chi2 > prevChi2 * (1.0 + 1e-9)) {
      ++metrics.nonMonotoneSteps;
      metrics.maxChi2Spike = std::max(metrics.maxChi2Spike, chi2 - prevChi2);
    }

    const double delta = chi2 - prevChi2;
    const int deltaSign = (delta > 1e-9) ? 1 : ((delta < -1e-9) ? -1 : 0);
    if (deltaSign != 0 && prevDeltaSign != 0 && deltaSign != prevDeltaSign) {
      ++metrics.chi2SignFlips;
    }
    if (deltaSign != 0) {
      prevDeltaSign = deltaSign;
    }

    if (metrics.itersToTolerance < 0 &&
        relativeChi2GainBelowTol(prevChi2, chi2, relTol)) {
      metrics.itersToTolerance = static_cast<int>(i) + 1;
    }

    prevChi2 = chi2;
  }

  metrics.chi2Range = maxChi2 - minChi2;
}

RunMetrics runGaussianReference(SparseOptimizer& optimizer, const Simulator& simulator,
                                const OptimizationConfig& optCfg, std::ostream& traceOut,
                                int testIdx) {
  RunMetrics metrics;
  metrics.variant = "gaussian";
  metrics.isGaussianReference = true;

  clearGpsRobustKernels(optimizer);
  metrics.chi2Initial = currentChi2(optimizer);

  metrics.active = runOptimizationPhase(optimizer, optCfg.maxIterations, optCfg.chi2RelTol,
                                        /*collectStats=*/true, &traceOut, testIdx, "gaussian", 0.0,
                                        "active");

  metrics.chi2AfterWarmup = metrics.chi2Initial;
  metrics.chi2Final = metrics.active.chi2End;
  metrics.meanPoseErr = meanPoseTranslationError(optimizer, simulator);
  return metrics;
}

RunMetrics runGndBeta(SparseOptimizer& optimizer, const Simulator& simulator, double beta,
                      const KernelParams& kernelParams, const OptimizationConfig& optCfg,
                      bool& gndActive, std::ostream& traceOut, int testIdx) {
  RunMetrics metrics;
  metrics.variant = "gnd";
  metrics.beta = beta;

  gndActive = false;
  assignGndKernels(optimizer, beta, kernelParams, &gndActive);
  metrics.chi2Initial = currentChi2(optimizer);

  metrics.warmup =
      runOptimizationPhase(optimizer, optCfg.gndWarmupIterations, optCfg.chi2RelTol,
                             /*collectStats=*/true, &traceOut, testIdx, "gnd", beta, "warmup");

  gndActive = true;
  metrics.active = runOptimizationPhase(optimizer, optCfg.maxIterations, optCfg.chi2RelTol,
                                        /*collectStats=*/true, &traceOut, testIdx, "gnd", beta,
                                        "active");

  metrics.chi2AfterWarmup = metrics.warmup.chi2End;
  metrics.chi2Final = metrics.active.chi2End;
  metrics.meanPoseErr = meanPoseTranslationError(optimizer, simulator);
  return metrics;
}

std::string outputTag(const RunMetrics& metrics) {
  if (metrics.isGaussianReference) {
    return "gaussian";
  }
  std::ostringstream oss;
  oss << metrics.beta;
  std::string s = oss.str();
  for (char& c : s) {
    if (c == '.') {
      c = 'p';
    }
  }
  return "beta_" + s;
}

void writeCsvHeader(std::ostream& out) {
  out << "test_idx,variant,beta,mean_pose_translation_error,"
      << "chi2_initial,chi2_after_warmup,chi2_final,chi2_gain_warmup,chi2_gain_active,"
      << "warmup_outer_iters,warmup_lm_inners,warmup_time_s,warmup_non_monotone_steps,"
      << "warmup_iters_to_tol,warmup_converged,warmup_hit_max_iters,warmup_chi2_sign_flips,"
      << "warmup_max_chi2_spike,warmup_chi2_range,warmup_solver_failed,"
      << "active_outer_iters,active_lm_inners,active_time_s,active_non_monotone_steps,"
      << "active_iters_to_tol,active_converged,active_hit_max_iters,active_chi2_sign_flips,"
      << "active_max_chi2_spike,active_chi2_range,active_solver_failed,"
      << "total_outer_iters,total_lm_inners,total_time_s,solver_failed\n";
}

void writePhaseColumns(std::ostream& out, const PhaseMetrics& phase) {
  out << phase.outerIters << "," << phase.lmInners << "," << phase.timeS << ","
      << phase.nonMonotoneSteps << "," << phase.itersToTolerance << "," << phase.converged << ","
      << phase.hitMaxIters << "," << phase.chi2SignFlips << "," << phase.maxChi2Spike << ","
      << phase.chi2Range << "," << phase.solverFailed << ",";
}

void writeCsvRow(std::ostream& out, int testIdx, const RunMetrics& metrics) {
  out << testIdx << "," << metrics.variant << ",";
  if (metrics.isGaussianReference) {
    out << ",";
  } else {
    out << metrics.beta << ",";
  }
  out << metrics.meanPoseErr << ","
      << metrics.chi2Initial << "," << metrics.chi2AfterWarmup << "," << metrics.chi2Final << ","
      << metrics.chi2GainWarmup() << "," << metrics.chi2GainActive() << ",";
  writePhaseColumns(out, metrics.warmup);
  writePhaseColumns(out, metrics.active);
  out << metrics.totalOuterIters() << "," << metrics.totalLmInners() << "," << metrics.totalTimeS()
      << "," << metrics.solverFailed() << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();

  const std::string defaultConfigPath =
      "Source/Examples/gnd_beta_study/beta_sweep_config.json";
  const std::string configPath = (argc > 1) ? argv[1] : defaultConfigPath;
  const BetaSweepConfig cfg = loadConfig(configPath);

  const SimulationConfig& simCfg = cfg.simulation;
  const OptimizationConfig& optCfg = cfg.optimization;
  const KernelParams& kernelParams = cfg.kernels;

  std::filesystem::create_directories(cfg.outputBaseDir);
  std::ofstream aggregate(cfg.outputBaseDir + "/beta_sweep_aggregate.csv");
  writeCsvHeader(aggregate);
  std::ofstream traceOut(cfg.outputBaseDir + "/optimization_trace.csv");
  writeTraceHeader(traceOut);

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
    optimizer.setAlgorithm(new OptimizationAlgorithmLevenberg(
        std::make_unique<SlamBlockSolver>(std::move(linearSolver))));

    ParameterSE2Offset* sensorOffset = new ParameterSE2Offset;
    sensorOffset->setOffset(simCfg.sensorOffset);
    sensorOffset->setId(0);
    optimizer.addParameter(sensorOffset);

    for (size_t i = 0; i < simulator.poses().size(); ++i) {
      const Simulator::GridPose& p = simulator.poses()[i];
      VertexSE2* robot = new VertexSE2;
      robot->setId(p.id);
      robot->setEstimate(p.simulatorPose);
      optimizer.addVertex(robot);

      if (static_cast<int>(i) % gps_period == 0) {
        if (Sampler::uniformRand(0.0, 1.0) > simCfg.transitionProb) {
          gps_degree = Sampler::uniformRand(-M_PI, M_PI);
        }
        const Eigen::Vector2d gps_noise(
            simCfg.gpsStd * sqrt(2) * cos(gps_degree),
            simCfg.gpsStd * sqrt(2) * sin(gps_degree));

        Eigen::Matrix<double, 2, 2> cov = Eigen::Matrix<double, 2, 2>::Zero();
        cov(0, 0) = simCfg.gpsStd * simCfg.gpsStd;
        cov(1, 1) = simCfg.gpsStd * simCfg.gpsStd;

        auto* gps = new EdgePlatformLocPrior;
        gps->vertices()[0] = robot;
        gps->setInformation(cov.inverse());
        gps->setMeasurement((p.truePose * simCfg.sensorOffset).toVector().head<2>() + gps_noise);
        gps->setParameterId(0, sensorOffset->id());
        optimizer.addEdge(gps);
      }
    }

    for (size_t i = 0; i < simulator.odometry().size(); ++i) {
      const Simulator::GridEdge& simEdge = simulator.odometry()[i];
      auto* odometry = new EdgeSE2;
      odometry->vertices()[0] = optimizer.vertex(simEdge.from);
      odometry->vertices()[1] = optimizer.vertex(simEdge.to);
      odometry->setMeasurement(simEdge.simulatorTransf);
      odometry->setInformation(simEdge.information);
      optimizer.addEdge(odometry);
    }

    for (size_t i = 0; i < simulator.landmarks().size(); ++i) {
      const Simulator::Landmark& l = simulator.landmarks()[i];
      auto* landmark = new VertexPointXY;
      landmark->setId(l.id);
      landmark->setEstimate(l.simulatedPose);
      optimizer.addVertex(landmark);
    }

    for (size_t i = 0; i < simulator.landmarkObservations().size(); ++i) {
      const Simulator::LandmarkEdge& simEdge = simulator.landmarkObservations()[i];
      auto* landmarkObservation = new EdgeSE2PointXY;
      landmarkObservation->vertices()[0] = optimizer.vertex(simEdge.from);
      landmarkObservation->vertices()[1] = optimizer.vertex(simEdge.to);
      landmarkObservation->setMeasurement(simEdge.simulatorMeas);
      landmarkObservation->setInformation(simEdge.information);
      landmarkObservation->setParameterId(0, sensorOffset->id());
      optimizer.addEdge(landmarkObservation);
    }

    std::stringstream dirStream;
    dirStream << cfg.outputBaseDir << "/test_" << testIdx;
    const std::string testDir = dirStream.str();
    std::filesystem::create_directories(testDir);

    dynamic_cast<VertexSE2*>(optimizer.vertex(0))->setFixed(true);
    optimizer.setVerbose(optCfg.verbose);
    optimizer.save((testDir + "/before.g2o").c_str());
    simulator.saveGroundTruth((testDir + "/gt.g2o").c_str());

    std::ofstream summary(testDir + "/beta_summary.csv");
    writeCsvHeader(summary);

    bool gndActive = false;

    if (cfg.includeGaussianReference) {
      resetEstimates(optimizer, simulator);
      RunMetrics gaussianRun =
          runGaussianReference(optimizer, simulator, optCfg, traceOut, testIdx);
      writeCsvRow(summary, testIdx, gaussianRun);
      writeCsvRow(aggregate, testIdx, gaussianRun);
      optimizer.save((testDir + "/" + outputTag(gaussianRun) + ".g2o").c_str());
      cerr << "test " << testIdx << " gaussian reference" << endl;
      cerr << "  chi2=" << gaussianRun.chi2Final
           << " mean_pose_err=" << gaussianRun.meanPoseErr
           << " time_s=" << gaussianRun.totalTimeS()
           << " lm_inners=" << gaussianRun.totalLmInners() << endl;
    }

    for (double beta : cfg.betaValues) {
      resetEstimates(optimizer, simulator);
      RunMetrics gndRun =
          runGndBeta(optimizer, simulator, beta, kernelParams, optCfg, gndActive, traceOut, testIdx);
      writeCsvRow(summary, testIdx, gndRun);
      writeCsvRow(aggregate, testIdx, gndRun);
      optimizer.save((testDir + "/" + outputTag(gndRun) + ".g2o").c_str());
      cerr << "test " << testIdx << " beta=" << beta << endl;
      cerr << "  chi2=" << gndRun.chi2Final << " mean_pose_err=" << gndRun.meanPoseErr
           << " time_s=" << gndRun.totalTimeS() << " lm_inners=" << gndRun.totalLmInners() << endl;
    }

    summary.close();
    optimizer.clear();

    if (simCfg.incrementGpsPeriod) {
      gps_period += 1;
    }
  }

  aggregate.close();
  traceOut.close();
  return 0;
}
