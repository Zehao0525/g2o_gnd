// Implementation of SlamSystemBase (included from slam_system_base.hpp).
// Do not compile this translation unit on its own.

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_block_matrix.h"

namespace g2o {

template <typename VertexType, typename EdgeType>
SlamSystemBase<VertexType, EdgeType>::SlamSystemBase(const std::string& filename) {
  optimizer_ = std::make_unique<SparseOptimizer>();
  loadConfig(filename);
  setupOptimizer();
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::loadConfig(const std::string& filename) {
  std::ifstream f(filename);
  if (!f) {
    throw std::runtime_error("Cannot open SLAM config file: " + filename);
  }
  nlohmann::json j;
  f >> j;

  verbose_ = j.value("verbose", false);
  optPeriod_ = j.value("optimization_period", 100);
  optimizationAlg_ = j.value("optimization_algorithm", "GaussNewton");
  optCountProcess_ = j["optimize_count"].value("process", 10);
  optCountStop_ = j["optimize_count"].value("stop", 10);
  optCountStopFix_ = j["optimize_count"].value("stop_fixed", 10);

  if (verbose_) {
    std::cout << "- Reading all other parameters.\n";
    std::cout << "- optimizationAlg_ = " << optimizationAlg_ << '\n';
    std::cout << "- optPeriod_ = " << optPeriod_ << '\n';
    std::cout << "- optCountProcess_ = " << optCountProcess_ << '\n';
    std::cout << "- optCountStop_ = " << optCountStop_ << '\n';
    std::cout << "- optCountStopFix_ = " << optCountStopFix_ << '\n';
  }
}

template <typename VertexType, typename EdgeType>
int SlamSystemBase<VertexType, EdgeType>::optimize(int maximumNumberOfOptimizationSteps) {
  if (verbose_) {
    optimizer_->setVerbose(true);
  }
  optimizer_->initializeOptimization();
  const int numIterations = optimizer_->optimize(maximumNumberOfOptimizationSteps);
  if (verbose_) {
    std::cout << "Final chi2: " << optimizer_->activeChi2() << '\n';
    std::cout << "Num Iterations: " << numIterations << '\n';
    std::cout << "Number of vertices: " << optimizer_->vertices().size() << '\n';
    std::cout << "Number of edges: " << optimizer_->edges().size() << '\n';
  }
  onAfterOptimize();
  return numIterations;
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::setFixOlderPlatformVertices(
    double unfixedTimeWindow) {
  unfixedTimeWindow_ = unfixedTimeWindow;
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::platformEstimateMarginals(
    EstimateType& x, CovarianceType& P) {
  if (!currentPlatformVertex_) {
    std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                 "currentPlatformVertex_ is null.\n";
    P.setZero();
    return;
  }

  x = currentPlatformVertex_->estimate();
  if (currentPlatformVertex_->fixed()) {
    std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                 "current vertex is fixed; skipping marginals.\n";
    P.setZero();
    return;
  }

  SparseBlockMatrix<MatrixX> spinv;
  const int idx = currentPlatformVertex_->hessianIndex();
  bool success = false;

  try {
    success = optimizer_->computeMarginals(spinv, currentPlatformVertex_);
    if (!success) {
      std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                   "computeMarginals reported failure.\n";
      P.setZero();
      return;
    }
  } catch (const std::exception& e) {
    std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                 "computeMarginals exception: "
              << e.what() << '\n';
    P.setZero();
    return;
  } catch (...) {
    std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                 "computeMarginals unknown error.\n";
    P.setZero();
    return;
  }

  const auto* block = spinv.block(idx, idx);
  if (block) {
    P = block->template topLeftCorner<Dimension, Dimension>();
  } else {
    std::cerr << "[WARN] SlamSystemBase::platformEstimateMarginals: "
                 "null covariance block.\n";
    P.setZero();
  }
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::platformEstimate(EstimateType& x) const {
  if (!currentPlatformVertex_) {
    std::cerr << "[WARN] SlamSystemBase::platformEstimate: "
                 "currentPlatformVertex_ is null.\n";
    return;
  }
  x = currentPlatformVertex_->estimate();
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::processEvents(
    const tutorial::EventPtrVector& events) {
  for (const auto& event : events) {
    if (!event) {
      std::cerr << "[WARN] SlamSystemBase::processEvents: "
                   "null event pointer; skipping.\n";
      continue;
    }
    processEvent(*event);
  }
  if (lastOptStep_ + optPeriod_ <= stepNumber_) {
    optimize(optCountProcess_);
    lastOptStep_ = stepNumber_;
  }
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::setVerbose(bool verbose) {
  verbose_ = verbose;
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::saveOptimizerResults(
    const std::string& fileName) const {
  optimizer_->save(fileName.c_str());
}

template <typename VertexType, typename EdgeType>
void SlamSystemBase<VertexType, EdgeType>::setupOptimizer() {
  auto linearSolver =
      std::make_unique<LinearSolverEigen<SlamBlockSolver::PoseMatrixType>>();
  // setAlgorithm takes a raw pointer and owns it (deleted via g2o's release()
  // when G2O_DELETE_IMPLICITLY_OWNED_OBJECTS is enabled — true in our builds).
  if (optimizationAlg_ == "GaussNewton") {
    auto algorithm = std::make_unique<OptimizationAlgorithmGaussNewton>(
        std::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    optimizer_->setAlgorithm(algorithm.release());
  } else if (optimizationAlg_ == "LevenbergMarquardt") {
    auto algorithm = std::make_unique<OptimizationAlgorithmLevenberg>(
        std::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    optimizer_->setAlgorithm(algorithm.release());
  } else {
    throw std::runtime_error("Unknown optimization_algorithm: " + optimizationAlg_);
  }
  platformVertices_.clear();
  processModelEdges_.clear();
}

}  // namespace g2o
