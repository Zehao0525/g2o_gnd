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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "g2o/core/block_solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "events_base.h"

namespace g2o {

using SlamBlockSolver = BlockSolver<BlockSolverTraits<-1, -1>>;
using SlamLinearSolver = LinearSolverEigen<SlamBlockSolver::PoseMatrixType>;

/**
 * Templated abstract base for a single-robot factor-graph SLAM system.
 * Method bodies live in slam_system_base_new.inl.
 */
template <typename VertexType, typename EdgeType>
class SlamSystemBase {
 public:
  using EstimateType = typename VertexType::EstimateType;
  static constexpr int Dimension = VertexType::Dimension;
  using CovarianceType = Eigen::Matrix<double, Dimension, Dimension>;

  /** Loads JSON config then wires the optimizer (RAII: usable after construction). */
  explicit SlamSystemBase(const std::string& filename);
  virtual ~SlamSystemBase() = default;

  SlamSystemBase(const SlamSystemBase&) = delete;
  SlamSystemBase& operator=(const SlamSystemBase&) = delete;
  SlamSystemBase(SlamSystemBase&&) = delete;
  SlamSystemBase& operator=(SlamSystemBase&&) = delete;

  /** Optimize graph; returns number of iterations performed. */
  int optimize(int maximumNumberOfOptimizationSteps);

  /** Access the optimizer (always valid after construction). */
  SparseOptimizer& optimizer() { return *optimizer_; }

  /** Access the optimizer (read-only; always valid after construction). */
  const SparseOptimizer& optimizer() const { return *optimizer_; }

  void setFixOlderPlatformVertices(double unfixedTimeWindow);

  /**
   * Platform pose estimate and marginal covariance (tangent / minimal coords).
   * Pose type follows the vertex (`SE2`, `Isometry3d`, …).
   * Covariance is Dimension×Dimension (3 for SE2, 6 for SE3).
   */
  void platformEstimateMarginals(EstimateType& x, CovarianceType& P);

  /** Pose only (no marginals). */
  void platformEstimate(EstimateType& x) const;

  /** Process an event vector (does not mutate the vector itself). */
  void processEvents(const tutorial::EventPtrVector& events);

  void setVerbose(bool verbose);

  /** Write the optimizer graph to a g2o file. */
  void saveOptimizerResults(const std::string& fileName) const;

  /** Initialize and start the SLAM system. */
  virtual void start() = 0;

  /** Stop the SLAM system and finalise result accumulation. */
  virtual void stop() = 0;

 protected:
  /** Read SLAM parameters from a JSON config file. Throws on I/O or parse failure. */
  void loadConfig(const std::string& filename);

  /** Wire the chosen algorithm; also resets platform / process-model containers. */
  void setupOptimizer();

  /** Process a single event. */
  virtual void processEvent(tutorial::Event& event) = 0;

  /**
   * Hook after every optimization pass via optimize().
   * Default: no-op. Derived classes can override.
   */
  virtual void onAfterOptimize() {}

  // Prefer in-class defaults; loadConfig overwrites values from JSON.
  bool verbose_ = false;

  int stepNumber_ = 0;
  int lastOptStep_ = -1;
  double currentTime_ = 0.0;
  bool initialized_ = false;
  bool componentsReady_ = false;

  int optPeriod_ = 100;
  std::string optimizationAlg_;
  int optCountProcess_ = 10;
  int optCountStop_ = 10;
  int optCountStopFix_ = 10;

  std::unique_ptr<SparseOptimizer> optimizer_;

  std::vector<VertexType*> platformVertices_;
  int vertexId_ = -1;

  std::vector<EdgeType*> processModelEdges_;
  int numProcessModelEdges_ = 0;

  /// Time window of recent platform poses kept unfixed (seconds). 0 = unset / unused.
  double unfixedTimeWindow_ = 0.0;

  VertexType* currentPlatformVertex_ = nullptr;
};

}  // namespace g2o

#include "slam_system_base_new.inl"
