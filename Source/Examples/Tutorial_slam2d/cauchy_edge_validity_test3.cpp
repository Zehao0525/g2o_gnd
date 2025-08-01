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
#include <iostream>
#include <random>

#include "edge_se2.h"
#include "edge_se2_pointxy.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "simulator_mc.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "rng_util.h"
#include "gnd_kernel.h"

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;

namespace g2o::tutorial {
  void forceLinkTypesTutorialSlam2d();  // Forward declaration
}

int main() {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();
  // Step 1: Create optimizer
  typedef BlockSolver<BlockSolverTraits<-1, -1>> SlamBlockSolver;
  typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

  auto linearSolver = std::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);

  OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(
      std::make_unique<SlamBlockSolver>(std::move(linearSolver)));

  SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  // Step 2: Load the graph
  std::ifstream ifs("cauchy_gauss_after.g2o");
  if (!ifs) {
    std::cerr << "Failed to open input file" << std::endl;
    return -1;
  }

  if (!optimizer.load(ifs)) {
    std::cerr << "Failed to load graph" << std::endl;
    return -1;
  }
  std::cout << "Loaded graph with " << optimizer.vertices().size() << " vertices and "
            << optimizer.edges().size() << " edges." << std::endl;

  // Step 3: Attach GNDKernel to all EdgeSE2PointXY
  for (auto edge : optimizer.edges()) {
    auto* obs = dynamic_cast<EdgeSE2PointXY*>(edge);
    if (obs) {
      auto rk = new GNDKernel(1.2, 8.0, 1e-3, 4.0);  // beta, scale, lnc, tail_std^2
      obs->setRobustKernel(rk);
    }
  }

  // Step 4: Fix the first pose (optional but usually needed)
  auto* firstPose = dynamic_cast<VertexSE2*>(optimizer.vertex(0));
  if (firstPose) {
    firstPose->setFixed(true);
  }

  // Step 5: Optimize
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(30);

  // Step 6: Save result
  optimizer.save("cauchy_cauchy_back_after.g2o");
  std::cout << "Saved optimized graph to cauchy_cauchy_back_after.g2o" << std::endl;

  return 0;
}
