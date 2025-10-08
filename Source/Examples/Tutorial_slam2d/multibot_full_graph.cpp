// se3_optimize_g2o.cpp
// Reads a G2O factor graph with SE3 vertices/edges, optimizes it with Levenberg–Marquardt
// using an Eigen linear solver, and writes the optimized graph back out.
// 
// Defaults:
//   Input : test_data/test1_new_data/test1_new.g2o
//   Output: test_results/multirobot/fullGraph.g2o
//
// Build (CMake):
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//
// CMakeLists.txt (drop this alongside the .cpp if you want a project):
// -------------------------------------------------------------------
// cmake_minimum_required(VERSION 3.14)
// project(se3_optimize_g2o LANGUAGES CXX)
// set(CMAKE_CXX_STANDARD 17)
// set(CMAKE_CXX_STANDARD_REQUIRED ON)
// find_package(Eigen3 REQUIRED)
// find_package(g2o REQUIRED COMPONENTS core stuff types_slam3d solver_eigen)
// add_executable(se3_optimize_g2o se3_optimize_g2o.cpp)
// target_link_libraries(se3_optimize_g2o PRIVATE g2o::core g2o::stuff g2o::types_slam3d g2o::solver_eigen Eigen3::Eigen)
// -------------------------------------------------------------------
//
// Quick build (no CMake, if pkg-config works):
//   g++ -O3 -std=c++17 se3_optimize_g2o.cpp -o se3_optimize_g2o \
//       $(pkg-config --cflags g2o) \
//       $(pkg-config --libs g2o) \
//       -lg2o_core -lg2o_stuff -lg2o_types_slam3d -lg2o_solver_eigen
//
// Usage:
//   ./se3_optimize_g2o [input.g2o] [output.g2o] [iters]
//   (Arguments optional; defaults shown above; iters default = 100)

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/stuff/logger.h>


// #include "g2o/core/block_solver.h"
// #include "g2o/core/factory.h"
// #include "g2o/core/optimization_algorithm_factory.h"
// #include "g2o/core/optimization_algorithm_levenberg.h"
// #include "g2o/core/sparse_optimizer.h"
// #include "g2o/solvers/eigen/linear_solver_eigen.h"

#include <Eigen/Core>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <limits>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string input_path  = "test_data/test1_new_data/test1_new_modified.g2o";
  std::string output_path = "test_results/multirobot/fullGraph.g2o";
  int iters = 100;

  if (argc > 1) input_path  = argv[1];
  if (argc > 2) output_path = argv[2];
  if (argc > 3) iters = std::max(1, std::atoi(argv[3]));

  // Ensure output directory exists
  try {
    fs::path outp(output_path);
    if (outp.has_parent_path()) fs::create_directories(outp.parent_path());
  } catch (const std::exception& e) {
    std::cerr << "[Error] Could not create output directories: " << e.what() << "\n";
    return 1;
  }

  // Set up optimizer
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);

  // Per user request: exact optimizer/solver setup
  // -------------------------------------------------------------
  // auto linearSolver = std::make_unique<LinearSolverEigen<BlockSolver<BlockSolverTraits<-1, -1>>::PoseMatrixType>>();
  // optimizer_->setAlgorithm(new OptimizationAlgorithmLevenberg(
  //             std::make_unique<BlockSolver<BlockSolverTraits<-1, -1>>>(std::move(linearSolver))
  //         ));
  // -------------------------------------------------------------
  using Block = g2o::BlockSolver< g2o::BlockSolverTraits<-1, -1> >;
  using LinearSolver = g2o::LinearSolverEigen<Block::PoseMatrixType>;
  auto linearSolver = std::make_unique<LinearSolver>();
  optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<Block>(std::move(linearSolver))
  ));

  // Load the .g2o graph
  std::cerr << "[Info] Loading: " << input_path << "\n";
  if (!optimizer.load(input_path.c_str())) {
    std::cerr << "[Error] Failed to load graph from '" << input_path << "'\n";
    return 1;
  }

  const auto& vmap = optimizer.vertices();
  const auto ne = optimizer.edges().size();
  std::cerr << "[Info] Loaded vertices: " << vmap.size() << ", edges: " << ne << "\n";
  if (vmap.empty()) {
    std::cerr << "[Error] Graph has no vertices.\n";
    return 1;
  }

  // Ensure the graph is well-posed: fix at least one vertex if none are fixed
  bool anyFixed = false;
  for (const auto& kv : vmap) {
    auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(kv.second);
    if (v && v->fixed()) { anyFixed = true; break; }
  }
  if (!anyFixed) {
    // Pick the smallest ID vertex to fix
    int minId = std::numeric_limits<int>::max();
    g2o::OptimizableGraph::Vertex* minV = nullptr;
    for (const auto& kv : vmap) {
      if (kv.first < minId) { minId = kv.first; minV = static_cast<g2o::OptimizableGraph::Vertex*>(kv.second); }
    }
    if (minV) {
      minV->setFixed(true);
      std::cerr << "[Info] No fixed vertex found. Fixing vertex id=" << minId << " to anchor the gauge.\n";
    }
  }

  // Optimize
  std::cerr << "[Info] Initializing optimization (iters=" << iters << ")...\n";
  optimizer.initializeOptimization();
  int its_done = optimizer.optimize(iters);
  if (its_done <= 0) {
    std::cerr << "[Warning] Optimizer returned " << its_done << ", results may be unchanged.\n";
  } else {
    std::cerr << "[Info] Optimization complete: performed " << its_done << " iterations.\n";
  }

  // Save optimized graph
  std::cerr << "[Info] Saving optimized graph to: " << output_path << "\n";
  if (!optimizer.save(output_path.c_str())) {
    std::cerr << "[Error] Failed to save graph to '" << output_path << "'\n";
    return 1;
  }

  std::cerr << "[Done] Wrote optimized graph to '" << output_path << "'\n";
  return 0;
}
