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
#include "edge_se2_pointxy.h"
#include "gnd_kernel.h"

using namespace std;
using namespace g2o;
using namespace g2o::tutorial;

namespace g2o::tutorial {
  void forceLinkTypesTutorialSlam2d();  // Forward declaration
}

std::mt19937 global_rng(42);

int main() {
  g2o::tutorial::forceLinkTypesTutorialSlam2d();
  // TODO simulate different sensor offset
  // simulate a robot observing landmarks while travelling on a grid
  SE2 sensorOffsetTransf(0.2, 0.1, -0.1);
  int numNodes = 300;
  MCSimulator simulator = MCSimulator(0.9, 20);
  simulator.simulate(numNodes, sensorOffsetTransf);

  /*********************************************************************************
   * creating the optimization problem
   ********************************************************************************/

  typedef BlockSolver<BlockSolverTraits<-1, -1> > SlamBlockSolver;
  typedef LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

  // allocating the optimizer
  SparseOptimizer optimizer;
  auto linearSolver = std::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);
  // OptimizationAlgorithmGaussNewton* solver =
  //     new OptimizationAlgorithmGaussNewton(
  //         std::make_unique<SlamBlockSolver>(std::move(linearSolver)));
  OptimizationAlgorithmLevenberg* solver =
      new OptimizationAlgorithmLevenberg(
          std::make_unique<SlamBlockSolver>(std::move(linearSolver)));

  optimizer.setAlgorithm(solver);

  // add the parameter representing the sensor offset
  ParameterSE2Offset* sensorOffset = new ParameterSE2Offset;
  sensorOffset->setOffset(sensorOffsetTransf);
  sensorOffset->setId(0);
  optimizer.addParameter(sensorOffset);

  // adding the odometry to the optimizer
  // first adding all the vertices
  cerr << "Optimization: Adding robot poses ... ";
  for (size_t i = 0; i < simulator.poses().size(); ++i) {
    const MCSimulator::GridPose& p = simulator.poses()[i];
    const SE2& t = p.simulatorPose;
    VertexSE2* robot = new VertexSE2;
    robot->setId(p.id);
    robot->setEstimate(t);
    optimizer.addVertex(robot);
  }
  cerr << "Number of poses added: " << simulator.poses().size() << endl;
  cerr << "done." << endl;

  // second add the odometry constraints
  cerr << "Optimization: Adding odometry measurements ... ";
  for (size_t i = 0; i < simulator.odometry().size(); ++i) {
    const MCSimulator::GridEdge& simEdge = simulator.odometry()[i];

    EdgeSE2* odometry = new EdgeSE2;
    odometry->vertices()[0] = optimizer.vertex(simEdge.from);
    odometry->vertices()[1] = optimizer.vertex(simEdge.to);
    odometry->setMeasurement(simEdge.simulatorTransf);
    odometry->setInformation(simEdge.information);
    optimizer.addEdge(odometry);
  }
  cerr << "Number of measurements added: " << simulator.odometry().size() << endl;
  cerr << "done." << endl;

  // add the landmark observations
  cerr << "Optimization: add landmark vertices ... ";
  for (size_t i = 0; i < simulator.landmarks().size(); ++i) {
    const MCSimulator::Landmark& l = simulator.landmarks()[i];
    VertexPointXY* landmark = new VertexPointXY;
    landmark->setId(l.id);
    landmark->setEstimate(l.simulatedPose);
    optimizer.addVertex(landmark);
  }
  cerr << "Number of landmarks added: " << simulator.landmarks().size() << endl;
  cerr << "done." << endl;

  cerr << "Optimization: add landmark observations ... ";
  for (size_t i = 0; i < simulator.landmarkObservations().size(); ++i) {
    const MCSimulator::LandmarkEdge& simEdge =
        simulator.landmarkObservations()[i];
    EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
    landmarkObservation->vertices()[0] = optimizer.vertex(simEdge.from);
    landmarkObservation->vertices()[1] = optimizer.vertex(simEdge.to);
    landmarkObservation->setMeasurement(simEdge.simulatorMeas);
    landmarkObservation->setInformation(simEdge.information);
    landmarkObservation->setParameterId(0, sensorOffset->id());
    auto rk = new g2o::GNDKernel(2.0, 8, 1e-12, 2.0*2.0);
    //rk->setDelta(1.0);  // Set based on expected noise
    landmarkObservation->setRobustKernel(rk);
    optimizer.addEdge(landmarkObservation);
  }
  cerr << "Number of observations added: " << simulator.landmarkObservations().size() << endl;
  cerr << "done." << endl;

  /*********************************************************************************
   * optimization
   ********************************************************************************/

  // dump initial state to the disk
  optimizer.save("cauchy_cauchy_before.g2o");

  // prepare and run the optimization
  // fix the first robot pose to account for gauge freedom
  VertexSE2* firstRobotPose = dynamic_cast<VertexSE2*>(optimizer.vertex(0));
  firstRobotPose->setFixed(true);
  optimizer.setVerbose(true);

  cerr << "Optimizing" << endl;
  optimizer.initializeOptimization();
  optimizer.optimize(1000);
  cerr << "done." << endl;

  optimizer.save("cauchy_cauchy_after.g2o");

  simulator.saveGroundTruth("cauchy_cauchy_gt.g2o");

  // freeing the graph memory
  optimizer.clear();

  return 0;
}
