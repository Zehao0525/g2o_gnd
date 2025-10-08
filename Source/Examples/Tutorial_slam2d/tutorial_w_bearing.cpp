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
#include <filesystem>

#include "edge_se2.h"
#include "edge_se2_pointxy.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/stuff/sampler.h"

#include "simulator_mc.h"
#include "simulator.h"
#include "types_tutorial_slam2d.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"
#include "edge_se2_pointxy.h"
#include "gnd_kernel.h"
#include "edge_platform_bearing_prior.h"
#include "edge_platform_loc_prior.h"

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
  int seed = 37;
  //MCSimulator simulator = MCSimulator(0.95, seed);

  // bearing noise
  double bearing_std_degrees = 5;
  double bearing_std = bearing_std_degrees * M_PI / 180;
  double transition_prob = 0.95;
  double bearing_mc_sign = 1;

  double gps_std = 2.5;
  double gps_degree = Sampler::uniformRand(-M_PI, M_PI);

  bool gndActive = false;

  int gps_period = 30;
  bool increment_gps_period = false;

  /*********************************************************************************
   * creating the optimization problem
   ********************************************************************************/
  int num_tests = 30;
  for(int testIdx=0;testIdx<num_tests;testIdx++){
    Simulator simulator = Simulator(seed+testIdx);
    simulator.simulate(numNodes, sensorOffsetTransf);
    Sampler::seedRand(seed+testIdx);
    gndActive = false;

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
      //const MCSimulator::GridPose& p = simulator.poses()[i];
      const Simulator::GridPose& p = simulator.poses()[i];
      const SE2& t = p.simulatorPose;
      VertexSE2* robot = new VertexSE2;
      robot->setId(p.id);
      robot->setEstimate(t);
      optimizer.addVertex(robot);


      // Right I'm adding a real bearing every 20 vertecies
      if(false){//(i+10)%20 == 0){
        EdgePlatformBearingPrior* bearing = new EdgePlatformBearingPrior;
        bearing->vertices()[0] = robot;

        if(Sampler::uniformRand(0.0, 1.0) > transition_prob){
          bearing_mc_sign *= -1;
        }
        //double bearing_noise =  bearing_mc_sign *  bearing_std;// Sampler::gaussRand(0.0,bearing_std);//  
        double bearing_noise =  Sampler::gaussRand(0.0,bearing_std);//  


        Eigen::Matrix<double, 1, 1> info;
        info(0, 0) = (bearing_std * bearing_std);
        bearing->setInformation(info.inverse());
        cerr << "Noisy bearing: " << normalize_theta((p.truePose*sensorOffsetTransf).toVector()[2] + bearing_noise) << endl;
        bearing->setMeasurement(normalize_theta((p.truePose*sensorOffsetTransf).toVector()[2] + bearing_noise));
        bearing->setParameterId(0, sensorOffset->id());
        auto rk = new g2o::ToggelableGNDKernel(2.0, 6, 1e-3, 2.0*2.0,&gndActive);
        bearing->setRobustKernel(rk);
        optimizer.addEdge(bearing);
      }

      // Right I'm adding a What about a GPS as well every 20 vertecies
      if(i%gps_period == 0){
        EdgePlatformLocPrior* gps = new EdgePlatformLocPrior;
        gps->vertices()[0] = robot;

        if(Sampler::uniformRand(0.0, 1.0) > transition_prob){
          gps_degree = Sampler::uniformRand(-M_PI, M_PI);
        }
        Eigen::Vector2d gps_noise =  Eigen::Vector2d(gps_std * sqrt(2) * cos(gps_degree), gps_std * sqrt(2) * sin(gps_degree));//  


        Eigen::Matrix<double, 2, 2> cov;
        cov.fill(0.);
        cov(0, 0) = (gps_std * gps_std);
        cov(1, 1) = (gps_std * gps_std);
        gps->setInformation(cov.inverse());
        cerr << "Noisy bearing: " << ((p.truePose*sensorOffsetTransf).toVector().head<2>() + gps_noise) << endl;
        gps->setMeasurement((p.truePose*sensorOffsetTransf).toVector().head<2>() + gps_noise);
        gps->setParameterId(0, sensorOffset->id());
        auto rk = new g2o::ToggelableGNDKernel(3.0, 6, 1e-3, 2.0*2.0, &gndActive);
        gps->setRobustKernel(rk);
        optimizer.addEdge(gps);
      }
    }
    cerr << "Number of poses added: " << simulator.poses().size() << endl;
    cerr << "done." << endl;

    // second add the odometry constraints
    cerr << "Optimization: Adding odometry measurements ... ";
    for (size_t i = 0; i < simulator.odometry().size(); ++i) {
      //const MCSimulator::GridEdge& simEdge = simulator.odometry()[i];
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

    // add the landmark observations
    cerr << "Optimization: add landmark vertices ... ";
    for (size_t i = 0; i < simulator.landmarks().size(); ++i) {
      //const MCSimulator::Landmark& l = simulator.landmarks()[i];
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
      //const MCSimulator::LandmarkEdge& simEdge = simulator.landmarkObservations()[i];
      EdgeSE2PointXY* landmarkObservation = new EdgeSE2PointXY;
      landmarkObservation->vertices()[0] = optimizer.vertex(simEdge.from);
      landmarkObservation->vertices()[1] = optimizer.vertex(simEdge.to);
      landmarkObservation->setMeasurement(simEdge.simulatorMeas);
      landmarkObservation->setInformation(simEdge.information);
      landmarkObservation->setParameterId(0, sensorOffset->id());
      //auto rk = new g2o::ToggelableGNDKernel(1.1, 6, 1e-12, 2.0*2.0,&gndActive);
      //landmarkObservation->setRobustKernel(rk);
      optimizer.addEdge(landmarkObservation);
    }
    cerr << "Number of observations added: " << simulator.landmarkObservations().size() << endl;
    cerr << "done." << endl;

    /*********************************************************************************
     * optimization
     ********************************************************************************/

    // dump initial state to the disk
      std::stringstream dirStream;
    dirStream << "test_results/exp1_test_results_3/test_" << testIdx;
    std::string testDir = dirStream.str();

    optimizer.save((testDir + "/twb_before.g2o").c_str());

    // prepare and run the optimization
    // fix the first robot pose to account for gauge freedom
    VertexSE2* firstRobotPose = dynamic_cast<VertexSE2*>(optimizer.vertex(0));
    firstRobotPose->setFixed(true);
    optimizer.setVerbose(true);



    // Ensure directory exists
    std::filesystem::create_directories(testDir);

    cerr << "Optimizing" << endl;
    optimizer.initializeOptimization();
    gndActive = false;
    optimizer.optimize(20);
    optimizer.save((testDir + "/twb_gauss.g2o").c_str());
    gndActive = true;
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cerr << "done." << endl;

    optimizer.save((testDir + "/twb_gnd.g2o").c_str());

    simulator.saveGroundTruth((testDir + "/twb_gt.g2o").c_str());

    // freeing the graph memory
    optimizer.clear();

    if(increment_gps_period) gps_period += 1;
  }

  return 0;
}
